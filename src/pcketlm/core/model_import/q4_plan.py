"""Header-only Q4 artifact planning for large safetensors models."""

from __future__ import annotations

import ctypes
import json
import math
import re
import struct
from pathlib import Path


Q4_FORMAT = "pcketlm-q4"
Q4_SCHEME = "per-channel-symmetric-v1"
FLOATING_SAFE_DTYPES = {"BF16", "F16", "F32", "F64"}
FP8_DTYPES = {"F8_E4M3", "F8_E4M3FN", "F8_E5M2"}
SCALE_SUFFIXES = (".scale_inv", "_scale_inv", ".scale", "_scale")
_DTYPE_ALIASES = {
    "BFLOAT16": "BF16",
    "BFloat16": "BF16",
    "FLOAT16": "F16",
    "FP16": "F16",
    "HALF": "F16",
    "FLOAT32": "F32",
    "FP32": "F32",
    "FLOAT64": "F64",
    "FP64": "F64",
    "FP8_E4M3": "F8_E4M3",
    "FP8_E4M3FN": "F8_E4M3FN",
    "FLOAT8_E4M3": "F8_E4M3",
    "FLOAT8_E4M3FN": "F8_E4M3FN",
    "FP8_E5M2": "F8_E5M2",
    "FLOAT8_E5M2": "F8_E5M2",
}
_LAYER_EXPERT_RE = re.compile(r"(?:^|\.)layers\.(\d+)\..*\.experts\.(\d+)\.")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _weight_map(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in (payload.get("weight_map") or {}).items()}


def _read_safetensors_header(path: Path) -> dict[str, dict]:
    """Read safetensors metadata without touching tensor payload bytes."""
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"{path} is not a valid safetensors file: missing header size.")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_payload = handle.read(header_size)
    header = json.loads(header_payload.decode("utf-8"))
    return {str(key): value for key, value in header.items() if key != "__metadata__"}


def _numel_from_shape(shape: list[int]) -> int:
    return int(math.prod(shape)) if shape else 1


def _canonical_dtype(dtype: str) -> str:
    normalized = str(dtype).strip()
    upper = normalized.upper()
    return _DTYPE_ALIASES.get(normalized, _DTYPE_ALIASES.get(upper, upper))


def _is_fp8_dtype(dtype: str) -> bool:
    return _canonical_dtype(dtype) in FP8_DTYPES


def _dtype_byte_width(dtype: str) -> int | None:
    canonical = _canonical_dtype(dtype)
    if canonical in FP8_DTYPES or canonical in {"I8", "U8", "BOOL"}:
        return 1
    if canonical in {"BF16", "F16", "I16", "U16"}:
        return 2
    if canonical in {"F32", "I32", "U32"}:
        return 4
    if canonical in {"F64", "I64", "U64"}:
        return 8
    return None


def _q4_estimate_for_shape(shape: list[int]) -> tuple[int, int, int]:
    numel = _numel_from_shape(shape)
    rows = int(shape[0]) if shape else 1
    packed_bytes = (numel + 1) // 2
    scale_bytes = rows * 2
    return packed_bytes + scale_bytes, packed_bytes, scale_bytes


def _is_expert_tensor_name(tensor_name: str) -> bool:
    lowered = tensor_name.lower()
    return ".experts." in lowered or "block_sparse_moe.experts" in lowered


def _scale_partner_name(tensor_name: str, all_tensor_names: set[str]) -> str | None:
    for suffix in SCALE_SUFFIXES:
        if tensor_name.endswith(suffix):
            partner = tensor_name[: -len(suffix)]
            if partner in all_tensor_names:
                return partner
    return None


def _tensor_layer_and_expert(tensor_name: str) -> tuple[int | None, int | None]:
    expert_match = _LAYER_EXPERT_RE.search(tensor_name)
    if expert_match:
        return int(expert_match.group(1)), int(expert_match.group(2))
    layer_match = _LAYER_RE.search(tensor_name)
    if layer_match:
        return int(layer_match.group(1)), None
    return None, None


def _read_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _detect_system_ram_bytes() -> int | None:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except (AttributeError, OSError):
        return None
    return None


def _top_k_from_config(config: dict) -> int:
    for key in ("num_experts_per_tok", "moe_top_k", "top_k"):
        value = config.get(key)
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue
    return 2


def _total_experts_from_config(config: dict) -> int | None:
    for key in ("n_routed_experts", "num_local_experts", "num_experts"):
        value = config.get(key)
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue
    return None


def _recommend_path(has_fp8_weights: bool, total_fp8_runtime_bytes: int, active_layer_bytes: int, ram_bytes: int | None) -> dict:
    if not has_fp8_weights:
        return {
            "path": "compact_q4",
            "reason": "Source tensors are standard floating point, so the existing compact Q4 path remains the fit-first route.",
            "q4_requantization_warning": None,
        }
    if ram_bytes is None:
        return {
            "path": "FP8 native paged",
            "reason": "DeepSeek-style FP8 source is too large to assume full residency; use FP8 weights and scales with paged layer/expert residency.",
            "q4_requantization_warning": "Q4 re-quantization from FP8 is lossy and is not the recommended first path.",
        }
    if total_fp8_runtime_bytes <= int(ram_bytes * 0.75):
        return {
            "path": "FP8 native, no conversion needed",
            "reason": "FP8 weights plus scales fit comfortably in system RAM.",
            "q4_requantization_warning": "Q4 re-quantization from FP8 is lossy and unnecessary here.",
        }
    if active_layer_bytes <= int(ram_bytes * 0.5):
        return {
            "path": "FP8 native paged",
            "reason": "Full FP8 residency does not fit, but one active layer/expert working set can fit with paging.",
            "q4_requantization_warning": "Q4 re-quantization from FP8 is lossy and is not the recommended first path.",
        }
    return {
        "path": "stream FP8 from disk with paged residency",
        "reason": "Even the active FP8 working set is large for this machine, so the runtime should page FP8 tensors from disk.",
        "q4_requantization_warning": "Q4 re-quantization from FP8 is lossy and should be a fallback only.",
    }


def plan_model_dir_to_q4(model_dir: Path, output_dir: Path | None = None) -> dict:
    """Estimate Q4 artifact size from safetensors headers only.

    This catches missing shards and estimates compact artifact size before a
    conversion or runtime load attempts to touch giant tensor payloads.
    """
    model_dir = model_dir.resolve()
    weight_map = _weight_map(model_dir / "model.safetensors.index.json")
    tensors_by_shard: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        tensors_by_shard.setdefault(shard_name, []).append(tensor_name)

    missing_shards: list[str] = []
    total_original_bytes = 0
    total_q4_bytes = 0
    total_packed_bytes = 0
    total_scale_bytes = 0
    q4_tensor_count = 0
    skipped_tensor_count = 0
    fp8_tensor_count = 0
    fp8_scale_tensor_count = 0
    non_fp8_tensor_count = 0
    fp8_weight_bytes = 0
    fp8_scale_bytes = 0
    non_fp8_non_scale_bytes = 0
    fp8_dequant_bf16_bytes = 0
    dtype_bytes: dict[str, int] = {}
    dtype_tensor_counts: dict[str, int] = {}
    expert_original_bytes = 0
    expert_q4_bytes = 0
    non_expert_original_bytes = 0
    non_expert_q4_bytes = 0
    scale_pairs: list[dict] = []
    role_counts: dict[str, int] = {}
    layer_non_expert_runtime_bytes: dict[int, int] = {}
    layer_expert_runtime_bytes: dict[int, dict[int, int]] = {}
    persistent_runtime_bytes = 0
    largest_tensors: list[dict] = []
    all_tensor_names = set(weight_map)
    config = _read_config(model_dir)
    top_k = _top_k_from_config(config)
    total_experts = _total_experts_from_config(config)

    for shard_name, tensor_names in sorted(tensors_by_shard.items()):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            missing_shards.append(shard_name)
            continue
        header = _read_safetensors_header(shard_path)
        for tensor_name in sorted(tensor_names):
            metadata = header.get(tensor_name)
            if metadata is None:
                skipped_tensor_count += 1
                continue
            dtype = _canonical_dtype(str(metadata.get("dtype", "")))
            shape = [int(value) for value in metadata.get("shape", [])]
            offsets = metadata.get("data_offsets") or [0, 0]
            original_bytes = int(offsets[1]) - int(offsets[0])
            partner_name = _scale_partner_name(tensor_name, all_tensor_names)
            role = "scale_companion" if partner_name else "weight"
            if partner_name:
                scale_pairs.append({"scale": tensor_name, "weight": partner_name, "dtype": dtype, "shape": shape})
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + original_bytes
            dtype_tensor_counts[dtype] = dtype_tensor_counts.get(dtype, 0) + 1
            role_counts[role] = role_counts.get(role, 0) + 1
            total_original_bytes += original_bytes
            is_fp8 = _is_fp8_dtype(dtype)
            runtime_bytes = original_bytes
            layer_index, expert_index = _tensor_layer_and_expert(tensor_name)
            if runtime_bytes:
                if layer_index is None:
                    persistent_runtime_bytes += runtime_bytes
                elif expert_index is None:
                    layer_non_expert_runtime_bytes[layer_index] = (
                        layer_non_expert_runtime_bytes.get(layer_index, 0) + runtime_bytes
                    )
                else:
                    expert_bytes = layer_expert_runtime_bytes.setdefault(layer_index, {})
                    expert_bytes[expert_index] = expert_bytes.get(expert_index, 0) + runtime_bytes
            if is_fp8 and role == "weight":
                fp8_tensor_count += 1
                fp8_weight_bytes += original_bytes
                fp8_dequant_bf16_bytes += _numel_from_shape(shape) * 2
            elif role == "scale_companion":
                fp8_scale_tensor_count += 1
                fp8_scale_bytes += original_bytes
            else:
                non_fp8_tensor_count += 1
                non_fp8_non_scale_bytes += original_bytes
                fp8_dequant_bf16_bytes += original_bytes
            if dtype not in FLOATING_SAFE_DTYPES and not is_fp8:
                skipped_tensor_count += 1
                continue
            if role == "scale_companion":
                skipped_tensor_count += 1
                continue
            q4_bytes, packed_bytes, scale_bytes = _q4_estimate_for_shape(shape)
            q4_tensor_count += 1
            total_q4_bytes += q4_bytes
            total_packed_bytes += packed_bytes
            total_scale_bytes += scale_bytes
            if _is_expert_tensor_name(tensor_name):
                expert_original_bytes += original_bytes
                expert_q4_bytes += q4_bytes
            else:
                non_expert_original_bytes += original_bytes
                non_expert_q4_bytes += q4_bytes
            largest_tensors.append(
                {
                    "name": tensor_name,
                    "dtype": dtype,
                    "shape": shape,
                    "original_bytes": original_bytes,
                    "estimated_q4_bytes": q4_bytes,
                    "expert": _is_expert_tensor_name(tensor_name),
                    "role": role,
                }
            )

    largest_tensors.sort(key=lambda item: int(item["original_bytes"]), reverse=True)
    output_dir = output_dir.resolve() if output_dir is not None else model_dir.parent / "artifacts" / "q4"
    active_layer_estimates: list[dict] = []
    for layer_index in sorted(set(layer_non_expert_runtime_bytes) | set(layer_expert_runtime_bytes)):
        experts = layer_expert_runtime_bytes.get(layer_index, {})
        avg_expert_bytes = 0 if not experts else int(round(sum(experts.values()) / len(experts)))
        active_expert_bytes = avg_expert_bytes * min(top_k, len(experts) or top_k)
        non_expert_bytes = layer_non_expert_runtime_bytes.get(layer_index, 0)
        active_layer_estimates.append(
            {
                "layer": layer_index,
                "non_expert_fp8_and_scale_bytes": int(non_expert_bytes),
                "average_expert_fp8_and_scale_bytes": int(avg_expert_bytes),
                "active_expert_count": int(min(top_k, len(experts) or top_k)),
                "active_expert_fp8_and_scale_bytes": int(active_expert_bytes),
                "active_layer_fp8_and_scale_bytes": int(non_expert_bytes + active_expert_bytes),
                "observed_expert_count": len(experts),
            }
        )
    max_active_layer_bytes = max(
        (int(item["active_layer_fp8_and_scale_bytes"]) for item in active_layer_estimates),
        default=0,
    )
    avg_active_layer_bytes = (
        int(round(sum(int(item["active_layer_fp8_and_scale_bytes"]) for item in active_layer_estimates) / len(active_layer_estimates)))
        if active_layer_estimates
        else 0
    )
    max_active_layer_estimate = max(
        active_layer_estimates,
        key=lambda item: int(item["active_layer_fp8_and_scale_bytes"]),
        default=None,
    )
    estimated_fp8_runtime_bytes = fp8_weight_bytes + fp8_scale_bytes + non_fp8_non_scale_bytes
    estimated_fp8_paged_working_set_bytes = persistent_runtime_bytes + max_active_layer_bytes
    ram_bytes = _detect_system_ram_bytes()
    recommendation = _recommend_path(
        fp8_weight_bytes > 0,
        estimated_fp8_runtime_bytes,
        estimated_fp8_paged_working_set_bytes,
        ram_bytes,
    )
    compression_ratio = 0.0 if total_original_bytes <= 0 else round(total_q4_bytes / total_original_bytes, 6)
    return {
        "format": Q4_FORMAT,
        "scheme": Q4_SCHEME,
        "source_model_dir": str(model_dir),
        "planned_output_dir": str(output_dir),
        "source_tensor_count": len(weight_map),
        "source_shard_count": len(tensors_by_shard),
        "q4_tensor_count": q4_tensor_count,
        "skipped_tensor_count": skipped_tensor_count,
        "dtype_tensor_counts": dtype_tensor_counts,
        "dtype_bytes": dtype_bytes,
        "role_counts": role_counts,
        "fp8_tensor_count": fp8_tensor_count,
        "fp8_scale_tensor_count": fp8_scale_tensor_count,
        "non_fp8_tensor_count": non_fp8_tensor_count,
        "fp8_weight_bytes": int(fp8_weight_bytes),
        "fp8_scale_bytes": int(fp8_scale_bytes),
        "non_fp8_non_scale_bytes": int(non_fp8_non_scale_bytes),
        "scale_pair_count": len(scale_pairs),
        "scale_pairs_sample": scale_pairs[:10],
        "fp8_native": fp8_weight_bytes > 0,
        "missing_shards": missing_shards,
        "ready_for_conversion": not missing_shards and q4_tensor_count > 0 and fp8_weight_bytes == 0,
        "ready_for_fp8_runtime_planning": not missing_shards and fp8_weight_bytes > 0,
        "total_original_bytes": int(total_original_bytes),
        "estimated_total_q4_bytes": int(total_q4_bytes),
        "estimated_packed_bytes": int(total_packed_bytes),
        "estimated_scale_bytes": int(total_scale_bytes),
        "estimated_q4_from_source_bytes": int(total_q4_bytes),
        "estimated_q4_from_source_lossy": fp8_weight_bytes > 0,
        "estimated_fp8_repacked_artifact_bytes": int(estimated_fp8_runtime_bytes),
        "estimated_fp8_runtime_bytes": int(estimated_fp8_runtime_bytes),
        "estimated_bf16_dequantized_runtime_bytes": int(fp8_dequant_bf16_bytes),
        "persistent_fp8_and_scale_bytes": int(persistent_runtime_bytes),
        "max_active_layer_fp8_and_scale_bytes": int(max_active_layer_bytes),
        "avg_active_layer_fp8_and_scale_bytes": int(avg_active_layer_bytes),
        "max_active_layer_estimate": max_active_layer_estimate,
        "estimated_fp8_paged_working_set_bytes": int(estimated_fp8_paged_working_set_bytes),
        "system_ram_bytes": ram_bytes,
        "moe_top_k": top_k,
        "configured_total_experts": total_experts,
        "active_layer_estimates_sample": active_layer_estimates[:5],
        "estimated_expert_original_bytes": int(expert_original_bytes),
        "estimated_expert_q4_bytes": int(expert_q4_bytes),
        "estimated_non_expert_original_bytes": int(non_expert_original_bytes),
        "estimated_non_expert_q4_bytes": int(non_expert_q4_bytes),
        "compression_ratio": compression_ratio,
        "disk_required_bytes_with_10pct_headroom": int(math.ceil(total_q4_bytes * 1.10)),
        "recommended_path": recommendation["path"],
        "recommendation_reason": recommendation["reason"],
        "q4_requantization_warning": recommendation["q4_requantization_warning"],
        "conversion_blockers": (
            ["FP8-native source requires FP8-aware conversion/runtime planning; direct Q4 conversion is lossy and not recommended."]
            if fp8_weight_bytes > 0
            else []
        ),
        "largest_tensors": largest_tensors[:10],
    }
