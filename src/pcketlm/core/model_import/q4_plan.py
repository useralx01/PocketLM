"""Header-only Q4 artifact planning for large safetensors models."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path


Q4_FORMAT = "pcketlm-q4"
Q4_SCHEME = "per-channel-symmetric-v1"
FLOATING_SAFE_DTYPES = {"BF16", "F16", "F32", "F64"}


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


def _q4_estimate_for_shape(shape: list[int]) -> tuple[int, int, int]:
    numel = _numel_from_shape(shape)
    rows = int(shape[0]) if shape else 1
    packed_bytes = (numel + 1) // 2
    scale_bytes = rows * 2
    return packed_bytes + scale_bytes, packed_bytes, scale_bytes


def _is_expert_tensor_name(tensor_name: str) -> bool:
    lowered = tensor_name.lower()
    return ".experts." in lowered or "block_sparse_moe.experts" in lowered


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
    expert_original_bytes = 0
    expert_q4_bytes = 0
    non_expert_original_bytes = 0
    non_expert_q4_bytes = 0
    largest_tensors: list[dict] = []

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
            dtype = str(metadata.get("dtype", ""))
            shape = [int(value) for value in metadata.get("shape", [])]
            offsets = metadata.get("data_offsets") or [0, 0]
            original_bytes = int(offsets[1]) - int(offsets[0])
            total_original_bytes += original_bytes
            if dtype not in FLOATING_SAFE_DTYPES:
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
                }
            )

    largest_tensors.sort(key=lambda item: int(item["original_bytes"]), reverse=True)
    output_dir = output_dir.resolve() if output_dir is not None else model_dir.parent / "artifacts" / "q4"
    compression_ratio = 0.0 if total_original_bytes <= 0 else round(total_q4_bytes / total_original_bytes, 6)
    return {
        "format": Q4_FORMAT,
        "scheme": Q4_SCHEME,
        "source_model_dir": str(model_dir),
        "planned_output_dir": str(output_dir),
        "source_tensor_count": len(weight_map),
        "q4_tensor_count": q4_tensor_count,
        "skipped_tensor_count": skipped_tensor_count,
        "missing_shards": missing_shards,
        "ready_for_conversion": not missing_shards and q4_tensor_count > 0,
        "total_original_bytes": int(total_original_bytes),
        "estimated_total_q4_bytes": int(total_q4_bytes),
        "estimated_packed_bytes": int(total_packed_bytes),
        "estimated_scale_bytes": int(total_scale_bytes),
        "estimated_expert_original_bytes": int(expert_original_bytes),
        "estimated_expert_q4_bytes": int(expert_q4_bytes),
        "estimated_non_expert_original_bytes": int(non_expert_original_bytes),
        "estimated_non_expert_q4_bytes": int(non_expert_q4_bytes),
        "compression_ratio": compression_ratio,
        "disk_required_bytes_with_10pct_headroom": int(math.ceil(total_q4_bytes * 1.10)),
        "largest_tensors": largest_tensors[:10],
    }
