"""Offline pcketlm Q4 streaming artifact builder.

The artifact stores Q4 bytes on disk and fp16/bf16 scales separately. Runtime
loads dequantize once into the normal tensor residency path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pcketlm.core.model_import.q4_plan import Q4_FORMAT, Q4_SCHEME, plan_model_dir_to_q4


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values [-8, 7] two per uint8 byte."""
    flat = values.detach().to(torch.int16).flatten()
    if flat.numel() == 0:
        return torch.empty((0,), dtype=torch.uint8)
    if int(flat.min().item()) < -8 or int(flat.max().item()) > 7:
        raise ValueError("int4 values must be in the range [-8, 7].")
    encoded = torch.bitwise_and(flat, 0x0F).to(torch.uint8)
    if encoded.numel() % 2:
        encoded = torch.cat([encoded, torch.zeros((1,), dtype=torch.uint8)])
    low = encoded[0::2]
    high = torch.bitwise_left_shift(encoded[1::2], 4)
    return torch.bitwise_or(low, high).contiguous()


def unpack_int4(packed: torch.Tensor, value_count: int) -> torch.Tensor:
    """Unpack signed int4 values from uint8 bytes."""
    bytes_flat = packed.detach().to(torch.uint8).flatten()
    low = torch.bitwise_and(bytes_flat, 0x0F)
    high = torch.bitwise_and(torch.bitwise_right_shift(bytes_flat, 4), 0x0F)
    unsigned = torch.empty((bytes_flat.numel() * 2,), dtype=torch.int16)
    unsigned[0::2] = low.to(torch.int16)
    unsigned[1::2] = high.to(torch.int16)
    signed = torch.where(unsigned >= 8, unsigned - 16, unsigned)
    return signed[:value_count].to(torch.int8).contiguous()


def quantize_tensor_to_q4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Quantize a tensor with per-output-channel symmetric Q4 scales."""
    original_shape = [int(value) for value in tensor.shape]
    working = tensor.detach().cpu().float()
    if working.ndim == 0:
        matrix = working.reshape(1, 1)
    elif working.ndim == 1:
        matrix = working.reshape(working.shape[0], 1)
    else:
        matrix = working.reshape(working.shape[0], -1)
    max_abs = matrix.abs().amax(dim=1)
    scales = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
    quantized = torch.round(matrix / scales[:, None]).clamp(-7, 7).to(torch.int8)
    packed = pack_int4(quantized.flatten())
    scale_tensor = scales.to(torch.float16).contiguous()
    metadata = {
        "shape": original_shape,
        "dtype": _catalog_dtype_for_tensor(tensor),
        "scheme": Q4_SCHEME,
        "numel": int(working.numel()),
        "packed_numel": int(packed.numel()),
        "scale_shape": [int(value) for value in scale_tensor.shape],
    }
    return packed, scale_tensor, metadata


def dequantize_q4_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: list[int],
    *,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize Q4 bytes back to a normal torch tensor."""
    value_count = int(math.prod(shape)) if shape else 1
    quantized = unpack_int4(packed, value_count).to(torch.float32)
    if not shape:
        matrix = quantized.reshape(1, 1)
    elif len(shape) == 1:
        matrix = quantized.reshape(shape[0], 1)
    else:
        matrix = quantized.reshape(shape[0], -1)
    restored = matrix * scales.detach().cpu().float().reshape(-1, 1)
    return restored.reshape(shape).to(dtype=dtype).contiguous()


def _catalog_dtype_for_tensor(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.bfloat16:
        return "BF16"
    if tensor.dtype == torch.float16:
        return "F16"
    if tensor.dtype == torch.float32:
        return "F32"
    return str(tensor.dtype).replace("torch.", "")


def _weight_map(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in (payload.get("weight_map") or {}).items()}


def quantize_model_dir_to_q4(model_dir: Path, output_dir: Path) -> dict:
    """Quantize all floating tensors in a safetensors model directory."""
    model_dir = model_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = _weight_map(model_dir / "model.safetensors.index.json")
    tensors_by_shard: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        tensors_by_shard.setdefault(shard_name, []).append(tensor_name)

    manifest = {
        "format": Q4_FORMAT,
        "scheme": Q4_SCHEME,
        "source_model_dir": str(model_dir),
        "tensors": {},
        "weight_map": {},
        "scale_map": {},
    }
    total_original_bytes = 0
    total_q4_bytes = 0

    for shard_name, tensor_names in sorted(tensors_by_shard.items()):
        q4_tensors: dict[str, torch.Tensor] = {}
        scale_tensors: dict[str, torch.Tensor] = {}
        q4_shard_name = shard_name.replace(".safetensors", ".q4.safetensors")
        scale_shard_name = shard_name.replace(".safetensors", ".scales.safetensors")
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
            for tensor_name in sorted(tensor_names):
                tensor = handle.get_tensor(tensor_name)
                if not tensor.dtype.is_floating_point:
                    continue
                packed, scales, metadata = quantize_tensor_to_q4(tensor)
                q4_tensors[tensor_name] = packed
                scale_tensors[tensor_name] = scales
                original_bytes = int(tensor.nelement() * tensor.element_size())
                q4_bytes = int(packed.nelement() * packed.element_size() + scales.nelement() * scales.element_size())
                total_original_bytes += original_bytes
                total_q4_bytes += q4_bytes
                manifest["tensors"][tensor_name] = {
                    **metadata,
                    "q4_shard": q4_shard_name,
                    "scale_shard": scale_shard_name,
                    "original_nbytes": original_bytes,
                    "q4_nbytes": q4_bytes,
                }
                manifest["weight_map"][tensor_name] = q4_shard_name
                manifest["scale_map"][tensor_name] = scale_shard_name
        if q4_tensors:
            save_file(q4_tensors, str(output_dir / q4_shard_name), metadata={"format": Q4_FORMAT, "scheme": Q4_SCHEME})
            save_file(scale_tensors, str(output_dir / scale_shard_name), metadata={"format": Q4_FORMAT, "scheme": Q4_SCHEME})

    manifest["total_original_bytes"] = int(total_original_bytes)
    manifest["total_q4_bytes"] = int(total_q4_bytes)
    manifest["compression_ratio"] = 0.0 if total_original_bytes <= 0 else round(total_q4_bytes / total_original_bytes, 6)
    (output_dir / "q4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pcketlm Q4 streaming artifacts.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate the Q4 artifact from safetensors headers without reading tensor payloads or writing files.",
    )
    args = parser.parse_args()
    if args.dry_run:
        plan = plan_model_dir_to_q4(args.model_dir, args.output_dir)
        print(
            json.dumps(
                {
                    key: plan[key]
                    for key in (
                        "format",
                        "scheme",
                        "ready_for_conversion",
                        "source_tensor_count",
                        "q4_tensor_count",
                        "missing_shards",
                        "total_original_bytes",
                        "estimated_total_q4_bytes",
                        "estimated_expert_q4_bytes",
                        "estimated_non_expert_q4_bytes",
                        "compression_ratio",
                        "disk_required_bytes_with_10pct_headroom",
                    )
                },
                indent=2,
            )
        )
        return 0 if plan["ready_for_conversion"] else 2
    manifest = quantize_model_dir_to_q4(args.model_dir, args.output_dir)
    print(json.dumps({key: manifest[key] for key in ("format", "scheme", "total_original_bytes", "total_q4_bytes", "compression_ratio")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
