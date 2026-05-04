"""Build offline row8-packed fp16/bf16 weight artifacts for native GEMV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


PACKED_FORMAT = "pcketlm-row8-packed-gemv"
ROW_BLOCK = 8


def pack_rows8_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Pack a 2D fp16/bf16 tensor into row8 interleaved uint16 storage."""
    if tensor.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("row8 packing requires torch.float16 or torch.bfloat16 tensors")
    if tensor.ndim != 2:
        raise ValueError("row8 packing requires a 2D tensor")
    source = tensor.detach().cpu().contiguous().view(torch.uint16)
    rows = int(source.shape[0])
    cols = int(source.shape[1])
    row_blocks = (rows + ROW_BLOCK - 1) // ROW_BLOCK
    packed = torch.zeros((row_blocks, cols, ROW_BLOCK), dtype=torch.uint16)
    for block in range(row_blocks):
        row_start = block * ROW_BLOCK
        row_end = min(row_start + ROW_BLOCK, rows)
        block_rows = row_end - row_start
        packed[block, :, :block_rows] = source[row_start:row_end, :].transpose(0, 1)
    return packed.reshape(-1).contiguous()


def unpack_rows8_tensor(packed: torch.Tensor, rows: int, cols: int, *, dtype: torch.dtype) -> torch.Tensor:
    """Unpack row8 interleaved uint16 storage back to the original 2D tensor."""
    if dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("row8 unpack requires torch.float16 or torch.bfloat16 dtype")
    row_blocks = (int(rows) + ROW_BLOCK - 1) // ROW_BLOCK
    packed_view = packed.detach().cpu().contiguous().view(torch.uint16).reshape(row_blocks, int(cols), ROW_BLOCK)
    out = torch.empty((int(rows), int(cols)), dtype=torch.uint16)
    for block in range(row_blocks):
        row_start = block * ROW_BLOCK
        row_end = min(row_start + ROW_BLOCK, int(rows))
        block_rows = row_end - row_start
        out[row_start:row_end, :] = packed_view[block, :, :block_rows].transpose(0, 1)
    return out.view(dtype)


def _catalog_dtype_for_tensor(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.bfloat16:
        return "BF16"
    if tensor.dtype == torch.float16:
        return "F16"
    return str(tensor.dtype).replace("torch.", "")


def _torch_dtype_from_catalog(value: str) -> torch.dtype:
    normalized = str(value).upper()
    if normalized == "BF16":
        return torch.bfloat16
    if normalized == "F16":
        return torch.float16
    raise ValueError(f"unsupported packed dtype: {value}")


def _weight_map(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in (payload.get("weight_map") or {}).items()}


def _selected_tensor_names(weight_map: dict[str, str], patterns: list[str]) -> list[str]:
    if not patterns:
        return sorted(name for name in weight_map if name.endswith(".weight"))
    selected: list[str] = []
    for name in sorted(weight_map):
        if any(pattern in name for pattern in patterns):
            selected.append(name)
    return selected


def build_row8_packed_artifact(model_dir: Path, output_dir: Path, *, include_patterns: list[str] | None = None) -> dict:
    model_dir = model_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = _weight_map(model_dir / "model.safetensors.index.json")
    selected_names = _selected_tensor_names(weight_map, include_patterns or [])
    packed_path = output_dir / "row8_packed.bin"
    manifest = {
        "format": PACKED_FORMAT,
        "row_block": ROW_BLOCK,
        "source_model_dir": str(model_dir),
        "data_file": packed_path.name,
        "tensors": {},
        "tensor_count": 0,
        "total_original_bytes": 0,
        "total_packed_bytes": 0,
    }
    offset = 0
    with packed_path.open("wb") as out_handle:
        for tensor_name in selected_names:
            shard_name = weight_map[tensor_name]
            with safe_open(model_dir / shard_name, framework="pt", device="cpu") as shard:
                tensor = shard.get_tensor(tensor_name)
            if tensor.ndim != 2 or tensor.dtype not in {torch.float16, torch.bfloat16}:
                continue
            packed = pack_rows8_tensor(tensor)
            packed_bytes = packed.numpy().tobytes(order="C")
            out_handle.write(packed_bytes)
            original_bytes = int(tensor.numel() * tensor.element_size())
            nbytes = len(packed_bytes)
            manifest["tensors"][tensor_name] = {
                "dtype": _catalog_dtype_for_tensor(tensor),
                "shape": [int(value) for value in tensor.shape],
                "offset": offset,
                "nbytes": nbytes,
                "source_shard": shard_name,
                "original_nbytes": original_bytes,
            }
            manifest["tensor_count"] += 1
            manifest["total_original_bytes"] += original_bytes
            manifest["total_packed_bytes"] += nbytes
            offset += nbytes
    manifest["size_ratio"] = (
        0.0
        if int(manifest["total_original_bytes"]) <= 0
        else round(int(manifest["total_packed_bytes"]) / int(manifest["total_original_bytes"]), 6)
    )
    (output_dir / "row8_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_row8_packed_tensor(artifact_dir: Path, tensor_name: str) -> tuple[torch.Tensor, dict]:
    manifest = json.loads((artifact_dir / "row8_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["tensors"][tensor_name]
    data = artifact_dir / str(manifest["data_file"])
    with data.open("rb") as handle:
        handle.seek(int(entry["offset"]))
        payload = handle.read(int(entry["nbytes"]))
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint16).clone()
    return tensor, entry


def load_row8_unpacked_tensor(artifact_dir: Path, tensor_name: str) -> torch.Tensor:
    packed, entry = load_row8_packed_tensor(artifact_dir, tensor_name)
    return unpack_rows8_tensor(
        packed,
        int(entry["shape"][0]),
        int(entry["shape"][1]),
        dtype=_torch_dtype_from_catalog(str(entry["dtype"])),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pcketlm row8 packed GEMV artifacts.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include", action="append", default=[], help="Substring filter; may be repeated.")
    args = parser.parse_args()
    manifest = build_row8_packed_artifact(args.model_dir, args.output_dir, include_patterns=list(args.include))
    print(json.dumps({key: manifest[key] for key in ("format", "tensor_count", "total_packed_bytes", "size_ratio")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
