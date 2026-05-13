"""FP8-native source helpers for paged DeepSeek-style safetensors."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch

from pcketlm.core.runtime.tensor_catalog import (
    TensorCatalogEntry,
    find_tensor_catalog_entry,
    is_fp8_dtype,
    load_tensor_catalog,
    load_tensor_entry_index,
)


@dataclass(slots=True)
class FP8TensorPair:
    """One FP8 weight and its scale companion loaded from source shards."""

    model_id: str
    weight_name: str
    scale_name: str | None
    weight_dtype: str
    weight_shape: list[int]
    weight_nbytes: int
    scale_shape: list[int] = field(default_factory=list)
    scale_nbytes: int = 0
    fp8_bytes: torch.Tensor | None = None
    scale_tensor: torch.Tensor | None = None
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "weight_name": self.weight_name,
            "scale_name": self.scale_name,
            "weight_dtype": self.weight_dtype,
            "weight_shape": list(self.weight_shape),
            "weight_nbytes": self.weight_nbytes,
            "scale_shape": list(self.scale_shape),
            "scale_nbytes": self.scale_nbytes,
            "payload_loaded": self.fp8_bytes is not None,
            "scale_loaded": self.scale_tensor is not None,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class FP8LayerWorkingSet:
    """Header-only working-set estimate for one layer and selected experts."""

    model_id: str
    layer_index: int
    selected_experts: list[int]
    tensor_count: int
    fp8_weight_count: int
    scale_count: int
    total_nbytes: int
    fp8_weight_bytes: int
    scale_bytes: int
    non_fp8_bytes: int
    tensor_names: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "layer_index": self.layer_index,
            "selected_experts": list(self.selected_experts),
            "tensor_count": self.tensor_count,
            "fp8_weight_count": self.fp8_weight_count,
            "scale_count": self.scale_count,
            "total_nbytes": self.total_nbytes,
            "fp8_weight_bytes": self.fp8_weight_bytes,
            "scale_bytes": self.scale_bytes,
            "non_fp8_bytes": self.non_fp8_bytes,
            "tensor_names": list(self.tensor_names),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


def fp8_source_status(model_id: str) -> dict:
    """Return FP8 source readiness from the persisted tensor catalog."""
    catalog = load_tensor_catalog(model_id)
    return {
        "model_id": model_id,
        "ready": catalog.ready and catalog.fp8_weight_count > 0 and catalog.fp8_pair_count > 0,
        "catalog_ready": catalog.ready,
        "fp8_weight_count": catalog.fp8_weight_count,
        "fp8_scale_count": catalog.fp8_scale_count,
        "fp8_pair_count": catalog.fp8_pair_count,
        "fp8_weight_bytes": catalog.fp8_weight_bytes,
        "fp8_scale_bytes": catalog.fp8_scale_bytes,
        "blockers": list(catalog.blockers),
    }


def plan_fp8_layer_working_set(
    model_id: str,
    layer_index: int,
    selected_experts: list[int] | None = None,
) -> FP8LayerWorkingSet:
    """Plan the tensors needed for one FP8 paged layer without loading payloads."""
    selected = [] if selected_experts is None else [int(value) for value in selected_experts]
    entries = load_tensor_entry_index(model_id)
    if not entries:
        return FP8LayerWorkingSet(
            model_id=model_id,
            layer_index=layer_index,
            selected_experts=selected,
            tensor_count=0,
            fp8_weight_count=0,
            scale_count=0,
            total_nbytes=0,
            fp8_weight_bytes=0,
            scale_bytes=0,
            non_fp8_bytes=0,
            blockers=["Tensor catalog is not ready."],
            ready=False,
        )

    selected_set = set(selected)
    chosen: list[TensorCatalogEntry] = []
    for entry in entries.values():
        if entry.layer_index != layer_index:
            continue
        if entry.expert_index is not None and selected_set and entry.expert_index not in selected_set:
            continue
        chosen.append(entry)

    fp8_weight_bytes = sum(entry.data_nbytes for entry in chosen if is_fp8_dtype(entry.dtype))
    scale_bytes = sum(entry.data_nbytes for entry in chosen if entry.tensor_role == "scale_companion")
    non_fp8_bytes = sum(
        entry.data_nbytes
        for entry in chosen
        if not is_fp8_dtype(entry.dtype) and entry.tensor_role != "scale_companion"
    )
    blockers: list[str] = []
    for entry in chosen:
        if is_fp8_dtype(entry.dtype) and not entry.scale_tensor_name:
            blockers.append(f"FP8 tensor {entry.tensor_name} has no scale companion in the catalog.")

    return FP8LayerWorkingSet(
        model_id=model_id,
        layer_index=layer_index,
        selected_experts=selected,
        tensor_count=len(chosen),
        fp8_weight_count=sum(1 for entry in chosen if is_fp8_dtype(entry.dtype)),
        scale_count=sum(1 for entry in chosen if entry.tensor_role == "scale_companion"),
        total_nbytes=sum(entry.data_nbytes for entry in chosen),
        fp8_weight_bytes=fp8_weight_bytes,
        scale_bytes=scale_bytes,
        non_fp8_bytes=non_fp8_bytes,
        tensor_names=[entry.tensor_name for entry in sorted(chosen, key=lambda item: item.tensor_name)],
        blockers=blockers,
        ready=bool(chosen) and not blockers,
    )


def load_fp8_weight_pair(model_id: str, weight_name: str, *, load_payload: bool = True) -> FP8TensorPair:
    """Load one FP8 weight as raw bytes plus its scale tensor."""
    weight = find_tensor_catalog_entry(model_id, weight_name)
    if weight is None:
        return FP8TensorPair(
            model_id=model_id,
            weight_name=weight_name,
            scale_name=None,
            weight_dtype="unknown",
            weight_shape=[],
            weight_nbytes=0,
            blockers=[f"Tensor {weight_name} is not present in the tensor catalog."],
            ready=False,
        )
    blockers: list[str] = []
    if not is_fp8_dtype(weight.dtype):
        blockers.append(f"Tensor {weight_name} is {weight.dtype}, not FP8.")
    if not weight.scale_tensor_name:
        blockers.append(f"Tensor {weight_name} has no scale companion.")
    scale = find_tensor_catalog_entry(model_id, weight.scale_tensor_name) if weight.scale_tensor_name else None
    if weight.scale_tensor_name and scale is None:
        blockers.append(f"Scale tensor {weight.scale_tensor_name} is not present in the tensor catalog.")

    fp8_bytes = None
    scale_tensor = None
    if load_payload and not blockers and scale is not None:
        fp8_bytes = _read_tensor_as_uint8(weight)
        scale_tensor = _read_scale_tensor(scale)

    return FP8TensorPair(
        model_id=model_id,
        weight_name=weight.tensor_name,
        scale_name=weight.scale_tensor_name,
        weight_dtype=weight.dtype,
        weight_shape=list(weight.shape),
        weight_nbytes=weight.data_nbytes,
        scale_shape=[] if scale is None else list(scale.shape),
        scale_nbytes=0 if scale is None else scale.data_nbytes,
        fp8_bytes=fp8_bytes,
        scale_tensor=scale_tensor,
        blockers=blockers,
        ready=not blockers and (not load_payload or (fp8_bytes is not None and scale_tensor is not None)),
    )


def _read_tensor_as_uint8(entry: TensorCatalogEntry) -> torch.Tensor:
    raw = _read_tensor_bytes(entry)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(tuple(entry.shape)).contiguous()


def _read_scale_tensor(entry: TensorCatalogEntry) -> torch.Tensor:
    raw = _read_tensor_bytes(entry)
    dtype = _torch_dtype_for_scale(entry.dtype)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(tuple(entry.shape)).clone()


def _torch_dtype_for_scale(dtype: str) -> torch.dtype:
    if dtype == "F32":
        return torch.float32
    if dtype == "BF16":
        return torch.bfloat16
    if dtype == "F16":
        return torch.float16
    return torch.float32


def _read_tensor_bytes(entry: TensorCatalogEntry) -> bytes:
    base_offset = _safetensors_data_base_offset(str(entry.shard_path.resolve()), _path_mtime_ns(entry.shard_path))
    with entry.shard_path.open("rb") as handle:
        handle.seek(base_offset + entry.data_offset_start)
        return handle.read(entry.data_nbytes)


def _path_mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


@lru_cache(maxsize=128)
def _safetensors_data_base_offset(path: str, mtime_ns: int) -> int:
    del mtime_ns
    with Path(path).open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
    return 8 + int(header_length)
