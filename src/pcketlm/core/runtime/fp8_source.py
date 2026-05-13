"""FP8-native source helpers for paged DeepSeek-style safetensors."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

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


@dataclass(slots=True)
class FP8DequantizedTensor:
    """One FP8 weight dequantized with its block scale tensor."""

    model_id: str
    weight_name: str
    scale_name: str | None
    shape: list[int]
    dtype: str
    loaded_nbytes: int
    tensor: torch.Tensor | None = None
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "weight_name": self.weight_name,
            "scale_name": self.scale_name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "loaded_nbytes": self.loaded_nbytes,
            "tensor_materialized": self.tensor is not None,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class FP8ExpertMLPResult:
    """One selected expert MLP run using dequantized FP8 weights."""

    model_id: str
    layer_index: int
    expert_index: int
    hidden_shape: list[int]
    output_shape: list[int]
    loaded_weight_bytes: int
    dequantized_weight_bytes: int
    output_tensor: torch.Tensor | None = None
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "layer_index": self.layer_index,
            "expert_index": self.expert_index,
            "hidden_shape": list(self.hidden_shape),
            "output_shape": list(self.output_shape),
            "loaded_weight_bytes": self.loaded_weight_bytes,
            "dequantized_weight_bytes": self.dequantized_weight_bytes,
            "output_materialized": self.output_tensor is not None,
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


def dequantize_fp8_block_scaled(
    fp8_bytes: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    block_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 E4M3 weights with DeepSeek-style 128x128 block scales."""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn.")
    if fp8_bytes.dtype != torch.uint8:
        raise TypeError("fp8_bytes must be a uint8 tensor containing raw E4M3 bytes.")
    if fp8_bytes.ndim != 2:
        raise ValueError("FP8 block dequant currently supports 2D weight tensors only.")
    if scale_inv.ndim != 2:
        raise ValueError("scale_inv must be a 2D block-scale tensor.")

    rows, cols = int(fp8_bytes.shape[0]), int(fp8_bytes.shape[1])
    expected_rows = (rows + block_size - 1) // block_size
    expected_cols = (cols + block_size - 1) // block_size
    if int(scale_inv.shape[0]) != expected_rows or int(scale_inv.shape[1]) != expected_cols:
        raise ValueError(
            "scale_inv shape "
            f"{list(scale_inv.shape)} does not match FP8 weight shape {list(fp8_bytes.shape)} "
            f"with block_size={block_size}; expected {[expected_rows, expected_cols]}."
        )

    fp8_values = fp8_bytes.contiguous().view(torch.float8_e4m3fn).to(torch.float32)
    expanded_scale = (
        scale_inv.to(torch.float32)
        .repeat_interleave(block_size, dim=0)
        .repeat_interleave(block_size, dim=1)[:rows, :cols]
    )
    return (fp8_values * expanded_scale).to(dtype=dtype).contiguous()


def dequantize_fp8_weight_pair(
    pair: FP8TensorPair,
    *,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 128,
) -> FP8DequantizedTensor:
    """Dequantize a loaded FP8 weight pair to a normal torch tensor."""
    blockers = list(pair.blockers)
    tensor = None
    if pair.fp8_bytes is None or pair.scale_tensor is None:
        blockers.append("FP8 payload and scale tensor must be loaded before dequantization.")
    if not blockers and pair.fp8_bytes is not None and pair.scale_tensor is not None:
        try:
            tensor = dequantize_fp8_block_scaled(pair.fp8_bytes, pair.scale_tensor, block_size=block_size, dtype=dtype)
        except (RuntimeError, TypeError, ValueError) as exc:
            blockers.append(str(exc))

    return FP8DequantizedTensor(
        model_id=pair.model_id,
        weight_name=pair.weight_name,
        scale_name=pair.scale_name,
        shape=list(pair.weight_shape),
        dtype=str(dtype).replace("torch.", ""),
        loaded_nbytes=pair.weight_nbytes + pair.scale_nbytes,
        tensor=tensor,
        blockers=blockers,
        ready=not blockers and tensor is not None,
    )


def load_dequantized_fp8_weight(
    model_id: str,
    weight_name: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 128,
) -> FP8DequantizedTensor:
    """Load one FP8 weight pair and dequantize it to a normal torch tensor."""
    pair = load_fp8_weight_pair(model_id, weight_name)
    return dequantize_fp8_weight_pair(pair, dtype=dtype, block_size=block_size)


def run_fp8_expert_mlp(
    model_id: str,
    layer_index: int,
    expert_index: int,
    hidden: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> FP8ExpertMLPResult:
    """Run one selected expert MLP using dequantized FP8 gate/up/down weights."""
    prefix = f"model.layers.{int(layer_index)}.mlp.experts.{int(expert_index)}"
    names = {
        "gate": f"{prefix}.gate_proj.weight",
        "up": f"{prefix}.up_proj.weight",
        "down": f"{prefix}.down_proj.weight",
    }
    loaded = {role: load_dequantized_fp8_weight(model_id, name, dtype=dtype) for role, name in names.items()}
    blockers = [blocker for item in loaded.values() for blocker in item.blockers]
    output = None
    if not blockers and all(item.tensor is not None for item in loaded.values()):
        gate = loaded["gate"].tensor
        up = loaded["up"].tensor
        down = loaded["down"].tensor
        if gate is None or up is None or down is None:
            blockers.append("One or more expert weights failed to materialize.")
        else:
            working = hidden.to(dtype=torch.float32)
            gate_out = F.linear(working, gate.to(torch.float32))
            up_out = F.linear(working, up.to(torch.float32))
            expert_hidden = F.silu(gate_out) * up_out
            output = F.linear(expert_hidden, down.to(torch.float32)).to(dtype=dtype).contiguous()

    dequantized_bytes = sum(0 if item.tensor is None else item.tensor.nelement() * item.tensor.element_size() for item in loaded.values())
    loaded_bytes = sum(item.loaded_nbytes for item in loaded.values())
    return FP8ExpertMLPResult(
        model_id=model_id,
        layer_index=int(layer_index),
        expert_index=int(expert_index),
        hidden_shape=[int(value) for value in hidden.shape],
        output_shape=[] if output is None else [int(value) for value in output.shape],
        loaded_weight_bytes=int(loaded_bytes),
        dequantized_weight_bytes=int(dequantized_bytes),
        output_tensor=output,
        blockers=blockers,
        ready=not blockers and output is not None,
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
