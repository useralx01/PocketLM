"""FP8-native source helpers for paged DeepSeek-style safetensors."""

from __future__ import annotations

import struct
import json
import math
import os
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from pcketlm.native import (
    fp8_e4m3_block_dual_linear_f32,
    fp8_e4m3_block_linear_f32,
    lm_head_topk_u16,
    native_fp16_loader_available,
    native_fp16_matmul_available,
    native_fp8_dual_linear_available,
    native_fp8_linear_available,
    native_read_tensor_bytes,
)
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


@dataclass(slots=True)
class FP8DenseMLPResult:
    """One dense MLP run using dequantized FP8 gate/up/down weights."""

    model_id: str
    layer_index: int
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
            "hidden_shape": list(self.hidden_shape),
            "output_shape": list(self.output_shape),
            "loaded_weight_bytes": self.loaded_weight_bytes,
            "dequantized_weight_bytes": self.dequantized_weight_bytes,
            "output_materialized": self.output_tensor is not None,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class FP8RouterResult:
    """DeepSeek-style router top-k result for one FP8 MoE layer."""

    model_id: str
    layer_index: int
    hidden_shape: list[int]
    weights_shape: list[int]
    indices_shape: list[int]
    selected_experts: list[int]
    scoring_func: str
    route_scale: float
    n_groups: int
    topk_groups: int
    bias_loaded: bool
    weights_tensor: torch.Tensor | None = None
    indices_tensor: torch.Tensor | None = None
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "layer_index": self.layer_index,
            "hidden_shape": list(self.hidden_shape),
            "weights_shape": list(self.weights_shape),
            "indices_shape": list(self.indices_shape),
            "selected_experts": list(self.selected_experts),
            "scoring_func": self.scoring_func,
            "route_scale": self.route_scale,
            "n_groups": self.n_groups,
            "topk_groups": self.topk_groups,
            "bias_loaded": self.bias_loaded,
            "weights_materialized": self.weights_tensor is not None,
            "indices_materialized": self.indices_tensor is not None,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class FP8MoEResult:
    """Selected-expert MoE result using FP8 routed and shared experts."""

    model_id: str
    layer_index: int
    hidden_shape: list[int]
    output_shape: list[int]
    selected_experts: list[int]
    routed_weight_bytes: int
    shared_weight_bytes: int
    dequantized_weight_bytes: int
    router: FP8RouterResult | None = None
    output_tensor: torch.Tensor | None = None
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "layer_index": self.layer_index,
            "hidden_shape": list(self.hidden_shape),
            "output_shape": list(self.output_shape),
            "selected_experts": list(self.selected_experts),
            "routed_weight_bytes": self.routed_weight_bytes,
            "shared_weight_bytes": self.shared_weight_bytes,
            "dequantized_weight_bytes": self.dequantized_weight_bytes,
            "router": None if self.router is None else self.router.to_dict(),
            "output_materialized": self.output_tensor is not None,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class FP8AttentionResult:
    """Single-token DeepSeek MLA attention result from FP8 source weights."""

    model_id: str
    layer_index: int
    hidden_shape: list[int]
    output_shape: list[int]
    loaded_weight_bytes: int
    dequantized_weight_bytes: int
    output_tensor: torch.Tensor | None = None
    kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "layer_index": self.layer_index,
            "hidden_shape": list(self.hidden_shape),
            "output_shape": list(self.output_shape),
            "loaded_weight_bytes": self.loaded_weight_bytes,
            "dequantized_weight_bytes": self.dequantized_weight_bytes,
            "output_materialized": self.output_tensor is not None,
            "cache_sequence_length": 0 if self.kv_cache is None else int(self.kv_cache[0].shape[1]),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class FP8BlockResult:
    """Single-token DeepSeek block proof with FP8 MLA attention and MoE FFN."""

    model_id: str
    layer_index: int
    hidden_shape: list[int]
    output_shape: list[int]
    attention: FP8AttentionResult | None = None
    moe: FP8MoEResult | None = None
    dense_mlp: FP8DenseMLPResult | None = None
    output_tensor: torch.Tensor | None = None
    next_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "layer_index": self.layer_index,
            "hidden_shape": list(self.hidden_shape),
            "output_shape": list(self.output_shape),
            "attention": None if self.attention is None else self.attention.to_dict(),
            "moe": None if self.moe is None else self.moe.to_dict(),
            "dense_mlp": None if self.dense_mlp is None else self.dense_mlp.to_dict(),
            "output_materialized": self.output_tensor is not None,
            "cache_sequence_length": 0 if self.next_kv_cache is None else int(self.next_kv_cache[0].shape[1]),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class FP8DecodeTailResult:
    """Final norm plus streamed lm_head top-k proof for an FP8 DeepSeek hidden state."""

    model_id: str
    input_shape: list[int]
    top_token_ids: list[int]
    top_logits: list[float]
    chunk_rows: int
    chunk_count: int
    loaded_lm_head_bytes: int
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "input_shape": list(self.input_shape),
            "top_token_ids": list(self.top_token_ids),
            "top_logits": list(self.top_logits),
            "chunk_rows": self.chunk_rows,
            "chunk_count": self.chunk_count,
            "loaded_lm_head_bytes": self.loaded_lm_head_bytes,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class FP8TokenEmbeddingResult:
    """One token embedding row loaded from the DeepSeek source."""

    model_id: str
    token_id: int
    output_shape: list[int]
    loaded_bytes: int
    output_tensor: torch.Tensor | None = None
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "token_id": self.token_id,
            "output_shape": list(self.output_shape),
            "loaded_bytes": self.loaded_bytes,
            "output_materialized": self.output_tensor is not None,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class FP8TokenForwardResult:
    """Single-token FP8 forward proof over a bounded layer range."""

    model_id: str
    token_id: int
    start_layer: int
    layer_count: int
    executed_layers: list[int]
    hidden_shape: list[int]
    embedding: FP8TokenEmbeddingResult | None = None
    tail: FP8DecodeTailResult | None = None
    step_summaries: list[dict] = field(default_factory=list)
    output_tensor: torch.Tensor | None = None
    next_kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "token_id": self.token_id,
            "start_layer": self.start_layer,
            "layer_count": self.layer_count,
            "executed_layers": list(self.executed_layers),
            "hidden_shape": list(self.hidden_shape),
            "embedding": None if self.embedding is None else self.embedding.to_dict(),
            "tail": None if self.tail is None else self.tail.to_dict(),
            "step_summaries": [dict(item) for item in self.step_summaries],
            "output_materialized": self.output_tensor is not None,
            "cache_sequence_lengths": {
                str(key): int(value[0].shape[1]) for key, value in self.next_kv_caches.items()
            },
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class FP8DecodeLoopResult:
    """Small FP8 prompt/decode loop using carried per-layer KV state."""

    model_id: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    start_layer: int
    layer_count: int
    positions_completed: int
    final_top_token_ids: list[int]
    final_top_logits: list[float]
    step_summaries: list[dict] = field(default_factory=list)
    next_kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
            "start_layer": self.start_layer,
            "layer_count": self.layer_count,
            "positions_completed": self.positions_completed,
            "final_top_token_ids": list(self.final_top_token_ids),
            "final_top_logits": list(self.final_top_logits),
            "step_summaries": [dict(item) for item in self.step_summaries],
            "cache_sequence_lengths": {
                str(key): int(value[0].shape[1]) for key, value in self.next_kv_caches.items()
            },
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class FP8PromptPrefillResult:
    """Layer-wise prompt prefill result for a bounded FP8 decode run."""

    model_id: str
    prompt_token_ids: list[int]
    start_layer: int
    layer_count: int
    executed_layers: list[int]
    hidden_shape: list[int]
    tail: FP8DecodeTailResult | None = None
    output_tensor: torch.Tensor | None = None
    next_kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    step_summaries: list[dict] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False


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
    output, loaded_bytes, dequantized_bytes, blockers = _run_fp8_mlp_prefix(model_id, prefix, hidden, dtype=dtype)
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


def run_fp8_dense_mlp(
    model_id: str,
    layer_index: int,
    hidden: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> FP8DenseMLPResult:
    """Run one dense DeepSeek MLP layer using dequantized FP8 gate/up/down weights."""
    prefix = f"model.layers.{int(layer_index)}.mlp"
    output, loaded_bytes, dequantized_bytes, blockers = _run_fp8_mlp_prefix(model_id, prefix, hidden, dtype=dtype)
    return FP8DenseMLPResult(
        model_id=model_id,
        layer_index=int(layer_index),
        hidden_shape=[int(value) for value in hidden.shape],
        output_shape=[] if output is None else [int(value) for value in output.shape],
        loaded_weight_bytes=int(loaded_bytes),
        dequantized_weight_bytes=int(dequantized_bytes),
        output_tensor=output,
        blockers=blockers,
        ready=not blockers and output is not None,
    )


def run_fp8_router(
    model_id: str,
    layer_index: int,
    hidden: torch.Tensor,
) -> FP8RouterResult:
    """Run DeepSeek V3 compatible sigmoid/grouped top-k routing for one layer."""
    from pcketlm.core.runtime.tensor_loader import load_tensor_by_name

    route_config = _load_deepseek_route_config(model_id)
    router_name = f"model.layers.{int(layer_index)}.mlp.gate.weight"
    bias_name = f"model.layers.{int(layer_index)}.mlp.gate.e_score_correction_bias"
    router = load_tensor_by_name(model_id, router_name)
    bias = load_tensor_by_name(model_id, bias_name)
    blockers = list(router.blockers)
    bias_tensor = None
    if bias.ready and bias.tensor is not None:
        bias_tensor = bias.tensor.float()
    elif bias.blockers and "not present" not in " ".join(bias.blockers):
        blockers.extend(bias.blockers)

    weights = None
    indices = None
    if router.tensor is None:
        blockers.append(f"Router tensor {router_name} did not materialize.")
    if not blockers and router.tensor is not None:
        flat_hidden = hidden.reshape(-1, hidden.shape[-1]).float()
        scores = F.linear(flat_hidden, router.tensor.float())
        if route_config["scoring_func"] == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        choice_scores = scores + bias_tensor if bias_tensor is not None else scores
        if route_config["n_groups"] > 1:
            choice_scores = _mask_deepseek_route_groups(
                choice_scores,
                batch_size=flat_hidden.shape[0],
                n_groups=route_config["n_groups"],
                topk_groups=route_config["topk_groups"],
                use_bias=bias_tensor is not None,
            )
        effective_top_k = max(1, min(int(route_config["top_k"]), choice_scores.shape[-1]))
        indices = torch.topk(choice_scores, effective_top_k, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if route_config["scoring_func"] == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = (weights * float(route_config["route_scale"])).to(dtype=hidden.dtype)

    selected = [] if indices is None else sorted({int(value) for value in indices.reshape(-1).tolist()})
    return FP8RouterResult(
        model_id=model_id,
        layer_index=int(layer_index),
        hidden_shape=[int(value) for value in hidden.shape],
        weights_shape=[] if weights is None else [int(value) for value in weights.shape],
        indices_shape=[] if indices is None else [int(value) for value in indices.shape],
        selected_experts=selected,
        scoring_func=str(route_config["scoring_func"]),
        route_scale=float(route_config["route_scale"]),
        n_groups=int(route_config["n_groups"]),
        topk_groups=int(route_config["topk_groups"]),
        bias_loaded=bias_tensor is not None,
        weights_tensor=weights,
        indices_tensor=indices,
        blockers=blockers,
        ready=not blockers and weights is not None and indices is not None,
    )


def run_fp8_moe(
    model_id: str,
    layer_index: int,
    hidden: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    router: FP8RouterResult | None = None,
) -> FP8MoEResult:
    """Run DeepSeek-style selected routed experts plus shared experts for one MoE layer."""
    router_result = router or run_fp8_router(model_id, layer_index, hidden)
    blockers = list(router_result.blockers)
    output = None
    routed_loaded_bytes = 0
    shared_loaded_bytes = 0
    dequantized_bytes = 0
    selected: list[int] = []
    if router_result.indices_tensor is None or router_result.weights_tensor is None:
        blockers.append("Router did not produce selected experts and route weights.")

    if not blockers and router_result.indices_tensor is not None and router_result.weights_tensor is not None:
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_output = torch.zeros_like(flat_hidden, dtype=dtype)
        route_indices = router_result.indices_tensor
        route_weights = router_result.weights_tensor.float()
        selected = sorted({int(value) for value in route_indices.reshape(-1).tolist()})
        expert_jobs: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for expert_index in selected:
            row_indices, top_indices = torch.where(route_indices == int(expert_index))
            if row_indices.numel() != 0:
                expert_jobs.append((int(expert_index), row_indices, top_indices, flat_hidden[row_indices].to(dtype=dtype)))

        expert_results: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None, int, int, list[str]]] = []
        workers = _fp8_moe_expert_workers()
        if workers > 1 and len(expert_jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(expert_jobs))) as executor:
                futures = {
                    executor.submit(
                        _run_fp8_mlp_prefix,
                        model_id,
                        f"model.layers.{int(layer_index)}.mlp.experts.{int(expert_index)}",
                        expert_hidden,
                        dtype=dtype,
                    ): (expert_index, row_indices, top_indices)
                    for expert_index, row_indices, top_indices, expert_hidden in expert_jobs
                }
                for future in as_completed(futures):
                    expert_index, row_indices, top_indices = futures[future]
                    expert_output, loaded_bytes, expert_dequant_bytes, expert_blockers = future.result()
                    expert_results.append(
                        (expert_index, row_indices, top_indices, expert_output, loaded_bytes, expert_dequant_bytes, expert_blockers)
                    )
        else:
            for expert_index, row_indices, top_indices, expert_hidden in expert_jobs:
                expert_output, loaded_bytes, expert_dequant_bytes, expert_blockers = _run_fp8_mlp_prefix(
                    model_id,
                    f"model.layers.{int(layer_index)}.mlp.experts.{int(expert_index)}",
                    expert_hidden,
                    dtype=dtype,
                )
                expert_results.append(
                    (expert_index, row_indices, top_indices, expert_output, loaded_bytes, expert_dequant_bytes, expert_blockers)
                )

        for expert_index, row_indices, top_indices, expert_output, loaded_bytes, expert_dequant_bytes, expert_blockers in sorted(
            expert_results, key=lambda item: item[0]
        ):
            routed_loaded_bytes += loaded_bytes
            dequantized_bytes += expert_dequant_bytes
            blockers.extend(expert_blockers)
            if expert_output is not None:
                route = route_weights[row_indices, top_indices].view(-1, 1).to(dtype=dtype)
                flat_output[row_indices] += expert_output * route

        shared_output, loaded_bytes, shared_dequant_bytes, shared_blockers = _run_fp8_mlp_prefix(
            model_id,
            f"model.layers.{int(layer_index)}.mlp.shared_experts",
            flat_hidden.to(dtype=dtype),
            dtype=dtype,
        )
        shared_loaded_bytes += loaded_bytes
        dequantized_bytes += shared_dequant_bytes
        blockers.extend(shared_blockers)
        if shared_output is not None:
            flat_output = flat_output + shared_output
        if not blockers:
            output = flat_output.reshape_as(hidden).contiguous()

    return FP8MoEResult(
        model_id=model_id,
        layer_index=int(layer_index),
        hidden_shape=[int(value) for value in hidden.shape],
        output_shape=[] if output is None else [int(value) for value in output.shape],
        selected_experts=selected,
        routed_weight_bytes=int(routed_loaded_bytes),
        shared_weight_bytes=int(shared_loaded_bytes),
        dequantized_weight_bytes=int(dequantized_bytes),
        router=router_result,
        output_tensor=output,
        blockers=blockers,
        ready=not blockers and output is not None,
    )


def run_fp8_single_token_attention(
    model_id: str,
    layer_index: int,
    hidden: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    start_pos: int = 0,
    previous_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> FP8AttentionResult:
    """Run one-token DeepSeek MLA attention from FP8 source tensors.

    This is a correctness bridge for the decode path: sequence length must be one,
    so causal masking and KV-cache extension are intentionally outside this helper.
    """
    if not _streamed_fp8_attention_enabled():
        return _run_fp8_single_token_attention_materialized(
            model_id,
            layer_index,
            hidden,
            dtype=dtype,
            start_pos=start_pos,
            previous_kv_cache=previous_kv_cache,
        )

    deepseek_config = _load_deepseek_config(model_id)
    blockers: list[str] = []
    hidden_3d = hidden.reshape(1, 1, hidden.shape[-1]) if hidden.ndim == 2 else hidden
    if hidden_3d.ndim != 3 or int(hidden_3d.shape[1]) < 1:
        blockers.append("FP8 attention requires hidden shape [batch, seq, hidden] with at least one token.")
    if previous_kv_cache is not None and int(hidden_3d.shape[1]) != 1:
        blockers.append("FP8 cached attention currently supports one appended token at a time.")

    prefix = f"model.layers.{int(layer_index)}.self_attn"
    weight_names = {
        "q_a": f"{prefix}.q_a_proj.weight",
        "q_b": f"{prefix}.q_b_proj.weight",
        "kv_a": f"{prefix}.kv_a_proj_with_mqa.weight",
        "kv_b": f"{prefix}.kv_b_proj.weight",
        "o": f"{prefix}.o_proj.weight",
    }
    kv_b_loaded = load_dequantized_fp8_weight(model_id, weight_names["kv_b"], dtype=dtype)
    blockers.extend(kv_b_loaded.blockers)
    q_norm = _load_regular_tensor(model_id, f"{prefix}.q_a_layernorm.weight")
    kv_norm = _load_regular_tensor(model_id, f"{prefix}.kv_a_layernorm.weight")
    blockers.extend(q_norm[1])
    blockers.extend(kv_norm[1])

    output = None
    loaded_bytes = int(kv_b_loaded.loaded_nbytes)
    dequantized_bytes = 0 if kv_b_loaded.tensor is None else int(kv_b_loaded.tensor.nelement() * kv_b_loaded.tensor.element_size())
    if not blockers and kv_b_loaded.tensor is not None and q_norm[0] is not None and kv_norm[0] is not None:
        kv_b = kv_b_loaded.tensor
        if kv_b is None:
            blockers.append("Attention kv_b weight failed to materialize.")
        else:
            working = hidden_3d.float()
            q_low, q_a_loaded, q_a_dequant, q_a_blockers = _run_fp8_linear_streamed(
                model_id, weight_names["q_a"], working, dtype=dtype
            )
            loaded_bytes += q_a_loaded
            dequantized_bytes += q_a_dequant
            blockers.extend(q_a_blockers)
        if not blockers and q_low is not None:
            q_low = _rms_norm_any(q_low.float(), q_norm[0].float(), float(deepseek_config["rms_norm_eps"]))
            q, q_b_loaded_nbytes, q_b_dequant, q_b_blockers = _run_fp8_linear_streamed(
                model_id, weight_names["q_b"], q_low, dtype=dtype
            )
            loaded_bytes += q_b_loaded_nbytes
            dequantized_bytes += q_b_dequant
            blockers.extend(q_b_blockers)
        if not blockers and q is not None:
            n_heads = int(deepseek_config["num_attention_heads"])
            qk_nope = int(deepseek_config["qk_nope_head_dim"])
            qk_rope = int(deepseek_config["qk_rope_head_dim"])
            v_head_dim = int(deepseek_config["v_head_dim"])
            kv_lora_rank = int(deepseek_config["kv_lora_rank"])
            q = q.float()
            seq_len = int(hidden_3d.shape[1])
            q = q.view(int(hidden_3d.shape[0]), seq_len, n_heads, qk_nope + qk_rope)
            q_nope, q_pe = torch.split(q, [qk_nope, qk_rope], dim=-1)
            kv, kv_a_loaded, kv_a_dequant, kv_a_blockers = _run_fp8_linear_streamed(
                model_id, weight_names["kv_a"], working, dtype=dtype
            )
            loaded_bytes += kv_a_loaded
            dequantized_bytes += kv_a_dequant
            blockers.extend(kv_a_blockers)
        if not blockers and kv is not None:
            kv = kv.float()
            kv_latent, k_pe = torch.split(kv, [kv_lora_rank, qk_rope], dim=-1)
            q_pe = _apply_rope_real(q_pe, start_pos=int(start_pos), config=deepseek_config)
            k_pe = _apply_rope_real(k_pe.unsqueeze(2), start_pos=int(start_pos), config=deepseek_config).squeeze(2)
            kv_latent = _rms_norm_any(kv_latent, kv_norm[0].float(), float(deepseek_config["rms_norm_eps"]))
            if previous_kv_cache is not None:
                previous_kv, previous_pe = previous_kv_cache
                kv_cache = torch.cat([previous_kv.to(kv_latent.dtype), kv_latent], dim=1)
                pe_cache = torch.cat([previous_pe.to(k_pe.dtype), k_pe], dim=1)
            else:
                kv_cache = kv_latent
                pe_cache = k_pe
            wkv_b = kv_b.float().view(n_heads, qk_nope + v_head_dim, kv_lora_rank)
            q_nope_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope])
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope_absorbed, kv_cache)
                + torch.einsum("bshr,btr->bsht", q_pe, pe_cache)
            ) * float(deepseek_config["softmax_scale"])
            if previous_kv_cache is None and seq_len > 1:
                causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=scores.device).triu(1)
                scores = scores.masked_fill(causal_mask.view(1, seq_len, 1, seq_len), float("-inf"))
            probs = scores.softmax(dim=-1, dtype=torch.float32).to(dtype=working.dtype)
            attention_latent = torch.einsum("bsht,btc->bshc", probs, kv_cache)
            attention_heads = torch.einsum("bshc,hdc->bshd", attention_latent, wkv_b[:, -v_head_dim:])
            output, o_loaded, o_dequant, o_blockers = _run_fp8_linear_streamed(
                model_id, weight_names["o"], attention_heads.flatten(2), dtype=dtype
            )
            loaded_bytes += o_loaded
            dequantized_bytes += o_dequant
            blockers.extend(o_blockers)
            if output is not None:
                output = output.to(dtype=dtype).contiguous()
                next_cache = (kv_cache.detach().contiguous(), pe_cache.detach().contiguous())

    return FP8AttentionResult(
        model_id=model_id,
        layer_index=int(layer_index),
        hidden_shape=[int(value) for value in hidden.shape],
        output_shape=[] if output is None else [int(value) for value in output.shape],
        loaded_weight_bytes=int(loaded_bytes),
        dequantized_weight_bytes=int(dequantized_bytes),
        output_tensor=output,
        kv_cache=locals().get("next_cache"),
        blockers=blockers,
        ready=not blockers and output is not None,
    )


def run_fp8_single_token_block(
    model_id: str,
    layer_index: int,
    hidden: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    start_pos: int = 0,
    previous_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> FP8BlockResult:
    """Run one DeepSeek block for a single token using FP8 attention and FP8 MoE."""
    hidden_3d = hidden.reshape(1, 1, hidden.shape[-1]) if hidden.ndim == 2 else hidden
    config = _load_deepseek_config(model_id)
    input_norm = _load_regular_tensor(model_id, f"model.layers.{int(layer_index)}.input_layernorm.weight")
    post_norm = _load_regular_tensor(model_id, f"model.layers.{int(layer_index)}.post_attention_layernorm.weight")
    blockers = list(input_norm[1]) + list(post_norm[1])
    attention = None
    moe = None
    dense_mlp = None
    output = None
    if input_norm[0] is None:
        blockers.append("Input layer norm did not materialize.")
    if post_norm[0] is None:
        blockers.append("Post-attention layer norm did not materialize.")
    if not blockers and input_norm[0] is not None and post_norm[0] is not None:
        normed = _rms_norm_any(hidden_3d.to(dtype=dtype), input_norm[0].float(), float(config["rms_norm_eps"]))
        attention = run_fp8_single_token_attention(
            model_id,
            layer_index,
            normed,
            dtype=dtype,
            start_pos=int(start_pos),
            previous_kv_cache=previous_kv_cache,
        )
        blockers.extend(attention.blockers)
        if attention.output_tensor is not None:
            after_attention = hidden_3d.to(dtype=dtype) + attention.output_tensor
            ffn_input = _rms_norm_any(after_attention, post_norm[0].float(), float(config["rms_norm_eps"]))
            if int(layer_index) < int(config["first_k_dense_replace"]):
                dense_mlp = run_fp8_dense_mlp(model_id, layer_index, ffn_input, dtype=dtype)
                blockers.extend(dense_mlp.blockers)
                if dense_mlp.output_tensor is not None and not blockers:
                    output = (after_attention + dense_mlp.output_tensor).contiguous()
            else:
                moe = run_fp8_moe(model_id, layer_index, ffn_input, dtype=dtype)
                blockers.extend(moe.blockers)
                if moe.output_tensor is not None and not blockers:
                    output = (after_attention + moe.output_tensor).contiguous()

    return FP8BlockResult(
        model_id=model_id,
        layer_index=int(layer_index),
        hidden_shape=[int(value) for value in hidden.shape],
        output_shape=[] if output is None else [int(value) for value in output.shape],
        attention=attention,
        moe=moe,
        dense_mlp=dense_mlp,
        output_tensor=output,
        next_kv_cache=None if attention is None else attention.kv_cache,
        blockers=blockers,
        ready=not blockers and output is not None,
    )


def run_fp8_decode_tail_topk(
    model_id: str,
    hidden: torch.Tensor,
    *,
    top_k: int = 5,
    chunk_rows: int = 2048,
) -> FP8DecodeTailResult:
    """Run final norm and stream lm_head chunks to get top-k logits."""
    norm_tensor, blockers = _load_regular_tensor(model_id, "model.norm.weight")
    hidden_2d = hidden.reshape(-1, hidden.shape[-1])[-1:].to(dtype=torch.bfloat16)
    config = _load_deepseek_config(model_id)
    top_values: torch.Tensor | None = None
    top_indices: torch.Tensor | None = None
    chunk_count = 0
    loaded_bytes = 0
    if norm_tensor is None:
        blockers.append("Final norm tensor did not materialize.")
    lm_head_entry = find_tensor_catalog_entry(model_id, "lm_head.weight")
    if lm_head_entry is None:
        blockers.append("lm_head.weight is not present in the tensor catalog.")
    if lm_head_entry is not None and lm_head_entry.dtype not in {"BF16", "F16", "F32"}:
        blockers.append(f"lm_head.weight has unsupported dtype {lm_head_entry.dtype}.")

    if not blockers and norm_tensor is not None and lm_head_entry is not None:
        normalized_u16 = _rms_norm_any(hidden_2d, norm_tensor.float(), float(config["rms_norm_eps"]))
        normalized = normalized_u16.float()
        vocab_size = int(lm_head_entry.shape[0])
        hidden_size = int(lm_head_entry.shape[1])
        if normalized.shape[-1] != hidden_size:
            blockers.append(
                f"Hidden size {normalized.shape[-1]} does not match lm_head hidden size {hidden_size}."
            )
        else:
            keep_k = max(1, min(int(top_k), vocab_size))
            rows_per_chunk = max(1, int(chunk_rows))
            for start in range(0, vocab_size, rows_per_chunk):
                end = min(vocab_size, start + rows_per_chunk)
                weight = _read_tensor_rows(lm_head_entry, start, end)
                loaded_bytes += (end - start) * hidden_size * _dtype_element_size(lm_head_entry.dtype)
                if (
                    _native_lm_head_topk_enabled()
                    and lm_head_entry.dtype in {"BF16", "F16"}
                    and native_fp16_matmul_available()
                ):
                    native_hidden = normalized_u16.to(dtype=_torch_dtype_for_scale(lm_head_entry.dtype))
                    values, indices = lm_head_topk_u16(
                        native_hidden,
                        weight,
                        top_k=min(keep_k, int(weight.shape[0])),
                        token_offset=start,
                    )
                else:
                    chunk_logits = F.linear(normalized, weight.float()).reshape(-1)
                    values, indices = torch.topk(chunk_logits, min(keep_k, chunk_logits.numel()))
                    indices = indices + start
                if top_values is None or top_indices is None:
                    top_values = values
                    top_indices = indices
                else:
                    merged_values = torch.cat([top_values, values])
                    merged_indices = torch.cat([top_indices, indices])
                    top_values, positions = torch.topk(merged_values, keep_k)
                    top_indices = merged_indices[positions]
                chunk_count += 1

    token_ids = [] if top_indices is None else [int(value) for value in top_indices.tolist()]
    logits = [] if top_values is None else [float(value) for value in top_values.tolist()]
    return FP8DecodeTailResult(
        model_id=model_id,
        input_shape=[int(value) for value in hidden.shape],
        top_token_ids=token_ids,
        top_logits=logits,
        chunk_rows=int(chunk_rows),
        chunk_count=int(chunk_count),
        loaded_lm_head_bytes=int(loaded_bytes),
        blockers=blockers,
        ready=not blockers and bool(token_ids),
    )


def load_fp8_token_embedding(model_id: str, token_id: int) -> FP8TokenEmbeddingResult:
    """Load a single embedding row for one token id."""
    entry = find_tensor_catalog_entry(model_id, "model.embed_tokens.weight")
    blockers: list[str] = []
    tensor = None
    loaded_bytes = 0
    if entry is None:
        blockers.append("model.embed_tokens.weight is not present in the tensor catalog.")
    elif len(entry.shape) != 2:
        blockers.append("model.embed_tokens.weight is not a 2D tensor.")
    elif int(token_id) < 0 or int(token_id) >= int(entry.shape[0]):
        blockers.append(f"Token id {int(token_id)} is outside embedding vocab size {int(entry.shape[0])}.")
    elif entry.dtype not in {"BF16", "F16", "F32"}:
        blockers.append(f"Embedding dtype {entry.dtype} is not supported for row loading.")
    else:
        row = _read_tensor_rows(entry, int(token_id), int(token_id) + 1)
        loaded_bytes = int(entry.shape[1]) * _dtype_element_size(entry.dtype)
        tensor = row.reshape(1, 1, int(entry.shape[1])).contiguous()

    return FP8TokenEmbeddingResult(
        model_id=model_id,
        token_id=int(token_id),
        output_shape=[] if tensor is None else [int(value) for value in tensor.shape],
        loaded_bytes=int(loaded_bytes),
        output_tensor=tensor,
        blockers=blockers,
        ready=not blockers and tensor is not None,
    )


def _load_fp8_prompt_embeddings(model_id: str, token_ids: list[int]) -> tuple[torch.Tensor | None, list[str]]:
    rows: list[torch.Tensor] = []
    blockers: list[str] = []
    for token_id in token_ids:
        loaded = load_fp8_token_embedding(model_id, int(token_id))
        blockers.extend(loaded.blockers)
        if loaded.output_tensor is not None:
            rows.append(loaded.output_tensor)
    if blockers or len(rows) != len(token_ids):
        return None, blockers or ["One or more prompt embedding rows failed to materialize."]
    return torch.cat(rows, dim=1).contiguous(), []


def _run_fp8_prefill_block(
    model_id: str,
    layer_index: int,
    hidden: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> FP8BlockResult:
    hidden_3d = hidden.reshape(1, -1, hidden.shape[-1]) if hidden.ndim == 2 else hidden
    config = _load_deepseek_config(model_id)
    input_norm = _load_regular_tensor(model_id, f"model.layers.{int(layer_index)}.input_layernorm.weight")
    post_norm = _load_regular_tensor(model_id, f"model.layers.{int(layer_index)}.post_attention_layernorm.weight")
    blockers = list(input_norm[1]) + list(post_norm[1])
    attention = None
    moe = None
    dense_mlp = None
    output = None
    next_cache = None
    if not blockers and input_norm[0] is not None and post_norm[0] is not None:
        normed = _rms_norm_any(hidden_3d.to(dtype=dtype), input_norm[0], float(config["rms_norm_eps"]))
        attention = _run_fp8_single_token_attention_materialized(
            model_id,
            layer_index,
            normed,
            dtype=dtype,
            start_pos=0,
            previous_kv_cache=None,
        )
        blockers.extend(attention.blockers)
        if attention.output_tensor is not None:
            hidden_after_attn = hidden_3d.to(dtype=dtype) + attention.output_tensor.to(dtype=dtype)
            post_normed = _rms_norm_any(hidden_after_attn, post_norm[0], float(config["rms_norm_eps"]))
            if int(layer_index) < int(config["first_k_dense_replace"]):
                dense_mlp = run_fp8_dense_mlp(model_id, layer_index, post_normed, dtype=dtype)
                blockers.extend(dense_mlp.blockers)
                if dense_mlp.output_tensor is not None:
                    output = hidden_after_attn + dense_mlp.output_tensor.to(dtype=dtype)
            else:
                moe = run_fp8_moe(model_id, layer_index, post_normed, dtype=dtype)
                blockers.extend(moe.blockers)
                if moe.output_tensor is not None:
                    output = hidden_after_attn + moe.output_tensor.to(dtype=dtype)
            next_cache = attention.kv_cache

    return FP8BlockResult(
        model_id=model_id,
        layer_index=int(layer_index),
        hidden_shape=[int(value) for value in hidden.shape],
        output_shape=[] if output is None else [int(value) for value in output.shape],
        attention=attention,
        moe=moe,
        dense_mlp=dense_mlp,
        output_tensor=None if output is None else output.contiguous(),
        next_kv_cache=next_cache,
        blockers=blockers,
        ready=not blockers and output is not None,
    )


def run_fp8_prompt_prefill(
    model_id: str,
    token_ids: list[int],
    *,
    start_layer: int = 0,
    layer_count: int = 1,
    include_tail: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> FP8PromptPrefillResult:
    """Run a bounded prompt prefill layer-wise so prompt weights are loaded once per layer."""
    prompt = [int(value) for value in token_ids]
    hidden, blockers = _load_fp8_prompt_embeddings(model_id, prompt)
    executed_layers: list[int] = []
    summaries: list[dict] = []
    caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    tail = None
    if not prompt:
        blockers.append("At least one token id is required.")
    if hidden is None:
        blockers.append("Prompt embeddings did not materialize.")

    if not blockers and hidden is not None:
        for layer_index in range(int(start_layer), int(start_layer) + max(0, int(layer_count))):
            layer_start = time.perf_counter()
            step = _run_fp8_prefill_block(model_id, layer_index, hidden.to(dtype=dtype), dtype=dtype)
            summaries.append(
                {
                    "layer_index": int(layer_index),
                    "ready": bool(step.ready),
                    "output_shape": list(step.output_shape),
                    "ffn_type": "dense" if step.dense_mlp is not None else "moe",
                    "selected_experts": [] if step.moe is None else list(step.moe.selected_experts),
                    "cache_sequence_length": 0 if step.next_kv_cache is None else int(step.next_kv_cache[0].shape[1]),
                    "elapsed_seconds": float(time.perf_counter() - layer_start),
                    "blockers": list(step.blockers),
                }
            )
            blockers.extend(step.blockers)
            if not step.ready or step.output_tensor is None:
                break
            hidden = step.output_tensor
            if step.next_kv_cache is not None:
                caches[int(layer_index)] = step.next_kv_cache
            executed_layers.append(int(layer_index))
        if include_tail and not blockers and hidden is not None:
            tail = run_fp8_decode_tail_topk(model_id, hidden[:, -1:, :])
            blockers.extend(tail.blockers)

    return FP8PromptPrefillResult(
        model_id=model_id,
        prompt_token_ids=prompt,
        start_layer=int(start_layer),
        layer_count=int(layer_count),
        executed_layers=executed_layers,
        hidden_shape=[] if hidden is None else [int(value) for value in hidden.shape],
        tail=tail,
        output_tensor=hidden,
        next_kv_caches=caches,
        step_summaries=summaries,
        blockers=blockers,
        ready=not blockers and hidden is not None and (not include_tail or (tail is not None and tail.ready)),
    )


def run_fp8_single_token_forward(
    model_id: str,
    token_id: int,
    *,
    start_layer: int = 0,
    layer_count: int = 1,
    include_tail: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    position: int = 0,
    previous_kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> FP8TokenForwardResult:
    """Run a bounded single-token FP8 forward path from embedding through layers."""
    embedding = load_fp8_token_embedding(model_id, token_id)
    blockers = list(embedding.blockers)
    hidden = embedding.output_tensor
    executed_layers: list[int] = []
    step_summaries: list[dict] = []
    next_kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    tail = None
    if hidden is None:
        blockers.append("Token embedding did not materialize.")

    if not blockers and hidden is not None:
        for layer_index in range(int(start_layer), int(start_layer) + max(0, int(layer_count))):
            step = run_fp8_single_token_block(
                model_id,
                layer_index,
                hidden.to(dtype=dtype),
                dtype=dtype,
                start_pos=int(position),
                previous_kv_cache=None if previous_kv_caches is None else previous_kv_caches.get(layer_index),
            )
            step_summaries.append(
                {
                    "layer_index": int(layer_index),
                    "ready": bool(step.ready),
                    "output_shape": list(step.output_shape),
                    "ffn_type": "dense" if step.dense_mlp is not None else "moe",
                    "selected_experts": [] if step.moe is None else list(step.moe.selected_experts),
                    "cache_sequence_length": 0 if step.next_kv_cache is None else int(step.next_kv_cache[0].shape[1]),
                    "blockers": list(step.blockers),
                }
            )
            blockers.extend(step.blockers)
            if not step.ready or step.output_tensor is None:
                break
            hidden = step.output_tensor
            if step.next_kv_cache is not None:
                next_kv_caches[int(layer_index)] = step.next_kv_cache
            executed_layers.append(int(layer_index))
        if include_tail and not blockers and hidden is not None:
            tail = run_fp8_decode_tail_topk(model_id, hidden)
            blockers.extend(tail.blockers)

    return FP8TokenForwardResult(
        model_id=model_id,
        token_id=int(token_id),
        start_layer=int(start_layer),
        layer_count=int(layer_count),
        executed_layers=executed_layers,
        hidden_shape=[] if hidden is None else [int(value) for value in hidden.shape],
        embedding=embedding,
        tail=tail,
        step_summaries=step_summaries,
        output_tensor=hidden,
        next_kv_caches=next_kv_caches,
        blockers=blockers,
        ready=not blockers and hidden is not None and (not include_tail or (tail is not None and tail.ready)),
    )


def run_fp8_decode_loop(
    model_id: str,
    token_ids: list[int],
    *,
    start_layer: int = 0,
    layer_count: int = 1,
    max_new_tokens: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> FP8DecodeLoopResult:
    """Run a small greedy decode loop using the FP8 KV-carrying token step."""
    prompt = [int(value) for value in token_ids]
    generated: list[int] = []
    blockers: list[str] = []
    summaries: list[dict] = []
    caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    final_top_ids: list[int] = []
    final_top_logits: list[float] = []
    position = 0
    current_tokens = list(prompt)
    if not current_tokens:
        blockers.append("At least one token id is required.")

    if not blockers and len(prompt) > 1 and _fp8_prompt_prefill_enabled():
        prefill_start = time.perf_counter()
        prefill = run_fp8_prompt_prefill(
            model_id,
            prompt,
            start_layer=start_layer,
            layer_count=layer_count,
            include_tail=int(max_new_tokens) > 0,
            dtype=dtype,
        )
        summaries.append(
            {
                "position": 0,
                "phase": "prompt_prefill",
                "token_ids": list(prompt),
                "ready": bool(prefill.ready),
                "executed_layers": list(prefill.executed_layers),
                "tail_top_token_ids": [] if prefill.tail is None else list(prefill.tail.top_token_ids),
                "cache_sequence_lengths": {
                    str(key): int(value[0].shape[1]) for key, value in prefill.next_kv_caches.items()
                },
                "elapsed_seconds": float(time.perf_counter() - prefill_start),
                "blockers": list(prefill.blockers),
            }
        )
        blockers.extend(prefill.blockers)
        if prefill.ready:
            caches = dict(prefill.next_kv_caches)
            if prefill.tail is not None:
                final_top_ids = list(prefill.tail.top_token_ids)
                final_top_logits = list(prefill.tail.top_logits)
            position = len(prompt)

    while not blockers and position < len(prompt) + max(0, int(max_new_tokens)):
        if position < len(prompt):
            token_id = prompt[position]
            include_tail = position == len(prompt) - 1 and int(max_new_tokens) > 0
            phase = "prompt"
        else:
            if not final_top_ids:
                blockers.append("Cannot continue generation because no previous tail token is available.")
                break
            token_id = int(final_top_ids[0])
            generated.append(token_id)
            current_tokens.append(token_id)
            include_tail = len(generated) < int(max_new_tokens)
            phase = "generate"
        step_start = time.perf_counter()
        step = run_fp8_single_token_forward(
            model_id,
            token_id,
            start_layer=start_layer,
            layer_count=layer_count,
            include_tail=include_tail,
            dtype=dtype,
            position=position,
            previous_kv_caches=caches,
        )
        summaries.append(
            {
                "position": int(position),
                "phase": phase,
                "token_id": int(token_id),
                "ready": bool(step.ready),
                "executed_layers": list(step.executed_layers),
                "tail_top_token_ids": [] if step.tail is None else list(step.tail.top_token_ids),
                "cache_sequence_lengths": {
                    str(key): int(value[0].shape[1]) for key, value in step.next_kv_caches.items()
                },
                "elapsed_seconds": float(time.perf_counter() - step_start),
                "blockers": list(step.blockers),
            }
        )
        blockers.extend(step.blockers)
        if not step.ready:
            break
        caches = dict(step.next_kv_caches)
        if step.tail is not None:
            final_top_ids = list(step.tail.top_token_ids)
            final_top_logits = list(step.tail.top_logits)
        position += 1

    return FP8DecodeLoopResult(
        model_id=model_id,
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        start_layer=int(start_layer),
        layer_count=int(layer_count),
        positions_completed=int(position),
        final_top_token_ids=final_top_ids,
        final_top_logits=final_top_logits,
        step_summaries=summaries,
        next_kv_caches=caches,
        blockers=blockers,
        ready=not blockers and position >= len(prompt),
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


def _dtype_element_size(dtype: str) -> int:
    return torch.empty((), dtype=_torch_dtype_for_scale(dtype)).element_size()


def _read_tensor_bytes(entry: TensorCatalogEntry) -> bytes:
    base_offset = _safetensors_data_base_offset(str(entry.shard_path.resolve()), _path_mtime_ns(entry.shard_path))
    with entry.shard_path.open("rb") as handle:
        handle.seek(base_offset + entry.data_offset_start)
        return handle.read(entry.data_nbytes)


def _read_tensor_rows(entry: TensorCatalogEntry, start_row: int, end_row: int) -> torch.Tensor:
    if len(entry.shape) != 2:
        raise ValueError(f"Tensor {entry.tensor_name} is not a 2D row-readable tensor.")
    dtype = _torch_dtype_for_scale(entry.dtype)
    row_count = max(0, int(end_row) - int(start_row))
    cols = int(entry.shape[1])
    row_bytes = cols * torch.empty((), dtype=dtype).element_size()
    base_offset = _safetensors_data_base_offset(str(entry.shard_path.resolve()), _path_mtime_ns(entry.shard_path))
    if native_fp16_loader_available():
        out = torch.empty((row_count, cols), dtype=dtype)
        native_read_tensor_bytes(
            entry.shard_path,
            base_offset + entry.data_offset_start + int(start_row) * row_bytes,
            row_count * row_bytes,
            out,
        )
        return out
    with entry.shard_path.open("rb") as handle:
        handle.seek(base_offset + entry.data_offset_start + int(start_row) * row_bytes)
        raw = handle.read(row_count * row_bytes)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(row_count, cols).clone()


def _read_fp8_weight_rows(entry: TensorCatalogEntry, start_row: int, end_row: int) -> torch.Tensor:
    if len(entry.shape) != 2:
        raise ValueError(f"Tensor {entry.tensor_name} is not a 2D row-readable tensor.")
    row_count = max(0, int(end_row) - int(start_row))
    cols = int(entry.shape[1])
    row_bytes = cols
    base_offset = _safetensors_data_base_offset(str(entry.shard_path.resolve()), _path_mtime_ns(entry.shard_path))
    if native_fp16_loader_available():
        out = torch.empty((row_count, cols), dtype=torch.uint8)
        native_read_tensor_bytes(
            entry.shard_path,
            base_offset + entry.data_offset_start + int(start_row) * row_bytes,
            row_count * row_bytes,
            out,
        )
        return out
    with entry.shard_path.open("rb") as handle:
        handle.seek(base_offset + entry.data_offset_start + int(start_row) * row_bytes)
        raw = handle.read(row_count * row_bytes)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(row_count, cols).contiguous()


def _read_scale_rows_for_weight_chunk(scale: TensorCatalogEntry, start_row: int, end_row: int) -> torch.Tensor:
    scale_start = int(start_row) // 128
    scale_end = (int(end_row) + 127) // 128
    return _read_tensor_rows(scale, scale_start, scale_end).to(dtype=torch.float32)


def _run_fp8_mlp_prefix(
    model_id: str,
    prefix: str,
    hidden: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, int, int, list[str]]:
    gate_name = f"{prefix}.gate_proj.weight"
    up_name = f"{prefix}.up_proj.weight"
    down_name = f"{prefix}.down_proj.weight"
    gate_out, up_out, gate_up_loaded, gate_up_dequant, gate_up_blockers = _run_fp8_dual_linear_streamed(
        model_id, gate_name, up_name, hidden, dtype=dtype
    )
    gate_loaded = gate_up_loaded
    up_loaded = 0
    gate_dequant = gate_up_dequant
    up_dequant = 0
    gate_blockers = gate_up_blockers
    up_blockers: list[str] = []
    blockers = [*gate_blockers, *up_blockers]
    output = None
    down_loaded = 0
    down_dequant = 0
    if gate_out is None or up_out is None:
        blockers.append(f"One or more FP8 MLP gate/up outputs failed for {prefix}.")
    if not blockers and gate_out is not None and up_out is not None:
        expert_hidden = F.silu(gate_out.float()) * up_out.float()
        output, down_loaded, down_dequant, down_blockers = _run_fp8_linear_streamed(
            model_id, down_name, expert_hidden, dtype=dtype
        )
        blockers.extend(down_blockers)

    loaded_bytes = gate_loaded + up_loaded + down_loaded
    dequantized_bytes = gate_dequant + up_dequant + down_dequant
    return output, int(loaded_bytes), int(dequantized_bytes), blockers


def _run_fp8_dual_linear_streamed(
    model_id: str,
    weight_name_a: str,
    weight_name_b: str,
    hidden: torch.Tensor,
    *,
    dtype: torch.dtype,
    chunk_rows: int = 2048,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int, list[str]]:
    if not (_native_fp8_linear_enabled() and native_fp8_dual_linear_available()):
        output_a, loaded_a, dequant_a, blockers_a = _run_fp8_linear_streamed(
            model_id, weight_name_a, hidden, dtype=dtype, chunk_rows=chunk_rows
        )
        output_b, loaded_b, dequant_b, blockers_b = _run_fp8_linear_streamed(
            model_id, weight_name_b, hidden, dtype=dtype, chunk_rows=chunk_rows
        )
        return output_a, output_b, loaded_a + loaded_b, dequant_a + dequant_b, [*blockers_a, *blockers_b]

    weight_a = find_tensor_catalog_entry(model_id, weight_name_a)
    weight_b = find_tensor_catalog_entry(model_id, weight_name_b)
    blockers: list[str] = []
    if weight_a is None:
        blockers.append(f"Tensor {weight_name_a} is not present in the tensor catalog.")
    if weight_b is None:
        blockers.append(f"Tensor {weight_name_b} is not present in the tensor catalog.")
    if blockers or weight_a is None or weight_b is None:
        return None, None, 0, 0, blockers
    if not is_fp8_dtype(weight_a.dtype):
        blockers.append(f"Tensor {weight_name_a} is {weight_a.dtype}, not FP8.")
    if not is_fp8_dtype(weight_b.dtype):
        blockers.append(f"Tensor {weight_name_b} is {weight_b.dtype}, not FP8.")
    if list(weight_a.shape) != list(weight_b.shape):
        blockers.append(f"Dual FP8 tensors {weight_name_a} and {weight_name_b} have different shapes.")
    if not weight_a.scale_tensor_name:
        blockers.append(f"Tensor {weight_name_a} has no scale companion.")
    if not weight_b.scale_tensor_name:
        blockers.append(f"Tensor {weight_name_b} has no scale companion.")
    scale_a = find_tensor_catalog_entry(model_id, weight_a.scale_tensor_name) if weight_a.scale_tensor_name else None
    scale_b = find_tensor_catalog_entry(model_id, weight_b.scale_tensor_name) if weight_b.scale_tensor_name else None
    if scale_a is None:
        blockers.append(f"Scale tensor {weight_a.scale_tensor_name} is not present in the tensor catalog.")
    if scale_b is None:
        blockers.append(f"Scale tensor {weight_b.scale_tensor_name} is not present in the tensor catalog.")
    if len(weight_a.shape) != 2:
        blockers.append(f"Tensor {weight_name_a} is not a 2D weight tensor.")
    if blockers or scale_a is None or scale_b is None:
        return None, None, 0, 0, blockers

    flat_hidden = hidden.reshape(-1, int(hidden.shape[-1])).float()
    out_rows = int(weight_a.shape[0])
    in_cols = int(weight_a.shape[1])
    if flat_hidden.shape[-1] != in_cols:
        return None, None, 0, 0, [f"Hidden size {flat_hidden.shape[-1]} does not match input size {in_cols}."]

    rows_per_chunk = max(128, int(chunk_rows))
    if rows_per_chunk % 128 != 0:
        rows_per_chunk = ((rows_per_chunk + 127) // 128) * 128
    output_a = torch.empty((flat_hidden.shape[0], out_rows), dtype=dtype)
    output_b = torch.empty((flat_hidden.shape[0], out_rows), dtype=dtype)
    loaded_bytes = 0
    dequantized_bytes = 0
    for start in range(0, out_rows, rows_per_chunk):
        end = min(out_rows, start + rows_per_chunk)
        fp8_rows_a = _read_fp8_weight_rows(weight_a, start, end)
        fp8_rows_b = _read_fp8_weight_rows(weight_b, start, end)
        scale_rows_a = _read_scale_rows_for_weight_chunk(scale_a, start, end)
        scale_rows_b = _read_scale_rows_for_weight_chunk(scale_b, start, end)
        loaded_bytes += (
            int(fp8_rows_a.nelement() * fp8_rows_a.element_size())
            + int(fp8_rows_b.nelement() * fp8_rows_b.element_size())
            + int(scale_rows_a.nelement() * scale_rows_a.element_size())
            + int(scale_rows_b.nelement() * scale_rows_b.element_size())
        )
        native_a, native_b = fp8_e4m3_block_dual_linear_f32(
            fp8_rows_a, scale_rows_a, fp8_rows_b, scale_rows_b, flat_hidden
        )
        dequantized_bytes += int(
            2 * fp8_rows_a.shape[0] * fp8_rows_a.shape[1] * torch.empty((), dtype=dtype).element_size()
        )
        output_a[:, start:end] = native_a.to(dtype=dtype)
        output_b[:, start:end] = native_b.to(dtype=dtype)

    return (
        output_a.reshape(*hidden.shape[:-1], out_rows).contiguous(),
        output_b.reshape(*hidden.shape[:-1], out_rows).contiguous(),
        int(loaded_bytes),
        int(dequantized_bytes),
        blockers,
    )


def _run_fp8_linear_streamed(
    model_id: str,
    weight_name: str,
    hidden: torch.Tensor,
    *,
    dtype: torch.dtype,
    chunk_rows: int = 2048,
) -> tuple[torch.Tensor | None, int, int, list[str]]:
    weight = find_tensor_catalog_entry(model_id, weight_name)
    blockers: list[str] = []
    if weight is None:
        return None, 0, 0, [f"Tensor {weight_name} is not present in the tensor catalog."]
    if not is_fp8_dtype(weight.dtype):
        return None, 0, 0, [f"Tensor {weight_name} is {weight.dtype}, not FP8."]
    if not weight.scale_tensor_name:
        return None, 0, 0, [f"Tensor {weight_name} has no scale companion."]
    scale = find_tensor_catalog_entry(model_id, weight.scale_tensor_name)
    if scale is None:
        return None, 0, 0, [f"Scale tensor {weight.scale_tensor_name} is not present in the tensor catalog."]
    if len(weight.shape) != 2:
        return None, 0, 0, [f"Tensor {weight_name} is not a 2D weight tensor."]

    flat_hidden = hidden.reshape(-1, int(hidden.shape[-1])).float()
    out_rows = int(weight.shape[0])
    in_cols = int(weight.shape[1])
    if flat_hidden.shape[-1] != in_cols:
        return None, 0, 0, [f"Hidden size {flat_hidden.shape[-1]} does not match {weight_name} input size {in_cols}."]

    rows_per_chunk = max(128, int(chunk_rows))
    if rows_per_chunk % 128 != 0:
        rows_per_chunk = ((rows_per_chunk + 127) // 128) * 128
    output = torch.empty((flat_hidden.shape[0], out_rows), dtype=dtype)
    loaded_bytes = 0
    dequantized_bytes = 0
    for start in range(0, out_rows, rows_per_chunk):
        end = min(out_rows, start + rows_per_chunk)
        fp8_rows = _read_fp8_weight_rows(weight, start, end)
        scale_rows = _read_scale_rows_for_weight_chunk(scale, start, end)
        loaded_bytes += int(fp8_rows.nelement() * fp8_rows.element_size()) + int(
            scale_rows.nelement() * scale_rows.element_size()
        )
        if _native_fp8_linear_enabled() and native_fp8_linear_available():
            native_out = fp8_e4m3_block_linear_f32(fp8_rows, scale_rows, flat_hidden)
            dequantized_bytes += int(fp8_rows.shape[0] * fp8_rows.shape[1] * torch.empty((), dtype=dtype).element_size())
            output[:, start:end] = native_out.to(dtype=dtype)
        else:
            dequantized = dequantize_fp8_block_scaled(fp8_rows, scale_rows, dtype=dtype)
            dequantized_bytes += int(dequantized.nelement() * dequantized.element_size())
            output[:, start:end] = F.linear(flat_hidden, dequantized.float()).to(dtype=dtype)

    return output.reshape(*hidden.shape[:-1], out_rows).contiguous(), loaded_bytes, dequantized_bytes, blockers


def _mask_deepseek_route_groups(
    scores: torch.Tensor,
    *,
    batch_size: int,
    n_groups: int,
    topk_groups: int,
    use_bias: bool,
) -> torch.Tensor:
    grouped = scores.view(batch_size, int(n_groups), -1)
    if use_bias:
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
    else:
        group_scores = grouped.amax(dim=-1)
    group_indices = group_scores.topk(max(1, min(int(topk_groups), int(n_groups))), dim=-1)[1]
    mask = grouped.new_ones(batch_size, int(n_groups), dtype=torch.bool).scatter_(1, group_indices, False)
    return grouped.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)


def _load_deepseek_route_config(model_id: str) -> dict:
    catalog = load_tensor_catalog(model_id)
    config_path = catalog.model_dir / "config.json"
    payload = {}
    if config_path.exists():
        payload = _load_json_payload(str(config_path), config_path.stat().st_mtime_ns)
    return {
        "top_k": int(payload.get("num_experts_per_tok", payload.get("n_activated_experts", 1)) or 1),
        "n_groups": int(payload.get("n_group", payload.get("n_expert_groups", 1)) or 1),
        "topk_groups": int(payload.get("topk_group", payload.get("n_limited_groups", 1)) or 1),
        "scoring_func": str(payload.get("scoring_func", payload.get("score_func", "softmax"))),
        "route_scale": float(payload.get("routed_scaling_factor", payload.get("route_scale", 1.0)) or 1.0),
    }


def _load_deepseek_config(model_id: str) -> dict:
    catalog = load_tensor_catalog(model_id)
    config_path = catalog.model_dir / "config.json"
    payload = {}
    if config_path.exists():
        payload = _load_json_payload(str(config_path), config_path.stat().st_mtime_ns)
    qk_nope = int(payload.get("qk_nope_head_dim", 128) or 128)
    qk_rope = int(payload.get("qk_rope_head_dim", 64) or 64)
    original_seq_len = int(payload.get("original_max_position_embeddings", 4096) or 4096)
    rope_factor = float((payload.get("rope_scaling") or {}).get("factor", payload.get("rope_factor", 1.0)) or 1.0)
    mscale_all_dim = float((payload.get("rope_scaling") or {}).get("mscale_all_dim", 1.0) or 1.0)
    softmax_scale = (qk_nope + qk_rope) ** -0.5
    max_position = int(payload.get("max_position_embeddings", original_seq_len) or original_seq_len)
    if max_position > original_seq_len and rope_factor > 1.0:
        import math

        mscale = 0.1 * mscale_all_dim * math.log(rope_factor) + 1.0
        softmax_scale *= mscale * mscale
    return {
        "num_attention_heads": int(payload.get("num_attention_heads", catalog.num_attention_heads or 1) or 1),
        "qk_nope_head_dim": qk_nope,
        "qk_rope_head_dim": qk_rope,
        "v_head_dim": int(payload.get("v_head_dim", 128) or 128),
        "kv_lora_rank": int(payload.get("kv_lora_rank", 512) or 512),
        "rms_norm_eps": float(payload.get("rms_norm_eps", 1e-6) or 1e-6),
        "softmax_scale": float(softmax_scale),
        "first_k_dense_replace": int(payload.get("first_k_dense_replace", payload.get("n_dense_layers", 0)) or 0),
        "num_hidden_layers": int(payload.get("num_hidden_layers", catalog.num_hidden_layers or catalog.layer_count) or 0),
        "rope_theta": float(payload.get("rope_theta", 10000.0) or 10000.0),
    }


def _apply_rope_real(x: torch.Tensor, *, start_pos: int, config: dict) -> torch.Tensor:
    if x.shape[-1] == 0:
        return x
    dim = int(x.shape[-1])
    if dim % 2 != 0:
        return x
    device = x.device
    positions = torch.arange(int(start_pos), int(start_pos) + int(x.shape[1]), dtype=torch.float32, device=device)
    freqs = 1.0 / (
        float(config["rope_theta"]) ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    angles = torch.outer(positions, freqs)
    cos = angles.cos().view(1, x.shape[1], *([1] * (x.ndim - 3)), -1)
    sin = angles.sin().view(1, x.shape[1], *([1] * (x.ndim - 3)), -1)
    pair = x.float().reshape(*x.shape[:-1], dim // 2, 2)
    x0 = pair[..., 0]
    x1 = pair[..., 1]
    rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)
    return rotated.to(dtype=x.dtype)


def _load_regular_tensor(model_id: str, tensor_name: str) -> tuple[torch.Tensor | None, list[str]]:
    from pcketlm.core.runtime.tensor_loader import load_tensor_by_name

    loaded = load_tensor_by_name(model_id, tensor_name)
    if not loaded.ready or loaded.tensor is None:
        return None, list(loaded.blockers) or [f"Tensor {tensor_name} did not materialize."]
    return loaded.tensor, []


def _rms_norm_any(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output_dtype = hidden_states.dtype
    hidden_float = hidden_states.float()
    variance = hidden_float.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_float * torch.rsqrt(variance + eps)
    view_shape = [1] * (hidden_states.ndim - 1) + [-1]
    return (normalized * weight.float().view(*view_shape)).to(dtype=output_dtype)


def _streamed_fp8_attention_enabled() -> bool:
    return os.environ.get("PCKETLM_ENABLE_STREAMED_FP8_ATTENTION", "").strip().lower() in {"1", "true", "yes", "on"}


def _fp8_prompt_prefill_enabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_FP8_PROMPT_PREFILL", "").strip().lower() not in {"1", "true", "yes", "on"}


def _native_fp8_linear_enabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_FP8_LINEAR", "").strip().lower() not in {"1", "true", "yes", "on"}


def _native_lm_head_topk_enabled() -> bool:
    return os.environ.get("PCKETLM_ENABLE_NATIVE_LM_HEAD_TOPK", "").strip().lower() in {"1", "true", "yes", "on"}


def _fp8_moe_expert_workers() -> int:
    raw = os.environ.get("PCKETLM_FP8_MOE_EXPERT_WORKERS", "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _run_fp8_single_token_attention_materialized(
    model_id: str,
    layer_index: int,
    hidden: torch.Tensor,
    *,
    dtype: torch.dtype,
    start_pos: int,
    previous_kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
) -> FP8AttentionResult:
    deepseek_config = _load_deepseek_config(model_id)
    blockers: list[str] = []
    hidden_3d = hidden.reshape(1, 1, hidden.shape[-1]) if hidden.ndim == 2 else hidden
    if hidden_3d.ndim != 3 or int(hidden_3d.shape[1]) < 1:
        blockers.append("FP8 attention requires hidden shape [batch, seq, hidden] with at least one token.")
    if previous_kv_cache is not None and int(hidden_3d.shape[1]) != 1:
        blockers.append("FP8 cached attention currently supports one appended token at a time.")

    prefix = f"model.layers.{int(layer_index)}.self_attn"
    weight_names = {
        "q_a": f"{prefix}.q_a_proj.weight",
        "q_b": f"{prefix}.q_b_proj.weight",
        "kv_a": f"{prefix}.kv_a_proj_with_mqa.weight",
        "kv_b": f"{prefix}.kv_b_proj.weight",
        "o": f"{prefix}.o_proj.weight",
    }
    loaded = {role: load_dequantized_fp8_weight(model_id, name, dtype=dtype) for role, name in weight_names.items()}
    blockers.extend(blocker for item in loaded.values() for blocker in item.blockers)
    q_norm = _load_regular_tensor(model_id, f"{prefix}.q_a_layernorm.weight")
    kv_norm = _load_regular_tensor(model_id, f"{prefix}.kv_a_layernorm.weight")
    blockers.extend(q_norm[1])
    blockers.extend(kv_norm[1])

    output = None
    next_cache = None
    if not blockers and all(item.tensor is not None for item in loaded.values()) and q_norm[0] is not None and kv_norm[0] is not None:
        q_a = loaded["q_a"].tensor
        q_b = loaded["q_b"].tensor
        kv_a = loaded["kv_a"].tensor
        kv_b = loaded["kv_b"].tensor
        o_proj = loaded["o"].tensor
        if q_a is None or q_b is None or kv_a is None or kv_b is None or o_proj is None:
            blockers.append("One or more attention weights failed to materialize.")
        else:
            working = hidden_3d.float()
            q_low = F.linear(working, q_a.float())
            q_low = _rms_norm_any(q_low, q_norm[0].float(), float(deepseek_config["rms_norm_eps"]))
            q = F.linear(q_low, q_b.float())
            n_heads = int(deepseek_config["num_attention_heads"])
            qk_nope = int(deepseek_config["qk_nope_head_dim"])
            qk_rope = int(deepseek_config["qk_rope_head_dim"])
            v_head_dim = int(deepseek_config["v_head_dim"])
            kv_lora_rank = int(deepseek_config["kv_lora_rank"])
            seq_len = int(hidden_3d.shape[1])
            q = q.view(int(hidden_3d.shape[0]), seq_len, n_heads, qk_nope + qk_rope)
            q_nope, q_pe = torch.split(q, [qk_nope, qk_rope], dim=-1)
            kv = F.linear(working, kv_a.float())
            kv_latent, k_pe = torch.split(kv, [kv_lora_rank, qk_rope], dim=-1)
            q_pe = _apply_rope_real(q_pe, start_pos=int(start_pos), config=deepseek_config)
            k_pe = _apply_rope_real(k_pe.unsqueeze(2), start_pos=int(start_pos), config=deepseek_config).squeeze(2)
            kv_latent = _rms_norm_any(kv_latent, kv_norm[0].float(), float(deepseek_config["rms_norm_eps"]))
            if previous_kv_cache is not None:
                previous_kv, previous_pe = previous_kv_cache
                kv_cache = torch.cat([previous_kv.to(kv_latent.dtype), kv_latent], dim=1)
                pe_cache = torch.cat([previous_pe.to(k_pe.dtype), k_pe], dim=1)
            else:
                kv_cache = kv_latent
                pe_cache = k_pe
            wkv_b = kv_b.float().view(n_heads, qk_nope + v_head_dim, kv_lora_rank)
            q_nope_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope])
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope_absorbed, kv_cache)
                + torch.einsum("bshr,btr->bsht", q_pe, pe_cache)
            ) * float(deepseek_config["softmax_scale"])
            if previous_kv_cache is None and seq_len > 1:
                causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=scores.device).triu(1)
                scores = scores.masked_fill(causal_mask.view(1, seq_len, 1, seq_len), float("-inf"))
            probs = scores.softmax(dim=-1, dtype=torch.float32).to(dtype=working.dtype)
            attention_latent = torch.einsum("bsht,btc->bshc", probs, kv_cache)
            attention_heads = torch.einsum("bshc,hdc->bshd", attention_latent, wkv_b[:, -v_head_dim:])
            output = F.linear(attention_heads.flatten(2), o_proj.float()).to(dtype=dtype).contiguous()
            next_cache = (kv_cache.detach().contiguous(), pe_cache.detach().contiguous())

    dequantized_bytes = sum(
        0 if item.tensor is None else item.tensor.nelement() * item.tensor.element_size()
        for item in loaded.values()
    )
    loaded_bytes = sum(item.loaded_nbytes for item in loaded.values())
    return FP8AttentionResult(
        model_id=model_id,
        layer_index=int(layer_index),
        hidden_shape=[int(value) for value in hidden.shape],
        output_shape=[] if output is None else [int(value) for value in output.shape],
        loaded_weight_bytes=int(loaded_bytes),
        dequantized_weight_bytes=int(dequantized_bytes),
        output_tensor=output,
        kv_cache=next_cache,
        blockers=blockers,
        ready=not blockers and output is not None,
    )


@lru_cache(maxsize=16)
def _load_json_payload(path: str, mtime_ns: int) -> dict:
    del mtime_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path_mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


@lru_cache(maxsize=128)
def _safetensors_data_base_offset(path: str, mtime_ns: int) -> int:
    del mtime_ns
    with Path(path).open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
    return 8 + int(header_length)
