"""FP8-native source helpers for paged DeepSeek-style safetensors."""

from __future__ import annotations

import struct
import json
import math
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
        for expert_index in selected:
            row_indices, top_indices = torch.where(route_indices == int(expert_index))
            if row_indices.numel() == 0:
                continue
            expert_hidden = flat_hidden[row_indices].to(dtype=dtype)
            expert_output, loaded_bytes, expert_dequant_bytes, expert_blockers = _run_fp8_mlp_prefix(
                model_id,
                f"model.layers.{int(layer_index)}.mlp.experts.{int(expert_index)}",
                expert_hidden,
                dtype=dtype,
            )
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
    deepseek_config = _load_deepseek_config(model_id)
    blockers: list[str] = []
    hidden_3d = hidden.reshape(1, 1, hidden.shape[-1]) if hidden.ndim == 2 else hidden
    if hidden_3d.ndim != 3 or int(hidden_3d.shape[1]) != 1:
        blockers.append("FP8 attention proof currently supports exactly one token.")

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
            q = q.view(int(hidden_3d.shape[0]), 1, n_heads, qk_nope + qk_rope)
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
        normalized = _rms_norm_any(hidden_2d, norm_tensor.float(), float(config["rms_norm_eps"])).float()
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
                weight = _read_tensor_rows(lm_head_entry, start, end).float()
                loaded_bytes += (end - start) * hidden_size * _dtype_element_size(lm_head_entry.dtype)
                chunk_logits = F.linear(normalized, weight).reshape(-1)
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
    with entry.shard_path.open("rb") as handle:
        handle.seek(base_offset + entry.data_offset_start + int(start_row) * row_bytes)
        raw = handle.read(row_count * row_bytes)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(row_count, cols).clone()


def _run_fp8_mlp_prefix(
    model_id: str,
    prefix: str,
    hidden: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, int, int, list[str]]:
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
            blockers.append(f"One or more FP8 MLP weights failed to materialize for {prefix}.")
        else:
            working = hidden.to(dtype=torch.float32)
            gate_out = F.linear(working, gate.to(torch.float32))
            up_out = F.linear(working, up.to(torch.float32))
            expert_hidden = F.silu(gate_out) * up_out
            output = F.linear(expert_hidden, down.to(torch.float32)).to(dtype=dtype).contiguous()

    dequantized_bytes = sum(
        0 if item.tensor is None else item.tensor.nelement() * item.tensor.element_size()
        for item in loaded.values()
    )
    loaded_bytes = sum(item.loaded_nbytes for item in loaded.values())
    return output, int(loaded_bytes), int(dequantized_bytes), blockers


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
