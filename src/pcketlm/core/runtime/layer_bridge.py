"""Minimal CPU-only layer-forward bridge built from real loaded execution units."""

from __future__ import annotations

import json
import math
import os
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open

from pcketlm.core.runtime.tensor_loader import open_scoped_tensor_handle, scoped_tensor_handle_cache
from pcketlm.core.runtime.tensor_residency import (
    TensorResidencyPolicy,
    expert_residency_snapshot,
    fp16_packed_expert_cache_scope,
    load_resident_tensor,
    load_resident_tensors,
    record_expert_activation,
)
from pcketlm.core.runtime.tensor_catalog import TensorCatalogEntry, find_tensor_catalog_entry, load_tensor_catalog
from pcketlm.core.runtime.tokenizer_runtime import (
    decode_token_ids_to_text,
    load_generation_settings,
    prepare_prompt_text,
)
from pcketlm.core.storage.paths import original_model_root


DEFAULT_LM_HEAD_CHUNK_ROWS = 8192
DEFAULT_RUNTIME_MATH_DTYPE = "bfloat16"
DEFAULT_TORCH_THREAD_CAP = 14
CANCEL_BLOCKER = "Generation canceled by user."
_ROPE_CACHE_MAX_ENTRIES = 32
_rope_cache: OrderedDict[tuple[int, float, str, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
_CAUSAL_MASK_CACHE_MAX_ENTRIES = 32
_causal_mask_cache: OrderedDict[tuple[str, tuple[int, ...], tuple[int, ...]], torch.Tensor] = OrderedDict()


def _cancel_requested(should_cancel: Callable[[], bool] | None) -> bool:
    return bool(should_cancel and should_cancel())


def _trim_generated_text_at_stop_string(generated_text: str, stop_strings: list[str]) -> tuple[str, str | None]:
    """Remove the first generated stop marker from visible assistant text."""
    matches = [(generated_text.find(value), value) for value in stop_strings if value and value in generated_text]
    if not matches:
        return generated_text, None
    index, value = min(matches, key=lambda match: match[0])
    return generated_text[:index], value


def _auto_torch_thread_count(cpu_count: int | None = None) -> int:
    available = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    if available >= 6:
        return max(1, min(DEFAULT_TORCH_THREAD_CAP, available - 2))
    return max(1, min(DEFAULT_TORCH_THREAD_CAP, available))


def runtime_torch_thread_count() -> int:
    """Return the configured PyTorch CPU thread count for runtime math."""
    requested = os.environ.get("PCKETLM_TORCH_THREADS", "auto").strip().lower()
    if requested in {"", "auto"}:
        return _auto_torch_thread_count()
    try:
        return max(1, int(requested))
    except ValueError:
        return _auto_torch_thread_count()


def configure_runtime_threads() -> int:
    """Apply the runtime's PyTorch CPU thread setting."""
    target = runtime_torch_thread_count()
    if torch.get_num_threads() != target:
        torch.set_num_threads(target)
    return torch.get_num_threads()


@dataclass(slots=True)
class LayerBridgeModelConfig:
    """The model config values needed for the first layer bridge."""

    model_id: str
    config_path: Path
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    model_type: str
    vocab_size: int
    rms_norm_eps: float
    hidden_act: str
    source_dtype: str
    max_position_embeddings: int
    rope_theta: float
    bos_token_id: int | None
    pad_token_id: int | None
    eos_token_ids: list[int] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "config_path": str(self.config_path),
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "norm_topk_prob": self.norm_topk_prob,
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_act": self.hidden_act,
            "source_dtype": self.source_dtype,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token_ids": list(self.eos_token_ids),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class LayerBridgeResult:
    """Result of one minimal layer-forward bridge execution."""

    model_id: str
    layer_index: int
    input_mode: str
    input_shape: list[int]
    output_shape: list[int]
    output_dtype: str
    loaded_unit_ids: list[str] = field(default_factory=list)
    attention_head_dim: int = 0
    cache_sequence_length: int = 0
    output_mean_abs: float = 0.0
    output_l2_norm: float = 0.0
    blockers: list[str] = field(default_factory=list)
    ready: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    output_tensor: torch.Tensor | None = None
    next_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    native_kv_session: Any | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "layer_index": self.layer_index,
            "input_mode": self.input_mode,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "loaded_unit_ids": list(self.loaded_unit_ids),
            "attention_head_dim": self.attention_head_dim,
            "cache_sequence_length": self.cache_sequence_length,
            "output_mean_abs": self.output_mean_abs,
            "output_l2_norm": self.output_l2_norm,
            "blockers": list(self.blockers),
            "ready": self.ready,
            "timings": dict(self.timings),
        }


@dataclass(slots=True)
class LayerBridgeStepSummary:
    """Compact summary of one executed layer-bridge step."""

    layer_index: int
    output_shape: list[int]
    output_dtype: str
    output_mean_abs: float
    output_l2_norm: float
    loaded_unit_ids: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "output_mean_abs": self.output_mean_abs,
            "output_l2_norm": self.output_l2_norm,
            "loaded_unit_ids": list(self.loaded_unit_ids),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class LayerBridgeStackResult:
    """Result of chaining multiple minimal layer-bridge steps."""

    model_id: str
    start_layer: int
    layer_count: int
    input_mode: str
    input_shape: list[int]
    output_shape: list[int]
    output_dtype: str
    executed_layers: list[int] = field(default_factory=list)
    step_summaries: list[LayerBridgeStepSummary] = field(default_factory=list)
    cache_sequence_lengths: dict[str, int] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    output_tensor: torch.Tensor | None = None
    next_kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    next_native_kv_sessions: dict[int, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "start_layer": self.start_layer,
            "layer_count": self.layer_count,
            "input_mode": self.input_mode,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "executed_layers": list(self.executed_layers),
            "step_summaries": [summary.to_dict() for summary in self.step_summaries],
            "cache_sequence_lengths": {str(key): int(value) for key, value in self.cache_sequence_lengths.items()},
            "blockers": list(self.blockers),
            "ready": self.ready,
            "timings": dict(self.timings),
        }


@dataclass(slots=True)
class TokenEntryBridgeResult:
    """Result of entering the runtime from token ids and then executing the layer bridge stack."""

    model_id: str
    token_ids: list[int]
    input_shape: list[int]
    embedding_shape: list[int]
    output_shape: list[int]
    output_dtype: str
    start_layer: int
    layer_count: int
    executed_layers: list[int] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False
    output_tensor: torch.Tensor | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "token_ids": list(self.token_ids),
            "input_shape": list(self.input_shape),
            "embedding_shape": list(self.embedding_shape),
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "start_layer": self.start_layer,
            "layer_count": self.layer_count,
            "executed_layers": list(self.executed_layers),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class DecodeTailResult:
    """Result of the final norm + streamed lm_head decode tail."""

    model_id: str
    input_shape: list[int]
    normalized_shape: list[int]
    logits_shape: list[int]
    logits_dtype: str
    vocab_size: int
    chunk_rows: int
    chunk_count: int
    top_token_ids: list[int] = field(default_factory=list)
    top_logits: list[float] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False
    logits: torch.Tensor | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "input_shape": list(self.input_shape),
            "normalized_shape": list(self.normalized_shape),
            "logits_shape": list(self.logits_shape),
            "logits_dtype": self.logits_dtype,
            "vocab_size": self.vocab_size,
            "chunk_rows": self.chunk_rows,
            "chunk_count": self.chunk_count,
            "top_token_ids": list(self.top_token_ids),
            "top_logits": list(self.top_logits),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class TokenDecodeStepResult:
    """Result of token entry + bridge stack + decode tail."""

    model_id: str
    token_ids: list[int]
    context_token_ids: list[int]
    context_mode: str
    history_window: int
    selection_policy: str
    chosen_token_id: int | None
    start_layer: int
    layer_count: int
    embedding_shape: list[int]
    hidden_shape: list[int]
    logits_shape: list[int]
    logits_dtype: str
    executed_layers: list[int] = field(default_factory=list)
    top_token_ids: list[int] = field(default_factory=list)
    top_logits: list[float] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False
    logits: torch.Tensor | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "token_ids": list(self.token_ids),
            "context_token_ids": list(self.context_token_ids),
            "context_mode": self.context_mode,
            "history_window": self.history_window,
            "selection_policy": self.selection_policy,
            "chosen_token_id": self.chosen_token_id,
            "start_layer": self.start_layer,
            "layer_count": self.layer_count,
            "embedding_shape": list(self.embedding_shape),
            "hidden_shape": list(self.hidden_shape),
            "logits_shape": list(self.logits_shape),
            "logits_dtype": self.logits_dtype,
            "executed_layers": list(self.executed_layers),
            "top_token_ids": list(self.top_token_ids),
            "top_logits": list(self.top_logits),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class RepeatedDecodeStepSummary:
    """One greedy decode step inside the first repeated loop."""

    step_index: int
    input_token_id: int
    chosen_token_id: int | None
    logits_shape: list[int]
    top_token_ids: list[int] = field(default_factory=list)
    top_logits: list[float] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "input_token_id": self.input_token_id,
            "chosen_token_id": self.chosen_token_id,
            "logits_shape": list(self.logits_shape),
            "top_token_ids": list(self.top_token_ids),
            "top_logits": list(self.top_logits),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class RepeatedDecodeLoopResult:
    """Result of the first repeated greedy decode loop."""

    model_id: str
    seed_token_id: int
    steps_requested: int
    steps_completed: int
    strategy: str
    selection_policy: str
    context_mode: str
    history_window: int
    layer_count: int
    generated_token_ids: list[int] = field(default_factory=list)
    step_summaries: list[RepeatedDecodeStepSummary] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "seed_token_id": self.seed_token_id,
            "steps_requested": self.steps_requested,
            "steps_completed": self.steps_completed,
            "strategy": self.strategy,
            "selection_policy": self.selection_policy,
            "context_mode": self.context_mode,
            "history_window": self.history_window,
            "layer_count": self.layer_count,
            "generated_token_ids": list(self.generated_token_ids),
            "step_summaries": [summary.to_dict() for summary in self.step_summaries],
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class KVDecodeStepResult:
    """Result of one first-generation KV-carrying decode step."""

    model_id: str
    input_token_id: int
    cache_sequence_lengths: dict[str, int]
    logits_shape: list[int]
    logits_dtype: str
    executed_layers: list[int] = field(default_factory=list)
    chosen_token_id: int | None = None
    top_token_ids: list[int] = field(default_factory=list)
    top_logits: list[float] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    logits: torch.Tensor | None = None
    next_kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    next_decode_state: KVDecodeState | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "input_token_id": self.input_token_id,
            "cache_sequence_lengths": {str(key): int(value) for key, value in self.cache_sequence_lengths.items()},
            "logits_shape": list(self.logits_shape),
            "logits_dtype": self.logits_dtype,
            "executed_layers": list(self.executed_layers),
            "chosen_token_id": self.chosen_token_id,
            "top_token_ids": list(self.top_token_ids),
            "top_logits": list(self.top_logits),
            "blockers": list(self.blockers),
            "ready": self.ready,
            "timings": dict(self.timings),
            "next_decode_state": None if self.next_decode_state is None else self.next_decode_state.to_dict(),
        }


@dataclass(slots=True)
class KVDecodeLoopResult:
    """Result of the first KV-carrying repeated decode loop."""

    model_id: str
    seed_token_id: int
    steps_requested: int
    steps_completed: int
    strategy: str
    selection_policy: str
    layer_count: int
    stop_reason: str | None = None
    generated_token_ids: list[int] = field(default_factory=list)
    cache_sequence_lengths: dict[str, int] = field(default_factory=dict)
    step_summaries: list[RepeatedDecodeStepSummary] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "seed_token_id": self.seed_token_id,
            "steps_requested": self.steps_requested,
            "steps_completed": self.steps_completed,
            "strategy": self.strategy,
            "selection_policy": self.selection_policy,
            "layer_count": self.layer_count,
            "stop_reason": self.stop_reason,
            "generated_token_ids": list(self.generated_token_ids),
            "cache_sequence_lengths": {str(key): int(value) for key, value in self.cache_sequence_lengths.items()},
            "step_summaries": [summary.to_dict() for summary in self.step_summaries],
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class TokenSelectionResult:
    """Result of choosing the next token from one logits vector."""

    policy: str
    chosen_token_id: int | None
    top_token_ids: list[int] = field(default_factory=list)
    top_logits: list[float] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "chosen_token_id": self.chosen_token_id,
            "top_token_ids": list(self.top_token_ids),
            "top_logits": list(self.top_logits),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class DecodeBenchmarkCaseResult:
    """One comparable decode run inside a small benchmark."""

    label: str
    strategy: str
    steps_completed: int = 0
    final_token_id: int | None = None
    unique_token_count: int = 0
    stop_reason: str | None = None
    generated_token_ids: list[int] = field(default_factory=list)
    cache_sequence_lengths: dict[str, int] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "strategy": self.strategy,
            "steps_completed": self.steps_completed,
            "final_token_id": self.final_token_id,
            "unique_token_count": self.unique_token_count,
            "stop_reason": self.stop_reason,
            "generated_token_ids": list(self.generated_token_ids),
            "cache_sequence_lengths": {str(key): int(value) for key, value in self.cache_sequence_lengths.items()},
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class DecodeBenchmarkResult:
    """Small comparative benchmark across multiple decode paths."""

    model_id: str
    seed_token_id: int
    steps_requested: int
    layer_count: int
    history_window: int
    sample_seed: int | None
    cases: list[DecodeBenchmarkCaseResult] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "seed_token_id": self.seed_token_id,
            "steps_requested": self.steps_requested,
            "layer_count": self.layer_count,
            "history_window": self.history_window,
            "sample_seed": self.sample_seed,
            "case_count": len(self.cases),
            "ready_case_count": sum(1 for case in self.cases if case.ready),
            "cases": [case.to_dict() for case in self.cases],
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class PromptDecodeLoopResult:
    """Result of the first real text-prompt entry path."""

    model_id: str
    prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str
    full_text: str
    steps_requested: int
    max_new_tokens: int
    min_new_tokens: int
    steps_completed: int
    strategy: str
    stop_reason: str | None = None
    stop_token_ids: list[int] = field(default_factory=list)
    stop_strings: list[str] = field(default_factory=list)
    cache_sequence_lengths: dict[str, int] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    token_summaries: list[dict] = field(default_factory=list)
    prefix_reuse: dict = field(default_factory=dict)
    reusable_token_ids: list[int] = field(default_factory=list)
    configured_layer_count: int = 0
    prompt_layer_count: int = 0
    layers_executed: int = 0
    expected_layers_executed: int = 0
    anti_cheat_passed: bool = False
    final_decode_state: KVDecodeState | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "prompt": self.prompt,
            "prompt_token_ids": list(self.prompt_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
            "generated_text": self.generated_text,
            "full_text": self.full_text,
            "steps_requested": self.steps_requested,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "steps_completed": self.steps_completed,
            "strategy": self.strategy,
            "stop_reason": self.stop_reason,
            "stop_token_ids": list(self.stop_token_ids),
            "stop_strings": list(self.stop_strings),
            "cache_sequence_lengths": {str(key): int(value) for key, value in self.cache_sequence_lengths.items()},
            "blockers": list(self.blockers),
            "ready": self.ready,
            "timings": dict(self.timings),
            "token_summaries": [dict(summary) for summary in self.token_summaries],
            "prefix_reuse": dict(self.prefix_reuse),
            "reusable_token_count": len(self.reusable_token_ids),
            "configured_layer_count": int(self.configured_layer_count),
            "prompt_layer_count": int(self.prompt_layer_count),
            "layers_executed": int(self.layers_executed),
            "expected_layers_executed": int(self.expected_layers_executed),
            "anti_cheat_passed": bool(self.anti_cheat_passed),
        }


@dataclass(slots=True)
class KVDecodeState:
    """Minimal explicit carried decode state for the K/V-aware runtime path."""

    model_id: str
    next_token_id: int
    next_position: int
    generated_token_ids: list[int] = field(default_factory=list)
    cache_sequence_lengths: dict[str, int] = field(default_factory=dict)
    kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    native_kv_sessions: dict[int, Any] = field(default_factory=dict)
    finished: bool = False
    stop_reason: str | None = None
    ready: bool = False
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "next_token_id": self.next_token_id,
            "next_position": self.next_position,
            "generated_token_ids": list(self.generated_token_ids),
            "cache_sequence_lengths": {str(key): int(value) for key, value in self.cache_sequence_lengths.items()},
            "finished": self.finished,
            "stop_reason": self.stop_reason,
            "ready": self.ready,
            "blockers": list(self.blockers),
        }


def load_layer_bridge_config(model_id: str) -> LayerBridgeModelConfig:
    """Load the minimal set of config values needed for the CPU bridge."""
    config_path = original_model_root(model_id) / "config.json"
    generation_config_path = original_model_root(model_id) / "generation_config.json"
    config_mtime = config_path.stat().st_mtime_ns if config_path.exists() else 0
    generation_mtime = generation_config_path.stat().st_mtime_ns if generation_config_path.exists() else 0
    return _load_layer_bridge_config_cached(model_id, str(config_path), config_mtime, str(generation_config_path), generation_mtime)


@lru_cache(maxsize=16)
def _load_layer_bridge_config_cached(
    model_id: str,
    config_path_string: str,
    config_mtime_ns: int,
    generation_config_path_string: str,
    generation_mtime_ns: int,
) -> LayerBridgeModelConfig:
    """Build the layer bridge config once per unchanged metadata file set."""
    del config_mtime_ns, generation_mtime_ns
    config_path = Path(config_path_string)
    generation_config_path = Path(generation_config_path_string)
    if not config_path.exists():
        return LayerBridgeModelConfig(
            model_id=model_id,
            config_path=config_path,
            hidden_size=0,
            num_hidden_layers=0,
            num_attention_heads=0,
            num_key_value_heads=0,
            head_dim=0,
            intermediate_size=0,
            moe_intermediate_size=0,
            num_experts=0,
            num_experts_per_tok=0,
            norm_topk_prob=False,
            model_type="unknown",
            vocab_size=0,
            rms_norm_eps=0.0,
            hidden_act="unknown",
            source_dtype="unknown",
            max_position_embeddings=0,
            rope_theta=10000.0,
            bos_token_id=None,
            pad_token_id=None,
            eos_token_ids=[],
            blockers=[f"Missing model config at {config_path}."],
            ready=False,
        )

    payload = _load_json_payload(str(config_path), config_path.stat().st_mtime_ns)
    generation_payload: dict = {}
    if generation_config_path.exists():
        generation_payload = _load_json_payload(str(generation_config_path), generation_config_path.stat().st_mtime_ns)
    blockers: list[str] = []
    required_fields = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "rms_norm_eps",
        "hidden_act",
    ]
    for field_name in required_fields:
        if field_name not in payload:
            blockers.append(f"Missing config field {field_name} in {config_path.name}.")

    hidden_act = str(payload.get("hidden_act", "unknown"))
    if hidden_act != "silu":
        blockers.append(
            f"Minimal layer bridge currently supports hidden_act=silu only, not {hidden_act}."
        )

    hidden_size = int(payload.get("hidden_size", 0))
    num_attention_heads = int(payload.get("num_attention_heads", 0))
    num_key_value_heads = int(payload.get("num_key_value_heads", 0))
    configured_head_dim = int(payload.get("head_dim", 0) or 0)
    effective_head_dim = configured_head_dim or (hidden_size // num_attention_heads if num_attention_heads > 0 else 0)
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        blockers.append("Attention head counts must be positive.")
    elif effective_head_dim <= 0:
        blockers.append("Attention head dimension must be positive.")
    elif configured_head_dim <= 0 and hidden_size % num_attention_heads != 0:
        blockers.append(
            f"Hidden size {hidden_size} must be divisible by attention heads {num_attention_heads}."
        )
    elif num_attention_heads % num_key_value_heads != 0:
        blockers.append(
            f"Attention heads {num_attention_heads} must be divisible by KV heads {num_key_value_heads}."
        )

    raw_eos = generation_payload.get("eos_token_id", payload.get("eos_token_id"))
    eos_token_ids: list[int]
    if raw_eos is None:
        eos_token_ids = []
    elif isinstance(raw_eos, list):
        eos_token_ids = [int(value) for value in raw_eos]
    else:
        eos_token_ids = [int(raw_eos)]

    num_experts = int(payload.get("num_experts", payload.get("num_local_experts", 0)) or 0)
    moe_intermediate_size = int(payload.get("moe_intermediate_size", 0) or 0)
    if moe_intermediate_size <= 0 and num_experts > 0:
        moe_intermediate_size = int(payload.get("intermediate_size", 0) or 0)

    norm_topk_payload = payload.get("norm_topk_prob")
    norm_topk_prob = bool(norm_topk_payload) if norm_topk_payload is not None else bool(
        num_experts > 0 and int(payload.get("num_experts_per_tok", 0) or 0) > 1
    )

    return LayerBridgeModelConfig(
        model_id=model_id,
        config_path=config_path,
        hidden_size=hidden_size,
        num_hidden_layers=int(payload.get("num_hidden_layers", 0)),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=effective_head_dim,
        intermediate_size=int(payload.get("intermediate_size", 0)),
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=int(payload.get("num_experts_per_tok", 0) or 0),
        norm_topk_prob=norm_topk_prob,
        model_type=str(payload.get("model_type", "unknown")),
        vocab_size=int(payload.get("vocab_size", 0)),
        rms_norm_eps=float(payload.get("rms_norm_eps", 0.0)),
        hidden_act=hidden_act,
        source_dtype=str(payload.get("torch_dtype", "unknown")),
        max_position_embeddings=int(payload.get("max_position_embeddings", 0)),
        rope_theta=float(payload.get("rope_theta", 10000.0)),
        bos_token_id=None if payload.get("bos_token_id") is None else int(payload.get("bos_token_id")),
        pad_token_id=None if generation_payload.get("pad_token_id", payload.get("pad_token_id")) is None else int(generation_payload.get("pad_token_id", payload.get("pad_token_id"))),
        eos_token_ids=eos_token_ids,
        blockers=blockers,
        ready=not blockers,
    )


@lru_cache(maxsize=32)
def _load_json_payload(path: str, mtime_ns: int) -> dict:
    """Read small JSON metadata with automatic invalidation when the file changes."""
    del mtime_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output_dtype = hidden_states.dtype
    hidden_float = hidden_states.float()
    variance = hidden_float.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_float * torch.rsqrt(variance + eps)
    return (normalized * weight.float().view(1, 1, -1)).to(dtype=output_dtype)


def _run_moe_mlp(
    *,
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    expert_tensors: dict[int, dict[str, torch.Tensor]],
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Run router + top-k expert FFN for one MoE layer."""
    output_dtype = hidden_states.dtype
    router_logits = F.linear(hidden_states.float(), router_weight.float())
    router_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    effective_top_k = max(1, min(int(top_k), router_probs.shape[-1]))
    routing_weights, selected_experts = torch.topk(router_probs, effective_top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    if (
        hidden_states.ndim == 3
        and list(hidden_states.shape[:2]) == [1, 1]
        and output_dtype in {torch.float16, torch.bfloat16}
        and os.environ.get("PCKETLM_DISABLE_NATIVE_MOE", "").strip().lower() not in {"1", "true", "yes", "on"}
    ):
        selected_ids = [int(value) for value in selected_experts.reshape(-1).tolist()]
        if all(expert_id in expert_tensors for expert_id in selected_ids):
            try:
                from pcketlm.native import moe_selected_forward_u16

                gate_stack = torch.stack(
                    [expert_tensors[expert_id]["gate_proj"].to(dtype=output_dtype) for expert_id in selected_ids],
                    dim=0,
                )
                up_stack = torch.stack(
                    [expert_tensors[expert_id]["up_proj"].to(dtype=output_dtype) for expert_id in selected_ids],
                    dim=0,
                )
                down_stack = torch.stack(
                    [expert_tensors[expert_id]["down_proj"].to(dtype=output_dtype) for expert_id in selected_ids],
                    dim=0,
                )
                native_out = moe_selected_forward_u16(
                    hidden_states.reshape(1, -1).to(dtype=output_dtype),
                    gate_stack,
                    up_stack,
                    down_stack,
                    routing_weights.reshape(1, -1).float(),
                )
                return native_out.view_as(hidden_states), sorted(set(selected_ids)), selected_experts
            except Exception:
                pass

    combined = torch.zeros_like(hidden_states.float())
    touched: set[int] = set()
    for expert_index, tensors in expert_tensors.items():
        mask = selected_experts == int(expert_index)
        if not bool(mask.any()):
            continue
        touched.add(int(expert_index))
        gate_weight = tensors["gate_proj"].float()
        up_weight = tensors["up_proj"].float()
        down_weight = tensors["down_proj"].float()
        expert_hidden = F.silu(F.linear(hidden_states.float(), gate_weight)) * F.linear(hidden_states.float(), up_weight)
        expert_output = F.linear(expert_hidden, down_weight)
        expert_weight = torch.where(mask, routing_weights, torch.zeros_like(routing_weights)).sum(dim=-1, keepdim=True)
        combined = combined + expert_output * expert_weight
    return combined.to(dtype=output_dtype), sorted(touched), selected_experts


def _repeat_kv(hidden_states: torch.Tensor, repeat_count: int) -> torch.Tensor:
    if repeat_count == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeat_count, dim=1)


def _attention_impl_name() -> str:
    requested = os.environ.get("PCKETLM_ATTENTION_IMPL", "sdp").strip().lower()
    return "manual" if requested in {"manual", "classic"} else "sdp"


def _causal_attention_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    query_key = tuple(int(value) for value in query_positions.detach().cpu().tolist())
    key_key = tuple(int(value) for value in key_positions.detach().cpu().tolist())
    cache_key = (str(device), query_key, key_key)
    cached = _causal_mask_cache.get(cache_key)
    if cached is not None:
        _causal_mask_cache.move_to_end(cache_key)
        return cached
    mask = key_positions.view(1, 1, 1, -1) <= query_positions.view(1, 1, -1, 1)
    mask = mask.to(device=device)
    _causal_mask_cache[cache_key] = mask
    if len(_causal_mask_cache) > _CAUSAL_MASK_CACHE_MAX_ENTRIES:
        _causal_mask_cache.popitem(last=False)
    return mask


def _attention_context(
    q_states: torch.Tensor,
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    causal_mask: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if _attention_impl_name() == "sdp":
        try:
            return F.scaled_dot_product_attention(
                q_states,
                k_states,
                v_states,
                attn_mask=causal_mask,
                dropout_p=0.0,
            )
        except RuntimeError:
            pass
    attention_scores = torch.matmul(q_states, k_states.transpose(-1, -2)) / math.sqrt(head_dim)
    attention_scores = attention_scores.masked_fill(~causal_mask, torch.finfo(attention_scores.dtype).min)
    attention_probs = torch.softmax(attention_scores, dim=-1)
    return torch.matmul(attention_probs, v_states)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    half_dim = hidden_states.shape[-1] // 2
    return torch.cat(
        [-hidden_states[..., half_dim:], hidden_states[..., :half_dim]],
        dim=-1,
    )


def _apply_rotary_position_embedding(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    output_dtype = hidden_states.dtype
    head_dim = hidden_states.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head dimension, got {head_dim}.")

    position_key = tuple(int(value) for value in positions.detach().cpu().tolist())
    cache_key = (head_dim, float(theta), str(hidden_states.device), position_key)
    cached = _rope_cache.get(cache_key)
    if cached is None:
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=hidden_states.device) / head_dim)
        )
        freqs = torch.outer(positions.to(device=hidden_states.device, dtype=torch.float32), inv_freq)
        rotary = torch.cat([freqs, freqs], dim=-1).view(1, 1, positions.shape[0], head_dim)
        cos = rotary.cos()
        sin = rotary.sin()
        _rope_cache[cache_key] = (cos, sin)
        if len(_rope_cache) > _ROPE_CACHE_MAX_ENTRIES:
            _rope_cache.popitem(last=False)
    else:
        cos, sin = cached
        _rope_cache.move_to_end(cache_key)
    hidden_float = hidden_states.float()
    rotated = (hidden_float * cos) + (_rotate_half(hidden_float) * sin)
    return rotated.to(dtype=output_dtype)


def _synthetic_hidden_state(hidden_size: int) -> torch.Tensor:
    return torch.linspace(-0.5, 0.5, hidden_size, dtype=torch.float32).view(1, 1, hidden_size)


def runtime_math_dtype_name() -> str:
    """Return the current runtime math dtype name."""
    requested = os.environ.get("PCKETLM_RUNTIME_MATH_DTYPE", DEFAULT_RUNTIME_MATH_DTYPE).strip().lower()
    if requested in {"bf16", "bfloat16", "torch.bfloat16"}:
        return "bfloat16"
    return "float32"


def _runtime_math_dtype() -> torch.dtype:
    return torch.bfloat16 if runtime_math_dtype_name() == "bfloat16" else torch.float32


def _fp16_decode_expert_packed_cache_enabled() -> bool:
    return os.environ.get("PCKETLM_ENABLE_FP16_DECODE_EXPERT_PACKED_CACHE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_moe_config(config: LayerBridgeModelConfig) -> bool:
    return config.num_experts > 0 and config.num_experts_per_tok > 0


def _tensor_entry_exists(model_id: str, tensor_name: str) -> bool:
    return find_tensor_catalog_entry(model_id, tensor_name) is not None


def _moe_router_tensor_name(model_id: str, layer_index: int) -> str:
    """Return the router tensor name for the model's MoE layout."""
    candidates = (
        f"model.layers.{layer_index}.mlp.gate.weight",
        f"model.layers.{layer_index}.block_sparse_moe.gate.weight",
    )
    for candidate in candidates:
        if _tensor_entry_exists(model_id, candidate):
            return candidate
    return candidates[0]


def _moe_expert_tensor_name_map(model_id: str, layer_index: int, expert_index: int) -> dict[str, str]:
    """Map canonical expert roles to tensor names for the model's MoE layout."""
    qwen_prefix = f"model.layers.{layer_index}.mlp.experts.{expert_index}"
    mixtral_prefix = f"model.layers.{layer_index}.block_sparse_moe.experts.{expert_index}"
    candidates = (
        {
            "gate_proj": f"{qwen_prefix}.gate_proj.weight",
            "up_proj": f"{qwen_prefix}.up_proj.weight",
            "down_proj": f"{qwen_prefix}.down_proj.weight",
        },
        {
            "gate_proj": f"{mixtral_prefix}.w1.weight",
            "up_proj": f"{mixtral_prefix}.w3.weight",
            "down_proj": f"{mixtral_prefix}.w2.weight",
        },
    )
    for candidate in candidates:
        if all(_tensor_entry_exists(model_id, tensor_name) for tensor_name in candidate.values()):
            return candidate
    return candidates[0]


def _layer_prefetch_enabled() -> bool:
    if os.environ.get("PCKETLM_DISABLE_LAYER_PREFETCH", "0").strip().lower() in {"1", "true", "yes"}:
        return False
    return os.environ.get("PCKETLM_ENABLE_LAYER_PREFETCH", "0").strip().lower() in {"1", "true", "yes"}


def _native_layer_enabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_LAYER", "").strip().lower() not in {"1", "true", "yes", "on"}


def _native_dense_prefill_enabled() -> bool:
    return os.environ.get("PCKETLM_ENABLE_NATIVE_DENSE_PREFILL", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_packed_gemv_layer_enabled() -> bool:
    return os.environ.get("PCKETLM_ENABLE_NATIVE_PACKED_GEMV_LAYER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _layer_tensor_names(layer_index: int) -> list[str]:
    return [
        f"model.layers.{layer_index}.input_layernorm.weight",
        f"model.layers.{layer_index}.post_attention_layernorm.weight",
        f"model.layers.{layer_index}.self_attn.q_proj.weight",
        f"model.layers.{layer_index}.self_attn.q_proj.bias",
        f"model.layers.{layer_index}.self_attn.k_proj.weight",
        f"model.layers.{layer_index}.self_attn.k_proj.bias",
        f"model.layers.{layer_index}.self_attn.v_proj.weight",
        f"model.layers.{layer_index}.self_attn.v_proj.bias",
        f"model.layers.{layer_index}.self_attn.o_proj.weight",
        f"model.layers.{layer_index}.mlp.gate_proj.weight",
        f"model.layers.{layer_index}.mlp.up_proj.weight",
        f"model.layers.{layer_index}.mlp.down_proj.weight",
    ]


def _recommended_prompt_layer_count(
    config: LayerBridgeModelConfig,
    prompt_token_count: int,
    max_new_tokens: int,
) -> int:
    """Use the full configured layer stack for default prompt runs."""
    del prompt_token_count, max_new_tokens
    if not config.ready or config.num_hidden_layers <= 0:
        return 2

    return config.num_hidden_layers


def initialize_kv_decode_state(model_id: str, seed_token_id: int) -> KVDecodeState:
    """Create the first explicit decode state for the K/V-aware runtime path."""
    blockers: list[str] = []
    if seed_token_id < 0:
        blockers.append("Seed token id must be 0 or greater.")
    return KVDecodeState(
        model_id=model_id,
        next_token_id=seed_token_id,
        next_position=0,
        generated_token_ids=[seed_token_id],
        cache_sequence_lengths={},
        kv_caches={},
        finished=False,
        stop_reason=None,
        ready=not blockers,
        blockers=blockers,
    )


def _load_required_tensor(
    model_id: str,
    tensor_name: str,
    blockers: list[str],
    dtype: torch.dtype | None = None,
    policy: TensorResidencyPolicy | None = None,
) -> torch.Tensor | None:
    loaded = load_resident_tensor(
        model_id,
        tensor_name,
        dtype=_runtime_math_dtype() if dtype is None else dtype,
        policy=policy,
    )
    blockers.extend(loaded.blockers)
    if not loaded.ready or loaded.tensor is None:
        blockers.append(f"Required tensor {tensor_name} could not be loaded.")
        return None
    return loaded.tensor


def _reshape_past_kv_for_native(
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
    config: LayerBridgeModelConfig,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    head_dim = config.head_dim
    kv_width = config.num_key_value_heads * head_dim
    if past_key_value is None:
        empty = torch.empty((0, kv_width), dtype=dtype)
        return empty, empty, 0
    past_keys, past_values = past_key_value
    past_length = int(past_keys.shape[-2])
    k_native = (
        past_keys.detach()
        .cpu()
        .to(dtype=dtype)
        .reshape(1, config.num_key_value_heads, past_length, head_dim)
        .squeeze(0)
        .permute(1, 0, 2)
        .contiguous()
        .view(past_length, kv_width)
    )
    v_native = (
        past_values.detach()
        .cpu()
        .to(dtype=dtype)
        .reshape(1, config.num_key_value_heads, past_length, head_dim)
        .squeeze(0)
        .permute(1, 0, 2)
        .contiguous()
        .view(past_length, kv_width)
    )
    return k_native, v_native, past_length


def _reshape_native_kv_for_python(
    k_native: torch.Tensor,
    v_native: torch.Tensor,
    config: LayerBridgeModelConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_len = int(k_native.shape[0])
    head_dim = config.head_dim
    k_python = (
        k_native.reshape(total_len, config.num_key_value_heads, head_dim)
        .permute(1, 0, 2)
        .contiguous()
        .view(1, config.num_key_value_heads, total_len, head_dim)
    )
    v_python = (
        v_native.reshape(total_len, config.num_key_value_heads, head_dim)
        .permute(1, 0, 2)
        .contiguous()
        .view(1, config.num_key_value_heads, total_len, head_dim)
    )
    return k_python, v_python


def _try_native_dense_decode_bridge(
    model_id: str,
    layer_index: int,
    hidden_states: torch.Tensor,
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
    native_kv_session: Any | None,
    config: LayerBridgeModelConfig,
    tensor_policy: TensorResidencyPolicy | None,
    collect_metrics: bool,
    timings: dict[str, float],
    native_kv_commit: bool = True,
    prefetched_tensors: dict[str, torch.Tensor] | None = None,
) -> LayerBridgeResult | None:
    if not _native_layer_enabled() or _is_moe_config(config):
        return None
    if hidden_states.ndim != 3 or list(hidden_states.shape[:2]) != [1, 1]:
        return None
    math_dtype = _runtime_math_dtype()
    if math_dtype not in {torch.float16, torch.bfloat16}:
        return None
    try:
        from pcketlm.native import NativeKvSession, native_fp16_kv_available
        from pcketlm.native import cached_pack_weight_rows8

        if not native_fp16_kv_available():
            return None
    except Exception:
        return None

    head_dim = config.head_dim
    kv_width = config.num_key_value_heads * head_dim
    required_names = [
        f"model.layers.{layer_index}.input_layernorm.weight",
        f"model.layers.{layer_index}.post_attention_layernorm.weight",
        f"model.layers.{layer_index}.self_attn.q_proj.weight",
        f"model.layers.{layer_index}.self_attn.k_proj.weight",
        f"model.layers.{layer_index}.self_attn.v_proj.weight",
        f"model.layers.{layer_index}.self_attn.o_proj.weight",
        f"model.layers.{layer_index}.mlp.gate_proj.weight",
        f"model.layers.{layer_index}.mlp.up_proj.weight",
        f"model.layers.{layer_index}.mlp.down_proj.weight",
    ]
    optional_names = [
        f"model.layers.{layer_index}.self_attn.q_proj.bias",
        f"model.layers.{layer_index}.self_attn.k_proj.bias",
        f"model.layers.{layer_index}.self_attn.v_proj.bias",
        f"model.layers.{layer_index}.self_attn.q_norm.weight",
        f"model.layers.{layer_index}.self_attn.k_norm.weight",
    ]
    present_optional = [name for name in optional_names if _tensor_entry_exists(model_id, name)]
    tensors: dict[str, torch.Tensor] = {}
    tensor_names = required_names + present_optional
    if prefetched_tensors is not None and all(tensor_name in prefetched_tensors for tensor_name in tensor_names):
        tensors = {tensor_name: prefetched_tensors[tensor_name].to(dtype=math_dtype) for tensor_name in tensor_names}
    else:
        load_started = time.perf_counter()
        loaded = load_resident_tensors(
            model_id,
            tensor_names,
            dtype=math_dtype,
            policy=tensor_policy,
        )
        timings["load_tensors"] = round(timings.get("load_tensors", 0.0) + (time.perf_counter() - load_started), 4)
        for tensor_name in tensor_names:
            loaded_slice = loaded[tensor_name]
            if not loaded_slice.ready or loaded_slice.tensor is None:
                return None
            tensors[tensor_name] = loaded_slice.tensor.to(dtype=math_dtype)

    session = None
    try:
        if native_kv_session is None:
            past_k_native, past_v_native, past_length = _reshape_past_kv_for_native(past_key_value, config, math_dtype)
            max_seq_len = max(config.max_position_embeddings, past_length + 1, 1)
            session = NativeKvSession(
                layer_count=1,
                max_seq_len=max_seq_len,
                kv_width=kv_width,
                dtype=math_dtype,
            )
            if past_length:
                session.append_committed(0, past_k_native, past_v_native, count=past_length)
        else:
            session = native_kv_session
            past_length = session.committed_length(0) + session.tentative_length(0)
        native_started = time.perf_counter()
        hidden_flat = hidden_states.reshape(-1).to(dtype=math_dtype)
        input_norm = tensors[f"model.layers.{layer_index}.input_layernorm.weight"]
        post_norm = tensors[f"model.layers.{layer_index}.post_attention_layernorm.weight"]
        q_weight = tensors[f"model.layers.{layer_index}.self_attn.q_proj.weight"]
        k_weight = tensors[f"model.layers.{layer_index}.self_attn.k_proj.weight"]
        v_weight = tensors[f"model.layers.{layer_index}.self_attn.v_proj.weight"]
        o_weight = tensors[f"model.layers.{layer_index}.self_attn.o_proj.weight"]
        gate_weight = tensors[f"model.layers.{layer_index}.mlp.gate_proj.weight"]
        up_weight = tensors[f"model.layers.{layer_index}.mlp.up_proj.weight"]
        down_weight = tensors[f"model.layers.{layer_index}.mlp.down_proj.weight"]
        common_kwargs = {
            "rms_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "q_bias": tensors.get(f"model.layers.{layer_index}.self_attn.q_proj.bias"),
            "k_bias": tensors.get(f"model.layers.{layer_index}.self_attn.k_proj.bias"),
            "v_bias": tensors.get(f"model.layers.{layer_index}.self_attn.v_proj.bias"),
            "q_norm_weight": tensors.get(f"model.layers.{layer_index}.self_attn.q_norm.weight"),
            "k_norm_weight": tensors.get(f"model.layers.{layer_index}.self_attn.k_norm.weight"),
        }
        if _native_packed_gemv_layer_enabled() and hasattr(session, "dense_layer_decode_packed_rows8"):
            pack_started = time.perf_counter()
            q_packed = cached_pack_weight_rows8(f"{model_id}:{layer_index}:q", q_weight)
            k_packed = cached_pack_weight_rows8(f"{model_id}:{layer_index}:k", k_weight)
            v_packed = cached_pack_weight_rows8(f"{model_id}:{layer_index}:v", v_weight)
            o_packed = cached_pack_weight_rows8(f"{model_id}:{layer_index}:o", o_weight)
            gate_packed = cached_pack_weight_rows8(f"{model_id}:{layer_index}:gate", gate_weight)
            up_packed = cached_pack_weight_rows8(f"{model_id}:{layer_index}:up", up_weight)
            down_packed = cached_pack_weight_rows8(f"{model_id}:{layer_index}:down", down_weight)
            timings["pack_weights"] = round(
                timings.get("pack_weights", 0.0) + (time.perf_counter() - pack_started),
                4,
            )
            output = session.dense_layer_decode_packed_rows8(
                0,
                hidden_flat,
                input_norm,
                post_norm,
                q_packed,
                k_packed,
                v_packed,
                o_packed,
                gate_packed,
                up_packed,
                down_packed,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                **common_kwargs,
            )
        else:
            output = session.dense_layer_decode_fp16(
                0,
                hidden_flat,
                input_norm,
                post_norm,
                q_weight,
                k_weight,
                v_weight,
                o_weight,
                gate_weight,
                up_weight,
                down_weight,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                **common_kwargs,
            )
        if native_kv_commit:
            session.commit(1)
        timings["native_layer"] = round(timings.get("native_layer", 0.0) + (time.perf_counter() - native_started), 4)
        if native_kv_session is None:
            next_kv_cache = None
        else:
            next_kv_cache = None
    except Exception:
        if native_kv_session is None and session is not None:
            try:
                session.close()
            except Exception:
                pass
        return None

    layer_output = output.view(1, 1, config.hidden_size).to(dtype=math_dtype)
    output_mean_abs = float(layer_output.abs().mean().item()) if collect_metrics else 0.0
    output_l2_norm = float(torch.linalg.vector_norm(layer_output.float()).item()) if collect_metrics else 0.0
    return LayerBridgeResult(
        model_id=model_id,
        layer_index=layer_index,
        input_mode="provided",
        input_shape=[int(value) for value in hidden_states.shape],
        output_shape=[int(value) for value in layer_output.shape],
        output_dtype=str(layer_output.dtype),
        loaded_unit_ids=[
            f"layer-{layer_index:02d}-layer_norm",
            f"layer-{layer_index:02d}-attention",
            f"layer-{layer_index:02d}-mlp",
        ],
        attention_head_dim=head_dim,
        cache_sequence_length=past_length + 1,
        output_mean_abs=output_mean_abs,
        output_l2_norm=output_l2_norm,
        blockers=[],
        ready=True,
        timings=timings,
        output_tensor=layer_output,
        next_kv_cache=next_kv_cache,
        native_kv_session=session,
    )


def _try_native_dense_prefill_bridge(
    *,
    model_id: str,
    layer_index: int,
    hidden_states: torch.Tensor,
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
    native_kv_session: Any | None,
    config: LayerBridgeModelConfig,
    tensor_policy: TensorResidencyPolicy | None,
    collect_metrics: bool,
    timings: dict[str, float],
    prefetched_tensors: dict[str, torch.Tensor] | None = None,
) -> LayerBridgeResult | None:
    if not _native_layer_enabled() or _is_moe_config(config):
        return None
    if not _native_dense_prefill_enabled():
        return None
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1 or hidden_states.shape[1] <= 1:
        return None
    math_dtype = _runtime_math_dtype()
    if math_dtype not in {torch.float16, torch.bfloat16}:
        return None
    try:
        from pcketlm.native import NativeKvSession, native_fp16_kv_available

        if not native_fp16_kv_available():
            return None
    except Exception:
        return None

    head_dim = config.head_dim
    kv_width = config.num_key_value_heads * head_dim
    required_names = [
        f"model.layers.{layer_index}.input_layernorm.weight",
        f"model.layers.{layer_index}.post_attention_layernorm.weight",
        f"model.layers.{layer_index}.self_attn.q_proj.weight",
        f"model.layers.{layer_index}.self_attn.k_proj.weight",
        f"model.layers.{layer_index}.self_attn.v_proj.weight",
        f"model.layers.{layer_index}.self_attn.o_proj.weight",
        f"model.layers.{layer_index}.mlp.gate_proj.weight",
        f"model.layers.{layer_index}.mlp.up_proj.weight",
        f"model.layers.{layer_index}.mlp.down_proj.weight",
    ]
    optional_names = [
        f"model.layers.{layer_index}.self_attn.q_proj.bias",
        f"model.layers.{layer_index}.self_attn.k_proj.bias",
        f"model.layers.{layer_index}.self_attn.v_proj.bias",
        f"model.layers.{layer_index}.self_attn.q_norm.weight",
        f"model.layers.{layer_index}.self_attn.k_norm.weight",
    ]
    present_optional = [name for name in optional_names if _tensor_entry_exists(model_id, name)]
    tensors: dict[str, torch.Tensor] = {}
    tensor_names = required_names + present_optional
    if prefetched_tensors is not None and all(tensor_name in prefetched_tensors for tensor_name in tensor_names):
        tensors = {tensor_name: prefetched_tensors[tensor_name].to(dtype=math_dtype) for tensor_name in tensor_names}
    else:
        load_started = time.perf_counter()
        loaded = load_resident_tensors(
            model_id,
            tensor_names,
            dtype=math_dtype,
            policy=tensor_policy,
        )
        timings["load_tensors"] = round(timings.get("load_tensors", 0.0) + (time.perf_counter() - load_started), 4)
        for tensor_name in tensor_names:
            loaded_slice = loaded[tensor_name]
            if not loaded_slice.ready or loaded_slice.tensor is None:
                return None
            tensors[tensor_name] = loaded_slice.tensor.to(dtype=math_dtype)

    session = None
    try:
        if native_kv_session is None:
            past_k_native, past_v_native, past_length = _reshape_past_kv_for_native(past_key_value, config, math_dtype)
            max_seq_len = max(config.max_position_embeddings, past_length + int(hidden_states.shape[1]), 1)
            session = NativeKvSession(
                layer_count=1,
                max_seq_len=max_seq_len,
                kv_width=kv_width,
                dtype=math_dtype,
            )
            if past_length:
                session.append_committed(0, past_k_native, past_v_native, count=past_length)
        else:
            session = native_kv_session
            past_length = session.committed_length(0) + session.tentative_length(0)
        native_started = time.perf_counter()
        output = session.dense_layer_prefill_fp16(
            0,
            hidden_states.reshape(int(hidden_states.shape[1]), config.hidden_size).to(dtype=math_dtype),
            tensors[f"model.layers.{layer_index}.input_layernorm.weight"],
            tensors[f"model.layers.{layer_index}.post_attention_layernorm.weight"],
            tensors[f"model.layers.{layer_index}.self_attn.q_proj.weight"],
            tensors[f"model.layers.{layer_index}.self_attn.k_proj.weight"],
            tensors[f"model.layers.{layer_index}.self_attn.v_proj.weight"],
            tensors[f"model.layers.{layer_index}.self_attn.o_proj.weight"],
            tensors[f"model.layers.{layer_index}.mlp.gate_proj.weight"],
            tensors[f"model.layers.{layer_index}.mlp.up_proj.weight"],
            tensors[f"model.layers.{layer_index}.mlp.down_proj.weight"],
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rms_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            q_bias=tensors.get(f"model.layers.{layer_index}.self_attn.q_proj.bias"),
            k_bias=tensors.get(f"model.layers.{layer_index}.self_attn.k_proj.bias"),
            v_bias=tensors.get(f"model.layers.{layer_index}.self_attn.v_proj.bias"),
            q_norm_weight=tensors.get(f"model.layers.{layer_index}.self_attn.q_norm.weight"),
            k_norm_weight=tensors.get(f"model.layers.{layer_index}.self_attn.k_norm.weight"),
        )
        timings["native_layer"] = round(timings.get("native_layer", 0.0) + (time.perf_counter() - native_started), 4)
    except Exception:
        if native_kv_session is None and session is not None:
            try:
                session.close()
            except Exception:
                pass
        return None

    layer_output = output.view(1, int(hidden_states.shape[1]), config.hidden_size).to(dtype=math_dtype)
    output_mean_abs = float(layer_output.abs().mean().item()) if collect_metrics else 0.0
    output_l2_norm = float(torch.linalg.vector_norm(layer_output.float()).item()) if collect_metrics else 0.0
    return LayerBridgeResult(
        model_id=model_id,
        layer_index=layer_index,
        input_mode="provided",
        input_shape=[int(value) for value in hidden_states.shape],
        output_shape=[int(value) for value in layer_output.shape],
        output_dtype=str(layer_output.dtype),
        loaded_unit_ids=[
            f"layer-{layer_index:02d}-layer_norm",
            f"layer-{layer_index:02d}-attention",
            f"layer-{layer_index:02d}-mlp",
        ],
        attention_head_dim=head_dim,
        cache_sequence_length=past_length + int(hidden_states.shape[1]),
        output_mean_abs=output_mean_abs,
        output_l2_norm=output_l2_norm,
        blockers=[],
        ready=True,
        timings=timings,
        output_tensor=layer_output,
        next_kv_cache=None,
        native_kv_session=session,
    )


def _try_native_attention_decode_bridge(
    model_id: str,
    layer_index: int,
    normed_input: torch.Tensor,
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
    native_kv_session: Any | None,
    config: LayerBridgeModelConfig,
    tensor_policy: TensorResidencyPolicy | None,
    timings: dict[str, float],
    native_kv_commit: bool = True,
) -> tuple[torch.Tensor, int, Any] | None:
    if not _native_layer_enabled() or os.environ.get("PCKETLM_DISABLE_NATIVE_ATTENTION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    if normed_input.ndim != 3 or list(normed_input.shape[:2]) != [1, 1]:
        return None
    math_dtype = _runtime_math_dtype()
    if math_dtype not in {torch.float16, torch.bfloat16}:
        return None
    try:
        from pcketlm.native import NativeKvSession, native_fp16_kv_available

        if not native_fp16_kv_available():
            return None
    except Exception:
        return None

    head_dim = config.head_dim
    kv_width = config.num_key_value_heads * head_dim
    required_names = [
        f"model.layers.{layer_index}.self_attn.q_proj.weight",
        f"model.layers.{layer_index}.self_attn.k_proj.weight",
        f"model.layers.{layer_index}.self_attn.v_proj.weight",
        f"model.layers.{layer_index}.self_attn.o_proj.weight",
    ]
    optional_names = [
        f"model.layers.{layer_index}.self_attn.q_proj.bias",
        f"model.layers.{layer_index}.self_attn.k_proj.bias",
        f"model.layers.{layer_index}.self_attn.v_proj.bias",
        f"model.layers.{layer_index}.self_attn.q_norm.weight",
        f"model.layers.{layer_index}.self_attn.k_norm.weight",
    ]
    present_optional = [name for name in optional_names if _tensor_entry_exists(model_id, name)]
    load_started = time.perf_counter()
    loaded = load_resident_tensors(
        model_id,
        required_names + present_optional,
        dtype=math_dtype,
        policy=tensor_policy,
    )
    timings["load_tensors"] = round(timings.get("load_tensors", 0.0) + (time.perf_counter() - load_started), 4)
    tensors: dict[str, torch.Tensor] = {}
    for tensor_name in required_names + present_optional:
        loaded_slice = loaded[tensor_name]
        if not loaded_slice.ready or loaded_slice.tensor is None:
            return None
        tensors[tensor_name] = loaded_slice.tensor.to(dtype=math_dtype)

    session = None
    try:
        if native_kv_session is None:
            past_k_native, past_v_native, past_length = _reshape_past_kv_for_native(past_key_value, config, math_dtype)
            max_seq_len = max(config.max_position_embeddings, past_length + 1, 1)
            session = NativeKvSession(
                layer_count=1,
                max_seq_len=max_seq_len,
                kv_width=kv_width,
                dtype=math_dtype,
            )
            if past_length:
                session.append_committed(0, past_k_native, past_v_native, count=past_length)
        else:
            session = native_kv_session
            past_length = session.committed_length(0) + session.tentative_length(0)
        native_started = time.perf_counter()
        output = session.attention_decode_fp16(
            0,
            normed_input.reshape(-1).to(dtype=math_dtype),
            tensors[f"model.layers.{layer_index}.self_attn.q_proj.weight"],
            tensors[f"model.layers.{layer_index}.self_attn.k_proj.weight"],
            tensors[f"model.layers.{layer_index}.self_attn.v_proj.weight"],
            tensors[f"model.layers.{layer_index}.self_attn.o_proj.weight"],
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            head_dim=config.head_dim,
            q_bias=tensors.get(f"model.layers.{layer_index}.self_attn.q_proj.bias"),
            k_bias=tensors.get(f"model.layers.{layer_index}.self_attn.k_proj.bias"),
            v_bias=tensors.get(f"model.layers.{layer_index}.self_attn.v_proj.bias"),
            q_norm_weight=tensors.get(f"model.layers.{layer_index}.self_attn.q_norm.weight"),
            k_norm_weight=tensors.get(f"model.layers.{layer_index}.self_attn.k_norm.weight"),
            rms_eps=config.rms_norm_eps,
        )
        if native_kv_commit:
            session.commit(1)
        timings["native_attention"] = round(
            timings.get("native_attention", 0.0) + (time.perf_counter() - native_started),
            4,
        )
    except Exception:
        if native_kv_session is None and session is not None:
            try:
                session.close()
            except Exception:
                pass
        return None

    return output.view(1, 1, config.hidden_size).to(dtype=math_dtype), past_length + 1, session


def _find_embedding_entry(model_id: str) -> TensorCatalogEntry | None:
    catalog = load_tensor_catalog(model_id)
    if not catalog.ready:
        return None
    for entry in catalog.tensors:
        if entry.tensor_name == "model.embed_tokens.weight":
            return entry
    return None


def _find_tensor_entry(model_id: str, tensor_name: str) -> TensorCatalogEntry | None:
    return find_tensor_catalog_entry(model_id, tensor_name)


def select_next_token(
    logits: torch.Tensor,
    policy: str = "greedy",
    top_k: int = 5,
    top_p: float = 1.0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    recent_token_ids: list[int] | None = None,
    sample_seed: int | None = None,
) -> TokenSelectionResult:
    """Choose the next token from one logits vector."""
    blockers: list[str] = []
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != 1:
        blockers.append(
            f"Token selection expects logits shaped [1, 1, vocab], got {list(logits.shape)}."
        )
        return TokenSelectionResult(
            policy=policy,
            chosen_token_id=None,
            blockers=blockers,
            ready=False,
        )
    if top_k <= 0:
        blockers.append("top_k must be at least 1.")
    if top_p <= 0 or top_p > 1:
        blockers.append("top_p must be greater than 0 and no more than 1.")
    if temperature <= 0:
        blockers.append("temperature must be greater than 0.")
    if repetition_penalty <= 0:
        blockers.append("repetition_penalty must be greater than 0.")
    if blockers:
        return TokenSelectionResult(
            policy=policy,
            chosen_token_id=None,
            blockers=blockers,
            ready=False,
        )

    logits_vector = logits.view(-1).float().clone()
    if repetition_penalty != 1.0 and recent_token_ids:
        vocab_size = logits_vector.shape[0]
        for token_id in set(int(value) for value in recent_token_ids):
            if token_id < 0 or token_id >= vocab_size:
                continue
            value = logits_vector[token_id]
            logits_vector[token_id] = value * repetition_penalty if value < 0 else value / repetition_penalty
    k = min(top_k, logits_vector.shape[0])
    top_logits, top_token_ids = torch.topk(logits_vector, k=k)

    if policy == "greedy":
        chosen_token_id = int(top_token_ids[0].item())
    elif policy == "top-k-sample":
        sorted_logits, sorted_indices = torch.sort(top_logits, descending=True)
        sorted_token_ids = top_token_ids[sorted_indices]
        probabilities = torch.softmax(sorted_logits / temperature, dim=-1)
        if top_p < 1.0:
            cumulative = torch.cumsum(probabilities, dim=-1)
            keep_mask = cumulative <= top_p
            keep_mask[0] = True
            sorted_logits = sorted_logits[keep_mask]
            sorted_token_ids = sorted_token_ids[keep_mask]
            probabilities = torch.softmax(sorted_logits / temperature, dim=-1)
        generator = None
        if sample_seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(sample_seed)
        sampled_index = int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())
        chosen_token_id = int(sorted_token_ids[sampled_index].item())
    else:
        return TokenSelectionResult(
            policy=policy,
            chosen_token_id=None,
            top_token_ids=[int(value) for value in top_token_ids.tolist()],
            top_logits=[float(value) for value in top_logits.tolist()],
            blockers=[f"Unknown token-selection policy: {policy}."],
            ready=False,
        )

    return TokenSelectionResult(
        policy=policy,
        chosen_token_id=chosen_token_id,
        top_token_ids=[int(value) for value in top_token_ids.tolist()],
        top_logits=[float(value) for value in top_logits.tolist()],
        blockers=[],
        ready=True,
    )


def _can_select_from_topk(policy: str) -> bool:
    return policy in {"greedy", "top-k-sample"}


def select_next_token_from_topk(
    top_token_ids: list[int],
    top_logits: list[float],
    *,
    policy: str = "greedy",
    top_p: float = 1.0,
    temperature: float = 1.0,
    sample_seed: int | None = None,
) -> TokenSelectionResult:
    """Choose the next token from already-streamed top-k candidates."""
    blockers: list[str] = []
    if not top_token_ids or not top_logits:
        blockers.append("Token selection requires at least one streamed top-k candidate.")
    if len(top_token_ids) != len(top_logits):
        blockers.append("Streamed top-k token ids and logits must have the same length.")
    if top_p <= 0 or top_p > 1:
        blockers.append("top_p must be greater than 0 and no more than 1.")
    if temperature <= 0:
        blockers.append("temperature must be greater than 0.")
    if blockers:
        return TokenSelectionResult(policy=policy, chosen_token_id=None, blockers=blockers, ready=False)

    candidate_logits = torch.tensor(top_logits, dtype=torch.float32)
    candidate_token_ids = torch.tensor(top_token_ids, dtype=torch.long)
    sorted_logits, sorted_indices = torch.sort(candidate_logits, descending=True)
    sorted_token_ids = candidate_token_ids[sorted_indices]

    if policy == "greedy":
        chosen_token_id = int(sorted_token_ids[0].item())
    elif policy == "top-k-sample":
        probabilities = torch.softmax(sorted_logits / temperature, dim=-1)
        if top_p < 1.0:
            cumulative = torch.cumsum(probabilities, dim=-1)
            keep_mask = cumulative <= top_p
            keep_mask[0] = True
            sorted_logits = sorted_logits[keep_mask]
            sorted_token_ids = sorted_token_ids[keep_mask]
            probabilities = torch.softmax(sorted_logits / temperature, dim=-1)
        generator = None
        if sample_seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(sample_seed)
        sampled_index = int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())
        chosen_token_id = int(sorted_token_ids[sampled_index].item())
    else:
        return TokenSelectionResult(
            policy=policy,
            chosen_token_id=None,
            top_token_ids=[int(value) for value in sorted_token_ids.tolist()],
            top_logits=[float(value) for value in sorted_logits.tolist()],
            blockers=[f"Unknown token-selection policy: {policy}."],
            ready=False,
        )

    return TokenSelectionResult(
        policy=policy,
        chosen_token_id=chosen_token_id,
        top_token_ids=[int(value) for value in sorted_token_ids.tolist()],
        top_logits=[float(value) for value in sorted_logits.tolist()],
        blockers=[],
        ready=True,
    )


def load_token_entry_hidden_state(model_id: str, token_ids: list[int]) -> tuple[torch.Tensor | None, list[str]]:
    """Load one or more embedding rows as the initial hidden state without loading the full table."""
    blockers: list[str] = []
    if not token_ids:
        return None, ["At least one token id is required for token entry."]

    entry = _find_embedding_entry(model_id)
    if entry is None:
        return None, ["Embedding table is not present in the tensor catalog yet."]

    vocab_size = entry.shape[0] if entry.shape else 0
    hidden_size = entry.shape[1] if len(entry.shape) > 1 else 0
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab_size:
            blockers.append(f"Token id {token_id} is outside the embedding table range 0..{max(vocab_size - 1, 0)}.")

    if blockers:
        return None, blockers

    with safe_open(entry.shard_path, framework="pt", device="cpu") as handle:
        embedding_slice = handle.get_slice(entry.tensor_name)
        rows = embedding_slice[min(token_ids) : max(token_ids) + 1]

    selected_rows = []
    offset = min(token_ids)
    math_dtype = _runtime_math_dtype()
    for token_id in token_ids:
        selected_rows.append(rows[token_id - offset].to(dtype=math_dtype))

    hidden_state = torch.stack(selected_rows, dim=0).view(1, len(token_ids), hidden_size)
    return hidden_state, []


def build_history_summary_hidden_state(
    model_id: str,
    token_ids: list[int],
    history_window: int = 1,
) -> tuple[torch.Tensor | None, list[int], str, list[str]]:
    """Build a small explicit decode state from recent token embeddings."""
    blockers: list[str] = []
    if history_window <= 0:
        blockers.append("History window must be at least 1.")
        return None, [], "invalid", blockers
    if not token_ids:
        blockers.append("At least one token id is required to build decode history.")
        return None, [], "invalid", blockers

    context_token_ids = list(token_ids[-history_window:])
    hidden_state, entry_blockers = load_token_entry_hidden_state(model_id, context_token_ids)
    blockers.extend(entry_blockers)
    if blockers or hidden_state is None:
        return None, context_token_ids, "invalid", blockers

    if hidden_state.shape[1] == 1:
        return hidden_state, context_token_ids, "single-token-entry", []

    weights = torch.arange(1, hidden_state.shape[1] + 1, dtype=torch.float32).view(1, hidden_state.shape[1], 1)
    summary_hidden = (hidden_state * weights).sum(dim=1, keepdim=True) / weights.sum()
    return summary_hidden, context_token_ids, "history-summary-no-kv-cache", []


@torch.inference_mode()
def run_minimal_layer_forward_bridge(
    model_id: str,
    layer_index: int = 0,
    input_hidden: torch.Tensor | None = None,
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    position_offset: int | None = None,
    return_kv_cache: bool = False,
    collect_metrics: bool = True,
    tensor_policy: TensorResidencyPolicy | None = None,
    prefetched_tensors: dict[str, torch.Tensor] | None = None,
    native_kv_session: Any | None = None,
    native_kv_commit: bool = True,
) -> LayerBridgeResult:
    """Run one real CPU-only layer slice using real loaded layer tensors."""
    timings: dict[str, float] = {}

    def record_phase(name: str, started: float) -> None:
        timings[name] = round(timings.get(name, 0.0) + (time.perf_counter() - started), 4)

    def load_required(tensor_name: str) -> torch.Tensor | None:
        if prefetched_tensors is not None and tensor_name in prefetched_tensors:
            return prefetched_tensors[tensor_name]
        phase_started = time.perf_counter()
        tensor = _load_required_tensor(model_id, tensor_name, blockers, policy=tensor_policy)
        record_phase("load_tensors", phase_started)
        return tensor

    def load_required_many(tensor_names: list[str]) -> dict[str, torch.Tensor | None]:
        if prefetched_tensors is not None and all(tensor_name in prefetched_tensors for tensor_name in tensor_names):
            return {tensor_name: prefetched_tensors[tensor_name] for tensor_name in tensor_names}
        phase_started = time.perf_counter()
        loaded_by_name = load_resident_tensors(
            model_id,
            tensor_names,
            dtype=_runtime_math_dtype(),
            policy=tensor_policy,
        )
        tensors: dict[str, torch.Tensor | None] = {}
        for tensor_name in tensor_names:
            loaded = loaded_by_name[tensor_name]
            blockers.extend(loaded.blockers)
            if not loaded.ready or loaded.tensor is None:
                blockers.append(f"Required tensor {tensor_name} could not be loaded.")
                tensors[tensor_name] = None
            else:
                tensors[tensor_name] = loaded.tensor
        record_phase("load_tensors", phase_started)
        return tensors

    config = load_layer_bridge_config(model_id)
    blockers = list(config.blockers)
    if not config.ready:
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode="synthetic",
            input_shape=[],
            output_shape=[],
            output_dtype="unknown",
            blockers=blockers,
            cache_sequence_length=0,
            ready=False,
            timings=timings,
        )

    layer_norm_unit_id = f"layer-{layer_index:02d}-layer_norm"
    attention_unit_id = f"layer-{layer_index:02d}-attention"
    mlp_unit_id = f"layer-{layer_index:02d}-mlp"

    if blockers:
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode="synthetic",
            input_shape=[],
            output_shape=[],
            output_dtype="unknown",
            loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
            blockers=blockers,
            cache_sequence_length=0,
            ready=False,
            timings=timings,
        )

    math_dtype = _runtime_math_dtype()
    hidden_states = (
        input_hidden.detach().cpu().to(dtype=math_dtype)
        if input_hidden is not None
        else _synthetic_hidden_state(config.hidden_size).to(dtype=math_dtype)
    )
    input_mode = "provided" if input_hidden is not None else "synthetic"

    if hidden_states.ndim != 3:
        blockers.append(f"Layer bridge expects a rank-3 hidden state, got rank {hidden_states.ndim}.")
    elif list(hidden_states.shape[-1:]) != [config.hidden_size]:
        blockers.append(
            f"Layer bridge expects hidden size {config.hidden_size}, got {hidden_states.shape[-1]}."
        )
    if blockers:
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode=input_mode,
            input_shape=[int(value) for value in hidden_states.shape],
            output_shape=[],
            output_dtype="unknown",
            loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
            blockers=blockers,
            cache_sequence_length=0,
            ready=False,
            timings=timings,
        )

    batch_size, sequence_length, _hidden_size = hidden_states.shape
    head_dim = config.head_dim
    attention_projection_size = config.num_attention_heads * head_dim
    kv_repeat = config.num_attention_heads // config.num_key_value_heads

    native_result = _try_native_dense_decode_bridge(
        model_id=model_id,
        layer_index=layer_index,
        hidden_states=hidden_states,
        past_key_value=past_key_value,
        native_kv_session=native_kv_session,
        config=config,
        tensor_policy=tensor_policy,
        collect_metrics=collect_metrics,
        timings=timings,
        native_kv_commit=native_kv_commit,
        prefetched_tensors=prefetched_tensors,
    )
    if native_result is not None:
        if not return_kv_cache:
            native_result.next_kv_cache = None
        return native_result

    native_prefill_result = _try_native_dense_prefill_bridge(
        model_id=model_id,
        layer_index=layer_index,
        hidden_states=hidden_states,
        past_key_value=past_key_value,
        native_kv_session=native_kv_session,
        config=config,
        tensor_policy=tensor_policy,
        collect_metrics=collect_metrics,
        timings=timings,
        prefetched_tensors=prefetched_tensors,
    )
    if native_prefill_result is not None:
        if not return_kv_cache:
            native_prefill_result.next_kv_cache = None
        return native_prefill_result

    norm_tensor_names = [
        f"model.layers.{layer_index}.input_layernorm.weight",
        f"model.layers.{layer_index}.post_attention_layernorm.weight",
    ]
    norm_tensors = load_required_many(norm_tensor_names)
    input_norm_weight = norm_tensors[norm_tensor_names[0]]
    post_attention_norm_weight = norm_tensors[norm_tensor_names[1]]
    if blockers or input_norm_weight is None or post_attention_norm_weight is None:
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode=input_mode,
            input_shape=[int(value) for value in hidden_states.shape],
            output_shape=[],
            output_dtype="unknown",
            loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
            blockers=blockers,
            cache_sequence_length=0,
            ready=False,
            timings=timings,
        )

    phase_started = time.perf_counter()
    normed_input = _rms_norm(hidden_states, input_norm_weight, config.rms_norm_eps)
    record_phase("input_norm", phase_started)
    del input_norm_weight

    native_attention_payload = None
    if _is_moe_config(config) and return_kv_cache and sequence_length == 1:
        native_attention_payload = _try_native_attention_decode_bridge(
            model_id=model_id,
            layer_index=layer_index,
            normed_input=normed_input,
            past_key_value=past_key_value,
            native_kv_session=native_kv_session,
            config=config,
            tensor_policy=tensor_policy,
            timings=timings,
            native_kv_commit=native_kv_commit,
        )
    if native_attention_payload is not None:
        attention_output, cache_sequence_length, attention_native_session = native_attention_payload
        residual_after_attention = hidden_states + attention_output
        phase_started = time.perf_counter()
        normed_post_attention = _rms_norm(residual_after_attention, post_attention_norm_weight, config.rms_norm_eps)
        record_phase("post_attention_norm", phase_started)
        del post_attention_norm_weight, attention_output, normed_input

        router_name = _moe_router_tensor_name(model_id, layer_index)
        router_weight = load_required(router_name)
        if blockers or router_weight is None:
            return LayerBridgeResult(
                model_id=model_id,
                layer_index=layer_index,
                input_mode=input_mode,
                input_shape=[int(value) for value in hidden_states.shape],
                output_shape=[],
                output_dtype="unknown",
                loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
                blockers=blockers,
                attention_head_dim=head_dim,
                cache_sequence_length=cache_sequence_length,
                ready=False,
                timings=timings,
            )
        with torch.no_grad():
            router_probs = torch.softmax(F.linear(normed_post_attention.float(), router_weight.float()), dim=-1)
            selected_experts = torch.topk(
                router_probs,
                max(1, min(config.num_experts_per_tok, config.num_experts)),
                dim=-1,
            ).indices
            selected_expert_ids = sorted({int(value) for value in selected_experts.detach().cpu().flatten().tolist()})
        expert_tensor_names: list[str] = []
        expert_name_maps: dict[int, dict[str, str]] = {}
        for expert_index in selected_expert_ids:
            record_expert_activation(layer_index, expert_index)
            expert_name_map = _moe_expert_tensor_name_map(model_id, layer_index, expert_index)
            expert_name_maps[expert_index] = expert_name_map
            expert_tensor_names.extend(expert_name_map.values())
        with fp16_packed_expert_cache_scope(enabled=_fp16_decode_expert_packed_cache_enabled()):
            loaded_experts = load_required_many(expert_tensor_names)
        expert_tensors: dict[int, dict[str, torch.Tensor]] = {}
        for expert_index in selected_expert_ids:
            expert_name_map = expert_name_maps[expert_index]
            gate_weight = loaded_experts.get(expert_name_map["gate_proj"])
            up_weight = loaded_experts.get(expert_name_map["up_proj"])
            down_weight = loaded_experts.get(expert_name_map["down_proj"])
            if gate_weight is None or up_weight is None or down_weight is None:
                blockers.append(f"Expert {expert_index} in layer {layer_index} did not load all required tensors.")
                continue
            expert_tensors[expert_index] = {
                "gate_proj": gate_weight,
                "up_proj": up_weight,
                "down_proj": down_weight,
            }
        if blockers or not expert_tensors:
            return LayerBridgeResult(
                model_id=model_id,
                layer_index=layer_index,
                input_mode=input_mode,
                input_shape=[int(value) for value in hidden_states.shape],
                output_shape=[],
                output_dtype="unknown",
                loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
                blockers=blockers,
                attention_head_dim=head_dim,
                cache_sequence_length=cache_sequence_length,
                ready=False,
                timings=timings,
            )
        phase_started = time.perf_counter()
        mlp_output, _touched_experts, _selected_experts = _run_moe_mlp(
            hidden_states=normed_post_attention,
            router_weight=router_weight,
            expert_tensors=expert_tensors,
            top_k=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
        )
        record_phase("mlp", phase_started)
        del router_weight, loaded_experts, expert_tensors, normed_post_attention
        layer_output = residual_after_attention + mlp_output
        del residual_after_attention, mlp_output
        output_mean_abs = float(layer_output.abs().mean().item()) if collect_metrics else 0.0
        output_l2_norm = float(torch.linalg.vector_norm(layer_output).item()) if collect_metrics else 0.0
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode=input_mode,
            input_shape=[int(value) for value in hidden_states.shape],
            output_shape=[int(value) for value in layer_output.shape],
            output_dtype=str(layer_output.dtype),
            loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
            attention_head_dim=head_dim,
            cache_sequence_length=cache_sequence_length,
            output_mean_abs=output_mean_abs,
            output_l2_norm=output_l2_norm,
            blockers=[],
            ready=True,
            timings=timings,
            output_tensor=layer_output,
            next_kv_cache=None,
            native_kv_session=attention_native_session,
        )

    qkv_tensor_names = [
        f"model.layers.{layer_index}.self_attn.q_proj.weight",
        f"model.layers.{layer_index}.self_attn.k_proj.weight",
        f"model.layers.{layer_index}.self_attn.v_proj.weight",
    ]
    q_bias_name = f"model.layers.{layer_index}.self_attn.q_proj.bias"
    k_bias_name = f"model.layers.{layer_index}.self_attn.k_proj.bias"
    v_bias_name = f"model.layers.{layer_index}.self_attn.v_proj.bias"
    q_norm_name = f"model.layers.{layer_index}.self_attn.q_norm.weight"
    k_norm_name = f"model.layers.{layer_index}.self_attn.k_norm.weight"
    for optional_name in [q_bias_name, k_bias_name, v_bias_name, q_norm_name, k_norm_name]:
        if _tensor_entry_exists(model_id, optional_name):
            qkv_tensor_names.append(optional_name)
    qkv_tensors = load_required_many(qkv_tensor_names)
    q_weight = qkv_tensors[qkv_tensor_names[0]]
    q_bias = qkv_tensors.get(q_bias_name)
    if blockers or q_weight is None:
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode=input_mode,
            input_shape=[int(value) for value in hidden_states.shape],
            output_shape=[],
            output_dtype="unknown",
            loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
            blockers=blockers,
            cache_sequence_length=0,
            ready=False,
            timings=timings,
    )
    phase_started = time.perf_counter()
    q_states = F.linear(normed_input, q_weight, q_bias)
    record_phase("qkv_projection", phase_started)
    del q_weight, q_bias

    k_weight = qkv_tensors[f"model.layers.{layer_index}.self_attn.k_proj.weight"]
    k_bias = qkv_tensors.get(k_bias_name)
    if blockers or k_weight is None:
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode=input_mode,
            input_shape=[int(value) for value in hidden_states.shape],
            output_shape=[],
            output_dtype="unknown",
            loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
            blockers=blockers,
            cache_sequence_length=0,
            ready=False,
            timings=timings,
    )
    phase_started = time.perf_counter()
    k_states = F.linear(normed_input, k_weight, k_bias)
    record_phase("qkv_projection", phase_started)
    del k_weight, k_bias

    v_weight = qkv_tensors[f"model.layers.{layer_index}.self_attn.v_proj.weight"]
    v_bias = qkv_tensors.get(v_bias_name)
    if blockers or v_weight is None:
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode=input_mode,
            input_shape=[int(value) for value in hidden_states.shape],
            output_shape=[],
            output_dtype="unknown",
            loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
            blockers=blockers,
            cache_sequence_length=0,
            ready=False,
            timings=timings,
        )
    phase_started = time.perf_counter()
    v_states = F.linear(normed_input, v_weight, v_bias)
    record_phase("qkv_projection", phase_started)
    del v_weight, v_bias

    q_states = q_states.view(batch_size, sequence_length, config.num_attention_heads, head_dim).transpose(1, 2)
    k_states = k_states.view(batch_size, sequence_length, config.num_key_value_heads, head_dim).transpose(1, 2)
    v_states = v_states.view(batch_size, sequence_length, config.num_key_value_heads, head_dim).transpose(1, 2)
    q_norm_weight = qkv_tensors.get(q_norm_name)
    k_norm_weight = qkv_tensors.get(k_norm_name)
    if q_norm_weight is not None:
        q_states = _rms_norm(q_states, q_norm_weight, config.rms_norm_eps)
    if k_norm_weight is not None:
        k_states = _rms_norm(k_states, k_norm_weight, config.rms_norm_eps)
    del qkv_tensors

    next_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    past_length = 0
    cache_sequence_length = k_states.shape[-2]
    if past_key_value is not None:
        past_keys, past_values = past_key_value
        if list(past_keys.shape[:2]) != [batch_size, config.num_key_value_heads]:
            blockers.append(
                f"Past key cache has incompatible shape {list(past_keys.shape)} for layer {layer_index}."
            )
        elif list(past_values.shape[:2]) != [batch_size, config.num_key_value_heads]:
            blockers.append(
                f"Past value cache has incompatible shape {list(past_values.shape)} for layer {layer_index}."
            )
        elif past_keys.shape[-1] != head_dim or past_values.shape[-1] != head_dim:
            blockers.append(f"Past KV cache head dimension does not match expected head dim {head_dim}.")
        else:
            past_length = past_keys.shape[-2]
            k_states = torch.cat([past_keys.to(dtype=math_dtype), k_states], dim=-2)
            v_states = torch.cat([past_values.to(dtype=math_dtype), v_states], dim=-2)
            cache_sequence_length = k_states.shape[-2]

    if blockers:
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode=input_mode,
            input_shape=[int(value) for value in hidden_states.shape],
            output_shape=[],
            output_dtype="unknown",
            loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
            attention_head_dim=head_dim,
            cache_sequence_length=cache_sequence_length,
            blockers=blockers,
            ready=False,
            timings=timings,
        )

    position_start = past_length if position_offset is None else position_offset
    positions = torch.arange(position_start, position_start + sequence_length, dtype=torch.long)
    if config.max_position_embeddings > 0 and int(positions[-1].item()) >= config.max_position_embeddings:
        blockers.append(
            f"RoPE position {int(positions[-1].item())} exceeds configured max_position_embeddings {config.max_position_embeddings}."
        )
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode=input_mode,
            input_shape=[int(value) for value in hidden_states.shape],
            output_shape=[],
            output_dtype="unknown",
            loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
            attention_head_dim=head_dim,
            cache_sequence_length=cache_sequence_length,
            blockers=blockers,
            ready=False,
            timings=timings,
        )

    phase_started = time.perf_counter()
    q_states = _apply_rotary_position_embedding(q_states, positions, config.rope_theta)
    current_k_states = k_states[:, :, past_length:, :]
    current_k_states = _apply_rotary_position_embedding(current_k_states, positions, config.rope_theta)
    if past_length > 0:
        k_states = torch.cat([k_states[:, :, :past_length, :], current_k_states], dim=-2)
    else:
        k_states = current_k_states
    record_phase("rope", phase_started)

    if return_kv_cache:
        phase_started = time.perf_counter()
        next_kv_cache = (k_states.detach().cpu(), v_states.detach().cpu())
        record_phase("kv_cache", phase_started)

    phase_started = time.perf_counter()
    k_states = _repeat_kv(k_states, kv_repeat)
    v_states = _repeat_kv(v_states, kv_repeat)
    past_start = position_start - past_length
    if past_length > 0:
        past_positions = torch.arange(past_start, position_start, dtype=torch.long)
        key_positions = torch.cat([past_positions, positions], dim=0)
    else:
        key_positions = positions
    if sequence_length == 1 and past_length > 0:
        causal_mask = torch.ones(1, 1, 1, k_states.shape[-2], dtype=torch.bool, device=q_states.device)
    else:
        causal_mask = _causal_attention_mask(positions, key_positions, q_states.device)
    attention_context = _attention_context(q_states, k_states, v_states, causal_mask, head_dim)
    attention_context = (
        attention_context.transpose(1, 2)
        .contiguous()
        .view(batch_size, sequence_length, attention_projection_size)
    )
    record_phase("attention", phase_started)
    del q_states, k_states, v_states, causal_mask

    o_weight = load_required(f"model.layers.{layer_index}.self_attn.o_proj.weight")
    if blockers or o_weight is None:
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode=input_mode,
            input_shape=[int(value) for value in hidden_states.shape],
            output_shape=[],
            output_dtype="unknown",
            loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
            blockers=blockers,
            attention_head_dim=head_dim,
            cache_sequence_length=cache_sequence_length,
            ready=False,
            timings=timings,
        )
    phase_started = time.perf_counter()
    attention_output = F.linear(attention_context, o_weight)
    record_phase("o_projection", phase_started)
    del attention_context, o_weight

    residual_after_attention = hidden_states + attention_output
    del attention_output
    phase_started = time.perf_counter()
    normed_post_attention = _rms_norm(residual_after_attention, post_attention_norm_weight, config.rms_norm_eps)
    record_phase("post_attention_norm", phase_started)
    del post_attention_norm_weight

    if _is_moe_config(config):
        router_name = _moe_router_tensor_name(model_id, layer_index)
        router_weight = load_required(router_name)
        if blockers or router_weight is None:
            return LayerBridgeResult(
                model_id=model_id,
                layer_index=layer_index,
                input_mode=input_mode,
                input_shape=[int(value) for value in hidden_states.shape],
                output_shape=[],
                output_dtype="unknown",
                loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
                blockers=blockers,
                attention_head_dim=head_dim,
                cache_sequence_length=cache_sequence_length,
                ready=False,
                timings=timings,
            )
        with torch.no_grad():
            router_probs = torch.softmax(F.linear(normed_post_attention.float(), router_weight.float()), dim=-1)
            selected_experts = torch.topk(router_probs, max(1, min(config.num_experts_per_tok, config.num_experts)), dim=-1).indices
            selected_expert_ids = sorted({int(value) for value in selected_experts.detach().cpu().flatten().tolist()})
        expert_tensor_names: list[str] = []
        expert_name_maps: dict[int, dict[str, str]] = {}
        for expert_index in selected_expert_ids:
            record_expert_activation(layer_index, expert_index)
            expert_name_map = _moe_expert_tensor_name_map(model_id, layer_index, expert_index)
            expert_name_maps[expert_index] = expert_name_map
            expert_tensor_names.extend(expert_name_map.values())
        decode_expert_cache = bool(
            return_kv_cache
            and sequence_length == 1
            and past_length > 0
            and _fp16_decode_expert_packed_cache_enabled()
        )
        with fp16_packed_expert_cache_scope(enabled=decode_expert_cache):
            loaded_experts = load_required_many(expert_tensor_names)
        expert_tensors: dict[int, dict[str, torch.Tensor]] = {}
        for expert_index in selected_expert_ids:
            expert_name_map = expert_name_maps[expert_index]
            gate_weight = loaded_experts.get(expert_name_map["gate_proj"])
            up_weight = loaded_experts.get(expert_name_map["up_proj"])
            down_weight = loaded_experts.get(expert_name_map["down_proj"])
            if gate_weight is None or up_weight is None or down_weight is None:
                blockers.append(f"Expert {expert_index} in layer {layer_index} did not load all required tensors.")
                continue
            expert_tensors[expert_index] = {
                "gate_proj": gate_weight,
                "up_proj": up_weight,
                "down_proj": down_weight,
            }
        if blockers or not expert_tensors:
            return LayerBridgeResult(
                model_id=model_id,
                layer_index=layer_index,
                input_mode=input_mode,
                input_shape=[int(value) for value in hidden_states.shape],
                output_shape=[],
                output_dtype="unknown",
                loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
                blockers=blockers,
                attention_head_dim=head_dim,
                cache_sequence_length=cache_sequence_length,
                ready=False,
                timings=timings,
            )
        phase_started = time.perf_counter()
        mlp_output, _touched_experts, _selected_experts = _run_moe_mlp(
            hidden_states=normed_post_attention,
            router_weight=router_weight,
            expert_tensors=expert_tensors,
            top_k=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
        )
        record_phase("mlp", phase_started)
        del router_weight, loaded_experts, expert_tensors, normed_post_attention
    else:
        mlp_tensor_names = [
            f"model.layers.{layer_index}.mlp.gate_proj.weight",
            f"model.layers.{layer_index}.mlp.up_proj.weight",
            f"model.layers.{layer_index}.mlp.down_proj.weight",
        ]
        mlp_tensors = load_required_many(mlp_tensor_names)
        gate_weight = mlp_tensors[mlp_tensor_names[0]]
        if blockers or gate_weight is None:
            return LayerBridgeResult(
                model_id=model_id,
                layer_index=layer_index,
                input_mode=input_mode,
                input_shape=[int(value) for value in hidden_states.shape],
                output_shape=[],
                output_dtype="unknown",
                loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
                blockers=blockers,
                attention_head_dim=head_dim,
                cache_sequence_length=cache_sequence_length,
                ready=False,
                timings=timings,
            )
        phase_started = time.perf_counter()
        gated = F.silu(F.linear(normed_post_attention, gate_weight))
        record_phase("mlp", phase_started)
        del gate_weight

        up_weight = mlp_tensors[mlp_tensor_names[1]]
        if blockers or up_weight is None:
            return LayerBridgeResult(
                model_id=model_id,
                layer_index=layer_index,
                input_mode=input_mode,
                input_shape=[int(value) for value in hidden_states.shape],
                output_shape=[],
                output_dtype="unknown",
                loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
                blockers=blockers,
                attention_head_dim=head_dim,
                cache_sequence_length=cache_sequence_length,
                ready=False,
                timings=timings,
            )
        phase_started = time.perf_counter()
        expanded = F.linear(normed_post_attention, up_weight)
        record_phase("mlp", phase_started)
        del up_weight, normed_post_attention

        down_weight = mlp_tensors[mlp_tensor_names[2]]
        if blockers or down_weight is None:
            return LayerBridgeResult(
                model_id=model_id,
                layer_index=layer_index,
                input_mode=input_mode,
                input_shape=[int(value) for value in hidden_states.shape],
                output_shape=[],
                output_dtype="unknown",
                loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
                blockers=blockers,
                attention_head_dim=head_dim,
                cache_sequence_length=cache_sequence_length,
                ready=False,
                timings=timings,
            )
        phase_started = time.perf_counter()
        mlp_output = F.linear(gated * expanded, down_weight)
        record_phase("mlp", phase_started)
        del down_weight, gated, expanded
        del mlp_tensors
    layer_output = residual_after_attention + mlp_output
    del residual_after_attention, mlp_output

    output_mean_abs = float(layer_output.abs().mean().item()) if collect_metrics else 0.0
    output_l2_norm = float(torch.linalg.vector_norm(layer_output).item()) if collect_metrics else 0.0

    return LayerBridgeResult(
        model_id=model_id,
        layer_index=layer_index,
        input_mode=input_mode,
        input_shape=[int(value) for value in hidden_states.shape],
        output_shape=[int(value) for value in layer_output.shape],
        output_dtype=str(layer_output.dtype),
        loaded_unit_ids=[layer_norm_unit_id, attention_unit_id, mlp_unit_id],
        attention_head_dim=head_dim,
        cache_sequence_length=cache_sequence_length,
        output_mean_abs=output_mean_abs,
        output_l2_norm=output_l2_norm,
        blockers=[],
        ready=True,
        timings=timings,
        output_tensor=layer_output,
        next_kv_cache=next_kv_cache,
    )


@torch.inference_mode()
def run_layer_bridge_stack(
    model_id: str,
    start_layer: int = 0,
    layer_count: int = 2,
    input_hidden: torch.Tensor | None = None,
    past_key_values: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    position_offset: int | None = None,
    return_kv_cache: bool = False,
    should_cancel: Callable[[], bool] | None = None,
    collect_step_summaries: bool = True,
    collect_metrics: bool = True,
    native_kv_sessions: dict[int, Any] | None = None,
    native_kv_commit: bool = True,
) -> LayerBridgeStackResult:
    """Run multiple minimal layer-forward bridge steps in sequence."""
    total_started = time.perf_counter()
    layer_times: dict[int, float] = {}
    operation_times: dict[str, float] = {}

    def finish_timings() -> dict[str, float]:
        payload: dict[str, float] = {
            "total": round(time.perf_counter() - total_started, 4),
            "layer_count": float(len(layer_times)),
        }
        for key, value in operation_times.items():
            payload[f"op_{key}"] = round(value, 4)
        if layer_times:
            slowest_layer, slowest_seconds = max(layer_times.items(), key=lambda item: item[1])
            payload["average_layer"] = round(sum(layer_times.values()) / len(layer_times), 4)
            payload["slowest_layer"] = float(slowest_layer)
            payload["slowest_layer_seconds"] = round(slowest_seconds, 4)
        return payload

    input_mode = "provided" if input_hidden is not None else "synthetic"
    config = load_layer_bridge_config(model_id)
    blockers = list(config.blockers)
    if layer_count <= 0:
        blockers.append("Layer stack count must be at least 1.")
    if start_layer < 0:
        blockers.append("Start layer must be 0 or greater.")
    if blockers:
        return LayerBridgeStackResult(
            model_id=model_id,
            start_layer=start_layer,
            layer_count=layer_count,
            input_mode=input_mode,
            input_shape=[],
            output_shape=[],
            output_dtype="unknown",
            cache_sequence_lengths={},
            blockers=blockers,
            ready=False,
            timings=finish_timings(),
        )

    current_hidden = input_hidden
    executed_layers: list[int] = []
    step_summaries: list[LayerBridgeStepSummary] = []
    first_input_shape: list[int] = []
    output_shape: list[int] = []
    output_dtype = "unknown"
    cache_sequence_lengths: dict[str, int] = {}
    next_kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    next_native_kv_sessions: dict[int, Any] = {}
    tensor_policy = TensorResidencyPolicy.from_environment(model_id)
    prefetch_enabled = _layer_prefetch_enabled() and layer_count > 1
    prefetch_executor: ThreadPoolExecutor | None = None
    prefetch_future: Future | None = None

    def load_prefetch(layer_index: int) -> dict[str, torch.Tensor]:
        loaded = load_resident_tensors(
            model_id,
            _layer_tensor_names(layer_index),
            dtype=_runtime_math_dtype(),
            policy=tensor_policy,
        )
        tensors: dict[str, torch.Tensor] = {}
        for tensor_name, loaded_slice in loaded.items():
            if loaded_slice.ready and loaded_slice.tensor is not None:
                tensors[tensor_name] = loaded_slice.tensor
        return tensors

    def submit_prefetch(layer_index: int) -> Future | None:
        if not prefetch_enabled or layer_index >= start_layer + layer_count:
            return None
        assert prefetch_executor is not None
        return prefetch_executor.submit(load_prefetch, layer_index)

    try:
        prefetch_executor = ThreadPoolExecutor(max_workers=1) if prefetch_enabled else None

        for layer_index in range(start_layer, start_layer + layer_count):
            if _cancel_requested(should_cancel):
                return LayerBridgeStackResult(
                    model_id=model_id,
                    start_layer=start_layer,
                    layer_count=layer_count,
                    input_mode=input_mode,
                    input_shape=first_input_shape,
                    output_shape=output_shape,
                    output_dtype=output_dtype,
                    executed_layers=executed_layers,
                    step_summaries=step_summaries,
                    cache_sequence_lengths=cache_sequence_lengths,
                    blockers=[CANCEL_BLOCKER],
                    ready=False,
                    timings=finish_timings(),
                    output_tensor=current_hidden if executed_layers else None,
                    next_kv_caches=next_kv_caches,
                    next_native_kv_sessions=next_native_kv_sessions,
                )
            prefetched_tensors: dict[str, torch.Tensor] | None = None
            if prefetch_future is not None:
                wait_started = time.perf_counter()
                prefetched_tensors = prefetch_future.result()
                operation_times["prefetch_wait"] = operation_times.get("prefetch_wait", 0.0) + (
                    time.perf_counter() - wait_started
                )
            prefetch_future = submit_prefetch(layer_index + 1)

            layer_started = time.perf_counter()
            result = run_minimal_layer_forward_bridge(
                model_id,
                layer_index=layer_index,
                input_hidden=current_hidden,
                past_key_value=None if past_key_values is None else past_key_values.get(layer_index),
                position_offset=position_offset,
                return_kv_cache=return_kv_cache,
                collect_metrics=collect_metrics,
                tensor_policy=tensor_policy,
                prefetched_tensors=prefetched_tensors,
                native_kv_session=None if native_kv_sessions is None else native_kv_sessions.get(layer_index),
                native_kv_commit=native_kv_commit,
            )
            layer_times[layer_index] = time.perf_counter() - layer_started
            for key, value in result.timings.items():
                operation_times[key] = operation_times.get(key, 0.0) + float(value)
            if not first_input_shape:
                first_input_shape = list(result.input_shape)
            cache_sequence_lengths[str(layer_index)] = result.cache_sequence_length

            if collect_step_summaries or not result.ready:
                step_summaries.append(
                    LayerBridgeStepSummary(
                        layer_index=layer_index,
                        output_shape=list(result.output_shape),
                        output_dtype=result.output_dtype,
                        output_mean_abs=result.output_mean_abs,
                        output_l2_norm=result.output_l2_norm,
                        loaded_unit_ids=list(result.loaded_unit_ids),
                        blockers=list(result.blockers),
                        ready=result.ready,
                    )
                )

            if not result.ready or result.output_tensor is None:
                blockers.extend(result.blockers)
                return LayerBridgeStackResult(
                    model_id=model_id,
                    start_layer=start_layer,
                    layer_count=layer_count,
                    input_mode=input_mode,
                    input_shape=first_input_shape,
                    output_shape=list(result.output_shape),
                    output_dtype=result.output_dtype,
                    executed_layers=executed_layers,
                    step_summaries=step_summaries,
                    cache_sequence_lengths=cache_sequence_lengths,
                    blockers=blockers or [f"Layer {layer_index} bridge failed."],
                    ready=False,
                    timings=finish_timings(),
                    next_native_kv_sessions=next_native_kv_sessions,
                )

            current_hidden = result.output_tensor
            executed_layers.append(layer_index)
            output_shape = list(result.output_shape)
            output_dtype = result.output_dtype
            if return_kv_cache and result.next_kv_cache is not None:
                next_kv_caches[layer_index] = result.next_kv_cache
            if return_kv_cache and result.native_kv_session is not None:
                next_native_kv_sessions[layer_index] = result.native_kv_session
    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=False, cancel_futures=True)

    return LayerBridgeStackResult(
        model_id=model_id,
        start_layer=start_layer,
        layer_count=layer_count,
        input_mode=input_mode,
        input_shape=first_input_shape,
        output_shape=output_shape,
        output_dtype=output_dtype,
        executed_layers=executed_layers,
        step_summaries=step_summaries,
        cache_sequence_lengths=cache_sequence_lengths,
        blockers=[],
        ready=len(executed_layers) == layer_count and current_hidden is not None,
        timings=finish_timings(),
        output_tensor=current_hidden,
        next_kv_caches=next_kv_caches,
        next_native_kv_sessions=next_native_kv_sessions,
    )


def run_token_entry_layer_bridge(
    model_id: str,
    token_ids: list[int],
    start_layer: int = 0,
    layer_count: int = 2,
) -> TokenEntryBridgeResult:
    """Start from token ids, build the initial hidden state, then run the layer bridge stack."""
    hidden_state, blockers = load_token_entry_hidden_state(model_id, token_ids)
    if blockers or hidden_state is None:
        return TokenEntryBridgeResult(
            model_id=model_id,
            token_ids=list(token_ids),
            input_shape=[],
            embedding_shape=[],
            output_shape=[],
            output_dtype="unknown",
            start_layer=start_layer,
            layer_count=layer_count,
            blockers=blockers or ["Token entry failed before the bridge stack could start."],
            ready=False,
        )

    stack_result = run_layer_bridge_stack(
        model_id,
        start_layer=start_layer,
        layer_count=layer_count,
        input_hidden=hidden_state,
    )
    return TokenEntryBridgeResult(
        model_id=model_id,
        token_ids=list(token_ids),
        input_shape=[1, len(token_ids)],
        embedding_shape=[int(value) for value in hidden_state.shape],
        output_shape=list(stack_result.output_shape),
        output_dtype=stack_result.output_dtype,
        start_layer=start_layer,
        layer_count=layer_count,
        executed_layers=list(stack_result.executed_layers),
        blockers=list(stack_result.blockers),
        ready=stack_result.ready,
        output_tensor=stack_result.output_tensor,
    )


@torch.inference_mode()
def run_decode_tail(
    model_id: str,
    hidden_state: torch.Tensor,
    lm_head_chunk_rows: int = DEFAULT_LM_HEAD_CHUNK_ROWS,
    top_k: int = 5,
    should_cancel: Callable[[], bool] | None = None,
    return_logits: bool = True,
    recent_token_ids: list[int] | None = None,
    repetition_penalty: float = 1.0,
) -> DecodeTailResult:
    """Run final norm and stream the lm_head in row chunks to produce real logits."""
    config = load_layer_bridge_config(model_id)
    blockers = list(config.blockers)
    if hidden_state.ndim != 3:
        blockers.append(f"Decode tail expects a rank-3 hidden state, got rank {hidden_state.ndim}.")
    elif hidden_state.shape[1] != 1:
        blockers.append("Decode tail currently supports single-token hidden states only.")
    elif hidden_state.shape[-1] != config.hidden_size:
        blockers.append(
            f"Decode tail expects hidden size {config.hidden_size}, got {hidden_state.shape[-1]}."
        )
    if lm_head_chunk_rows <= 0:
        blockers.append("lm_head chunk rows must be at least 1.")
    if top_k <= 0:
        blockers.append("top_k must be at least 1.")
    if repetition_penalty <= 0:
        blockers.append("repetition_penalty must be greater than 0.")
    if blockers:
        return DecodeTailResult(
            model_id=model_id,
            input_shape=[int(value) for value in hidden_state.shape] if hidden_state.ndim else [],
            normalized_shape=[],
            logits_shape=[],
            logits_dtype="unknown",
            vocab_size=config.vocab_size,
            chunk_rows=lm_head_chunk_rows,
            chunk_count=0,
            blockers=blockers,
            ready=False,
        )

    final_norm_weight = _load_required_tensor(model_id, "model.norm.weight", blockers)
    lm_head_entry = _find_tensor_entry(model_id, "lm_head.weight")
    if final_norm_weight is None:
        blockers.append("Final norm weight is not ready for decode tail.")
    if lm_head_entry is None:
        blockers.append("lm_head.weight is not present in the tensor catalog.")
    if blockers or final_norm_weight is None or lm_head_entry is None:
        return DecodeTailResult(
            model_id=model_id,
            input_shape=[int(value) for value in hidden_state.shape],
            normalized_shape=[],
            logits_shape=[],
            logits_dtype="unknown",
            vocab_size=config.vocab_size,
            chunk_rows=lm_head_chunk_rows,
            chunk_count=0,
            blockers=blockers,
            ready=False,
        )

    math_dtype = _runtime_math_dtype()
    normalized = _rms_norm(hidden_state.detach().cpu().to(dtype=math_dtype), final_norm_weight, config.rms_norm_eps)
    del final_norm_weight
    hidden_vector = normalized.view(1, config.hidden_size)
    logits_chunks: list[torch.Tensor] = []
    streamed_top_logits: torch.Tensor | None = None
    streamed_top_token_ids: torch.Tensor | None = None
    recent_ids = {int(value) for value in (recent_token_ids or [])}
    vocab_size = lm_head_entry.shape[0] if lm_head_entry.shape else config.vocab_size
    chunk_count = math.ceil(vocab_size / lm_head_chunk_rows)
    k = min(top_k, vocab_size)

    canceled_during_lm_head = False
    native_lm_head_topk = None
    native_lm_head_topk_enabled = os.environ.get("PCKETLM_ENABLE_NATIVE_LM_HEAD_TOPK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if native_lm_head_topk_enabled and not return_logits and repetition_penalty == 1.0:
        try:
            from pcketlm.native import lm_head_topk_u16 as native_lm_head_topk
        except Exception:
            native_lm_head_topk = None

    def stream_lm_head(handle) -> bool:
        nonlocal streamed_top_logits, streamed_top_token_ids
        lm_head_slice = handle.get_slice(lm_head_entry.tensor_name)
        for start in range(0, vocab_size, lm_head_chunk_rows):
            if _cancel_requested(should_cancel):
                return False
            end = min(start + lm_head_chunk_rows, vocab_size)
            weight_chunk = lm_head_slice[start:end].to(dtype=math_dtype)
            if native_lm_head_topk is not None:
                try:
                    chunk_top_logits, chunk_top_token_ids = native_lm_head_topk(
                        hidden_vector.reshape(-1).to(dtype=math_dtype),
                        weight_chunk,
                        top_k=min(k, end - start),
                        token_offset=start,
                    )
                    logits_chunk = None
                except Exception:
                    chunk_top_logits = None
                    chunk_top_token_ids = None
                    logits_chunk = F.linear(hidden_vector, weight_chunk)
            else:
                chunk_top_logits = None
                chunk_top_token_ids = None
                logits_chunk = F.linear(hidden_vector, weight_chunk)
            if return_logits:
                assert logits_chunk is not None
                logits_chunks.append(logits_chunk)
            else:
                if chunk_top_logits is None or chunk_top_token_ids is None:
                    assert logits_chunk is not None
                    chunk_vector = logits_chunk.view(-1).float()
                    if repetition_penalty != 1.0 and recent_ids:
                        for token_id in recent_ids:
                            if start <= token_id < end:
                                index = token_id - start
                                value = chunk_vector[index]
                                chunk_vector[index] = value * repetition_penalty if value < 0 else value / repetition_penalty
                    chunk_k = min(k, chunk_vector.shape[0])
                    chunk_top_logits, chunk_top_offsets = torch.topk(chunk_vector, k=chunk_k)
                    chunk_top_token_ids = chunk_top_offsets + start
                if streamed_top_logits is None or streamed_top_token_ids is None:
                    streamed_top_logits = chunk_top_logits
                    streamed_top_token_ids = chunk_top_token_ids
                else:
                    merged_logits = torch.cat([streamed_top_logits, chunk_top_logits])
                    merged_token_ids = torch.cat([streamed_top_token_ids, chunk_top_token_ids])
                    merged_top_logits, merged_indices = torch.topk(merged_logits, k=min(k, merged_logits.shape[0]))
                    streamed_top_logits = merged_top_logits
                    streamed_top_token_ids = merged_token_ids[merged_indices]
        return True

    scoped_lm_head_handle = open_scoped_tensor_handle(lm_head_entry.shard_path)
    if scoped_lm_head_handle is not None:
        canceled_during_lm_head = not stream_lm_head(scoped_lm_head_handle)
    else:
        with safe_open(lm_head_entry.shard_path, framework="pt", device="cpu") as handle:
            canceled_during_lm_head = not stream_lm_head(handle)

    if canceled_during_lm_head:
        return DecodeTailResult(
            model_id=model_id,
            input_shape=[int(value) for value in hidden_state.shape],
            normalized_shape=[int(value) for value in normalized.shape],
            logits_shape=[],
            logits_dtype="unknown",
            vocab_size=vocab_size,
            chunk_rows=lm_head_chunk_rows,
            chunk_count=chunk_count,
            blockers=[CANCEL_BLOCKER],
            ready=False,
        )

    if return_logits:
        logits = torch.cat(logits_chunks, dim=-1).view(1, 1, vocab_size)
        top_logits, top_token_ids = torch.topk(logits.view(vocab_size), k=k)
        logits_shape = [int(value) for value in logits.shape]
        logits_dtype = str(logits.dtype)
    else:
        logits = None
        top_logits = torch.empty(0, dtype=torch.float32) if streamed_top_logits is None else streamed_top_logits
        top_token_ids = torch.empty(0, dtype=torch.long) if streamed_top_token_ids is None else streamed_top_token_ids
        logits_shape = [1, 1, vocab_size]
        logits_dtype = str(math_dtype)
    return DecodeTailResult(
        model_id=model_id,
        input_shape=[int(value) for value in hidden_state.shape],
        normalized_shape=[int(value) for value in normalized.shape],
        logits_shape=logits_shape,
        logits_dtype=logits_dtype,
        vocab_size=vocab_size,
        chunk_rows=lm_head_chunk_rows,
        chunk_count=chunk_count,
        top_token_ids=[int(value) for value in top_token_ids.tolist()],
        top_logits=[float(value) for value in top_logits.tolist()],
        blockers=[],
        ready=True,
        logits=logits,
    )


def run_token_decode_step(
    model_id: str,
    token_ids: list[int],
    start_layer: int = 0,
    layer_count: int = 2,
    lm_head_chunk_rows: int = DEFAULT_LM_HEAD_CHUNK_ROWS,
    top_k: int = 5,
    history_window: int = 1,
    selection_policy: str = "greedy",
    temperature: float = 1.0,
    sample_seed: int | None = None,
) -> TokenDecodeStepResult:
    """Run token entry, the bridge stack, and the decode tail in one step."""
    history_hidden, context_token_ids, context_mode, history_blockers = build_history_summary_hidden_state(
        model_id,
        token_ids=token_ids,
        history_window=history_window,
    )
    if history_blockers or history_hidden is None:
        return TokenDecodeStepResult(
            model_id=model_id,
            token_ids=list(token_ids),
            context_token_ids=list(context_token_ids),
            context_mode=context_mode,
            history_window=history_window,
            selection_policy=selection_policy,
            chosen_token_id=None,
            start_layer=start_layer,
            layer_count=layer_count,
            embedding_shape=[],
            hidden_shape=[],
            logits_shape=[],
            logits_dtype="unknown",
            executed_layers=[],
            blockers=list(history_blockers) or ["Decode history could not be built before the bridge stack."],
            ready=False,
        )

    stack_result = run_layer_bridge_stack(
        model_id,
        start_layer=start_layer,
        layer_count=layer_count,
        input_hidden=history_hidden,
    )
    if not stack_result.ready or stack_result.output_tensor is None:
        return TokenDecodeStepResult(
            model_id=model_id,
            token_ids=list(token_ids),
            context_token_ids=list(context_token_ids),
            context_mode=context_mode,
            history_window=history_window,
            selection_policy=selection_policy,
            chosen_token_id=None,
            start_layer=start_layer,
            layer_count=layer_count,
            embedding_shape=[int(value) for value in history_hidden.shape],
            hidden_shape=list(stack_result.output_shape),
            logits_shape=[],
            logits_dtype="unknown",
            executed_layers=list(stack_result.executed_layers),
            blockers=list(stack_result.blockers) or ["Bridge stack failed before decode tail."],
            ready=False,
        )

    decode_result = run_decode_tail(
        model_id,
        stack_result.output_tensor,
        lm_head_chunk_rows=lm_head_chunk_rows,
        top_k=top_k,
    )
    selection_result = select_next_token(
        decode_result.logits,
        policy=selection_policy,
        top_k=top_k,
        temperature=temperature,
        sample_seed=sample_seed,
    )
    blockers = list(decode_result.blockers) + list(selection_result.blockers)
    return TokenDecodeStepResult(
        model_id=model_id,
        token_ids=list(token_ids),
        context_token_ids=list(context_token_ids),
        context_mode=context_mode,
        history_window=history_window,
        selection_policy=selection_policy,
        chosen_token_id=selection_result.chosen_token_id,
        start_layer=start_layer,
        layer_count=layer_count,
        embedding_shape=[int(value) for value in history_hidden.shape],
        hidden_shape=list(stack_result.output_shape),
        logits_shape=list(decode_result.logits_shape),
        logits_dtype=decode_result.logits_dtype,
        executed_layers=list(stack_result.executed_layers),
        top_token_ids=list(selection_result.top_token_ids),
        top_logits=list(selection_result.top_logits),
        blockers=blockers,
        ready=decode_result.ready and selection_result.ready,
        logits=decode_result.logits,
    )


def run_repeated_decode_loop(
    model_id: str,
    seed_token_id: int,
    steps: int = 2,
    start_layer: int = 0,
    layer_count: int = 2,
    lm_head_chunk_rows: int = DEFAULT_LM_HEAD_CHUNK_ROWS,
    top_k: int = 5,
    history_window: int = 1,
    selection_policy: str = "greedy",
    temperature: float = 1.0,
    sample_seed: int | None = None,
) -> RepeatedDecodeLoopResult:
    """Run the first greedy repeated decode loop on top of the token-to-logits path."""
    blockers: list[str] = []
    if steps <= 0:
        blockers.append("Repeated decode loop requires at least one step.")
    if seed_token_id < 0:
        blockers.append("Seed token id must be 0 or greater.")
    if blockers:
        return RepeatedDecodeLoopResult(
            model_id=model_id,
            seed_token_id=seed_token_id,
            steps_requested=steps,
            steps_completed=0,
            strategy=f"{selection_policy}-single-token-no-kv-cache",
            selection_policy=selection_policy,
            context_mode="invalid",
            history_window=history_window,
            layer_count=layer_count,
            generated_token_ids=[seed_token_id],
            blockers=blockers,
            ready=False,
        )

    current_token_id = seed_token_id
    generated_token_ids = [seed_token_id]
    step_summaries: list[RepeatedDecodeStepSummary] = []
    for step_index in range(steps):
        step_result = run_token_decode_step(
            model_id,
            token_ids=generated_token_ids,
            start_layer=start_layer,
            layer_count=layer_count,
            lm_head_chunk_rows=lm_head_chunk_rows,
            top_k=top_k,
            history_window=history_window,
            selection_policy=selection_policy,
            temperature=temperature,
            sample_seed=None if sample_seed is None else sample_seed + step_index,
        )
        if not step_result.ready or not step_result.top_token_ids:
            blockers.extend(step_result.blockers or [f"Decode step {step_index} did not return any candidate tokens."])
            step_summaries.append(
                RepeatedDecodeStepSummary(
                    step_index=step_index,
                    input_token_id=current_token_id,
                    chosen_token_id=None,
                    logits_shape=list(step_result.logits_shape),
                    top_token_ids=list(step_result.top_token_ids),
                    top_logits=list(step_result.top_logits),
                    blockers=list(step_result.blockers),
                    ready=False,
                )
            )
            return RepeatedDecodeLoopResult(
                model_id=model_id,
                seed_token_id=seed_token_id,
                steps_requested=steps,
                steps_completed=step_index,
                strategy=f"{selection_policy}-single-token-no-kv-cache",
                selection_policy=selection_policy,
                context_mode=step_result.context_mode,
                history_window=history_window,
                layer_count=layer_count,
                generated_token_ids=generated_token_ids,
                step_summaries=step_summaries,
                blockers=blockers,
                ready=False,
            )

        chosen_token_id = step_result.chosen_token_id
        step_summaries.append(
            RepeatedDecodeStepSummary(
                step_index=step_index,
                input_token_id=current_token_id,
                chosen_token_id=chosen_token_id,
                logits_shape=list(step_result.logits_shape),
                top_token_ids=list(step_result.top_token_ids),
                top_logits=list(step_result.top_logits),
                blockers=[],
                ready=True,
            )
        )
        generated_token_ids.append(chosen_token_id)
        current_token_id = chosen_token_id

    blockers.append(
        "This repeated loop now carries a small recent-token history summary, but it still does not implement true causal attention or KV-cache state."
    )
    return RepeatedDecodeLoopResult(
        model_id=model_id,
        seed_token_id=seed_token_id,
        steps_requested=steps,
        steps_completed=steps,
        strategy=f"{selection_policy}-single-token-no-kv-cache",
        selection_policy=selection_policy,
        context_mode=step_result.context_mode if step_summaries else "single-token-entry",
        history_window=history_window,
        layer_count=layer_count,
        generated_token_ids=generated_token_ids,
        step_summaries=step_summaries,
        blockers=blockers,
        ready=True,
    )


@torch.inference_mode()
def run_kv_decode_step(
    model_id: str,
    input_token_id: int,
    kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    decode_state: KVDecodeState | None = None,
    forced_next_token_id: int | None = None,
    start_layer: int = 0,
    layer_count: int = 2,
    lm_head_chunk_rows: int = DEFAULT_LM_HEAD_CHUNK_ROWS,
    top_k: int = 5,
    top_p: float = 1.0,
    selection_policy: str = "greedy",
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_token_ids: list[int] | None = None,
    sample_seed: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
    collect_layer_details: bool = True,
) -> KVDecodeStepResult:
    """Run one decode step while carrying real projected K/V state between steps."""
    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    def finish_timings() -> dict[str, float]:
        payload = dict(timings)
        payload["total"] = round(time.perf_counter() - total_started, 4)
        return payload

    def record_phase(name: str, started: float) -> None:
        timings[name] = round(timings.get(name, 0.0) + (time.perf_counter() - started), 4)

    effective_model_id = model_id if decode_state is None else decode_state.model_id
    effective_input_token_id = input_token_id if decode_state is None else decode_state.next_token_id
    effective_kv_caches = kv_caches if decode_state is None else decode_state.kv_caches
    effective_native_kv_sessions = {} if decode_state is None else decode_state.native_kv_sessions
    position_offset = None if decode_state is None else decode_state.next_position

    if decode_state is not None and not decode_state.ready:
        return KVDecodeStepResult(
            model_id=effective_model_id,
            input_token_id=effective_input_token_id,
            cache_sequence_lengths=dict(decode_state.cache_sequence_lengths),
            logits_shape=[],
            logits_dtype="unknown",
            blockers=list(decode_state.blockers) or ["Decode state is not ready for KV-aware decode."],
            ready=False,
            timings=finish_timings(),
        )
    if decode_state is not None and decode_state.finished:
        return KVDecodeStepResult(
            model_id=effective_model_id,
            input_token_id=effective_input_token_id,
            cache_sequence_lengths=dict(decode_state.cache_sequence_lengths),
            logits_shape=[],
            logits_dtype="unknown",
            blockers=[f"Decode state is already finished ({decode_state.stop_reason or 'unknown-reason'})."],
            ready=False,
            timings=finish_timings(),
            next_decode_state=decode_state,
        )

    config = load_layer_bridge_config(effective_model_id)
    if not config.ready:
        return KVDecodeStepResult(
            model_id=effective_model_id,
            input_token_id=effective_input_token_id,
            cache_sequence_lengths={},
            logits_shape=[],
            logits_dtype="unknown",
            blockers=list(config.blockers) or ["Layer bridge config is not ready for KV-aware decode."],
            ready=False,
            timings=finish_timings(),
        )
    if position_offset is not None and config.max_position_embeddings > 0 and position_offset >= config.max_position_embeddings:
        return KVDecodeStepResult(
            model_id=effective_model_id,
            input_token_id=effective_input_token_id,
            cache_sequence_lengths={} if decode_state is None else dict(decode_state.cache_sequence_lengths),
            logits_shape=[],
            logits_dtype="unknown",
            blockers=[
                f"Decode position {position_offset} exceeds configured max_position_embeddings {config.max_position_embeddings}."
            ],
            ready=False,
            timings=finish_timings(),
            next_decode_state=decode_state,
        )

    if _cancel_requested(should_cancel):
        return KVDecodeStepResult(
            model_id=effective_model_id,
            input_token_id=effective_input_token_id,
            cache_sequence_lengths={} if decode_state is None else dict(decode_state.cache_sequence_lengths),
            logits_shape=[],
            logits_dtype="unknown",
            blockers=[CANCEL_BLOCKER],
            ready=False,
            timings=finish_timings(),
            next_decode_state=decode_state,
        )

    phase_started = time.perf_counter()
    hidden_state, entry_blockers = load_token_entry_hidden_state(effective_model_id, [effective_input_token_id])
    record_phase("token_entry", phase_started)
    if entry_blockers or hidden_state is None:
        return KVDecodeStepResult(
            model_id=effective_model_id,
            input_token_id=effective_input_token_id,
            cache_sequence_lengths={},
            logits_shape=[],
            logits_dtype="unknown",
            blockers=list(entry_blockers) or ["Token entry failed before KV-aware decode."],
            ready=False,
            timings=finish_timings(),
        )

    phase_started = time.perf_counter()
    stack_result = run_layer_bridge_stack(
        effective_model_id,
        start_layer=start_layer,
        layer_count=layer_count,
        input_hidden=hidden_state,
        past_key_values=effective_kv_caches,
        position_offset=position_offset,
        return_kv_cache=True,
        native_kv_sessions=effective_native_kv_sessions,
        should_cancel=should_cancel,
        collect_step_summaries=collect_layer_details,
        collect_metrics=collect_layer_details,
    )
    record_phase("stack", phase_started)
    for key, value in stack_result.timings.items():
        timings[f"stack_{key}"] = round(float(value), 4)
    if not stack_result.ready or stack_result.output_tensor is None:
        return KVDecodeStepResult(
            model_id=effective_model_id,
            input_token_id=effective_input_token_id,
            cache_sequence_lengths=dict(stack_result.cache_sequence_lengths),
            logits_shape=[],
            logits_dtype="unknown",
            executed_layers=list(stack_result.executed_layers),
            blockers=list(stack_result.blockers) or ["KV-aware bridge stack failed before decode tail."],
            ready=False,
            timings=finish_timings(),
        )

    phase_started = time.perf_counter()
    decode_result = run_decode_tail(
        effective_model_id,
        stack_result.output_tensor,
        lm_head_chunk_rows=lm_head_chunk_rows,
        top_k=top_k,
        should_cancel=should_cancel,
        return_logits=not _can_select_from_topk(selection_policy),
        recent_token_ids=[] if decode_state is None else list(decode_state.generated_token_ids),
        repetition_penalty=repetition_penalty,
    )
    record_phase("decode_tail", phase_started)
    if not decode_result.ready:
        return KVDecodeStepResult(
            model_id=effective_model_id,
            input_token_id=effective_input_token_id,
            cache_sequence_lengths=dict(stack_result.cache_sequence_lengths),
            logits_shape=list(decode_result.logits_shape),
            logits_dtype=decode_result.logits_dtype,
            executed_layers=list(stack_result.executed_layers),
            blockers=list(decode_result.blockers) or ["KV-aware decode tail failed."],
            ready=False,
            timings=finish_timings(),
            next_decode_state=decode_state,
        )
    if _cancel_requested(should_cancel):
        return KVDecodeStepResult(
            model_id=effective_model_id,
            input_token_id=effective_input_token_id,
            cache_sequence_lengths=dict(stack_result.cache_sequence_lengths),
            logits_shape=list(decode_result.logits_shape),
            logits_dtype=decode_result.logits_dtype,
            executed_layers=list(stack_result.executed_layers),
            blockers=[CANCEL_BLOCKER],
            ready=False,
            timings=finish_timings(),
            next_decode_state=decode_state,
        )
    phase_started = time.perf_counter()
    if forced_next_token_id is not None:
        selection_result = TokenSelectionResult(
            policy="forced-prefill-token",
            chosen_token_id=forced_next_token_id,
            top_token_ids=list(decode_result.top_token_ids),
            top_logits=list(decode_result.top_logits),
            blockers=[],
            ready=True,
        )
    else:
        recent_token_ids = [] if decode_state is None else list(decode_state.generated_token_ids)
        if decode_result.logits is None and _can_select_from_topk(selection_policy):
            selection_result = select_next_token_from_topk(
                decode_result.top_token_ids,
                decode_result.top_logits,
                policy=selection_policy,
                top_p=top_p,
                temperature=temperature,
                sample_seed=sample_seed,
            )
        else:
            selection_result = select_next_token(
                decode_result.logits,
                policy=selection_policy,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                recent_token_ids=recent_token_ids,
                sample_seed=sample_seed,
            )
    record_phase("selection", phase_started)
    blockers = list(decode_result.blockers) + list(selection_result.blockers)
    blockers.append(
        "This K/V-aware step now applies RoPE to the carried key path, but it is still not yet a full production decode implementation."
    )
    generated_token_ids = (
        [effective_input_token_id, selection_result.chosen_token_id]
        if decode_state is None
        else list(decode_state.generated_token_ids) + [selection_result.chosen_token_id]
    )
    effective_stop_token_ids = list(config.eos_token_ids) if stop_token_ids is None else [int(value) for value in stop_token_ids]
    reached_eos = selection_result.chosen_token_id in effective_stop_token_ids
    stop_reason = "eos-token" if reached_eos and stop_token_ids is None else ("custom-stop-token" if reached_eos else None)
    next_decode_state = KVDecodeState(
        model_id=effective_model_id,
        next_token_id=selection_result.chosen_token_id,
        next_position=(0 if position_offset is None else position_offset) + 1,
        generated_token_ids=generated_token_ids,
        cache_sequence_lengths=dict(stack_result.cache_sequence_lengths),
        kv_caches=dict(stack_result.next_kv_caches),
        native_kv_sessions=dict(stack_result.next_native_kv_sessions),
        finished=reached_eos,
        stop_reason=stop_reason,
        ready=decode_result.ready and selection_result.ready,
        blockers=list(blockers),
    )
    return KVDecodeStepResult(
        model_id=effective_model_id,
        input_token_id=effective_input_token_id,
        cache_sequence_lengths=dict(stack_result.cache_sequence_lengths),
        logits_shape=list(decode_result.logits_shape),
        logits_dtype=decode_result.logits_dtype,
        executed_layers=list(stack_result.executed_layers),
        chosen_token_id=selection_result.chosen_token_id,
        top_token_ids=list(selection_result.top_token_ids),
        top_logits=list(selection_result.top_logits),
        blockers=blockers,
        ready=decode_result.ready and selection_result.ready,
        timings=finish_timings(),
        logits=decode_result.logits,
        next_kv_caches=dict(stack_result.next_kv_caches),
        next_decode_state=next_decode_state,
    )


def run_kv_decode_loop(
    model_id: str,
    seed_token_id: int,
    steps: int = 2,
    initial_state: KVDecodeState | None = None,
    start_layer: int = 0,
    layer_count: int = 2,
    lm_head_chunk_rows: int = DEFAULT_LM_HEAD_CHUNK_ROWS,
    top_k: int = 5,
    top_p: float = 1.0,
    selection_policy: str = "greedy",
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_token_ids: list[int] | None = None,
    sample_seed: int | None = None,
) -> KVDecodeLoopResult:
    """Run the first repeated decode loop that carries real projected K/V state."""
    decode_state = initial_state if initial_state is not None else initialize_kv_decode_state(model_id, seed_token_id)
    blockers: list[str] = list(decode_state.blockers)
    if steps <= 0:
        blockers.append("KV decode loop requires at least one step.")
    if blockers:
        return KVDecodeLoopResult(
            model_id=model_id,
            seed_token_id=seed_token_id,
            steps_requested=steps,
            steps_completed=0,
            strategy=f"{selection_policy}-kv-cache-rope",
            selection_policy=selection_policy,
            layer_count=layer_count,
            stop_reason="invalid-input",
            generated_token_ids=[seed_token_id],
            blockers=blockers,
            ready=False,
        )

    generated_token_ids = list(decode_state.generated_token_ids)
    step_summaries: list[RepeatedDecodeStepSummary] = []
    cache_sequence_lengths: dict[str, int] = {}

    for step_index in range(steps):
        step_result = run_kv_decode_step(
            model_id,
            input_token_id=decode_state.next_token_id,
            decode_state=decode_state,
            start_layer=start_layer,
            layer_count=layer_count,
            lm_head_chunk_rows=lm_head_chunk_rows,
            top_k=top_k,
            top_p=top_p,
            selection_policy=selection_policy,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            stop_token_ids=stop_token_ids,
            sample_seed=None if sample_seed is None else sample_seed + step_index,
        )
        cache_sequence_lengths = dict(step_result.cache_sequence_lengths)
        if not step_result.ready or step_result.chosen_token_id is None:
            blockers.extend(step_result.blockers or [f"KV decode step {step_index} did not return a chosen token."])
            step_summaries.append(
                RepeatedDecodeStepSummary(
                    step_index=step_index,
                    input_token_id=decode_state.next_token_id,
                    chosen_token_id=step_result.chosen_token_id,
                    logits_shape=list(step_result.logits_shape),
                    top_token_ids=list(step_result.top_token_ids),
                    top_logits=list(step_result.top_logits),
                    blockers=list(step_result.blockers),
                    ready=False,
                )
            )
            return KVDecodeLoopResult(
                model_id=model_id,
                seed_token_id=seed_token_id,
                steps_requested=steps,
                steps_completed=step_index,
                strategy=f"{selection_policy}-kv-cache-rope",
                selection_policy=selection_policy,
                layer_count=layer_count,
                stop_reason=decode_state.stop_reason,
                generated_token_ids=generated_token_ids,
                cache_sequence_lengths=cache_sequence_lengths,
                step_summaries=step_summaries,
                blockers=blockers,
                ready=False,
            )

        chosen_token_id = step_result.chosen_token_id
        step_summaries.append(
            RepeatedDecodeStepSummary(
                step_index=step_index,
                input_token_id=decode_state.next_token_id,
                chosen_token_id=chosen_token_id,
                logits_shape=list(step_result.logits_shape),
                top_token_ids=list(step_result.top_token_ids),
                top_logits=list(step_result.top_logits),
                blockers=list(step_result.blockers),
                ready=True,
            )
        )
        decode_state = step_result.next_decode_state
        if decode_state is None:
            blockers.append(f"KV decode step {step_index} did not return the next decode state.")
            return KVDecodeLoopResult(
                model_id=model_id,
                seed_token_id=seed_token_id,
                steps_requested=steps,
                steps_completed=step_index + 1,
                strategy=f"{selection_policy}-kv-cache-rope",
                selection_policy=selection_policy,
                layer_count=layer_count,
                stop_reason="missing-next-state",
                generated_token_ids=generated_token_ids,
                cache_sequence_lengths=cache_sequence_lengths,
                step_summaries=step_summaries,
                blockers=blockers,
                ready=False,
            )
        generated_token_ids = list(decode_state.generated_token_ids)
        if decode_state.finished:
            break

    stop_reason = decode_state.stop_reason if decode_state.finished else "step-limit"
    blockers.append(
        "This K/V-aware loop now carries real projected K/V tensors with RoPE applied on the live key path, but it is still not yet a full production decode implementation."
    )
    return KVDecodeLoopResult(
        model_id=model_id,
        seed_token_id=seed_token_id,
        steps_requested=steps,
        steps_completed=len(step_summaries),
        strategy=f"{selection_policy}-kv-cache-rope",
        selection_policy=selection_policy,
        layer_count=layer_count,
        stop_reason=stop_reason,
        generated_token_ids=generated_token_ids,
        cache_sequence_lengths=cache_sequence_lengths,
        step_summaries=step_summaries,
        blockers=blockers,
        ready=True,
    )


def run_decode_benchmark(
    model_id: str,
    seed_token_id: int,
    steps: int = 2,
    start_layer: int = 0,
    layer_count: int = 2,
    lm_head_chunk_rows: int = DEFAULT_LM_HEAD_CHUNK_ROWS,
    top_k: int = 5,
    history_window: int = 3,
    temperature: float = 1.0,
    sample_seed: int | None = 11,
) -> DecodeBenchmarkResult:
    """Compare the current small decode strategies on one fixed seed."""
    blockers: list[str] = []
    if steps <= 0:
        blockers.append("Decode benchmark requires at least one step.")
    if seed_token_id < 0:
        blockers.append("Seed token id must be 0 or greater.")
    if blockers:
        return DecodeBenchmarkResult(
            model_id=model_id,
            seed_token_id=seed_token_id,
            steps_requested=steps,
            layer_count=layer_count,
            history_window=history_window,
            sample_seed=sample_seed,
            blockers=blockers,
            ready=False,
        )

    cases: list[DecodeBenchmarkCaseResult] = []

    greedy_loop = run_repeated_decode_loop(
        model_id,
        seed_token_id=seed_token_id,
        steps=steps,
        start_layer=start_layer,
        layer_count=layer_count,
        lm_head_chunk_rows=lm_head_chunk_rows,
        top_k=top_k,
        history_window=history_window,
        selection_policy="greedy",
        temperature=temperature,
        sample_seed=sample_seed,
    )
    cases.append(
        DecodeBenchmarkCaseResult(
            label="history-greedy",
            strategy=greedy_loop.strategy,
            steps_completed=greedy_loop.steps_completed,
            final_token_id=None if not greedy_loop.generated_token_ids else greedy_loop.generated_token_ids[-1],
            unique_token_count=len(set(greedy_loop.generated_token_ids)),
            stop_reason="step-limit",
            generated_token_ids=list(greedy_loop.generated_token_ids),
            blockers=list(greedy_loop.blockers),
            ready=greedy_loop.ready,
        )
    )

    sampled_loop = run_repeated_decode_loop(
        model_id,
        seed_token_id=seed_token_id,
        steps=steps,
        start_layer=start_layer,
        layer_count=layer_count,
        lm_head_chunk_rows=lm_head_chunk_rows,
        top_k=top_k,
        history_window=history_window,
        selection_policy="top-k-sample",
        temperature=temperature,
        sample_seed=sample_seed,
    )
    cases.append(
        DecodeBenchmarkCaseResult(
            label="history-top-k-sample",
            strategy=sampled_loop.strategy,
            steps_completed=sampled_loop.steps_completed,
            final_token_id=None if not sampled_loop.generated_token_ids else sampled_loop.generated_token_ids[-1],
            unique_token_count=len(set(sampled_loop.generated_token_ids)),
            stop_reason="step-limit",
            generated_token_ids=list(sampled_loop.generated_token_ids),
            blockers=list(sampled_loop.blockers),
            ready=sampled_loop.ready,
        )
    )

    kv_loop = run_kv_decode_loop(
        model_id,
        seed_token_id=seed_token_id,
        steps=steps,
        start_layer=start_layer,
        layer_count=layer_count,
        lm_head_chunk_rows=lm_head_chunk_rows,
        top_k=top_k,
        top_p=1.0,
        selection_policy="greedy",
        temperature=temperature,
        sample_seed=sample_seed,
    )
    cases.append(
        DecodeBenchmarkCaseResult(
            label="kv-greedy",
            strategy=kv_loop.strategy,
            steps_completed=kv_loop.steps_completed,
            final_token_id=None if not kv_loop.generated_token_ids else kv_loop.generated_token_ids[-1],
            unique_token_count=len(set(kv_loop.generated_token_ids)),
            stop_reason=kv_loop.stop_reason,
            generated_token_ids=list(kv_loop.generated_token_ids),
            cache_sequence_lengths=dict(kv_loop.cache_sequence_lengths),
            blockers=list(kv_loop.blockers),
            ready=kv_loop.ready,
        )
    )

    all_ready = all(case.ready for case in cases)
    for case in cases:
        blockers.extend(case.blockers)

    return DecodeBenchmarkResult(
        model_id=model_id,
        seed_token_id=seed_token_id,
        steps_requested=steps,
        layer_count=layer_count,
        history_window=history_window,
        sample_seed=sample_seed,
        cases=cases,
        blockers=blockers,
        ready=all_ready,
    )


def run_prompt_decode_loop(
    model_id: str,
    prompt: str,
    steps: int = 1,
    max_new_tokens: int | None = None,
    start_layer: int = 0,
    layer_count: int | None = None,
    lm_head_chunk_rows: int = DEFAULT_LM_HEAD_CHUNK_ROWS,
    top_k: int = 5,
    top_p: float | None = None,
    selection_policy: str = "greedy",
    temperature: float = 1.0,
    repetition_penalty: float | None = None,
    min_new_tokens: int = 1,
    system_prompt: str | None = None,
    apply_chat_format: bool = True,
    stop_token_ids: list[int] | None = None,
    stop_strings: list[str] | None = None,
    sample_seed: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
    initial_decode_state: KVDecodeState | None = None,
    initial_token_ids: list[int] | None = None,
) -> PromptDecodeLoopResult:
    """Run one prompt generation with request-scoped safetensors handle reuse."""
    kwargs = {
        "steps": steps,
        "max_new_tokens": max_new_tokens,
        "start_layer": start_layer,
        "layer_count": layer_count,
        "lm_head_chunk_rows": lm_head_chunk_rows,
        "top_k": top_k,
        "top_p": top_p,
        "selection_policy": selection_policy,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "min_new_tokens": min_new_tokens,
        "system_prompt": system_prompt,
        "apply_chat_format": apply_chat_format,
        "stop_token_ids": stop_token_ids,
        "stop_strings": stop_strings,
        "sample_seed": sample_seed,
        "should_cancel": should_cancel,
        "initial_decode_state": initial_decode_state,
        "initial_token_ids": initial_token_ids,
    }
    effective_steps = steps if max_new_tokens is None else max_new_tokens
    use_scoped_handles = _use_scoped_safetensor_handles(model_id, effective_steps)
    if use_scoped_handles:
        with scoped_tensor_handle_cache():
            return _run_prompt_decode_loop(model_id, prompt, **kwargs)
    return _run_prompt_decode_loop(model_id, prompt, **kwargs)


def _use_scoped_safetensor_handles(model_id: str, effective_steps: int) -> bool:
    """Return whether one prompt call should reuse safetensors handles."""
    scoped_setting = os.environ.get("PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE", "auto").strip().lower()
    if scoped_setting in {"1", "true", "yes"}:
        return True
    if scoped_setting in {"0", "false", "no"}:
        return False
    normalized_model_id = model_id.strip().lower()
    if normalized_model_id in {
        "qwen2.5-32b-instruct",
        "qwen-2.5-32b-instruct",
        "qwen3-30b-a3b",
        "mixtral-8x7b-instruct-v01",
    }:
        return False
    return scoped_setting in {"", "auto"} and effective_steps <= 4


@torch.inference_mode()
def _run_prompt_decode_loop(
    model_id: str,
    prompt: str,
    steps: int = 1,
    max_new_tokens: int | None = None,
    start_layer: int = 0,
    layer_count: int | None = None,
    lm_head_chunk_rows: int = DEFAULT_LM_HEAD_CHUNK_ROWS,
    top_k: int = 5,
    top_p: float | None = None,
    selection_policy: str = "greedy",
    temperature: float = 1.0,
    repetition_penalty: float | None = None,
    min_new_tokens: int = 1,
    system_prompt: str | None = None,
    apply_chat_format: bool = True,
    stop_token_ids: list[int] | None = None,
    stop_strings: list[str] | None = None,
    sample_seed: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
    initial_decode_state: KVDecodeState | None = None,
    initial_token_ids: list[int] | None = None,
) -> PromptDecodeLoopResult:
    """Run the first real text-prompt entry path on top of the K/V decode session."""
    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    def finish_timings() -> dict[str, float]:
        payload = dict(timings)
        payload["total"] = round(time.perf_counter() - total_started, 4)
        return payload

    def record_phase(name: str, started: float) -> None:
        timings[name] = round(timings.get(name, 0.0) + (time.perf_counter() - started), 4)

    def add_nested_timings(prefix: str, source: dict[str, float]) -> None:
        for key, value in source.items():
            timing_key = f"{prefix}_{key}"
            timings[timing_key] = round(timings.get(timing_key, 0.0) + float(value), 4)

    token_summaries: list[dict] = []

    def record_token_summary(token_index: int, token_id: int, token_seconds: float) -> None:
        telemetry = expert_residency_snapshot()
        token_summaries.append(
            {
                "token_index": int(token_index),
                "token_id": int(token_id),
                "elapsed_seconds": round(float(token_seconds), 4),
                "expert_hits": int(telemetry.get("expert_hits", 0)),
                "expert_misses": int(telemetry.get("expert_misses", 0)),
                "expert_hit_rate": float(telemetry.get("expert_hit_rate", 0.0)),
                "expert_resident_count": int(telemetry.get("expert_resident_count", 0)),
                "expert_resident_bytes": int(telemetry.get("expert_resident_bytes", 0)),
            }
        )

    configure_runtime_threads()
    blockers: list[str] = []
    effective_max_new_tokens = steps if max_new_tokens is None else max_new_tokens
    if effective_max_new_tokens <= 0:
        blockers.append("Prompt decode loop requires at least one generation step.")
    if min_new_tokens <= 0:
        blockers.append("min_new_tokens must be at least 1.")
    effective_min_new_tokens = max(1, min(min_new_tokens, effective_max_new_tokens))
    effective_stop_strings = [value for value in (stop_strings or []) if value]
    phase_started = time.perf_counter()
    prepared_prompt_result = prepare_prompt_text(
        model_id,
        prompt,
        system_prompt=system_prompt,
        apply_chat_format=apply_chat_format,
    )
    record_phase("prepare_prompt", phase_started)
    prompt_token_ids = list(prepared_prompt_result.token_ids)
    blockers.extend(prepared_prompt_result.blockers)
    prefix_reuse: dict = {
        "enabled": initial_decode_state is not None and bool(initial_token_ids),
        "used": False,
        "matched_token_count": 0,
        "appended_token_count": 0,
        "summary": "No reusable prompt prefix was supplied.",
    }

    def canceled_result(
        *,
        generated_token_ids: list[int] | None = None,
        full_chain: list[int] | None = None,
        cache_sequence_lengths: dict[str, int] | None = None,
        steps_completed: int = 0,
    ) -> PromptDecodeLoopResult:
        generated = [] if generated_token_ids is None else list(generated_token_ids)
        chain = list(prompt_token_ids) + generated if full_chain is None else list(full_chain)
        generated_text, generated_blockers = decode_token_ids_to_text(model_id, generated)
        full_text, full_blockers = decode_token_ids_to_text(model_id, chain)
        return PromptDecodeLoopResult(
            model_id=model_id,
            prompt=prepared_prompt_result.prepared_prompt,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated,
            generated_text=generated_text,
            full_text=full_text,
            steps_requested=effective_max_new_tokens,
            max_new_tokens=effective_max_new_tokens,
            min_new_tokens=effective_min_new_tokens,
            steps_completed=steps_completed,
            strategy=f"{selection_policy}-prompt-kv-cache-rope",
            stop_reason="canceled",
            stop_token_ids=[] if stop_token_ids is None else [int(value) for value in stop_token_ids],
            stop_strings=list(effective_stop_strings),
            cache_sequence_lengths={} if cache_sequence_lengths is None else dict(cache_sequence_lengths),
            blockers=[CANCEL_BLOCKER] + list(generated_blockers) + list(full_blockers),
            ready=False,
            timings=finish_timings(),
        )

    if _cancel_requested(should_cancel):
        return canceled_result()

    if blockers or not prompt_token_ids:
        return PromptDecodeLoopResult(
            model_id=model_id,
            prompt=prepared_prompt_result.prepared_prompt,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=[],
            generated_text="",
            full_text=prompt,
            steps_requested=effective_max_new_tokens,
            max_new_tokens=effective_max_new_tokens,
            min_new_tokens=effective_min_new_tokens,
            steps_completed=0,
            strategy=f"{selection_policy}-prompt-kv-cache-rope",
            stop_token_ids=[] if stop_token_ids is None else [int(value) for value in stop_token_ids],
            stop_strings=list(effective_stop_strings),
            blockers=blockers,
            ready=False,
            timings=finish_timings(),
        )

    phase_started = time.perf_counter()
    generation_settings = load_generation_settings(model_id)
    effective_policy = selection_policy
    effective_top_k = top_k
    effective_top_p = top_p if top_p is not None else 1.0
    effective_temperature = temperature
    effective_repetition_penalty = (
        repetition_penalty
        if repetition_penalty is not None
        else (generation_settings.repetition_penalty if generation_settings.ready else 1.0)
    )
    config = load_layer_bridge_config(model_id)
    effective_layer_count = layer_count
    if effective_layer_count is None:
        effective_layer_count = _recommended_prompt_layer_count(
            config,
            len(prompt_token_ids),
            effective_max_new_tokens,
        )
    configured_layer_count = int(config.num_hidden_layers if config.ready else effective_layer_count)
    is_default_full_stack_run = layer_count is None and start_layer == 0 and configured_layer_count > 0
    if is_default_full_stack_run and effective_layer_count < configured_layer_count:
        raise RuntimeError(
            f"Anti-cheat guard refused a default full prompt run with {effective_layer_count} "
            f"layers for a {configured_layer_count}-layer model."
        )
    layers_executed_total = 0

    if generation_settings.ready:
        if top_k == 5:
            effective_top_k = generation_settings.top_k
        if top_p is None:
            effective_top_p = generation_settings.top_p
        if temperature == 1.0:
            effective_temperature = generation_settings.temperature
    record_phase("configure_generation", phase_started)

    supplied_prefix_ids = [] if initial_token_ids is None else [int(value) for value in initial_token_ids]
    try:
        max_prefix_append_tokens = max(1, int(os.environ.get("PCKETLM_PREFIX_REUSE_MAX_APPEND_TOKENS", "64")))
    except ValueError:
        max_prefix_append_tokens = 8
    suffix_token_count = len(prompt_token_ids) - len(supplied_prefix_ids)
    can_try_prefix = (
        initial_decode_state is not None
        and initial_decode_state.ready
        and initial_decode_state.model_id == model_id
        and bool(supplied_prefix_ids)
        and len(supplied_prefix_ids) < len(prompt_token_ids)
        and prompt_token_ids[: len(supplied_prefix_ids)] == supplied_prefix_ids
        and suffix_token_count <= max_prefix_append_tokens
    )
    if initial_decode_state is not None or supplied_prefix_ids:
        if can_try_prefix:
            prefix_reuse.update(
                {
                    "matched_token_count": len(supplied_prefix_ids),
                    "summary": "A reusable prompt prefix matched and will be extended.",
                }
            )
        else:
            summary = "Supplied prefix did not safely match this prompt; full prefill was used."
            if (
                initial_decode_state is not None
                and initial_decode_state.ready
                and initial_decode_state.model_id == model_id
                and bool(supplied_prefix_ids)
                and len(supplied_prefix_ids) < len(prompt_token_ids)
                and prompt_token_ids[: len(supplied_prefix_ids)] == supplied_prefix_ids
                and suffix_token_count > max_prefix_append_tokens
            ):
                summary = (
                    f"Matching prefix needed {suffix_token_count} append tokens, above the safe "
                    f"batched limit of {max_prefix_append_tokens}; full prefill was used."
                )
            prefix_reuse.update(
                {
                    "matched_token_count": len(supplied_prefix_ids)
                    if suffix_token_count > max_prefix_append_tokens
                    else 0,
                    "summary": summary,
                }
            )

    if can_try_prefix:
        phase_started = time.perf_counter()
        suffix_token_ids = prompt_token_ids[len(supplied_prefix_ids) :]
        token_hidden_state, token_hidden_blockers = load_token_entry_hidden_state(model_id, suffix_token_ids)
        append_blockers: list[str] = []
        stack_result = None
        current_state = initial_decode_state
        current_token_ids = list(supplied_prefix_ids)
        if token_hidden_blockers or token_hidden_state is None:
            append_blockers.extend(token_hidden_blockers or ["Prefix append token entry failed."])
        elif _cancel_requested(should_cancel):
            return canceled_result(
                full_chain=current_token_ids,
                cache_sequence_lengths=dict(current_state.cache_sequence_lengths),
            )
        else:
            stack_result = run_layer_bridge_stack(
                model_id,
                start_layer=start_layer,
                layer_count=effective_layer_count,
                input_hidden=token_hidden_state,
                past_key_values=current_state.kv_caches,
                position_offset=current_state.next_position,
                return_kv_cache=True,
                native_kv_sessions=current_state.native_kv_sessions,
                should_cancel=should_cancel,
                collect_step_summaries=False,
                collect_metrics=False,
            )
            add_nested_timings("prefix_append_stack", stack_result.timings)
            if not stack_result.ready or stack_result.output_tensor is None:
                append_blockers.extend(stack_result.blockers or ["Prefix append stack failed."])
            else:
                current_token_ids.extend(suffix_token_ids)
                current_state = KVDecodeState(
                    model_id=model_id,
                    next_token_id=suffix_token_ids[-1],
                    next_position=current_state.next_position + len(suffix_token_ids),
                    generated_token_ids=list(current_token_ids),
                    cache_sequence_lengths=dict(stack_result.cache_sequence_lengths),
                    kv_caches=dict(stack_result.next_kv_caches),
                    native_kv_sessions=dict(stack_result.next_native_kv_sessions),
                    ready=True,
                )
        record_phase("prefix_append", phase_started)
        prefix_reuse.update(
            {
                "used": not append_blockers and stack_result is not None and stack_result.output_tensor is not None,
                "appended_token_count": len(suffix_token_ids) if not append_blockers else 0,
                "summary": (
                    f"Reused {len(supplied_prefix_ids)} prompt tokens and batch-appended {len(suffix_token_ids)} new prompt tokens."
                    if not append_blockers and stack_result is not None and stack_result.output_tensor is not None
                    else "Prefix reuse was attempted but failed; generation stopped before unsafe output."
                ),
            }
        )
        if append_blockers or stack_result is None or stack_result.output_tensor is None:
            return PromptDecodeLoopResult(
                model_id=model_id,
                prompt=prepared_prompt_result.prepared_prompt,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=[],
                generated_text="",
                full_text=prompt,
                steps_requested=effective_max_new_tokens,
                max_new_tokens=effective_max_new_tokens,
                min_new_tokens=effective_min_new_tokens,
                steps_completed=0,
                strategy=f"{selection_policy}-prompt-kv-cache-rope",
                stop_token_ids=[] if stop_token_ids is None else [int(value) for value in stop_token_ids],
                stop_strings=list(effective_stop_strings),
                cache_sequence_lengths=dict(current_state.cache_sequence_lengths),
                blockers=append_blockers,
                ready=False,
                timings=finish_timings(),
                prefix_reuse=prefix_reuse,
            )
        prefill_stack = SimpleNamespace(
            ready=True,
            output_tensor=stack_result.output_tensor,
            cache_sequence_lengths=dict(current_state.cache_sequence_lengths),
            next_kv_caches=dict(current_state.kv_caches),
            blockers=[],
            timings={"prefix_reuse": round(time.perf_counter() - phase_started, 4)},
        )
    else:
        phase_started = time.perf_counter()
        prompt_hidden_state, prompt_hidden_blockers = load_token_entry_hidden_state(model_id, prompt_token_ids)
        record_phase("token_entry", phase_started)
        blockers.extend(prompt_hidden_blockers)
        if blockers or prompt_hidden_state is None:
            return PromptDecodeLoopResult(
                model_id=model_id,
                prompt=prepared_prompt_result.prepared_prompt,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=[],
                generated_text="",
                full_text=prompt,
                steps_requested=effective_max_new_tokens,
                max_new_tokens=effective_max_new_tokens,
                min_new_tokens=effective_min_new_tokens,
                steps_completed=0,
                strategy=f"{selection_policy}-prompt-kv-cache-rope",
                stop_token_ids=[] if stop_token_ids is None else [int(value) for value in stop_token_ids],
                stop_strings=list(effective_stop_strings),
                blockers=blockers,
                ready=False,
                timings=finish_timings(),
                prefix_reuse=prefix_reuse,
            )
        if _cancel_requested(should_cancel):
            return canceled_result()

        phase_started = time.perf_counter()
        prefill_stack = run_layer_bridge_stack(
            model_id,
            start_layer=start_layer,
            layer_count=effective_layer_count,
            input_hidden=prompt_hidden_state,
            return_kv_cache=True,
            should_cancel=should_cancel,
            collect_step_summaries=False,
            collect_metrics=False,
        )
        record_phase("prefill_stack", phase_started)
        add_nested_timings("prefill_stack", prefill_stack.timings)
        layers_executed_total += len(prefill_stack.executed_layers)
    if not prefill_stack.ready or prefill_stack.output_tensor is None:
        return PromptDecodeLoopResult(
            model_id=model_id,
            prompt=prepared_prompt_result.prepared_prompt,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=[],
            generated_text="",
            full_text=prompt,
            steps_requested=effective_max_new_tokens,
            max_new_tokens=effective_max_new_tokens,
            min_new_tokens=effective_min_new_tokens,
            steps_completed=0,
            strategy=f"{effective_policy}-prompt-kv-cache-rope",
            stop_token_ids=[] if stop_token_ids is None else [int(value) for value in stop_token_ids],
            stop_strings=list(effective_stop_strings),
            cache_sequence_lengths=dict(prefill_stack.cache_sequence_lengths),
            blockers=list(prefill_stack.blockers) or ["Prompt prefill stack failed before generation could start."],
            ready=False,
            timings=finish_timings(),
        )

    phase_started = time.perf_counter()
    prefill_tail = run_decode_tail(
        model_id,
        prefill_stack.output_tensor[:, -1:, :],
        lm_head_chunk_rows=lm_head_chunk_rows,
        top_k=effective_top_k,
        should_cancel=should_cancel,
        return_logits=not _can_select_from_topk(effective_policy),
        recent_token_ids=prompt_token_ids,
        repetition_penalty=effective_repetition_penalty,
    )
    record_phase("prefill_decode_tail", phase_started)
    if not prefill_tail.ready or (
        prefill_tail.logits is None
        and (not _can_select_from_topk(effective_policy) or not prefill_tail.top_token_ids)
    ):
        return PromptDecodeLoopResult(
            model_id=model_id,
            prompt=prepared_prompt_result.prepared_prompt,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=[],
            generated_text="",
            full_text=prompt,
            steps_requested=effective_max_new_tokens,
            max_new_tokens=effective_max_new_tokens,
            min_new_tokens=effective_min_new_tokens,
            steps_completed=0,
            strategy=f"{effective_policy}-prompt-kv-cache-rope",
            stop_token_ids=[] if stop_token_ids is None else [int(value) for value in stop_token_ids],
            stop_strings=list(effective_stop_strings),
            cache_sequence_lengths=dict(prefill_stack.cache_sequence_lengths),
            blockers=list(prefill_tail.blockers) or ["Prompt prefill decode tail failed before generation could start."],
            ready=False,
            timings=finish_timings(),
        )
    if _cancel_requested(should_cancel):
        return canceled_result(cache_sequence_lengths=dict(prefill_stack.cache_sequence_lengths))

    phase_started = time.perf_counter()
    if prefill_tail.logits is None and _can_select_from_topk(effective_policy):
        first_selection = select_next_token_from_topk(
            prefill_tail.top_token_ids,
            prefill_tail.top_logits,
            policy=effective_policy,
            top_p=effective_top_p,
            temperature=effective_temperature,
            sample_seed=sample_seed,
        )
    else:
        first_selection = select_next_token(
            prefill_tail.logits,
            policy=effective_policy,
            top_k=effective_top_k,
            top_p=effective_top_p,
            temperature=effective_temperature,
            repetition_penalty=effective_repetition_penalty,
            recent_token_ids=prompt_token_ids,
            sample_seed=sample_seed,
        )
    record_phase("first_selection", phase_started)
    if not first_selection.ready or first_selection.chosen_token_id is None:
        return PromptDecodeLoopResult(
            model_id=model_id,
            prompt=prepared_prompt_result.prepared_prompt,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=[],
            generated_text="",
            full_text=prompt,
            steps_requested=effective_max_new_tokens,
            max_new_tokens=effective_max_new_tokens,
            min_new_tokens=effective_min_new_tokens,
            steps_completed=0,
            strategy=f"{effective_policy}-prompt-kv-cache-rope",
            stop_token_ids=[] if stop_token_ids is None else [int(value) for value in stop_token_ids],
            stop_strings=list(effective_stop_strings),
            cache_sequence_lengths=dict(prefill_stack.cache_sequence_lengths),
            blockers=list(first_selection.blockers) or ["Prompt prefill token selection failed."],
            ready=False,
            timings=finish_timings(),
        )

    effective_stop_token_ids = list(config.eos_token_ids) if stop_token_ids is None else [int(value) for value in stop_token_ids]
    first_generated_token_id = first_selection.chosen_token_id
    first_generated_chain = list(prompt_token_ids) + [first_generated_token_id]
    record_token_summary(1, first_generated_token_id, time.perf_counter() - total_started)
    reached_stop = effective_min_new_tokens <= 1 and first_generated_token_id in effective_stop_token_ids
    first_stop_reason = "eos-token" if reached_stop and stop_token_ids is None else ("custom-stop-token" if reached_stop else None)

    decode_state = KVDecodeState(
        model_id=model_id,
        next_token_id=first_generated_token_id,
        next_position=len(prompt_token_ids),
        generated_token_ids=first_generated_chain,
        cache_sequence_lengths=dict(prefill_stack.cache_sequence_lengths),
        kv_caches=dict(prefill_stack.next_kv_caches),
        native_kv_sessions=dict(prefill_stack.next_native_kv_sessions),
        finished=reached_stop,
        stop_reason=first_stop_reason,
        ready=True,
        blockers=[],
    )

    if decode_state.finished or effective_max_new_tokens == 1:
        generated_token_ids = [first_generated_token_id]
        generated_text, decode_generated_blockers = decode_token_ids_to_text(model_id, generated_token_ids)
        full_text, decode_full_blockers = decode_token_ids_to_text(model_id, first_generated_chain)
        generated_text, triggered_stop_string = _trim_generated_text_at_stop_string(generated_text, effective_stop_strings)
        if not decode_state.finished and triggered_stop_string is not None:
            decode_state.stop_reason = "stop-string"
        return PromptDecodeLoopResult(
            model_id=model_id,
            prompt=prepared_prompt_result.prepared_prompt,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_text=generated_text,
            full_text=full_text,
            steps_requested=effective_max_new_tokens,
            max_new_tokens=effective_max_new_tokens,
            min_new_tokens=effective_min_new_tokens,
            steps_completed=1,
            strategy=f"{effective_policy}-prompt-kv-cache-rope",
            stop_reason=decode_state.stop_reason if decode_state.finished else "step-limit",
            stop_token_ids=[] if stop_token_ids is None else [int(value) for value in stop_token_ids],
            stop_strings=list(effective_stop_strings),
            cache_sequence_lengths=dict(prefill_stack.cache_sequence_lengths),
            blockers=list(prefill_stack.blockers) + list(prefill_tail.blockers) + list(decode_generated_blockers) + list(decode_full_blockers),
            ready=True,
            timings=finish_timings(),
            token_summaries=token_summaries,
            prefix_reuse=prefix_reuse,
            reusable_token_ids=list(prompt_token_ids),
            configured_layer_count=configured_layer_count,
            prompt_layer_count=effective_layer_count,
            layers_executed=layers_executed_total,
            expected_layers_executed=effective_layer_count,
            anti_cheat_passed=layers_executed_total >= effective_layer_count,
            final_decode_state=decode_state,
        )
    generated_token_ids = [first_generated_token_id]
    latest_chain = list(first_generated_chain)
    latest_cache_lengths = dict(prefill_stack.cache_sequence_lengths)
    generated_text = ""
    full_text = ""
    continuation_blockers: list[str] = []
    stop_reason: str | None = None
    decode_generated_blockers: list[str] = []
    decode_full_blockers: list[str] = []
    steps_completed = 1

    for step_index in range(1, effective_max_new_tokens):
        if _cancel_requested(should_cancel):
            return canceled_result(
                generated_token_ids=generated_token_ids,
                full_chain=latest_chain,
                cache_sequence_lengths=latest_cache_lengths,
                steps_completed=steps_completed,
            )
        effective_step_stop_token_ids = (
            effective_stop_token_ids
            if (steps_completed + 1) >= effective_min_new_tokens
            else None
        )
        phase_started = time.perf_counter()
        step_result = run_kv_decode_step(
            model_id,
            input_token_id=decode_state.next_token_id,
            decode_state=decode_state,
            start_layer=start_layer,
            layer_count=effective_layer_count,
            lm_head_chunk_rows=lm_head_chunk_rows,
            top_k=effective_top_k,
            top_p=effective_top_p,
            selection_policy=effective_policy,
            temperature=effective_temperature,
            repetition_penalty=effective_repetition_penalty,
            stop_token_ids=effective_step_stop_token_ids,
            sample_seed=None if sample_seed is None else sample_seed + step_index,
            should_cancel=should_cancel,
            collect_layer_details=False,
        )
        step_elapsed = time.perf_counter() - phase_started
        record_phase("continuation_steps", phase_started)
        add_nested_timings("continuation", step_result.timings)
        latest_cache_lengths = dict(step_result.cache_sequence_lengths)
        if not step_result.ready or step_result.chosen_token_id is None or step_result.next_decode_state is None:
            continuation_blockers.extend(step_result.blockers or [f"Prompt continuation step {step_index} failed."])
            break
        layers_executed_total += len(step_result.executed_layers)

        decode_state = step_result.next_decode_state
        latest_chain = list(decode_state.generated_token_ids)
        generated_token_ids = list(latest_chain[len(prompt_token_ids):])
        steps_completed = len(generated_token_ids)
        record_token_summary(steps_completed, generated_token_ids[-1], step_elapsed)

        generated_text, decode_generated_blockers = decode_token_ids_to_text(model_id, generated_token_ids)
        full_text, decode_full_blockers = decode_token_ids_to_text(model_id, latest_chain)
        if decode_generated_blockers or decode_full_blockers:
            continuation_blockers.extend(decode_generated_blockers + decode_full_blockers)
            break

        if steps_completed >= effective_min_new_tokens and effective_stop_strings:
            triggered = next((value for value in effective_stop_strings if value in generated_text), None)
            if triggered is not None:
                stop_reason = "stop-string"
                break

        if decode_state.finished:
            stop_reason = decode_state.stop_reason
            break

    if not generated_text:
        generated_text, decode_generated_blockers = decode_token_ids_to_text(model_id, generated_token_ids)
    if not full_text:
        full_text, decode_full_blockers = decode_token_ids_to_text(model_id, latest_chain)
    generated_text, _triggered_stop_string = _trim_generated_text_at_stop_string(generated_text, effective_stop_strings)
    if stop_reason is None:
        stop_reason = decode_state.stop_reason if decode_state.finished else ("step-limit" if steps_completed >= effective_max_new_tokens else None)
    expected_layers_executed = effective_layer_count * max(1, steps_completed)
    anti_cheat_blockers: list[str] = []
    if is_default_full_stack_run and layers_executed_total < expected_layers_executed:
        anti_cheat_blockers.append(
            f"Anti-cheat guard: executed {layers_executed_total} layer forwards, "
            f"expected {expected_layers_executed} for {steps_completed} generated token(s)."
        )
    blockers = (
        list(prefill_stack.blockers)
        + list(prefill_tail.blockers)
        + continuation_blockers
        + list(decode_generated_blockers)
        + list(decode_full_blockers)
        + anti_cheat_blockers
    )
    return PromptDecodeLoopResult(
        model_id=model_id,
        prompt=prepared_prompt_result.prepared_prompt,
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        generated_text=generated_text,
        full_text=full_text,
        steps_requested=effective_max_new_tokens,
        max_new_tokens=effective_max_new_tokens,
        min_new_tokens=effective_min_new_tokens,
        steps_completed=steps_completed,
        strategy=f"{effective_policy}-prompt-kv-cache-rope",
        stop_reason=stop_reason,
        stop_token_ids=[] if stop_token_ids is None else [int(value) for value in stop_token_ids],
        stop_strings=list(effective_stop_strings),
        cache_sequence_lengths=latest_cache_lengths,
        blockers=blockers,
        ready=not continuation_blockers and not anti_cheat_blockers,
        timings=finish_timings(),
        token_summaries=token_summaries,
        prefix_reuse=prefix_reuse,
        reusable_token_ids=list(latest_chain[:-1]) if len(latest_chain) > len(prompt_token_ids) else list(prompt_token_ids),
        configured_layer_count=configured_layer_count,
        prompt_layer_count=effective_layer_count,
        layers_executed=layers_executed_total,
        expected_layers_executed=expected_layers_executed,
        anti_cheat_passed=not anti_cheat_blockers,
        final_decode_state=decode_state,
    )
