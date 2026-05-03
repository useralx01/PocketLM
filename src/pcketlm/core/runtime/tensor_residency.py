"""Memory-capped tensor residency for converted runtime tensors."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from pcketlm.core.runtime.tensor_catalog import TensorCatalogEntry, find_tensor_catalog_entry
from pcketlm.core.runtime.tensor_loader import (
    LoadedTensorSlice,
    clear_tensor_handle_cache,
    load_tensor_by_name,
    load_tensors_by_name,
)

DEFAULT_TENSOR_CACHE_MB = 256
DEFAULT_FRONT_LAYER_COUNT = 12
DEFAULT_MAX_TENSOR_CACHE_MB = 32
DEFAULT_STICKY_RESIDENCY_STEPS = 1
DEFAULT_EXPERT_CACHE_MB = 256
DEFAULT_MAX_RESIDENT_EXPERTS_PER_LAYER = 4
DEFAULT_EXPERT_DECAY_RATE = 0.98
DEFAULT_ALWAYS_RESIDENT_TENSOR_MB = 16
DEFAULT_FP16_PACKED_CACHE_CAP_MB = 8 * 1024
BOOSTED_TENSOR_CACHE_MB = 288
BOOSTED_FRONT_LAYER_COUNT = 13
LOW_MEMORY_CACHE_MB = 128
LOW_MEMORY_FRONT_LAYER_COUNT = 6
LOW_MEMORY_GUARD_THRESHOLD_MB = 3 * 1024
MODEL_AWARE_SAFETY_MARGIN_MB = 4 * 1024
MODEL_AWARE_MAX_CACHE_MB = 4 * 1024
MEMORY_SNAPSHOT_CACHE_SECONDS = 2.0
_memory_snapshot_cache: tuple[float, int | None] = (0.0, None)


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or fallback)
    except ValueError:
        return fallback


def _env_enabled(name: str, fallback: str = "1") -> bool:
    return os.environ.get(name, fallback).strip().lower() not in {"0", "false", "no"}


def _env_is_set(name: str) -> bool:
    return os.environ.get(name, "").strip() != ""


def _env_float(name: str, fallback: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or fallback)
    except ValueError:
        return fallback


def _free_memory_bytes() -> int | None:
    global _memory_snapshot_cache
    now = time.monotonic()
    cached_at, cached_value = _memory_snapshot_cache
    if now - cached_at <= MEMORY_SNAPSHOT_CACHE_SECONDS:
        return cached_value
    try:
        from pcketlm.core.runtime.load_attempt import _memory_snapshot

        free_bytes = int(_memory_snapshot().free_bytes)
    except Exception:
        free_bytes = None
    _memory_snapshot_cache = (now, free_bytes)
    return free_bytes


@dataclass(frozen=True, slots=True)
class TensorResidencyPolicy:
    """Limits for keeping converted tensors resident between decode passes."""

    enabled: bool = True
    max_resident_bytes: int = 256 * 1024 * 1024
    max_tensor_bytes: int = 32 * 1024 * 1024
    all_layer_small_tensor_bytes: int = 1 * 1024 * 1024
    min_tensor_bytes: int = 1
    front_layer_count: int = 12
    tensor_cache_preset: str = "standard"
    memory_guard_active: bool = False
    adaptive_boost_active: bool = False
    model_aware_budget_active: bool = False
    free_memory_bytes: int | None = None
    sticky_residency_steps: int = DEFAULT_STICKY_RESIDENCY_STEPS
    expert_max_resident_bytes: int = DEFAULT_EXPERT_CACHE_MB * 1024 * 1024
    max_resident_experts_per_layer: int = DEFAULT_MAX_RESIDENT_EXPERTS_PER_LAYER
    expert_decay_rate: float = DEFAULT_EXPERT_DECAY_RATE
    expert_q4_residency: bool = False

    @classmethod
    def from_environment(cls, model_id: str | None = None) -> "TensorResidencyPolicy":
        requested_preset = os.environ.get("PCKETLM_TENSOR_CACHE_PRESET", "standard").strip().lower() or "standard"
        tensor_cache_preset = "boosted" if requested_preset in {"boost", "boosted", "high-ram", "high_ram"} else "standard"
        max_resident_mb = max(0, _env_int("PCKETLM_TENSOR_CACHE_MB", DEFAULT_TENSOR_CACHE_MB))
        max_tensor_mb = max(0, _env_int("PCKETLM_TENSOR_CACHE_TENSOR_MB", DEFAULT_MAX_TENSOR_CACHE_MB))
        all_layer_small_tensor_kb = max(0, _env_int("PCKETLM_TENSOR_CACHE_ALL_LAYER_SMALL_KB", 1024))
        front_layer_count = max(0, _env_int("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", DEFAULT_FRONT_LAYER_COUNT))
        memory_guard_active = False
        adaptive_boost_active = False
        model_aware_budget_active = False
        guard_threshold_mb = max(0, _env_int("PCKETLM_TENSOR_CACHE_LOW_MEMORY_GUARD_MB", LOW_MEMORY_GUARD_THRESHOLD_MB))
        free_memory_bytes = _free_memory_bytes() if _env_enabled("PCKETLM_TENSOR_CACHE_MEMORY_GUARD") else None
        if free_memory_bytes is not None and free_memory_bytes < guard_threshold_mb * 1024 * 1024:
            memory_guard_active = True
            if not _env_is_set("PCKETLM_TENSOR_CACHE_MB"):
                max_resident_mb = min(max_resident_mb, LOW_MEMORY_CACHE_MB)
            if not _env_is_set("PCKETLM_TENSOR_CACHE_FRONT_LAYERS"):
                front_layer_count = min(front_layer_count, LOW_MEMORY_FRONT_LAYER_COUNT)
        elif tensor_cache_preset == "boosted":
            adaptive_boost_active = True
            if not _env_is_set("PCKETLM_TENSOR_CACHE_MB"):
                max_resident_mb = max(max_resident_mb, BOOSTED_TENSOR_CACHE_MB)
            if not _env_is_set("PCKETLM_TENSOR_CACHE_FRONT_LAYERS"):
                front_layer_count = max(front_layer_count, BOOSTED_FRONT_LAYER_COUNT)
        elif model_id and free_memory_bytes is not None and not _env_is_set("PCKETLM_TENSOR_CACHE_MB"):
            try:
                from pcketlm.core.runtime.tensor_catalog import load_tensor_catalog

                catalog = load_tensor_catalog(model_id)
                deep_model = bool(catalog.num_hidden_layers and catalog.num_hidden_layers > DEFAULT_FRONT_LAYER_COUNT * 4)
            except Exception:
                deep_model = False
            available_mb = int(free_memory_bytes // (1024**2))
            model_budget_mb = max(0, min(available_mb - MODEL_AWARE_SAFETY_MARGIN_MB, MODEL_AWARE_MAX_CACHE_MB))
            if deep_model and model_budget_mb > max_resident_mb:
                max_resident_mb = model_budget_mb
                model_aware_budget_active = True
        expert_cache_mb = max(0, _env_int("PCKETLM_EXPERT_TENSOR_CACHE_MB", DEFAULT_EXPERT_CACHE_MB))
        moe_top_k = 0
        if (
            model_id
            and free_memory_bytes is not None
        ):
            try:
                from pcketlm.core.runtime.tensor_catalog import load_tensor_catalog

                catalog = load_tensor_catalog(model_id)
                is_moe_model = bool(catalog.num_experts and catalog.num_experts_per_tok)
                moe_top_k = int(catalog.num_experts_per_tok or 0)
            except Exception:
                is_moe_model = False
            if is_moe_model and not _env_is_set("PCKETLM_EXPERT_TENSOR_CACHE_MB"):
                free_mb = int(free_memory_bytes // (1024**2))
                adaptive_expert_mb = max(DEFAULT_EXPERT_CACHE_MB, min(max(0, free_mb - 2048), 2048))
                expert_cache_mb = max(expert_cache_mb, adaptive_expert_mb)

        expert_decay_rate = max(0.0, min(1.0, _env_float("PCKETLM_EXPERT_CACHE_DECAY", DEFAULT_EXPERT_DECAY_RATE)))
        expert_q4_residency = _env_enabled("PCKETLM_EXPERT_Q4_CACHE", "0")
        max_resident_experts_per_layer = max(
            0,
            _env_int("PCKETLM_MAX_RESIDENT_EXPERTS_PER_LAYER", DEFAULT_MAX_RESIDENT_EXPERTS_PER_LAYER),
        )
        if moe_top_k and not _env_is_set("PCKETLM_MAX_RESIDENT_EXPERTS_PER_LAYER"):
            q4_multiplier = 4 if expert_q4_residency else 1
            max_resident_experts_per_layer = max(max_resident_experts_per_layer, moe_top_k * q4_multiplier)

        return cls(
            enabled=os.environ.get("PCKETLM_TENSOR_CACHE", "1").strip().lower() not in {"0", "false", "no"},
            max_resident_bytes=max_resident_mb * 1024 * 1024,
            max_tensor_bytes=max_tensor_mb * 1024 * 1024,
            all_layer_small_tensor_bytes=all_layer_small_tensor_kb * 1024,
            front_layer_count=front_layer_count,
            tensor_cache_preset=tensor_cache_preset,
            memory_guard_active=memory_guard_active,
            adaptive_boost_active=adaptive_boost_active,
            model_aware_budget_active=model_aware_budget_active,
            free_memory_bytes=free_memory_bytes,
            sticky_residency_steps=max(0, _env_int("PCKETLM_TENSOR_CACHE_STICKY_STEPS", DEFAULT_STICKY_RESIDENCY_STEPS)),
            expert_max_resident_bytes=expert_cache_mb * 1024 * 1024,
            max_resident_experts_per_layer=max_resident_experts_per_layer,
            expert_decay_rate=expert_decay_rate,
            expert_q4_residency=expert_q4_residency,
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "max_resident_bytes": self.max_resident_bytes,
            "max_resident_mb": round(self.max_resident_bytes / (1024**2), 2),
            "max_tensor_bytes": self.max_tensor_bytes,
            "all_layer_small_tensor_bytes": self.all_layer_small_tensor_bytes,
            "front_layer_count": self.front_layer_count,
            "tensor_cache_preset": self.tensor_cache_preset,
            "memory_guard_active": self.memory_guard_active,
            "adaptive_boost_active": self.adaptive_boost_active,
            "model_aware_budget_active": self.model_aware_budget_active,
            "free_memory_bytes": self.free_memory_bytes,
            "free_memory_gb": None if self.free_memory_bytes is None else round(self.free_memory_bytes / (1024**3), 2),
            "sticky_residency_steps": self.sticky_residency_steps,
            "expert_max_resident_bytes": self.expert_max_resident_bytes,
            "expert_max_resident_mb": round(self.expert_max_resident_bytes / (1024**2), 2),
            "max_resident_experts_per_layer": self.max_resident_experts_per_layer,
            "expert_decay_rate": self.expert_decay_rate,
            "expert_q4_residency": self.expert_q4_residency,
        }


@dataclass(slots=True)
class TensorResidencyStats:
    """Small runtime counters for cache behavior."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    skips: int = 0
    resident_bytes: int = 0
    resident_count: int = 0
    expert_hits: int = 0
    expert_misses: int = 0
    expert_evictions: int = 0
    expert_resident_bytes: int = 0
    expert_resident_count: int = 0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "skips": self.skips,
            "resident_bytes": self.resident_bytes,
            "resident_count": self.resident_count,
            "expert_hits": self.expert_hits,
            "expert_misses": self.expert_misses,
            "expert_evictions": self.expert_evictions,
            "expert_resident_bytes": self.expert_resident_bytes,
            "expert_resident_count": self.expert_resident_count,
        }


@dataclass(slots=True)
class Fp16PackedCacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    disk_reads: int = 0
    resident_bytes: int = 0
    resident_count: int = 0
    budget_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "disk_reads": self.disk_reads,
            "resident_bytes": self.resident_bytes,
            "resident_mb": round(self.resident_bytes / (1024**2), 2),
            "resident_count": self.resident_count,
            "budget_bytes": self.budget_bytes,
            "budget_mb": round(self.budget_bytes / (1024**2), 2),
        }


@dataclass(slots=True)
class _QuantizedQ4Tensor:
    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


@dataclass(slots=True)
class _ResidentTensor:
    tensor: torch.Tensor | None
    entry: TensorCatalogEntry
    nbytes: int
    dtype: str
    loaded_step: int
    expert_key: tuple[int, int] | None = None
    q4: _QuantizedQ4Tensor | None = None


_CacheKey = tuple[str, str, str, int, int, str]
_Fp16PackedCacheKey = tuple[str, str, str, int, int, int]

_cache_lock = threading.RLock()
_resident_tensors: OrderedDict[_CacheKey, _ResidentTensor] = OrderedDict()
_fp16_packed_cache: OrderedDict[_Fp16PackedCacheKey, bytearray] = OrderedDict()
_stats = TensorResidencyStats()
_fp16_packed_stats = Fp16PackedCacheStats()
_resident_bytes = 0
_fp16_packed_cache_bytes = 0
_resident_expert_bytes = 0
_residency_step = 0
_expert_activation_counts: dict[tuple[int, int], int] = {}
_expert_activation_scores: dict[tuple[int, int], float] = {}
_current_step_experts: set[tuple[int, int]] = set()


def advance_tensor_residency_step(steps: int = 1) -> int:
    """Advance the logical decode step used by sticky residency."""
    global _residency_step
    with _cache_lock:
        _residency_step += max(1, int(steps))
        _current_step_experts.clear()
        return _residency_step


def current_tensor_residency_step() -> int:
    """Return the current logical decode step used by sticky residency."""
    with _cache_lock:
        return _residency_step


def record_expert_activation(layer_index: int, expert_index: int) -> None:
    """Record that an MoE expert was selected for the current decode step."""
    key = (int(layer_index), int(expert_index))
    with _cache_lock:
        _current_step_experts.add(key)
        _expert_activation_counts[key] = _expert_activation_counts.get(key, 0) + 1
        decay_rate = max(0.0, min(1.0, _env_float("PCKETLM_EXPERT_CACHE_DECAY", DEFAULT_EXPERT_DECAY_RATE)))
        if decay_rate < 1.0:
            stale_keys: list[tuple[int, int]] = []
            for score_key, score in _expert_activation_scores.items():
                decayed = score * decay_rate
                if decayed < 1e-6:
                    stale_keys.append(score_key)
                else:
                    _expert_activation_scores[score_key] = decayed
            for stale_key in stale_keys:
                _expert_activation_scores.pop(stale_key, None)
        _expert_activation_scores[key] = _expert_activation_scores.get(key, 0.0) + 1.0


def expert_residency_snapshot(top_k: int = 10) -> dict:
    """Return expert-cache counters for diagnostics."""
    with _cache_lock:
        total_activations = sum(_expert_activation_counts.values())
        hits = int(_stats.expert_hits)
        misses = int(_stats.expert_misses)
        denominator = hits + misses
        sorted_by_touch = sorted(
            _expert_activation_counts.items(),
            key=lambda item: (-_expert_activation_scores.get(item[0], 0.0), -item[1], item[0][0], item[0][1]),
        )
        sorted_least_touched = sorted(
            _expert_activation_counts.items(),
            key=lambda item: (item[1], item[0][0], item[0][1]),
        )
        def format_experts(items: list[tuple[tuple[int, int], int]]) -> list[dict]:
            return [
                {
                    "layer": int(layer),
                    "expert": int(expert),
                    "touches": int(count),
                    "score": round(float(_expert_activation_scores.get((layer, expert), 0.0)), 4),
                }
                for (layer, expert), count in items[: max(0, int(top_k))]
            ]

        return {
            "activated_experts": {f"{layer}:{expert}": count for (layer, expert), count in sorted(_expert_activation_counts.items())},
            "current_step_experts": [f"{layer}:{expert}" for layer, expert in sorted(_current_step_experts)],
            "total_expert_requests": denominator,
            "expert_hits": hits,
            "expert_misses": misses,
            "expert_hit_rate": 0.0 if denominator == 0 else round(hits / denominator, 4),
            "expert_activation_total": total_activations,
            "top_touched_experts": format_experts(sorted_by_touch),
            "least_touched_experts": format_experts(sorted_least_touched),
            "expert_resident_bytes": _resident_expert_bytes,
            "expert_resident_count": sum(1 for resident in _resident_tensors.values() if resident.expert_key is not None),
        }


def _find_tensor_entry(model_id: str, tensor_name: str) -> TensorCatalogEntry | None:
    return find_tensor_catalog_entry(model_id, tensor_name)


def _path_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _cache_key(model_id: str, entry: TensorCatalogEntry, dtype: torch.dtype) -> _CacheKey:
    shard_path = entry.shard_path.resolve()
    return (
        model_id,
        entry.tensor_name,
        str(shard_path),
        _path_mtime_ns(shard_path),
        int(entry.data_nbytes),
        str(dtype),
    )


def _fp16_packed_cache_key(model_id: str, entry: TensorCatalogEntry) -> _Fp16PackedCacheKey:
    shard_path = entry.shard_path.resolve()
    return (
        model_id,
        entry.tensor_name,
        str(shard_path),
        _path_mtime_ns(shard_path),
        int(entry.data_offset_start),
        int(entry.data_nbytes),
    )


def _fp16_packed_cache_enabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_FP16_PACKED_CACHE", "0").strip().lower() not in {"1", "true", "yes", "on"}


def _fp16_packed_cache_budget_bytes() -> int:
    explicit_mb = _env_int("PCKETLM_FP16_PACKED_CACHE_MB", -1)
    if explicit_mb >= 0:
        return max(0, explicit_mb) * 1024 * 1024
    cap_mb = max(0, _env_int("PCKETLM_FP16_PACKED_CACHE_CAP_MB", DEFAULT_FP16_PACKED_CACHE_CAP_MB))
    free_bytes = _free_memory_bytes()
    if free_bytes is None:
        return cap_mb * 1024 * 1024
    return min(int(free_bytes * 0.5), cap_mb * 1024 * 1024)


def _is_fp16_packed_always_resident(entry: TensorCatalogEntry) -> bool:
    component_group = (entry.component_group or "").lower()
    if component_group in {"attention", "router", "embeddings", "lm_head", "final_norm"}:
        return True
    tensor_name = entry.tensor_name
    return tensor_name == "model.norm.weight" or tensor_name.startswith("lm_head.") or tensor_name.startswith("model.embed_tokens.")


def _select_fp16_packed_eviction_key() -> _Fp16PackedCacheKey | None:
    for key in _fp16_packed_cache:
        entry = find_tensor_catalog_entry(key[0], key[1])
        if entry is not None and _is_fp16_packed_always_resident(entry):
            continue
        return key
    return None


def fp16_packed_cache_get_or_read(
    model_id: str,
    entry: TensorCatalogEntry,
    reader,
) -> bytearray:
    """Return raw fp16/BF16 safetensors bytes, caching them between tensor loads."""
    global _fp16_packed_cache_bytes
    if not _fp16_packed_cache_enabled():
        _fp16_packed_stats.disk_reads += 1
        return reader()

    key = _fp16_packed_cache_key(model_id, entry)
    budget_bytes = _fp16_packed_cache_budget_bytes()
    with _cache_lock:
        _fp16_packed_stats.budget_bytes = budget_bytes
        cached = _fp16_packed_cache.get(key)
        if cached is not None:
            _fp16_packed_cache.move_to_end(key)
            _fp16_packed_stats.hits += 1
            _fp16_packed_stats.resident_bytes = _fp16_packed_cache_bytes
            _fp16_packed_stats.resident_count = len(_fp16_packed_cache)
            return cached
        _fp16_packed_stats.misses += 1

    raw = reader()
    nbytes = len(raw)
    if nbytes <= 0:
        return raw

    with _cache_lock:
        _fp16_packed_stats.disk_reads += 1
        if key in _fp16_packed_cache:
            cached = _fp16_packed_cache[key]
            _fp16_packed_cache.move_to_end(key)
            _fp16_packed_stats.hits += 1
            return cached
        if not _is_fp16_packed_always_resident(entry) and nbytes > budget_bytes:
            _fp16_packed_stats.resident_bytes = _fp16_packed_cache_bytes
            _fp16_packed_stats.resident_count = len(_fp16_packed_cache)
            return raw
        while _fp16_packed_cache and _fp16_packed_cache_bytes + nbytes > budget_bytes:
            evict_key = _select_fp16_packed_eviction_key()
            if evict_key is None:
                break
            old = _fp16_packed_cache.pop(evict_key)
            _fp16_packed_cache_bytes -= len(old)
            _fp16_packed_stats.evictions += 1
        _fp16_packed_cache[key] = raw
        _fp16_packed_cache_bytes += nbytes
        _fp16_packed_stats.stores += 1
        _fp16_packed_stats.resident_bytes = _fp16_packed_cache_bytes
        _fp16_packed_stats.resident_count = len(_fp16_packed_cache)
        return raw


def clear_fp16_packed_cache() -> None:
    """Release raw fp16/BF16 packed bytes and reset packed-cache counters."""
    global _fp16_packed_cache_bytes, _fp16_packed_stats
    with _cache_lock:
        _fp16_packed_cache.clear()
        _fp16_packed_cache_bytes = 0
        _fp16_packed_stats = Fp16PackedCacheStats()


def fp16_packed_cache_stats() -> Fp16PackedCacheStats:
    """Return fp16 packed-cache counters for diagnostics."""
    with _cache_lock:
        stats = Fp16PackedCacheStats(
            hits=_fp16_packed_stats.hits,
            misses=_fp16_packed_stats.misses,
            stores=_fp16_packed_stats.stores,
            evictions=_fp16_packed_stats.evictions,
            disk_reads=_fp16_packed_stats.disk_reads,
            resident_bytes=_fp16_packed_cache_bytes,
            resident_count=len(_fp16_packed_cache),
            budget_bytes=_fp16_packed_stats.budget_bytes,
        )
        return stats


def _expert_key(entry: TensorCatalogEntry) -> tuple[int, int] | None:
    if entry.layer_index is None or entry.expert_index is None:
        return None
    return (int(entry.layer_index), int(entry.expert_index))


def _is_always_resident_entry(entry: TensorCatalogEntry) -> bool:
    if entry.expert_index is not None:
        return False
    max_always_resident_bytes = max(
        0,
        _env_int("PCKETLM_ALWAYS_RESIDENT_TENSOR_MB", DEFAULT_ALWAYS_RESIDENT_TENSOR_MB),
    ) * 1024 * 1024
    if int(entry.data_nbytes) > max_always_resident_bytes:
        return False
    component_group = (entry.component_group or "").lower()
    if component_group in {"router", "final_norm", "layer_norm", "rotary"}:
        return True
    tensor_name = entry.tensor_name
    return tensor_name == "model.norm.weight"


def _loaded_slice_from_resident(model_id: str, resident: _ResidentTensor) -> LoadedTensorSlice:
    entry = resident.entry
    tensor = resident.tensor if resident.tensor is not None else _dequantize_q4_tensor(resident.q4)
    if tensor is None:
        raise RuntimeError(f"resident tensor {entry.tensor_name} has no tensor storage")
    return LoadedTensorSlice(
        model_id=model_id,
        tensor_name=entry.tensor_name,
        shard_name=entry.shard_name,
        dtype=resident.dtype,
        shape=[int(value) for value in tensor.shape],
        tensor=tensor,
        layer_index=entry.layer_index,
        component_group=entry.component_group,
        loaded_nbytes=resident.nbytes,
        blockers=[],
        ready=True,
    )


def _loaded_slice_with_tensor(loaded: LoadedTensorSlice, tensor: torch.Tensor) -> LoadedTensorSlice:
    nbytes = tensor.element_size() * tensor.nelement()
    return LoadedTensorSlice(
        model_id=loaded.model_id,
        tensor_name=loaded.tensor_name,
        shard_name=loaded.shard_name,
        dtype=str(tensor.dtype),
        shape=[int(value) for value in tensor.shape],
        tensor=tensor,
        layer_index=loaded.layer_index,
        component_group=loaded.component_group,
        loaded_nbytes=nbytes,
        blockers=list(loaded.blockers),
        ready=loaded.ready,
        borrowed_from_live_handle=loaded.borrowed_from_live_handle,
    )


def _zero_copy_hot_tensors_enabled() -> bool:
    if os.environ.get("PCKETLM_DISABLE_ZERO_COPY_TENSORS", "0").strip().lower() in {"1", "true", "yes"}:
        return False
    return os.environ.get("PCKETLM_ENABLE_ZERO_COPY_TENSORS", "0").strip().lower() in {"1", "true", "yes"}


def _hot_tensor_for_compute(loaded: LoadedTensorSlice, dtype: torch.dtype) -> torch.Tensor:
    tensor = loaded.tensor.detach().cpu().to(dtype=dtype)
    if tensor.data_ptr() != loaded.tensor.data_ptr():
        return tensor
    if loaded.borrowed_from_live_handle and _zero_copy_hot_tensors_enabled():
        return tensor
    return tensor.clone()


def _quantize_q4_tensor(tensor: torch.Tensor) -> _QuantizedQ4Tensor:
    source = tensor.detach().cpu()
    shape = tuple(int(value) for value in source.shape)
    if source.numel() == 0:
        packed = torch.empty((0,), dtype=torch.uint8)
        scales = torch.empty((0,), dtype=torch.float16)
        return _QuantizedQ4Tensor(packed=packed, scales=scales, shape=shape, dtype=source.dtype, nbytes=0)

    rows = source.to(dtype=torch.float32).reshape(-1, shape[-1] if shape else 1)
    scales = rows.abs().amax(dim=1).clamp_min(1e-8) / 7.0
    quantized = torch.round(rows / scales[:, None]).clamp(-8, 7).to(torch.int16) + 8
    flat = quantized.reshape(-1).to(torch.uint8)
    if int(flat.numel()) % 2:
        flat = torch.cat([flat, torch.zeros((1,), dtype=torch.uint8)])
    low = flat[0::2]
    high = flat[1::2] << 4
    packed = (low | high).contiguous()
    scales = scales.to(dtype=torch.float16).contiguous()
    nbytes = int(packed.numel() * packed.element_size() + scales.numel() * scales.element_size())
    return _QuantizedQ4Tensor(packed=packed, scales=scales, shape=shape, dtype=source.dtype, nbytes=nbytes)


def _dequantize_q4_tensor(q4: _QuantizedQ4Tensor | None) -> torch.Tensor | None:
    if q4 is None:
        return None
    numel = 1
    for value in q4.shape:
        numel *= int(value)
    if numel == 0:
        return torch.empty(q4.shape, dtype=q4.dtype)
    packed = q4.packed
    unpacked = torch.empty((int(packed.numel()) * 2,), dtype=torch.int16)
    unpacked[0::2] = (packed & 0x0F).to(torch.int16)
    unpacked[1::2] = ((packed >> 4) & 0x0F).to(torch.int16)
    signed = unpacked[:numel].to(torch.float32) - 8.0
    last_dim = q4.shape[-1] if q4.shape else 1
    rows = signed.reshape(-1, last_dim)
    dequantized = rows * q4.scales.to(dtype=torch.float32)[:, None]
    return dequantized.reshape(q4.shape).to(dtype=q4.dtype)


def _tensor_for_residency_store(tensor: torch.Tensor, loaded: LoadedTensorSlice) -> torch.Tensor:
    if loaded.borrowed_from_live_handle and tensor.data_ptr() == loaded.tensor.data_ptr():
        return tensor.clone()
    return tensor


def _is_cacheable(tensor: torch.Tensor, entry: TensorCatalogEntry, policy: TensorResidencyPolicy) -> bool:
    nbytes = tensor.element_size() * tensor.nelement()
    if not policy.enabled:
        return False
    if _expert_key(entry) is not None:
        return nbytes <= policy.expert_max_resident_bytes
    if policy.min_tensor_bytes <= nbytes <= policy.all_layer_small_tensor_bytes:
        return True
    if entry.layer_index is not None and entry.layer_index >= policy.front_layer_count:
        return False
    return policy.min_tensor_bytes <= nbytes <= policy.max_tensor_bytes


def _store_resident_tensor(
    key: _CacheKey,
    entry: TensorCatalogEntry,
    tensor: torch.Tensor,
    policy: TensorResidencyPolicy,
) -> None:
    global _resident_bytes, _resident_expert_bytes
    expert_key = _expert_key(entry)
    q4: _QuantizedQ4Tensor | None = None
    resident_tensor: torch.Tensor | None = tensor
    if expert_key is not None and policy.expert_q4_residency:
        q4 = _quantize_q4_tensor(tensor)
        resident_tensor = None
        nbytes = q4.nbytes
    else:
        nbytes = tensor.element_size() * tensor.nelement()
    if expert_key is None:
        if nbytes > policy.max_resident_bytes:
            _stats.skips += 1
            return
    else:
        if nbytes > policy.expert_max_resident_bytes:
            _stats.skips += 1
            return

    with _cache_lock:
        if key in _resident_tensors:
            old = _resident_tensors.pop(key)
            _resident_bytes -= old.nbytes
            if old.expert_key is not None:
                _resident_expert_bytes -= old.nbytes

        if expert_key is None:
            while _resident_tensors and _non_expert_resident_bytes() + nbytes > policy.max_resident_bytes:
                evict_key = _select_non_expert_eviction_key(policy)
                if evict_key is None:
                    break
                old = _resident_tensors.pop(evict_key)
                _resident_bytes -= old.nbytes
                _stats.evictions += 1

        if expert_key is not None:
            while _resident_tensors and _resident_expert_bytes + nbytes > policy.expert_max_resident_bytes:
                evict_key = _select_expert_eviction_key(allow_current_step=False)
                if evict_key is None:
                    evict_key = _select_expert_eviction_key(allow_current_step=True)
                if evict_key is None:
                    break
                old = _resident_tensors.pop(evict_key)
                _resident_bytes -= old.nbytes
                _resident_expert_bytes -= old.nbytes
                _stats.evictions += 1
                _stats.expert_evictions += 1

        _resident_tensors[key] = _ResidentTensor(
            tensor=resident_tensor,
            entry=entry,
            nbytes=nbytes,
            dtype=str(tensor.dtype),
            loaded_step=_residency_step,
            expert_key=expert_key,
            q4=q4,
        )
        _resident_bytes += nbytes
        if expert_key is not None:
            _resident_expert_bytes += nbytes
            _enforce_expert_layer_cap(policy, expert_key[0])
        _stats.stores += 1
        _stats.resident_bytes = _resident_bytes
        _stats.resident_count = len(_resident_tensors)
        _stats.expert_resident_bytes = _resident_expert_bytes
        _stats.expert_resident_count = sum(1 for resident in _resident_tensors.values() if resident.expert_key is not None)


def _select_eviction_key(policy: TensorResidencyPolicy) -> _CacheKey | None:
    """Choose an eviction victim, preserving recently loaded tensors when possible."""
    if not _resident_tensors:
        return None
    expert_key = _select_expert_eviction_key()
    if expert_key is not None:
        return expert_key
    sticky_floor = _residency_step - int(policy.sticky_residency_steps)
    for key, resident in _resident_tensors.items():
        if _is_always_resident_entry(resident.entry):
            continue
        if resident.loaded_step < sticky_floor:
            return key
    for key, resident in _resident_tensors.items():
        if not _is_always_resident_entry(resident.entry):
            return key
    return None


def _non_expert_resident_bytes() -> int:
    return sum(resident.nbytes for resident in _resident_tensors.values() if resident.expert_key is None)


def _select_non_expert_eviction_key(policy: TensorResidencyPolicy) -> _CacheKey | None:
    sticky_floor = _residency_step - int(policy.sticky_residency_steps)
    for key, resident in _resident_tensors.items():
        if resident.expert_key is not None or _is_always_resident_entry(resident.entry):
            continue
        if resident.loaded_step < sticky_floor:
            return key
    for key, resident in _resident_tensors.items():
        if resident.expert_key is None and not _is_always_resident_entry(resident.entry):
            return key
    return None


def _select_expert_eviction_key(*, allow_current_step: bool = False) -> _CacheKey | None:
    candidates: list[tuple[float, int, _CacheKey]] = []
    for key, resident in _resident_tensors.items():
        if resident.expert_key is None:
            continue
        if not allow_current_step and resident.expert_key in _current_step_experts:
            continue
        activation_score = _expert_activation_scores.get(resident.expert_key, 0.0)
        candidates.append((activation_score, resident.loaded_step, key))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _select_expert_eviction_key_for_layer(layer_index: int) -> _CacheKey | None:
    candidates: list[tuple[float, int, _CacheKey]] = []
    for key, resident in _resident_tensors.items():
        if resident.expert_key is None or resident.expert_key[0] != layer_index:
            continue
        if resident.expert_key in _current_step_experts:
            continue
        activation_score = _expert_activation_scores.get(resident.expert_key, 0.0)
        candidates.append((activation_score, resident.loaded_step, key))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _enforce_expert_layer_cap(policy: TensorResidencyPolicy, layer_index: int) -> None:
    global _resident_bytes, _resident_expert_bytes
    if policy.max_resident_experts_per_layer <= 0:
        return

    def resident_expert_count_for_layer() -> int:
        return len(
            {
                resident.expert_key[1]
                for resident in _resident_tensors.values()
                if resident.expert_key is not None and resident.expert_key[0] == layer_index
            }
        )

    while resident_expert_count_for_layer() > policy.max_resident_experts_per_layer:
        evict_key = _select_expert_eviction_key_for_layer(layer_index)
        if evict_key is None:
            candidates: list[tuple[float, int, _CacheKey]] = []
            for key, resident in _resident_tensors.items():
                if resident.expert_key is None or resident.expert_key[0] != layer_index:
                    continue
                candidates.append((
                    _expert_activation_scores.get(resident.expert_key, 0.0),
                    resident.loaded_step,
                    key,
                ))
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]))
                evict_key = candidates[0][2]
        if evict_key is None:
            break
        old = _resident_tensors.pop(evict_key)
        _resident_bytes -= old.nbytes
        _resident_expert_bytes -= old.nbytes
        _stats.evictions += 1
        _stats.expert_evictions += 1


def load_resident_tensor(
    model_id: str,
    tensor_name: str,
    *,
    dtype: torch.dtype = torch.float32,
    policy: TensorResidencyPolicy | None = None,
) -> LoadedTensorSlice:
    """Load a tensor and keep its converted CPU form resident when it fits policy."""
    effective_policy = TensorResidencyPolicy.from_environment(model_id) if policy is None else policy
    entry = _find_tensor_entry(model_id, tensor_name)
    if entry is None:
        loaded = load_tensor_by_name(model_id, tensor_name)
        if loaded.tensor is None:
            return loaded
        return _loaded_slice_with_tensor(loaded, loaded.tensor.to(dtype=dtype))

    key = _cache_key(model_id, entry, dtype)
    if effective_policy.enabled:
        with _cache_lock:
            resident = _resident_tensors.get(key)
            if resident is not None:
                _resident_tensors.move_to_end(key)
                _stats.hits += 1
                if resident.expert_key is not None:
                    _stats.expert_hits += 1
                _stats.resident_bytes = _resident_bytes
                _stats.resident_count = len(_resident_tensors)
                return _loaded_slice_from_resident(model_id, resident)

    _stats.misses += 1
    if _expert_key(entry) is not None:
        _stats.expert_misses += 1
    loaded = load_tensor_by_name(model_id, tensor_name)
    if not loaded.ready or loaded.tensor is None:
        return loaded

    converted = _hot_tensor_for_compute(loaded, dtype)
    converted_slice = _loaded_slice_with_tensor(loaded, converted)
    if _is_cacheable(converted, entry, effective_policy):
        _store_resident_tensor(key, entry, _tensor_for_residency_store(converted, loaded), effective_policy)
    else:
        _stats.skips += 1
        _stats.resident_bytes = _resident_bytes
        _stats.resident_count = len(_resident_tensors)
    return converted_slice


def load_resident_tensors(
    model_id: str,
    tensor_names: list[str],
    *,
    dtype: torch.dtype = torch.float32,
    policy: TensorResidencyPolicy | None = None,
) -> dict[str, LoadedTensorSlice]:
    """Load several tensors through residency, batching misses by shard."""
    effective_policy = TensorResidencyPolicy.from_environment(model_id) if policy is None else policy
    results: dict[str, LoadedTensorSlice] = {}
    entries_by_name: dict[str, TensorCatalogEntry] = {}
    missing_names: list[str] = []

    for tensor_name in dict.fromkeys(tensor_names):
        entry = _find_tensor_entry(model_id, tensor_name)
        if entry is None:
            missing_names.append(tensor_name)
            continue

        entries_by_name[tensor_name] = entry
        key = _cache_key(model_id, entry, dtype)
        if effective_policy.enabled:
            with _cache_lock:
                resident = _resident_tensors.get(key)
                if resident is not None:
                    _resident_tensors.move_to_end(key)
                    _stats.hits += 1
                    if resident.expert_key is not None:
                        _stats.expert_hits += 1
                    _stats.resident_bytes = _resident_bytes
                    _stats.resident_count = len(_resident_tensors)
                    results[tensor_name] = _loaded_slice_from_resident(model_id, resident)
                    continue
        missing_names.append(tensor_name)

    if missing_names:
        _stats.misses += len(missing_names)
        loaded_by_name = load_tensors_by_name(model_id, missing_names)
        for tensor_name in missing_names:
            loaded = loaded_by_name[tensor_name]
            entry = entries_by_name.get(tensor_name)
            if entry is None or not loaded.ready or loaded.tensor is None:
                results[tensor_name] = loaded
                continue
            if _expert_key(entry) is not None:
                _stats.expert_misses += 1

            converted = _hot_tensor_for_compute(loaded, dtype)
            converted_slice = _loaded_slice_with_tensor(loaded, converted)
            if _is_cacheable(converted, entry, effective_policy):
                _store_resident_tensor(
                    _cache_key(model_id, entry, dtype),
                    entry,
                    _tensor_for_residency_store(converted, loaded),
                    effective_policy,
                )
            else:
                _stats.skips += 1
                _stats.resident_bytes = _resident_bytes
                _stats.resident_count = len(_resident_tensors)
            results[tensor_name] = converted_slice

    return {tensor_name: results[tensor_name] for tensor_name in tensor_names}


def clear_tensor_residency_cache() -> None:
    """Release resident tensors and reset cache counters."""
    global _memory_snapshot_cache, _resident_bytes, _resident_expert_bytes, _residency_step, _stats
    with _cache_lock:
        _resident_tensors.clear()
        _resident_bytes = 0
        _resident_expert_bytes = 0
        _residency_step = 0
        _expert_activation_counts.clear()
        _expert_activation_scores.clear()
        _current_step_experts.clear()
        _stats = TensorResidencyStats()
        _memory_snapshot_cache = (0.0, None)
    clear_fp16_packed_cache()
    clear_tensor_handle_cache()


def tensor_residency_stats() -> TensorResidencyStats:
    """Return a snapshot of residency counters."""
    with _cache_lock:
        return TensorResidencyStats(
            hits=_stats.hits,
            misses=_stats.misses,
            stores=_stats.stores,
            evictions=_stats.evictions,
            skips=_stats.skips,
            resident_bytes=_resident_bytes,
            resident_count=len(_resident_tensors),
            expert_hits=_stats.expert_hits,
            expert_misses=_stats.expert_misses,
            expert_evictions=_stats.expert_evictions,
            expert_resident_bytes=_resident_expert_bytes,
            expert_resident_count=sum(1 for resident in _resident_tensors.values() if resident.expert_key is not None),
        )


def tensor_residency_policy_snapshot() -> dict:
    """Return the current effective residency policy for status surfaces."""
    return TensorResidencyPolicy.from_environment().to_dict()
