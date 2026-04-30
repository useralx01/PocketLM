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

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "skips": self.skips,
            "resident_bytes": self.resident_bytes,
            "resident_count": self.resident_count,
        }


@dataclass(slots=True)
class _ResidentTensor:
    tensor: torch.Tensor
    entry: TensorCatalogEntry
    nbytes: int
    dtype: str
    loaded_step: int


_CacheKey = tuple[str, str, str, int, int, str]

_cache_lock = threading.RLock()
_resident_tensors: OrderedDict[_CacheKey, _ResidentTensor] = OrderedDict()
_stats = TensorResidencyStats()
_resident_bytes = 0
_residency_step = 0


def advance_tensor_residency_step(steps: int = 1) -> int:
    """Advance the logical decode step used by sticky residency."""
    global _residency_step
    with _cache_lock:
        _residency_step += max(1, int(steps))
        return _residency_step


def current_tensor_residency_step() -> int:
    """Return the current logical decode step used by sticky residency."""
    with _cache_lock:
        return _residency_step


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


def _loaded_slice_from_resident(model_id: str, resident: _ResidentTensor) -> LoadedTensorSlice:
    entry = resident.entry
    return LoadedTensorSlice(
        model_id=model_id,
        tensor_name=entry.tensor_name,
        shard_name=entry.shard_name,
        dtype=resident.dtype,
        shape=[int(value) for value in resident.tensor.shape],
        tensor=resident.tensor,
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
    )


def _is_cacheable(tensor: torch.Tensor, entry: TensorCatalogEntry, policy: TensorResidencyPolicy) -> bool:
    nbytes = tensor.element_size() * tensor.nelement()
    if not policy.enabled:
        return False
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
    global _resident_bytes
    nbytes = tensor.element_size() * tensor.nelement()
    if nbytes > policy.max_resident_bytes:
        _stats.skips += 1
        return

    with _cache_lock:
        if key in _resident_tensors:
            old = _resident_tensors.pop(key)
            _resident_bytes -= old.nbytes

        while _resident_tensors and _resident_bytes + nbytes > policy.max_resident_bytes:
            evict_key = _select_eviction_key(policy)
            if evict_key is None:
                break
            old = _resident_tensors.pop(evict_key)
            _resident_bytes -= old.nbytes
            _stats.evictions += 1

        _resident_tensors[key] = _ResidentTensor(
            tensor=tensor,
            entry=entry,
            nbytes=nbytes,
            dtype=str(tensor.dtype),
            loaded_step=_residency_step,
        )
        _resident_bytes += nbytes
        _stats.stores += 1
        _stats.resident_bytes = _resident_bytes
        _stats.resident_count = len(_resident_tensors)


def _select_eviction_key(policy: TensorResidencyPolicy) -> _CacheKey | None:
    """Choose an eviction victim, preserving recently loaded tensors when possible."""
    if not _resident_tensors:
        return None
    sticky_floor = _residency_step - int(policy.sticky_residency_steps)
    for key, resident in _resident_tensors.items():
        if resident.loaded_step < sticky_floor:
            return key
    return next(iter(_resident_tensors))


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
                _stats.resident_bytes = _resident_bytes
                _stats.resident_count = len(_resident_tensors)
                return _loaded_slice_from_resident(model_id, resident)

    _stats.misses += 1
    loaded = load_tensor_by_name(model_id, tensor_name)
    if not loaded.ready or loaded.tensor is None:
        return loaded

    converted = loaded.tensor.detach().cpu().to(dtype=dtype)
    if converted.data_ptr() == loaded.tensor.data_ptr():
        converted = converted.clone()
    converted_slice = _loaded_slice_with_tensor(loaded, converted)
    if _is_cacheable(converted, entry, effective_policy):
        _store_resident_tensor(key, entry, converted, effective_policy)
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

            converted = loaded.tensor.detach().cpu().to(dtype=dtype)
            if converted.data_ptr() == loaded.tensor.data_ptr():
                converted = converted.clone()
            converted_slice = _loaded_slice_with_tensor(loaded, converted)
            if _is_cacheable(converted, entry, effective_policy):
                _store_resident_tensor(_cache_key(model_id, entry, dtype), entry, converted, effective_policy)
            else:
                _stats.skips += 1
                _stats.resident_bytes = _resident_bytes
                _stats.resident_count = len(_resident_tensors)
            results[tensor_name] = converted_slice

    return {tensor_name: results[tensor_name] for tensor_name in tensor_names}


def clear_tensor_residency_cache() -> None:
    """Release resident tensors and reset cache counters."""
    global _memory_snapshot_cache, _resident_bytes, _residency_step, _stats
    with _cache_lock:
        _resident_tensors.clear()
        _resident_bytes = 0
        _residency_step = 0
        _stats = TensorResidencyStats()
        _memory_snapshot_cache = (0.0, None)
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
        )


def tensor_residency_policy_snapshot() -> dict:
    """Return the current effective residency policy for status surfaces."""
    return TensorResidencyPolicy.from_environment().to_dict()
