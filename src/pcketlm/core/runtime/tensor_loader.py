"""On-demand tensor loading from the persisted tensor catalog and execution plan."""

from __future__ import annotations

import os
import struct
import threading
import json
import math
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

from pcketlm.core.runtime.tensor_catalog import (
    TensorCatalogEntry,
    find_tensor_catalog_entry,
    load_tensor_catalog,
    load_tensor_entry_index,
)
from pcketlm.core.runtime.tensor_execution_plan import TensorExecutionUnit, load_tensor_execution_plan
from pcketlm.core.storage.paths import artifacts_root

_HANDLE_CACHE_MAX_ENTRIES = 16
_handle_cache: OrderedDict[tuple[str, int], object] = OrderedDict()
_scoped_handles = threading.local()
_load_stats_lock = threading.RLock()


@dataclass(slots=True)
class TensorLoadStats:
    """Cumulative tensor-load counters for runtime diagnostics."""

    single_load_calls: int = 0
    batch_load_calls: int = 0
    tensors_requested: int = 0
    tensors_loaded: int = 0
    loaded_nbytes: int = 0
    shard_opens: int = 0
    artifact_pack_opens: int = 0
    artifact_tensor_hits: int = 0
    scoped_handle_reuses: int = 0
    persistent_handle_reuses: int = 0
    live_handle_tensor_hits: int = 0
    q4_loads: int = 0
    q4_loaded_nbytes: int = 0
    native_fp16_loads: int = 0
    native_fp16_loaded_nbytes: int = 0

    def to_dict(self) -> dict:
        return {
            "single_load_calls": self.single_load_calls,
            "batch_load_calls": self.batch_load_calls,
            "tensors_requested": self.tensors_requested,
            "tensors_loaded": self.tensors_loaded,
            "loaded_nbytes": self.loaded_nbytes,
            "loaded_mb": round(self.loaded_nbytes / (1024**2), 2),
            "shard_opens": self.shard_opens,
            "artifact_pack_opens": self.artifact_pack_opens,
            "artifact_tensor_hits": self.artifact_tensor_hits,
            "scoped_handle_reuses": self.scoped_handle_reuses,
            "persistent_handle_reuses": self.persistent_handle_reuses,
            "live_handle_tensor_hits": self.live_handle_tensor_hits,
            "q4_loads": self.q4_loads,
            "q4_loaded_nbytes": self.q4_loaded_nbytes,
            "q4_loaded_mb": round(self.q4_loaded_nbytes / (1024**2), 2),
            "q4_loaded": self.q4_loads > 0,
            "native_fp16_loads": self.native_fp16_loads,
            "native_fp16_loaded_nbytes": self.native_fp16_loaded_nbytes,
            "native_fp16_loaded_mb": round(self.native_fp16_loaded_nbytes / (1024**2), 2),
            "native_fp16_loaded": self.native_fp16_loads > 0,
        }


_load_stats = TensorLoadStats()


def _update_load_stats(**updates: int) -> None:
    with _load_stats_lock:
        for key, value in updates.items():
            setattr(_load_stats, key, getattr(_load_stats, key) + int(value))


def tensor_load_stats_snapshot() -> TensorLoadStats:
    """Return a cumulative tensor-load stats snapshot."""
    with _load_stats_lock:
        return TensorLoadStats(
            single_load_calls=_load_stats.single_load_calls,
            batch_load_calls=_load_stats.batch_load_calls,
            tensors_requested=_load_stats.tensors_requested,
            tensors_loaded=_load_stats.tensors_loaded,
            loaded_nbytes=_load_stats.loaded_nbytes,
            shard_opens=_load_stats.shard_opens,
            artifact_pack_opens=_load_stats.artifact_pack_opens,
            artifact_tensor_hits=_load_stats.artifact_tensor_hits,
            scoped_handle_reuses=_load_stats.scoped_handle_reuses,
            persistent_handle_reuses=_load_stats.persistent_handle_reuses,
            live_handle_tensor_hits=_load_stats.live_handle_tensor_hits,
            q4_loads=_load_stats.q4_loads,
            q4_loaded_nbytes=_load_stats.q4_loaded_nbytes,
            native_fp16_loads=_load_stats.native_fp16_loads,
            native_fp16_loaded_nbytes=_load_stats.native_fp16_loaded_nbytes,
        )


def reset_tensor_load_stats() -> None:
    """Reset cumulative tensor-load stats."""
    global _load_stats
    with _load_stats_lock:
        _load_stats = TensorLoadStats()
    clear_runtime_pack_cache()


@dataclass(slots=True)
class LoadedTensorSlice:
    """One real tensor loaded on demand from the original shard."""

    model_id: str
    tensor_name: str
    shard_name: str
    dtype: str
    shape: list[int]
    tensor: torch.Tensor | None = None
    layer_index: int | None = None
    component_group: str = "other"
    loaded_nbytes: int = 0
    blockers: list[str] = field(default_factory=list)
    ready: bool = False
    borrowed_from_live_handle: bool = False
    q4_loaded: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "tensor_name": self.tensor_name,
            "shard_name": self.shard_name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layer_index": self.layer_index,
            "component_group": self.component_group,
            "loaded_nbytes": self.loaded_nbytes,
            "blockers": list(self.blockers),
            "ready": self.ready,
            "borrowed_from_live_handle": self.borrowed_from_live_handle,
            "q4_loaded": self.q4_loaded,
        }


@dataclass(slots=True)
class LoadedExecutionUnit:
    """One grouped execution unit loaded into real tensors."""

    model_id: str
    unit_id: str
    label: str
    phase: str
    tensor_count: int
    total_nbytes: int
    tensors: list[LoadedTensorSlice] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "unit_id": self.unit_id,
            "label": self.label,
            "phase": self.phase,
            "tensor_count": self.tensor_count,
            "total_nbytes": self.total_nbytes,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class TensorVerificationResult:
    """Verification result for one loaded tensor against the tensor catalog."""

    model_id: str
    tensor_name: str
    expected_dtype: str
    actual_dtype: str
    expected_shape: list[int]
    actual_shape: list[int]
    expected_nbytes: int
    actual_nbytes: int
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "tensor_name": self.tensor_name,
            "expected_dtype": self.expected_dtype,
            "actual_dtype": self.actual_dtype,
            "expected_shape": list(self.expected_shape),
            "actual_shape": list(self.actual_shape),
            "expected_nbytes": self.expected_nbytes,
            "actual_nbytes": self.actual_nbytes,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class ExecutionUnitVerificationResult:
    """Verification result for one loaded execution unit."""

    model_id: str
    unit_id: str
    tensor_count: int
    total_nbytes: int
    tensor_results: list[TensorVerificationResult] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "unit_id": self.unit_id,
            "tensor_count": self.tensor_count,
            "total_nbytes": self.total_nbytes,
            "tensor_results": [result.to_dict() for result in self.tensor_results],
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


def _find_tensor_entry(model_id: str, tensor_name: str) -> TensorCatalogEntry | None:
    return find_tensor_catalog_entry(model_id, tensor_name)


def _runtime_pack_enabled() -> bool:
    return os.environ.get("PCKETLM_RUNTIME_PACK", "1").strip().lower() in {"1", "true", "yes"}


@lru_cache(maxsize=16)
def _runtime_pack_info(model_id: str) -> tuple[Path, frozenset[str], dict] | None:
    if not _runtime_pack_enabled():
        return None
    from pcketlm.core.optimize.artifact_manifest import select_runtime_artifact_manifest

    free_memory_bytes = None
    try:
        from pcketlm.core.runtime.load_attempt import _memory_snapshot

        free_memory_bytes = int(_memory_snapshot().free_bytes)
    except Exception:
        free_memory_bytes = None

    runtime_preset = os.environ.get("PCKETLM_TENSOR_CACHE_PRESET", "standard")
    manifest = select_runtime_artifact_manifest(
        model_id,
        free_memory_bytes=free_memory_bytes,
        runtime_preset=runtime_preset,
    )
    if manifest is None or not manifest.ready or manifest.tensor_pack_path is None:
        return None
    if not manifest.tensor_pack_path.exists() or not manifest.packed_tensor_names:
        return None
    selection = {
        "artifact_id": manifest.artifact_id,
        "strategy": manifest.strategy,
        "profile_id": manifest.profile_id,
        "packed_tensor_count": manifest.packed_tensor_count,
        "packed_nbytes": manifest.packed_nbytes,
        "packed_mb": round(manifest.packed_nbytes / (1024**2), 2),
        "free_memory_gb": None if free_memory_bytes is None else round(free_memory_bytes / (1024**3), 2),
        "runtime_preset": runtime_preset,
    }
    return manifest.tensor_pack_path, frozenset(manifest.packed_tensor_names), selection


def clear_runtime_pack_cache() -> None:
    """Clear cached runtime-pack metadata."""
    _runtime_pack_info.cache_clear()


def _artifact_pack_path_for_tensor(model_id: str, tensor_name: str) -> Path | None:
    info = _runtime_pack_info(model_id)
    if info is None:
        return None
    pack_path, tensor_names, _selection = info
    return pack_path if tensor_name in tensor_names else None


def runtime_pack_selection_snapshot(model_id: str) -> dict:
    """Return the currently selected runtime pack for diagnostics."""
    info = _runtime_pack_info(model_id)
    if info is None:
        return {
            "enabled": _runtime_pack_enabled(),
            "selected": False,
            "summary": "No ready runtime pack is selected.",
        }
    pack_path, _tensor_names, selection = info
    return {
        "enabled": _runtime_pack_enabled(),
        "selected": True,
        "pack_path": str(pack_path),
        **selection,
        "summary": (
            f"Using runtime pack {selection['artifact_id']} with "
            f"{selection['packed_tensor_count']} tensors "
            f"({selection['packed_mb']} MB)."
        ),
    }


def _normalize_catalog_dtype(dtype: str) -> str:
    mapping = {
        "BF16": "torch.bfloat16",
        "F16": "torch.float16",
        "F32": "torch.float32",
        "I64": "torch.int64",
        "I32": "torch.int32",
    }
    return mapping.get(dtype, dtype)


def _loaded_slice_from_entry(
    model_id: str,
    entry: TensorCatalogEntry,
    tensor: torch.Tensor | None = None,
    blockers: list[str] | None = None,
    ready: bool = True,
    borrowed_from_live_handle: bool = False,
    q4_loaded: bool = False,
) -> LoadedTensorSlice:
    actual_blockers = [] if blockers is None else list(blockers)
    actual_shape = list(entry.shape) if tensor is None else [int(value) for value in tensor.shape]
    actual_dtype = entry.dtype if tensor is None else str(tensor.dtype)
    loaded_nbytes = 0 if tensor is None else tensor.element_size() * tensor.nelement()
    if tensor is not None and list(tensor.shape) != list(entry.shape):
        actual_blockers.append(
            f"Loaded tensor shape {list(tensor.shape)} does not match catalog shape {list(entry.shape)} for {entry.tensor_name}."
        )
    return LoadedTensorSlice(
        model_id=model_id,
        tensor_name=entry.tensor_name,
        shard_name=entry.shard_name,
        dtype=actual_dtype,
        shape=actual_shape,
        tensor=tensor,
        layer_index=entry.layer_index,
        component_group=entry.component_group,
        loaded_nbytes=loaded_nbytes,
        blockers=actual_blockers,
        ready=ready and not actual_blockers and tensor is not None,
        borrowed_from_live_handle=borrowed_from_live_handle,
        q4_loaded=q4_loaded,
    )


def _q4_artifact_root(model_id: str) -> Path:
    return artifacts_root(model_id) / "q4"


def _q4_manifest_path(model_id: str) -> Path:
    return _q4_artifact_root(model_id) / "q4_manifest.json"


@lru_cache(maxsize=16)
def _q4_manifest(model_id: str, mtime_ns: int) -> dict:
    del mtime_ns
    path = _q4_manifest_path(model_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_q4_manifest(model_id: str) -> dict:
    path = _q4_manifest_path(model_id)
    return _q4_manifest(model_id, path.stat().st_mtime_ns if path.exists() else 0)


def _q4_source_enabled(model_id: str) -> bool:
    source = os.environ.get("PCKETLM_TENSOR_SOURCE", "auto").strip().lower()
    if source in {"fp16", "bf16", "original", "safetensors"}:
        return False
    manifest = _load_q4_manifest(model_id)
    ready = manifest.get("format") == "pcketlm-q4" and bool(manifest.get("tensors"))
    if source == "q4":
        return ready
    return source == "auto" and ready


def q4_source_status(model_id: str) -> dict:
    """Return Q4 artifact readiness for diagnostics and tests."""
    manifest_path = _q4_manifest_path(model_id)
    manifest = _load_q4_manifest(model_id)
    ready = manifest.get("format") == "pcketlm-q4" and bool(manifest.get("tensors"))
    return {
        "enabled": _q4_source_enabled(model_id),
        "ready": ready,
        "manifest_path": str(manifest_path),
        "tensor_count": len(manifest.get("tensors") or {}),
        "total_original_bytes": int(manifest.get("total_original_bytes") or 0),
        "total_q4_bytes": int(manifest.get("total_q4_bytes") or 0),
        "compression_ratio": float(manifest.get("compression_ratio") or 0.0),
    }


def _torch_dtype_from_catalog(dtype: str) -> torch.dtype:
    normalized = _normalize_catalog_dtype(dtype)
    if normalized == "torch.bfloat16":
        return torch.bfloat16
    if normalized == "torch.float16":
        return torch.float16
    if normalized == "torch.float32":
        return torch.float32
    if normalized == "torch.int64":
        return torch.int64
    if normalized == "torch.int32":
        return torch.int32
    return torch.float16


def _native_fp16_load_enabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_FP16_LOAD", "0").strip().lower() not in {"1", "true", "yes", "on"}


@lru_cache(maxsize=64)
def _safetensors_data_base_offset(path: str, mtime_ns: int) -> int:
    del mtime_ns
    with Path(path).open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
    return 8 + int(header_length)


def _load_native_fp16_tensor(model_id: str, entry: TensorCatalogEntry) -> LoadedTensorSlice | None:
    """Load a catalog tensor by raw byte-copying into a contiguous torch buffer."""
    if not _native_fp16_load_enabled():
        return None
    if entry.dtype not in {"BF16", "F16", "F32", "I64", "I32"}:
        return None
    try:
        from pcketlm.native import native_copy_tensor_bytes, native_read_bytes, native_read_tensor_bytes

        tensor = torch.empty(tuple(int(value) for value in entry.shape), dtype=_torch_dtype_from_catalog(entry.dtype))
        absolute_offset = _safetensors_data_base_offset(
            str(entry.shard_path.resolve()),
            _path_mtime_ns(entry.shard_path),
        ) + int(entry.data_offset_start)
        if os.environ.get("PCKETLM_DISABLE_FP16_PACKED_CACHE", "0").strip().lower() in {"1", "true", "yes", "on"}:
            native_read_tensor_bytes(entry.shard_path, absolute_offset, int(entry.data_nbytes), tensor)
        else:
            from pcketlm.core.runtime.tensor_residency import fp16_packed_cache_get_or_read

            raw = fp16_packed_cache_get_or_read(
                model_id,
                entry,
                lambda: native_read_bytes(entry.shard_path, absolute_offset, int(entry.data_nbytes)),
            )
            native_copy_tensor_bytes(raw, tensor)
        _update_load_stats(native_fp16_loads=1, native_fp16_loaded_nbytes=int(entry.data_nbytes))
        return _loaded_slice_from_entry(model_id, entry, tensor, borrowed_from_live_handle=False)
    except Exception:
        return None


def _unpack_int4(packed: torch.Tensor, value_count: int) -> torch.Tensor:
    bytes_flat = packed.detach().to(torch.uint8).flatten()
    low = torch.bitwise_and(bytes_flat, 0x0F)
    high = torch.bitwise_and(torch.bitwise_right_shift(bytes_flat, 4), 0x0F)
    unsigned = torch.empty((bytes_flat.numel() * 2,), dtype=torch.int16)
    unsigned[0::2] = low.to(torch.int16)
    unsigned[1::2] = high.to(torch.int16)
    signed = torch.where(unsigned >= 8, unsigned - 16, unsigned)
    return signed[:value_count].to(torch.int8).contiguous()


def _python_dequantize_q4_tensor(packed: torch.Tensor, scales: torch.Tensor, shape: list[int]) -> torch.Tensor:
    value_count = int(math.prod(shape)) if shape else 1
    quantized = _unpack_int4(packed, value_count).to(torch.float32)
    if not shape:
        matrix = quantized.reshape(1, 1)
    elif len(shape) == 1:
        matrix = quantized.reshape(shape[0], 1)
    else:
        matrix = quantized.reshape(shape[0], -1)
    restored = matrix * scales.detach().cpu().float().reshape(-1, 1)
    return restored.reshape(shape).to(dtype=torch.float16).contiguous()


def _dequantize_q4_tensor(packed: torch.Tensor, scales: torch.Tensor, shape: list[int], dtype: str) -> torch.Tensor:
    del dtype
    if not os.environ.get("PCKETLM_DISABLE_NATIVE_Q4"):
        try:
            from pcketlm.native import q4_dequant_to_fp16

            num_channels, channel_size = _q4_dequant_shape_params(shape)
            return q4_dequant_to_fp16(packed, scales, num_channels, channel_size).reshape(shape).contiguous()
        except Exception:
            pass
    return _python_dequantize_q4_tensor(packed, scales, shape)


def _q4_dequant_shape_params(shape: list[int]) -> tuple[int, int]:
    value_count = int(math.prod(shape)) if shape else 1
    if not shape:
        return 1, 1
    if len(shape) == 1:
        return int(shape[0]), 1
    num_channels = int(shape[0])
    return num_channels, max(1, value_count // num_channels)


def _dequantize_q4_tensor_batch(
    payloads: list[tuple[TensorCatalogEntry, torch.Tensor, torch.Tensor, list[int]]],
) -> dict[str, torch.Tensor] | None:
    if not payloads or os.environ.get("PCKETLM_DISABLE_NATIVE_Q4"):
        return None
    try:
        from pcketlm.native import q4_dequant_many_to_fp16

        native_items = []
        shapes: list[list[int]] = []
        for _entry, packed, scales, shape in payloads:
            num_channels, channel_size = _q4_dequant_shape_params(shape)
            native_items.append((packed, scales, num_channels, channel_size))
            shapes.append(shape)
        outputs = q4_dequant_many_to_fp16(native_items)
        return {
            entry.tensor_name: output.reshape(shape).contiguous()
            for (entry, _packed, _scales, _shape), output, shape in zip(payloads, outputs, shapes)
        }
    except Exception:
        return None


def _q4_entry(model_id: str, tensor_name: str) -> dict | None:
    manifest = _load_q4_manifest(model_id)
    tensors = manifest.get("tensors") or {}
    payload = tensors.get(tensor_name)
    return payload if isinstance(payload, dict) else None


def _load_q4_tensor_from_source(
    model_id: str,
    entry: TensorCatalogEntry,
    load_packed: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    q4_path: Path | None = None,
    scale_path: Path | None = None,
) -> LoadedTensorSlice:
    payload = _q4_entry(model_id, entry.tensor_name)
    if payload is None:
        return _loaded_slice_from_entry(
            model_id,
            entry,
            blockers=[f"Q4 artifact does not contain tensor {entry.tensor_name}."],
            ready=False,
        )
    if q4_path is not None and scale_path is not None:
        from pcketlm.core.runtime.tensor_residency import q4_packed_cache_get_or_load

        packed, scales = q4_packed_cache_get_or_load(
            model_id,
            entry,
            q4_path,
            scale_path,
            load_packed,
        )
    else:
        packed, scales = load_packed()
    tensor = _dequantize_q4_tensor(packed, scales, [int(value) for value in payload.get("shape", entry.shape)], entry.dtype)
    _update_load_stats(q4_loads=1, q4_loaded_nbytes=int(packed.nelement() * packed.element_size()))
    return _loaded_slice_from_entry(model_id, entry, tensor, q4_loaded=True)


def _load_q4_tensor(model_id: str, entry: TensorCatalogEntry) -> LoadedTensorSlice | None:
    if not _q4_source_enabled(model_id):
        return None
    payload = _q4_entry(model_id, entry.tensor_name)
    if payload is None:
        return None
    root = _q4_artifact_root(model_id)
    q4_path = root / str(payload.get("q4_shard"))
    scale_path = root / str(payload.get("scale_shard"))
    try:
        with ExitStack() as stack:
            handles: dict[str, object] = {}

            def load_packed() -> tuple[torch.Tensor, torch.Tensor]:
                if "q4" not in handles:
                    handles["q4"] = stack.enter_context(safe_open(q4_path, framework="pt", device="cpu"))
                    handles["scale"] = stack.enter_context(safe_open(scale_path, framework="pt", device="cpu"))
                    _update_load_stats(shard_opens=2)
                return (
                    handles["q4"].get_tensor(entry.tensor_name),
                    handles["scale"].get_tensor(entry.tensor_name),
                )

            return _load_q4_tensor_from_source(model_id, entry, load_packed, q4_path, scale_path)
    except Exception as exc:  # pragma: no cover - defensive Q4 IO path
        return _loaded_slice_from_entry(
            model_id,
            entry,
            blockers=[f"Failed to load Q4 tensor {entry.tensor_name}: {exc}"],
            ready=False,
        )


def _handle_cache_enabled() -> bool:
    if os.environ.get("PCKETLM_DISABLE_PERSISTENT_HANDLES", "0").strip().lower() in {"1", "true", "yes"}:
        return False
    return os.environ.get("PCKETLM_SAFETENSOR_HANDLE_CACHE", "0").strip().lower() in {"1", "true", "yes"}


def _packed_layer_reads_enabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_PACKED_LAYER_READS", "0").strip().lower() not in {"1", "true", "yes"}


def _path_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _close_handle(handle: object) -> None:
    exit_method = getattr(handle, "__exit__", None)
    if callable(exit_method):
        exit_method(None, None, None)


def clear_tensor_handle_cache() -> None:
    """Close cached safetensors shard handles."""
    while _handle_cache:
        _key, handle = _handle_cache.popitem(last=False)
        _close_handle(handle)
    clear_runtime_pack_cache()


@contextmanager
def scoped_tensor_handle_cache():
    """Reuse safetensors handles within one local generation call, then close them."""
    if getattr(_scoped_handles, "handles", None) is not None:
        yield
        return

    handles: OrderedDict[tuple[str, int], object] = OrderedDict()
    _scoped_handles.handles = handles
    try:
        yield
    finally:
        while handles:
            _key, handle = handles.popitem(last=False)
            _close_handle(handle)
        try:
            del _scoped_handles.handles
        except AttributeError:  # pragma: no cover - defensive cleanup
            pass


def _open_scoped_shard_handle(path: Path, *, open_counter: str = "shard_opens"):
    if os.environ.get("PCKETLM_DISABLE_PERSISTENT_HANDLES", "0").strip().lower() in {"1", "true", "yes"}:
        return None
    handles = getattr(_scoped_handles, "handles", None)
    if handles is None:
        return None
    key = (str(path.resolve()), _path_mtime_ns(path))
    cached = handles.get(key)
    if cached is not None:
        handles.move_to_end(key)
        _update_load_stats(scoped_handle_reuses=1)
        return cached
    handle = safe_open(path, framework="pt", device="cpu")
    _update_load_stats(**{open_counter: 1})
    enter_method = getattr(handle, "__enter__", None)
    if callable(enter_method):
        handle = enter_method()
    handles[key] = handle
    return handle


def open_scoped_tensor_handle(path: Path, *, open_counter: str = "shard_opens"):
    """Return a request-scoped safetensors handle when a scoped cache is active."""
    return _open_scoped_shard_handle(path, open_counter=open_counter)


def _open_shard_handle(path: Path):
    if not _handle_cache_enabled():
        _update_load_stats(shard_opens=1)
        return safe_open(path, framework="pt", device="cpu")
    key = (str(path.resolve()), _path_mtime_ns(path))
    cached = _handle_cache.get(key)
    if cached is not None:
        _handle_cache.move_to_end(key)
        _update_load_stats(persistent_handle_reuses=1)
        return cached
    handle = safe_open(path, framework="pt", device="cpu")
    _update_load_stats(shard_opens=1)
    enter_method = getattr(handle, "__enter__", None)
    if callable(enter_method):
        handle = enter_method()
    _handle_cache[key] = handle
    while len(_handle_cache) > _HANDLE_CACHE_MAX_ENTRIES:
        _old_key, old_handle = _handle_cache.popitem(last=False)
        _close_handle(old_handle)
    return handle


def load_tensor_by_name(model_id: str, tensor_name: str) -> LoadedTensorSlice:
    """Load one real tensor by name from the original shard set."""
    _update_load_stats(single_load_calls=1, tensors_requested=1)
    entry = _find_tensor_entry(model_id, tensor_name)
    if entry is None:
        return LoadedTensorSlice(
            model_id=model_id,
            tensor_name=tensor_name,
            shard_name="",
            dtype="unknown",
            shape=[],
            blockers=[f"Tensor {tensor_name} is not present in the persisted tensor catalog."],
            ready=False,
        )

    try:
        q4_loaded = _load_q4_tensor(model_id, entry)
        if q4_loaded is not None:
            if q4_loaded.ready:
                _update_load_stats(tensors_loaded=1, loaded_nbytes=q4_loaded.loaded_nbytes)
            return q4_loaded
        scoped_handle = _open_scoped_shard_handle(entry.shard_path)
        artifact_pack_path = _artifact_pack_path_for_tensor(model_id, entry.tensor_name)
        if artifact_pack_path is not None:
            _update_load_stats(artifact_tensor_hits=1)
            artifact_handle = _open_scoped_shard_handle(artifact_pack_path, open_counter="artifact_pack_opens")
            if artifact_handle is not None:
                tensor = artifact_handle.get_tensor(entry.tensor_name)
                borrowed_from_live_handle = True
                _update_load_stats(live_handle_tensor_hits=1)
            else:
                borrowed_from_live_handle = False
                _update_load_stats(artifact_pack_opens=1)
                with safe_open(artifact_pack_path, framework="pt", device="cpu") as handle:
                    tensor = handle.get_tensor(entry.tensor_name)
        elif scoped_handle is not None:
            tensor = scoped_handle.get_tensor(entry.tensor_name)
            borrowed_from_live_handle = True
            _update_load_stats(live_handle_tensor_hits=1)
        elif _handle_cache_enabled():
            handle = _open_shard_handle(entry.shard_path)
            tensor = handle.get_tensor(entry.tensor_name)
            borrowed_from_live_handle = True
            _update_load_stats(live_handle_tensor_hits=1)
        else:
            native_loaded = _load_native_fp16_tensor(model_id, entry)
            if native_loaded is not None:
                if native_loaded.ready:
                    _update_load_stats(tensors_loaded=1, loaded_nbytes=native_loaded.loaded_nbytes)
                return native_loaded
            borrowed_from_live_handle = False
            _update_load_stats(shard_opens=1)
            with safe_open(entry.shard_path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(entry.tensor_name)
    except Exception as exc:  # pragma: no cover - safety guard
        return LoadedTensorSlice(
            model_id=model_id,
            tensor_name=entry.tensor_name,
            shard_name=entry.shard_name,
            dtype=entry.dtype,
            shape=list(entry.shape),
            layer_index=entry.layer_index,
            component_group=entry.component_group,
            blockers=[f"Failed to load tensor {entry.tensor_name}: {exc}"],
            ready=False,
        )

    loaded = _loaded_slice_from_entry(model_id, entry, tensor, borrowed_from_live_handle=borrowed_from_live_handle)
    if loaded.ready:
        _update_load_stats(tensors_loaded=1, loaded_nbytes=loaded.loaded_nbytes)
    return loaded


def load_tensors_by_name(model_id: str, tensor_names: list[str]) -> dict[str, LoadedTensorSlice]:
    """Load several real tensors while opening each safetensors shard only once."""
    unique_tensor_names = list(dict.fromkeys(tensor_names))
    _update_load_stats(batch_load_calls=1, tensors_requested=len(unique_tensor_names))
    catalog = load_tensor_catalog(model_id)
    if not catalog.ready:
        return {
            tensor_name: LoadedTensorSlice(
                model_id=model_id,
                tensor_name=tensor_name,
                shard_name="",
                dtype="unknown",
                shape=[],
                blockers=list(catalog.blockers) or ["Tensor catalog does not exist yet."],
                ready=False,
            )
            for tensor_name in tensor_names
        }

    entries_by_name = load_tensor_entry_index(model_id)
    results: dict[str, LoadedTensorSlice] = {}
    q4_groups: dict[tuple[Path, Path], list[TensorCatalogEntry]] = {}
    shard_groups: dict[Path, list[TensorCatalogEntry]] = {}
    artifact_groups: dict[Path, list[TensorCatalogEntry]] = {}
    for tensor_name in unique_tensor_names:
        entry = entries_by_name.get(tensor_name)
        if entry is None:
            results[tensor_name] = LoadedTensorSlice(
                model_id=model_id,
                tensor_name=tensor_name,
                shard_name="",
                dtype="unknown",
                shape=[],
                blockers=[f"Tensor {tensor_name} is not present in the persisted tensor catalog."],
                ready=False,
            )
            continue
        q4_payload = _q4_entry(model_id, entry.tensor_name) if _q4_source_enabled(model_id) else None
        if q4_payload is not None:
            root = _q4_artifact_root(model_id)
            q4_groups.setdefault(
                (root / str(q4_payload.get("q4_shard")), root / str(q4_payload.get("scale_shard"))),
                [],
            ).append(entry)
            continue
        artifact_pack_path = _artifact_pack_path_for_tensor(model_id, entry.tensor_name)
        if artifact_pack_path is not None:
            artifact_groups.setdefault(artifact_pack_path, []).append(entry)
            continue
        shard_groups.setdefault(entry.shard_path, []).append(entry)

    for (q4_path, scale_path), entries in q4_groups.items():
        try:
            with ExitStack() as stack:
                handles: dict[str, object] = {}

                def load_packed(entry: TensorCatalogEntry) -> tuple[torch.Tensor, torch.Tensor]:
                    if "q4" not in handles:
                        handles["q4"] = stack.enter_context(safe_open(q4_path, framework="pt", device="cpu"))
                        handles["scale"] = stack.enter_context(safe_open(scale_path, framework="pt", device="cpu"))
                        _update_load_stats(shard_opens=2)
                    return (
                        handles["q4"].get_tensor(entry.tensor_name),
                        handles["scale"].get_tensor(entry.tensor_name),
                    )

                q4_payloads: list[tuple[TensorCatalogEntry, torch.Tensor, torch.Tensor, list[int]]] = []
                for entry in entries:
                    payload = _q4_entry(model_id, entry.tensor_name)
                    if payload is None:
                        results[entry.tensor_name] = _loaded_slice_from_entry(
                            model_id,
                            entry,
                            blockers=[f"Q4 artifact does not contain tensor {entry.tensor_name}."],
                            ready=False,
                        )
                        continue
                    from pcketlm.core.runtime.tensor_residency import q4_packed_cache_get_or_load

                    packed, scales = q4_packed_cache_get_or_load(
                        model_id,
                        entry,
                        q4_path,
                        scale_path,
                        lambda entry=entry: load_packed(entry),
                    )
                    shape = [int(value) for value in payload.get("shape", entry.shape)]
                    q4_payloads.append((entry, packed, scales, shape))
                batched = _dequantize_q4_tensor_batch(q4_payloads)
                for entry, packed, scales, shape in q4_payloads:
                    tensor = None if batched is None else batched.get(entry.tensor_name)
                    if tensor is None:
                        tensor = _dequantize_q4_tensor(packed, scales, shape, entry.dtype)
                    _update_load_stats(q4_loads=1, q4_loaded_nbytes=int(packed.nelement() * packed.element_size()))
                    results[entry.tensor_name] = _loaded_slice_from_entry(model_id, entry, tensor, q4_loaded=True)
        except Exception as exc:  # pragma: no cover - defensive Q4 IO path
            for entry in entries:
                results[entry.tensor_name] = _loaded_slice_from_entry(
                    model_id,
                    entry,
                    blockers=[f"Failed to load Q4 tensor {entry.tensor_name}: {exc}"],
                    ready=False,
                )

    for pack_path, entries in artifact_groups.items():
        try:
            _update_load_stats(artifact_tensor_hits=len(entries))
            scoped_handle = _open_scoped_shard_handle(pack_path, open_counter="artifact_pack_opens")
            if scoped_handle is not None:
                _update_load_stats(live_handle_tensor_hits=len(entries))
                for entry in entries:
                    results[entry.tensor_name] = _loaded_slice_from_entry(
                        model_id,
                        entry,
                        scoped_handle.get_tensor(entry.tensor_name),
                        borrowed_from_live_handle=True,
                    )
            else:
                _update_load_stats(artifact_pack_opens=1)
                with safe_open(pack_path, framework="pt", device="cpu") as handle:
                    for entry in entries:
                        results[entry.tensor_name] = _loaded_slice_from_entry(
                            model_id,
                            entry,
                            handle.get_tensor(entry.tensor_name),
                        )
        except Exception as exc:  # pragma: no cover - defensive artifact IO path
            for entry in entries:
                results[entry.tensor_name] = _loaded_slice_from_entry(
                    model_id,
                    entry,
                    blockers=[f"Failed to load packed tensor {entry.tensor_name}: {exc}"],
                    ready=False,
                )

    for shard_path, entries in shard_groups.items():
        try:
            scoped_handle = _open_scoped_shard_handle(shard_path)
            if scoped_handle is not None:
                _update_load_stats(live_handle_tensor_hits=len(entries))
                for entry in entries:
                    results[entry.tensor_name] = _loaded_slice_from_entry(
                        model_id,
                        entry,
                        scoped_handle.get_tensor(entry.tensor_name),
                        borrowed_from_live_handle=True,
                    )
            elif _handle_cache_enabled():
                handle = _open_shard_handle(shard_path)
                _update_load_stats(live_handle_tensor_hits=len(entries))
                for entry in entries:
                    results[entry.tensor_name] = _loaded_slice_from_entry(
                        model_id,
                        entry,
                        handle.get_tensor(entry.tensor_name),
                        borrowed_from_live_handle=True,
                    )
            else:
                native_results: dict[str, LoadedTensorSlice] = {}
                native_failed = False
                for entry in entries:
                    native_loaded = _load_native_fp16_tensor(model_id, entry)
                    if native_loaded is None:
                        native_failed = True
                        break
                    native_results[entry.tensor_name] = native_loaded
                if not native_failed:
                    results.update(native_results)
                    continue
                _update_load_stats(shard_opens=1)
                with safe_open(shard_path, framework="pt", device="cpu") as handle:
                    for entry in entries:
                        results[entry.tensor_name] = _loaded_slice_from_entry(
                            model_id,
                            entry,
                            handle.get_tensor(entry.tensor_name),
                        )
        except Exception as exc:  # pragma: no cover - safety guard
            for entry in entries:
                results[entry.tensor_name] = _loaded_slice_from_entry(
                    model_id,
                    entry,
                    blockers=[f"Failed to load tensor {entry.tensor_name}: {exc}"],
                ready=False,
            )

    loaded_slices = [result for result in results.values() if result.ready]
    if loaded_slices:
        _update_load_stats(
            tensors_loaded=len(loaded_slices),
            loaded_nbytes=sum(result.loaded_nbytes for result in loaded_slices),
        )
    return {tensor_name: results[tensor_name] for tensor_name in tensor_names}


def verify_loaded_tensor(model_id: str, tensor_name: str) -> TensorVerificationResult:
    """Verify one loaded tensor against the persisted tensor catalog."""
    entry = _find_tensor_entry(model_id, tensor_name)
    if entry is None:
        return TensorVerificationResult(
            model_id=model_id,
            tensor_name=tensor_name,
            expected_dtype="unknown",
            actual_dtype="unknown",
            expected_shape=[],
            actual_shape=[],
            expected_nbytes=0,
            actual_nbytes=0,
            blockers=[f"Tensor {tensor_name} is not present in the persisted tensor catalog."],
            ready=False,
        )

    loaded = load_tensor_by_name(model_id, tensor_name)
    blockers = list(loaded.blockers)
    expected_dtype = _normalize_catalog_dtype(entry.dtype)
    actual_dtype = loaded.dtype
    expected_shape = list(entry.shape)
    actual_shape = list(loaded.shape)
    expected_nbytes = entry.data_nbytes
    actual_nbytes = loaded.loaded_nbytes

    if actual_dtype != expected_dtype:
        blockers.append(
            f"Loaded tensor dtype {actual_dtype} does not match catalog dtype {expected_dtype} for {tensor_name}."
        )
    if actual_shape != expected_shape:
        blockers.append(
            f"Loaded tensor shape {actual_shape} does not match catalog shape {expected_shape} for {tensor_name}."
        )
    if actual_nbytes != expected_nbytes:
        blockers.append(
            f"Loaded tensor size {actual_nbytes} bytes does not match catalog size {expected_nbytes} bytes for {tensor_name}."
        )

    return TensorVerificationResult(
        model_id=model_id,
        tensor_name=tensor_name,
        expected_dtype=expected_dtype,
        actual_dtype=actual_dtype,
        expected_shape=expected_shape,
        actual_shape=actual_shape,
        expected_nbytes=expected_nbytes,
        actual_nbytes=actual_nbytes,
        blockers=blockers,
        ready=not blockers,
    )


def _find_execution_unit(model_id: str, unit_id: str) -> TensorExecutionUnit | None:
    plan = load_tensor_execution_plan(model_id)
    if not plan.ready:
        return None
    for unit in plan.units:
        if unit.unit_id == unit_id:
            return unit
    return None


def load_execution_unit(model_id: str, unit_id: str) -> LoadedExecutionUnit:
    """Load one grouped execution unit into real CPU tensors."""
    unit = _find_execution_unit(model_id, unit_id)
    if unit is None:
        return LoadedExecutionUnit(
            model_id=model_id,
            unit_id=unit_id,
            label=unit_id,
            phase="unknown",
            tensor_count=0,
            total_nbytes=0,
            blockers=[f"Execution unit {unit_id} is not present in the persisted tensor execution plan."],
            ready=False,
        )

    loaded_tensors = [load_tensor_by_name(model_id, tensor_name) for tensor_name in unit.tensor_names]
    blockers = [blocker for tensor in loaded_tensors for blocker in tensor.blockers]
    total_nbytes = sum(tensor.loaded_nbytes for tensor in loaded_tensors)
    ready = bool(loaded_tensors) and not blockers and all(tensor.ready for tensor in loaded_tensors)

    return LoadedExecutionUnit(
        model_id=model_id,
        unit_id=unit.unit_id,
        label=unit.label,
        phase=unit.phase,
        tensor_count=len(loaded_tensors),
        total_nbytes=total_nbytes,
        tensors=loaded_tensors,
        blockers=blockers,
        ready=ready,
    )


def verify_execution_unit(model_id: str, unit_id: str) -> ExecutionUnitVerificationResult:
    """Verify one grouped execution unit against the execution plan and tensor catalog."""
    unit = _find_execution_unit(model_id, unit_id)
    if unit is None:
        return ExecutionUnitVerificationResult(
            model_id=model_id,
            unit_id=unit_id,
            tensor_count=0,
            total_nbytes=0,
            blockers=[f"Execution unit {unit_id} is not present in the persisted tensor execution plan."],
            ready=False,
        )

    results = [verify_loaded_tensor(model_id, tensor_name) for tensor_name in unit.tensor_names]
    blockers = [blocker for result in results for blocker in result.blockers]
    total_nbytes = sum(result.actual_nbytes for result in results)
    if len(results) != unit.tensor_count:
        blockers.append(
            f"Loaded tensor count {len(results)} does not match execution-plan tensor count {unit.tensor_count} for {unit_id}."
        )
    if total_nbytes != unit.total_nbytes:
        blockers.append(
            f"Loaded byte size {total_nbytes} does not match execution-plan size {unit.total_nbytes} for {unit_id}."
        )

    return ExecutionUnitVerificationResult(
        model_id=model_id,
        unit_id=unit_id,
        tensor_count=len(results),
        total_nbytes=total_nbytes,
        tensor_results=results,
        blockers=blockers,
        ready=not blockers and all(result.ready for result in results),
    )
