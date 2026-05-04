"""Runtime loader for offline row8 packed GEMV artifacts."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import torch

from pcketlm.core.storage.paths import model_root, streaming_model_root


ROW8_ARTIFACT_DIRNAME = "row8"
ROW8_MANIFEST_NAME = "row8_manifest.json"
_ROW8_TENSOR_CACHE: OrderedDict[tuple[str, str, str], tuple[torch.Tensor, dict, int]] = OrderedDict()
_ROW8_TENSOR_CACHE_BYTES = 0
_ROW8_NATIVE_TENSOR_CACHE: OrderedDict[tuple[str, str, str], tuple[object, dict, int]] = OrderedDict()
_ROW8_NATIVE_TENSOR_CACHE_BYTES = 0


@lru_cache(maxsize=16)
def _row8_manifest(model_id: str, artifact_name: str = ROW8_ARTIFACT_DIRNAME) -> dict:
    path = _row8_manifest_path(model_id, artifact_name)
    if not path.exists():
        return {"ready": False, "path": str(path), "tensors": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ready"] = True
    payload["path"] = str(path)
    return payload


def _row8_manifest_path(model_id: str, artifact_name: str) -> Path:
    candidates = (
        streaming_model_root(model_id) / "artifacts" / artifact_name / ROW8_MANIFEST_NAME,
        model_root(model_id) / "artifacts" / artifact_name / ROW8_MANIFEST_NAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def row8_artifact_status(model_id: str, artifact_name: str = ROW8_ARTIFACT_DIRNAME) -> dict:
    manifest = _row8_manifest(model_id, artifact_name)
    return {
        "ready": bool(manifest.get("ready")),
        "path": str(manifest.get("path", "")),
        "format": manifest.get("format"),
        "tensor_count": int(manifest.get("tensor_count", 0) or 0),
        "total_packed_bytes": int(manifest.get("total_packed_bytes", 0) or 0),
    }


def row8_tensor_available(model_id: str, tensor_name: str, artifact_name: str = ROW8_ARTIFACT_DIRNAME) -> bool:
    manifest = _row8_manifest(model_id, artifact_name)
    return bool(manifest.get("ready")) and str(tensor_name) in (manifest.get("tensors") or {})


def load_row8_packed_tensor(
    model_id: str,
    tensor_name: str,
    artifact_name: str = ROW8_ARTIFACT_DIRNAME,
) -> tuple[torch.Tensor, dict]:
    cache_key = (str(model_id), str(artifact_name), str(tensor_name))
    if _row8_tensor_cache_enabled():
        cached = _ROW8_TENSOR_CACHE.get(cache_key)
        if cached is not None:
            tensor, entry, _nbytes = cached
            _ROW8_TENSOR_CACHE.move_to_end(cache_key)
            # The native dense layer is expected to treat packed weights as
            # const, but returning a fresh view avoids pointer lifetime/reuse
            # surprises across ctypes calls while still avoiding disk I/O.
            return tensor.clone(), dict(entry)

    manifest = _row8_manifest(model_id, artifact_name)
    if not manifest.get("ready"):
        raise FileNotFoundError(str(manifest.get("path", "")))
    tensors = manifest.get("tensors") or {}
    if tensor_name not in tensors:
        raise KeyError(tensor_name)
    entry = dict(tensors[tensor_name])
    artifact_dir = Path(str(manifest["path"])).parent
    data_file = artifact_dir / str(manifest["data_file"])
    with data_file.open("rb") as handle:
        handle.seek(int(entry["offset"]))
        payload = handle.read(int(entry["nbytes"]))
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint16).clone()
    _store_row8_tensor_cache(cache_key, tensor, entry)
    return tensor, entry


def load_row8_native_packed_tensor(
    model_id: str,
    tensor_name: str,
    artifact_name: str = ROW8_ARTIFACT_DIRNAME,
) -> tuple[object, dict]:
    cache_key = (str(model_id), str(artifact_name), str(tensor_name))
    if _row8_tensor_cache_enabled():
        cached = _ROW8_NATIVE_TENSOR_CACHE.get(cache_key)
        if cached is not None:
            native_tensor, entry, _nbytes = cached
            _ROW8_NATIVE_TENSOR_CACHE.move_to_end(cache_key)
            return native_tensor, dict(entry)

    manifest = _row8_manifest(model_id, artifact_name)
    if not manifest.get("ready"):
        raise FileNotFoundError(str(manifest.get("path", "")))
    tensors = manifest.get("tensors") or {}
    if tensor_name not in tensors:
        raise KeyError(tensor_name)
    entry = dict(tensors[tensor_name])
    artifact_dir = Path(str(manifest["path"])).parent
    data_file = artifact_dir / str(manifest["data_file"])
    from pcketlm.native import load_native_row8_tensor

    native_tensor = load_native_row8_tensor(data_file, int(entry["offset"]), int(entry["nbytes"]))
    _store_row8_native_tensor_cache(cache_key, native_tensor, entry)
    return native_tensor, entry


def row8_tensor_cache_stats() -> dict[str, int | float]:
    return {
        "resident_count": len(_ROW8_TENSOR_CACHE),
        "resident_bytes": int(_ROW8_TENSOR_CACHE_BYTES),
        "resident_mb": round(_ROW8_TENSOR_CACHE_BYTES / (1024 * 1024), 3),
        "native_resident_count": len(_ROW8_NATIVE_TENSOR_CACHE),
        "native_resident_bytes": int(_ROW8_NATIVE_TENSOR_CACHE_BYTES),
        "native_resident_mb": round(_ROW8_NATIVE_TENSOR_CACHE_BYTES / (1024 * 1024), 3),
        "budget_bytes": int(_row8_tensor_cache_budget_bytes()),
        "budget_mb": round(_row8_tensor_cache_budget_bytes() / (1024 * 1024), 3),
    }


def clear_row8_manifest_cache() -> None:
    _row8_manifest.cache_clear()


def clear_row8_tensor_cache() -> None:
    global _ROW8_TENSOR_CACHE_BYTES, _ROW8_NATIVE_TENSOR_CACHE_BYTES
    _ROW8_TENSOR_CACHE.clear()
    _ROW8_TENSOR_CACHE_BYTES = 0
    _ROW8_NATIVE_TENSOR_CACHE.clear()
    _ROW8_NATIVE_TENSOR_CACHE_BYTES = 0


def _row8_tensor_cache_enabled() -> bool:
    if os.environ.get("PCKETLM_DISABLE_ROW8_TENSOR_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return _row8_tensor_cache_budget_bytes() > 0


def _row8_tensor_cache_budget_bytes() -> int:
    raw = os.environ.get("PCKETLM_ROW8_TENSOR_CACHE_MB", "1024").strip()
    try:
        mb = int(raw)
    except ValueError:
        mb = 1024
    return max(0, mb) * 1024 * 1024


def _store_row8_tensor_cache(cache_key: tuple[str, str, str], tensor: torch.Tensor, entry: dict) -> None:
    global _ROW8_TENSOR_CACHE_BYTES
    if not _row8_tensor_cache_enabled():
        return
    nbytes = int(tensor.numel() * tensor.element_size())
    budget = _row8_tensor_cache_budget_bytes()
    if nbytes > budget:
        return
    existing = _ROW8_TENSOR_CACHE.pop(cache_key, None)
    if existing is not None:
        _ROW8_TENSOR_CACHE_BYTES -= int(existing[2])
    _ROW8_TENSOR_CACHE[cache_key] = (tensor, dict(entry), nbytes)
    _ROW8_TENSOR_CACHE_BYTES += nbytes
    while _ROW8_TENSOR_CACHE_BYTES > budget and _ROW8_TENSOR_CACHE:
        _old_key, (_old_tensor, _old_entry, old_nbytes) = _ROW8_TENSOR_CACHE.popitem(last=False)
        _ROW8_TENSOR_CACHE_BYTES -= int(old_nbytes)


def _store_row8_native_tensor_cache(cache_key: tuple[str, str, str], native_tensor: object, entry: dict) -> None:
    global _ROW8_NATIVE_TENSOR_CACHE_BYTES
    if not _row8_tensor_cache_enabled():
        return
    nbytes = int(getattr(native_tensor, "nbytes"))
    budget = _row8_tensor_cache_budget_bytes()
    if nbytes > budget:
        return
    existing = _ROW8_NATIVE_TENSOR_CACHE.pop(cache_key, None)
    if existing is not None:
        _ROW8_NATIVE_TENSOR_CACHE_BYTES -= int(existing[2])
    _ROW8_NATIVE_TENSOR_CACHE[cache_key] = (native_tensor, dict(entry), nbytes)
    _ROW8_NATIVE_TENSOR_CACHE_BYTES += nbytes
    while _ROW8_NATIVE_TENSOR_CACHE_BYTES > budget and _ROW8_NATIVE_TENSOR_CACHE:
        _old_key, (_old_tensor, _old_entry, old_nbytes) = _ROW8_NATIVE_TENSOR_CACHE.popitem(last=False)
        _ROW8_NATIVE_TENSOR_CACHE_BYTES -= int(old_nbytes)
