"""Lossless FP8 packed artifact reader."""

from __future__ import annotations

import json
import mmap
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(slots=True)
class PackedTensorView:
    """Memory views for one packed tensor and its optional scale companion."""

    name: str
    dtype: str
    shape: list[int]
    fp8_bytes: memoryview
    scale_bytes: memoryview | None = None
    scale_name: str | None = None
    scale_dtype: str | None = None
    scale_shape: list[int] | None = None


@dataclass(frozen=True, slots=True)
class PackedSliceLocation:
    """Absolute pack-file byte range for native file reads."""

    path: Path
    byte_offset: int
    byte_length: int


@dataclass(frozen=True, slots=True)
class PackedSpanLocation:
    """One pack-file byte range that contains several tensor payloads."""

    path: Path
    byte_offset: int
    byte_length: int
    tensor_slices: dict[str, tuple[int, int]]


class FP8PackReader:
    """Read lossless FP8 pack files through long-lived mmap handles."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir).resolve()
        self.pack_dir = _default_pack_dir(self.model_dir)
        self.manifest_path = self.pack_dir / "pack_manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"FP8 pack manifest not found: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.tensors: dict[str, dict] = dict(self.manifest.get("tensors") or {})
        self._files: dict[int, object] = {}
        self._mmaps: dict[int, mmap.mmap] = {}
        self._native_files_seen: set[int] = set()
        self._stats = {
            "pack_files_open": 0,
            "mmap_bytes_resident": 0,
            "sequential_reads": 0,
            "scattered_reads": 0,
            "bytes_returned": 0,
        }

    def close(self) -> None:
        while self._mmaps:
            _index, mapped = self._mmaps.popitem()
            mapped.close()
        while self._files:
            _index, handle = self._files.popitem()
            handle.close()

    def telemetry(self) -> dict:
        return dict(self._stats)

    def get_tensor(self, name: str) -> PackedTensorView:
        entry = self._entry(name)
        raw = self._slice(entry["pack_file_index"], entry["byte_offset"], entry["byte_length"])
        scale_raw = None
        if entry.get("scale_name"):
            scale_raw = self._slice(
                entry["scale_pack_file_index"],
                entry["scale_byte_offset"],
                entry["scale_byte_length"],
            )
        self._stats["sequential_reads"] += 1
        self._stats["bytes_returned"] += int(entry["byte_length"]) + int(entry.get("scale_byte_length") or 0)
        return PackedTensorView(
            name=str(name),
            dtype=str(entry["dtype"]),
            shape=[int(value) for value in entry.get("shape", [])],
            fp8_bytes=raw,
            scale_bytes=scale_raw,
            scale_name=entry.get("scale_name"),
            scale_dtype=entry.get("scale_dtype"),
            scale_shape=None if entry.get("scale_shape") is None else [int(value) for value in entry["scale_shape"]],
        )

    def get_tensor_bytes(self, name: str) -> memoryview:
        entry = self._entry(name)
        raw = self.get_tensor_slice_bytes(name, 0, int(entry["byte_length"]))
        return raw

    def get_tensor_slice_bytes(self, name: str, relative_offset: int, byte_length: int) -> memoryview:
        entry = self._entry(name)
        self._validate_slice(name, entry, int(relative_offset), int(byte_length))
        raw = self._slice(
            entry["pack_file_index"],
            int(entry["byte_offset"]) + int(relative_offset),
            int(byte_length),
        )
        self._stats["sequential_reads"] += 1
        self._stats["bytes_returned"] += int(byte_length)
        return raw

    def get_tensor_slice_location(self, name: str, relative_offset: int, byte_length: int) -> PackedSliceLocation:
        entry = self._entry(name)
        self._validate_slice(name, entry, int(relative_offset), int(byte_length))
        index = int(entry["pack_file_index"])
        self._native_files_seen.add(index)
        self._stats["pack_files_open"] = len(set(self._mmaps) | self._native_files_seen)
        self._stats["sequential_reads"] += 1
        self._stats["bytes_returned"] += int(byte_length)
        return PackedSliceLocation(
            path=self.pack_dir / f"pack_{index:04d}.bin",
            byte_offset=int(entry["byte_offset"]) + int(relative_offset),
            byte_length=int(byte_length),
        )

    def get_tensor_span_location(self, names: list[str]) -> PackedSpanLocation:
        if not names:
            raise ValueError("Packed tensor span requires at least one tensor name.")
        entries = [(str(name), self._entry(str(name))) for name in names]
        pack_indexes = {int(entry["pack_file_index"]) for _name, entry in entries}
        if len(pack_indexes) != 1:
            raise ValueError("Packed tensor span crosses pack files.")
        index = pack_indexes.pop()
        start = min(int(entry["byte_offset"]) for _name, entry in entries)
        end = max(int(entry["byte_offset"]) + int(entry["byte_length"]) for _name, entry in entries)
        self._native_files_seen.add(index)
        self._stats["pack_files_open"] = len(set(self._mmaps) | self._native_files_seen)
        self._stats["sequential_reads"] += 1
        self._stats["bytes_returned"] += int(end - start)
        return PackedSpanLocation(
            path=self.pack_dir / f"pack_{index:04d}.bin",
            byte_offset=int(start),
            byte_length=int(end - start),
            tensor_slices={
                name: (int(entry["byte_offset"]) - int(start), int(entry["byte_length"]))
                for name, entry in entries
            },
        )

    def get_layer_experts(self, layer_idx: int, expert_ids: list[int]) -> list[PackedTensorView]:
        requested = {int(value) for value in expert_ids}
        prefix = f"model.layers.{int(layer_idx)}.mlp.experts."
        names: list[str] = []
        for name, entry in self.tensors.items():
            if not name.startswith(prefix):
                continue
            expert = entry.get("expert_index")
            if expert is not None and int(expert) in requested and entry.get("tensor_role") != "scale_companion":
                names.append(name)
        names.sort(key=lambda item: self._entry(item).get("pack_order", 0))
        return [self.get_tensor(name) for name in names]

    def _entry(self, name: str) -> dict:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"Tensor {name} is not present in FP8 pack manifest.") from exc

    def _validate_slice(self, name: str, entry: dict, relative_offset: int, byte_length: int) -> None:
        if int(relative_offset) < 0 or int(byte_length) < 0:
            raise ValueError("Packed tensor slice offset and length must be non-negative.")
        if int(relative_offset) + int(byte_length) > int(entry["byte_length"]):
            raise ValueError(f"Packed tensor slice exceeds tensor byte length for {name}.")

    def _mmap_for(self, pack_file_index: int) -> mmap.mmap:
        index = int(pack_file_index)
        mapped = self._mmaps.get(index)
        if mapped is not None:
            return mapped
        filename = f"pack_{index:04d}.bin"
        path = self.pack_dir / filename
        handle = path.open("rb")
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._files[index] = handle
        self._mmaps[index] = mapped
        self._stats["pack_files_open"] = len(set(self._mmaps) | self._native_files_seen)
        self._stats["mmap_bytes_resident"] += int(path.stat().st_size)
        return mapped

    def _slice(self, pack_file_index: int, byte_offset: int, byte_length: int) -> memoryview:
        mapped = self._mmap_for(int(pack_file_index))
        start = int(byte_offset)
        end = start + int(byte_length)
        return memoryview(mapped)[start:end]


def fp8_pack_enabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_FP8_PACK", "").strip().lower() not in {"1", "true", "yes", "on"}


def default_fp8_pack_dir(model_dir: Path) -> Path:
    return _default_pack_dir(Path(model_dir))


def fp8_pack_available(model_dir: Path) -> bool:
    return fp8_pack_enabled() and (default_fp8_pack_dir(model_dir) / "pack_manifest.json").exists()


def record_scattered_read(model_dir: Path | None, byte_count: int = 0) -> None:
    if model_dir is None:
        return
    key = str(Path(model_dir).resolve())
    reader = _READERS.get(key)
    if reader is not None:
        stats = reader._stats
        stats["scattered_reads"] += 1
        stats["bytes_returned"] += int(byte_count)
        return
    stats = _GLOBAL_STATS.setdefault(
        key,
        {
            "pack_files_open": 0,
            "mmap_bytes_resident": 0,
            "sequential_reads": 0,
            "scattered_reads": 0,
            "bytes_returned": 0,
        },
    )
    stats["scattered_reads"] += 1
    stats["bytes_returned"] += int(byte_count)


def reader_for_model_dir(model_dir: Path) -> FP8PackReader | None:
    if not fp8_pack_available(model_dir):
        return None
    key = str(Path(model_dir).resolve())
    reader = _READERS.get(key)
    if reader is None:
        reader = FP8PackReader(Path(model_dir))
        _READERS[key] = reader
    return reader


def fp8_pack_telemetry(model_dir: Path) -> dict:
    reader = _READERS.get(str(Path(model_dir).resolve()))
    if reader is None:
        stats = _GLOBAL_STATS.get(str(Path(model_dir).resolve()), {})
        return {
            "enabled": fp8_pack_enabled(),
            "available": fp8_pack_available(model_dir),
            "pack_files_open": int(stats.get("pack_files_open", 0)),
            "mmap_bytes_resident": int(stats.get("mmap_bytes_resident", 0)),
            "sequential_reads": int(stats.get("sequential_reads", 0)),
            "scattered_reads": int(stats.get("scattered_reads", 0)),
            "bytes_returned": int(stats.get("bytes_returned", 0)),
        }
    return {
        "enabled": fp8_pack_enabled(),
        "available": True,
        **reader.telemetry(),
    }


def clear_fp8_pack_readers() -> None:
    while _READERS:
        _key, reader = _READERS.popitem()
        reader.close()
    _GLOBAL_STATS.clear()
    _default_pack_dir.cache_clear()


@lru_cache(maxsize=64)
def _default_pack_dir(model_dir: Path) -> Path:
    override = os.environ.get("PCKETLM_FP8_PACK_DIR", "").strip()
    if override:
        return Path(override).resolve()
    return Path(model_dir).resolve() / "artifacts" / "fp8_pack"


_READERS: dict[str, FP8PackReader] = {}
_GLOBAL_STATS: dict[str, dict] = {}
