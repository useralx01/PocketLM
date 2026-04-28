"""Helpers for estimating source download state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pcketlm.core.model_import.inspect import inspect_qwen_source


@dataclass(slots=True)
class DownloadState:
    """Approximate download state for a source model folder."""

    status: str
    bytes_on_disk: int
    expected_bytes: int | None
    progress_pct: float | None
    expected_shards: int
    present_shards: int
    missing_core_files: list[str]


CORE_FILES = (
    "config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
)


def estimate_download_state(model_dir: Path) -> DownloadState:
    """Estimate the current download state for a model folder."""
    bytes_on_disk = sum(
        path.stat().st_size
        for path in model_dir.rglob("*")
        if path.is_file()
    ) if model_dir.exists() else 0

    missing_core_files = [name for name in CORE_FILES if not (model_dir / name).exists()]
    inspection = inspect_qwen_source(model_dir)

    expected_bytes = inspection.total_size_bytes
    progress_pct = round((bytes_on_disk / expected_bytes) * 100, 2) if expected_bytes else None

    if not model_dir.exists():
        status = "missing"
    elif missing_core_files and bytes_on_disk > 0:
        status = "partial"
    elif inspection.expected_shards > 0 and inspection.present_shards < inspection.expected_shards:
        status = "downloading"
    elif inspection.expected_shards > 0 and inspection.present_shards == inspection.expected_shards:
        status = "ready"
    elif bytes_on_disk > 0:
        status = "partial"
    else:
        status = "empty"

    return DownloadState(
        status=status,
        bytes_on_disk=bytes_on_disk,
        expected_bytes=expected_bytes,
        progress_pct=progress_pct,
        expected_shards=inspection.expected_shards,
        present_shards=inspection.present_shards,
        missing_core_files=missing_core_files,
    )
