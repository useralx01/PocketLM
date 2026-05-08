"""User-facing acquisition state helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pcketlm.core.model_import.download_state import estimate_download_state
from pcketlm.core.model_import.q4_plan import plan_model_dir_to_q4


def _to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024**3), 2)


def _build_progress_bar(progress_pct: float | None, width: int = 24) -> str:
    if progress_pct is None:
        return "[" + ("?" * width) + "]"

    clamped = max(0.0, min(progress_pct, 100.0))
    filled = int((clamped / 100.0) * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _build_summary(status: str, progress_pct: float | None, missing_core_files: list[str]) -> str:
    if status == "missing":
        return "The model folder does not exist yet, so no download has started."
    if status == "empty":
        return "The model folder exists, but it is still empty."
    if status == "ready":
        return "The source model looks complete and ready for import and runtime work."
    if status == "downloading" and progress_pct is not None:
        return f"The download is active and has reached about {progress_pct}% of the expected size."
    if status == "partial" and missing_core_files:
        joined = ", ".join(missing_core_files)
        return f"The source folder has real data, but it is not usable yet because key files are still missing: {joined}."
    return "The source folder has partial data, but pcketlm cannot call it ready yet."


def _build_next_step(status: str) -> str:
    if status in {"missing", "empty", "partial", "downloading"}:
        return "Let the download continue, then recheck acquisition state before import."
    return "The source is ready, so the next step is import and runtime readiness validation."


def _build_compact_q4_plan(model_dir: Path, status: str) -> dict | None:
    if status != "ready":
        return None
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    try:
        return plan_model_dir_to_q4(model_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


@dataclass(slots=True)
class AcquisitionSnapshot:
    """Product-facing acquisition summary for one source model."""

    model_dir: Path
    status: str
    bytes_on_disk: int
    bytes_on_disk_gb: float
    expected_bytes: int | None
    expected_bytes_gb: float | None
    progress_pct: float | None
    progress_bar: str
    expected_shards: int
    present_shards: int
    missing_core_files: list[str]
    plain_english_summary: str
    recommended_next_step: str
    compact_q4_plan: dict | None = None

    def to_dict(self) -> dict:
        """Serialize the acquisition snapshot."""
        return {
            "model_dir": str(self.model_dir),
            "status": self.status,
            "bytes_on_disk": self.bytes_on_disk,
            "bytes_on_disk_gb": self.bytes_on_disk_gb,
            "expected_bytes": self.expected_bytes,
            "expected_bytes_gb": self.expected_bytes_gb,
            "progress_pct": self.progress_pct,
            "progress_bar": self.progress_bar,
            "expected_shards": self.expected_shards,
            "present_shards": self.present_shards,
            "missing_core_files": list(self.missing_core_files),
            "plain_english_summary": self.plain_english_summary,
            "recommended_next_step": self.recommended_next_step,
            "compact_q4_plan": self.compact_q4_plan,
        }


def build_acquisition_snapshot(model_dir: Path) -> AcquisitionSnapshot:
    """Build a richer user-facing acquisition snapshot for a source model folder."""
    state = estimate_download_state(model_dir)
    compact_q4_plan = _build_compact_q4_plan(model_dir, state.status)
    recommended_next_step = _build_next_step(state.status)
    if compact_q4_plan and compact_q4_plan.get("ready_for_conversion"):
        q4_gb = _to_gb(int(compact_q4_plan.get("estimated_total_q4_bytes", 0))) or 0.0
        recommended_next_step = (
            f"The source is complete. Build the compact Q4 artifact first; estimated output is about {q4_gb} GB."
        )
    return AcquisitionSnapshot(
        model_dir=model_dir,
        status=state.status,
        bytes_on_disk=state.bytes_on_disk,
        bytes_on_disk_gb=_to_gb(state.bytes_on_disk) or 0.0,
        expected_bytes=state.expected_bytes,
        expected_bytes_gb=_to_gb(state.expected_bytes),
        progress_pct=state.progress_pct,
        progress_bar=_build_progress_bar(state.progress_pct),
        expected_shards=state.expected_shards,
        present_shards=state.present_shards,
        missing_core_files=list(state.missing_core_files),
        plain_english_summary=_build_summary(
            state.status,
            state.progress_pct,
            state.missing_core_files,
        ),
        recommended_next_step=recommended_next_step,
        compact_q4_plan=compact_q4_plan,
    )
