"""Runtime control decisions for staged streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.runtime.streaming_materialize import (
    StreamingMaterializationResult,
    materialize_window_schedule,
    verify_materialized_cache,
)
from pcketlm.core.runtime.streaming_reader import load_streaming_manifest
from pcketlm.core.runtime.streaming_residency import StreamingResidencyState, load_streaming_residency
from pcketlm.core.runtime.streaming_rotation import StreamingRotationResult, rotate_window_schedule
from pcketlm.core.runtime.window_schedule import WindowSchedule, load_window_schedule


@dataclass(slots=True)
class StreamingControlState:
    """One runtime control recommendation for staged streaming."""

    model_id: str
    action_key: str
    action_label: str
    summary: str
    rotation_step: int
    last_event: str
    current_hot_unit_ids: list[str] = field(default_factory=list)
    current_warm_unit_ids: list[str] = field(default_factory=list)
    overflow_head_unit_id: str | None = None
    refill_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    last_cache_hit_delta: int = 0
    last_cache_miss_delta: int = 0
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "action_key": self.action_key,
            "action_label": self.action_label,
            "summary": self.summary,
            "rotation_step": self.rotation_step,
            "last_event": self.last_event,
            "current_hot_unit_ids": list(self.current_hot_unit_ids),
            "current_warm_unit_ids": list(self.current_warm_unit_ids),
            "overflow_head_unit_id": self.overflow_head_unit_id,
            "refill_count": self.refill_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "last_cache_hit_delta": self.last_cache_hit_delta,
            "last_cache_miss_delta": self.last_cache_miss_delta,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class StreamingControlExecutionResult:
    """One executed runtime control step."""

    model_id: str
    action_key: str
    action_label: str
    executed: bool
    summary: str
    blockers: list[str] = field(default_factory=list)
    ready: bool = False
    control_state: StreamingControlState | None = None
    verification: StreamingMaterializationResult | None = None
    materialization: StreamingMaterializationResult | None = None
    rotation: StreamingRotationResult | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "action_key": self.action_key,
            "action_label": self.action_label,
            "executed": self.executed,
            "summary": self.summary,
            "blockers": list(self.blockers),
            "ready": self.ready,
            "control_state": self.control_state.to_dict() if self.control_state else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "materialization": self.materialization.to_dict() if self.materialization else None,
            "rotation": self.rotation.to_dict() if self.rotation else None,
        }


def _telemetry_from_residency(residency: StreamingResidencyState) -> dict:
    return {
        "rotation_step": residency.rotation_step,
        "last_event": residency.last_event,
        "current_hot_unit_ids": list(residency.current_hot_unit_ids),
        "current_warm_unit_ids": list(residency.current_warm_unit_ids),
        "overflow_head_unit_id": residency.overflow_head_unit_id,
        "refill_count": residency.refill_count,
        "cache_hit_count": residency.cache_hit_count,
        "cache_miss_count": residency.cache_miss_count,
        "last_cache_hit_delta": residency.last_cache_hit_delta,
        "last_cache_miss_delta": residency.last_cache_miss_delta,
    }


def build_streaming_control_state(model_id: str) -> StreamingControlState:
    """Decide the next staged-streaming control step from live runtime state."""
    manifest = load_streaming_manifest(model_id)
    schedule = load_window_schedule(model_id)
    residency = load_streaming_residency(model_id)
    blockers = list(dict.fromkeys(list(manifest.blockers) + list(schedule.blockers) + list(residency.blockers)))
    telemetry = _telemetry_from_residency(residency)

    if not manifest.ready or not schedule.ready:
        return StreamingControlState(
            model_id=model_id,
            action_key="await-streaming-metadata",
            action_label="Await Streaming Metadata",
            summary="Streaming metadata is not ready yet, so runtime control cannot advance.",
            blockers=blockers or ["Streaming metadata must be rebuilt before the stream can advance."],
            ready=False,
            **telemetry,
        )

    if residency.last_cache_miss_delta > 0:
        return StreamingControlState(
            model_id=model_id,
            action_key="repair-cache-window",
            action_label="Repair Cache Window",
            summary="The last cache verification found misses, so the next step is to rebuild the current cache window before advancing.",
            blockers=blockers,
            ready=True,
            **telemetry,
        )

    if not residency.ready or not residency.current_hot_unit_ids:
        return StreamingControlState(
            model_id=model_id,
            action_key="materialize-current-window",
            action_label="Materialize Current Window",
            summary="The current streaming window is not resident yet, so the next step is to materialize the hot and warm cache files.",
            blockers=blockers,
            ready=True,
            **telemetry,
        )

    if residency.current_warm_unit_ids or residency.overflow_head_unit_id:
        return StreamingControlState(
            model_id=model_id,
            action_key="rotate-forward",
            action_label="Rotate Forward",
            summary="The current cache window is healthy, so the next step is to advance the stream and refill warm cache from overflow.",
            blockers=blockers,
            ready=True,
            **telemetry,
        )

    return StreamingControlState(
        model_id=model_id,
        action_key="hold-position",
        action_label="Hold Position",
        summary="No warm or overflow units remain, so the stream cannot advance until a new schedule is built.",
        blockers=blockers,
        ready=False,
        **telemetry,
    )


def advance_streaming_runtime(model_id: str, model_dir: Path) -> StreamingControlExecutionResult:
    """Execute the next staged-streaming control step for one model."""
    control = build_streaming_control_state(model_id)

    if control.action_key in {"materialize-current-window", "repair-cache-window"}:
        materialization = materialize_window_schedule(model_id, model_dir)
        return StreamingControlExecutionResult(
            model_id=model_id,
            action_key=control.action_key,
            action_label=control.action_label,
            executed=True,
            summary=control.summary,
            blockers=list(materialization.blockers),
            ready=materialization.ready,
            control_state=control,
            materialization=materialization,
        )

    if control.action_key == "rotate-forward":
        rotation = rotate_window_schedule(model_id, model_dir)
        return StreamingControlExecutionResult(
            model_id=model_id,
            action_key=control.action_key,
            action_label=control.action_label,
            executed=True,
            summary=control.summary,
            blockers=list(rotation.blockers),
            ready=rotation.ready,
            control_state=control,
            rotation=rotation,
        )

    return StreamingControlExecutionResult(
        model_id=model_id,
        action_key=control.action_key,
        action_label=control.action_label,
        executed=False,
        summary=control.summary,
        blockers=list(control.blockers),
        ready=control.ready,
        control_state=control,
    )


def advance_streaming_runtime_safely(model_id: str, model_dir: Path) -> StreamingControlExecutionResult:
    """Verify cache health first, then repair or advance the stream."""
    verification = verify_materialized_cache(model_id)
    control = build_streaming_control_state(model_id)

    if control.action_key == "repair-cache-window":
        materialization = materialize_window_schedule(model_id, model_dir)
        blockers = list(dict.fromkeys(list(verification.blockers) + list(materialization.blockers)))
        return StreamingControlExecutionResult(
            model_id=model_id,
            action_key="safe-repair-cache-window",
            action_label="Safe Repair Cache Window",
            executed=True,
            summary="Verification found cache misses, so the current cache window was rebuilt instead of advancing the stream.",
            blockers=blockers,
            ready=materialization.ready and not blockers,
            control_state=control,
            verification=verification,
            materialization=materialization,
        )

    if control.action_key == "rotate-forward":
        rotation = rotate_window_schedule(model_id, model_dir)
        blockers = list(dict.fromkeys(list(verification.blockers) + list(rotation.blockers)))
        return StreamingControlExecutionResult(
            model_id=model_id,
            action_key="safe-rotate-forward",
            action_label="Safe Rotate Forward",
            executed=True,
            summary="Verification passed, so the stream advanced and the warm cache was refilled.",
            blockers=blockers,
            ready=verification.ready and rotation.ready and not blockers,
            control_state=control,
            verification=verification,
            rotation=rotation,
        )

    if control.action_key == "materialize-current-window":
        materialization = materialize_window_schedule(model_id, model_dir)
        blockers = list(dict.fromkeys(list(verification.blockers) + list(materialization.blockers)))
        return StreamingControlExecutionResult(
            model_id=model_id,
            action_key="safe-materialize-current-window",
            action_label="Safe Materialize Current Window",
            executed=True,
            summary="Verification ran first, then the current cache window was materialized.",
            blockers=blockers,
            ready=materialization.ready and not blockers,
            control_state=control,
            verification=verification,
            materialization=materialization,
        )

    return StreamingControlExecutionResult(
        model_id=model_id,
        action_key=f"safe-{control.action_key}",
        action_label=f"Safe {control.action_label}",
        executed=False,
        summary=control.summary,
        blockers=list(dict.fromkeys(list(verification.blockers) + list(control.blockers))),
        ready=verification.ready and control.ready,
        control_state=control,
        verification=verification,
    )
