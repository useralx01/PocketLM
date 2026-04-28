"""Persist and load staged-streaming residency state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.runtime.window_schedule import WindowSchedule, load_window_schedule
from pcketlm.core.storage.paths import streaming_model_root


@dataclass(slots=True)
class StreamingResidencyState:
    """Persistent streaming residency and rotation position state."""

    model_id: str
    residency_path: Path
    rotation_step: int
    last_event: str
    current_hot_unit_ids: list[str] = field(default_factory=list)
    current_warm_unit_ids: list[str] = field(default_factory=list)
    overflow_head_unit_id: str | None = None
    consumed_unit_ids: list[str] = field(default_factory=list)
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
            "residency_path": str(self.residency_path),
            "rotation_step": self.rotation_step,
            "last_event": self.last_event,
            "current_hot_unit_ids": list(self.current_hot_unit_ids),
            "current_warm_unit_ids": list(self.current_warm_unit_ids),
            "overflow_head_unit_id": self.overflow_head_unit_id,
            "consumed_unit_ids": list(self.consumed_unit_ids),
            "refill_count": self.refill_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "last_cache_hit_delta": self.last_cache_hit_delta,
            "last_cache_miss_delta": self.last_cache_miss_delta,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


def residency_state_path(model_id: str) -> Path:
    """Return the residency state file for one model."""
    return streaming_model_root(model_id) / "residency.json"


def _state_from_payload(payload: dict) -> StreamingResidencyState:
    return StreamingResidencyState(
        model_id=str(payload.get("model_id", "")),
        residency_path=Path(payload.get("residency_path", "")),
        rotation_step=int(payload.get("rotation_step", 0)),
        last_event=str(payload.get("last_event", "unknown")),
        current_hot_unit_ids=list(payload.get("current_hot_unit_ids", [])),
        current_warm_unit_ids=list(payload.get("current_warm_unit_ids", [])),
        overflow_head_unit_id=payload.get("overflow_head_unit_id"),
        consumed_unit_ids=list(payload.get("consumed_unit_ids", [])),
        refill_count=int(payload.get("refill_count", 0)),
        cache_hit_count=int(payload.get("cache_hit_count", 0)),
        cache_miss_count=int(payload.get("cache_miss_count", 0)),
        last_cache_hit_delta=int(payload.get("last_cache_hit_delta", 0)),
        last_cache_miss_delta=int(payload.get("last_cache_miss_delta", 0)),
        blockers=list(payload.get("blockers", [])),
        ready=bool(payload.get("ready", False)),
    )


def load_streaming_residency(model_id: str) -> StreamingResidencyState:
    """Load persisted staged-streaming residency state for one model."""
    path = residency_state_path(model_id)
    if not path.exists():
        return StreamingResidencyState(
            model_id=model_id,
            residency_path=path,
            rotation_step=0,
            last_event="missing",
            blockers=["Streaming residency state does not exist yet."],
            ready=False,
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["residency_path"] = str(path)
    return _state_from_payload(payload)


def write_streaming_residency(
    model_id: str,
    schedule: WindowSchedule,
    *,
    last_event: str,
    consumed_unit_ids: list[str] | None = None,
    previous_state: StreamingResidencyState | None = None,
    blockers: list[str] | None = None,
    cache_hits_delta: int = 0,
    cache_misses_delta: int = 0,
    refill_count_delta: int | None = None,
) -> Path:
    """Persist the current residency and rotation position for one model."""
    path = residency_state_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    previous = previous_state or load_streaming_residency(model_id)
    hot_units = list(getattr(schedule, "hot_window_units", []))
    warm_units = list(getattr(schedule, "warm_window_units", []))
    overflow_units = list(getattr(schedule, "overflow_units", []))
    schedule_ready = bool(getattr(schedule, "ready", False))
    previous_rotation_step = previous.rotation_step if previous.ready else 0
    if last_event == "rotation":
        rotation_step = previous_rotation_step + 1
    else:
        rotation_step = previous_rotation_step

    previous_refill_count = previous.refill_count if previous.ready else 0
    if refill_count_delta is not None:
        refill_count = previous_refill_count + refill_count_delta
    elif last_event == "rotation":
        refill_count = previous_refill_count + len(warm_units)
    else:
        refill_count = previous_refill_count

    last_hit_delta = cache_hits_delta if last_event == "verification" else 0
    last_miss_delta = cache_misses_delta if last_event == "verification" else 0

    payload = StreamingResidencyState(
        model_id=model_id,
        residency_path=path,
        rotation_step=rotation_step,
        last_event=last_event,
        current_hot_unit_ids=[unit.unit_id for unit in hot_units],
        current_warm_unit_ids=[unit.unit_id for unit in warm_units],
        overflow_head_unit_id=overflow_units[0].unit_id if overflow_units else None,
        consumed_unit_ids=list(consumed_unit_ids or []),
        refill_count=refill_count,
        cache_hit_count=(previous.cache_hit_count if previous.ready else 0) + cache_hits_delta,
        cache_miss_count=(previous.cache_miss_count if previous.ready else 0) + cache_misses_delta,
        last_cache_hit_delta=last_hit_delta,
        last_cache_miss_delta=last_miss_delta,
        blockers=list(blockers or []),
        ready=not (blockers or []) and schedule_ready,
    )
    path.write_text(json.dumps(payload.to_dict(), indent=2), encoding="utf-8")
    return path
