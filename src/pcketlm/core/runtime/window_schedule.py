"""Hot/warm window scheduling for staged streaming."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.runtime.streaming_reader import load_streaming_manifest
from pcketlm.core.runtime.streaming_units import build_streaming_unit_map
from pcketlm.core.storage.paths import streaming_model_root


@dataclass(slots=True)
class ScheduledUnit:
    """One scheduled streamable unit."""

    unit_id: str
    shard_name: str
    segment_gb: float
    segment_index: int
    total_segments: int

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "shard_name": self.shard_name,
            "segment_gb": self.segment_gb,
            "segment_index": self.segment_index,
            "total_segments": self.total_segments,
        }


@dataclass(slots=True)
class WindowSchedule:
    """One staged-streaming hot/warm window schedule."""

    model_id: str
    schedule_path: Path
    hot_window_target_bytes: int
    warm_window_target_bytes: int
    hot_window_bytes: int
    warm_window_bytes: int
    hot_window_units: list[ScheduledUnit] = field(default_factory=list)
    warm_window_units: list[ScheduledUnit] = field(default_factory=list)
    overflow_units: list[ScheduledUnit] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "schedule_path": str(self.schedule_path),
            "hot_window_target_bytes": self.hot_window_target_bytes,
            "warm_window_target_bytes": self.warm_window_target_bytes,
            "hot_window_bytes": self.hot_window_bytes,
            "warm_window_bytes": self.warm_window_bytes,
            "hot_window_units": [unit.to_dict() for unit in self.hot_window_units],
            "warm_window_units": [unit.to_dict() for unit in self.warm_window_units],
            "overflow_units": [unit.to_dict() for unit in self.overflow_units],
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


def _schedule_from_payload(payload: dict) -> WindowSchedule:
    return WindowSchedule(
        model_id=str(payload.get("model_id", "")),
        schedule_path=Path(payload.get("schedule_path", "")),
        hot_window_target_bytes=int(payload.get("hot_window_target_bytes", 0)),
        warm_window_target_bytes=int(payload.get("warm_window_target_bytes", 0)),
        hot_window_bytes=int(payload.get("hot_window_bytes", 0)),
        warm_window_bytes=int(payload.get("warm_window_bytes", 0)),
        hot_window_units=[
            ScheduledUnit(
                unit_id=str(unit.get("unit_id", "")),
                shard_name=str(unit.get("shard_name", "")),
                segment_gb=float(unit.get("segment_gb", 0.0)),
                segment_index=int(unit.get("segment_index", 0)),
                total_segments=int(unit.get("total_segments", 0)),
            )
            for unit in payload.get("hot_window_units", [])
        ],
        warm_window_units=[
            ScheduledUnit(
                unit_id=str(unit.get("unit_id", "")),
                shard_name=str(unit.get("shard_name", "")),
                segment_gb=float(unit.get("segment_gb", 0.0)),
                segment_index=int(unit.get("segment_index", 0)),
                total_segments=int(unit.get("total_segments", 0)),
            )
            for unit in payload.get("warm_window_units", [])
        ],
        overflow_units=[
            ScheduledUnit(
                unit_id=str(unit.get("unit_id", "")),
                shard_name=str(unit.get("shard_name", "")),
                segment_gb=float(unit.get("segment_gb", 0.0)),
                segment_index=int(unit.get("segment_index", 0)),
                total_segments=int(unit.get("total_segments", 0)),
            )
            for unit in payload.get("overflow_units", [])
        ],
        blockers=list(payload.get("blockers", [])),
        ready=bool(payload.get("ready", False)),
    )


def _scheduled_unit(unit) -> ScheduledUnit:
    return ScheduledUnit(
        unit_id=unit.unit_id,
        shard_name=unit.shard_name,
        segment_gb=unit.segment_gb,
        segment_index=unit.segment_index,
        total_segments=unit.total_segments,
    )


def _add_if_fits(units, current_bytes: int, target_bytes: int):
    selected = []
    for unit in units:
        if current_bytes + unit.segment_bytes <= target_bytes:
            selected.append(unit)
            current_bytes += unit.segment_bytes
        else:
            break
    return selected, current_bytes


def build_window_schedule(model_id: str, model_dir: Path) -> WindowSchedule:
    """Build the first hot/warm window schedule from the unit map and manifest."""
    manifest = load_streaming_manifest(model_id)
    unit_map = build_streaming_unit_map(model_id, model_dir)
    schedule_path = streaming_model_root(model_id) / "schedule.json"
    blockers = list(manifest.blockers) + list(unit_map.blockers)

    if not manifest.ready or not unit_map.ready:
        return WindowSchedule(
            model_id=model_id,
            schedule_path=schedule_path,
            hot_window_target_bytes=manifest.working_window_bytes,
            warm_window_target_bytes=manifest.prefetch_window_bytes,
            hot_window_bytes=0,
            warm_window_bytes=0,
            blockers=blockers or ["Streaming manifest or unit map is not ready yet."],
            ready=False,
        )

    hot_candidates = [unit for unit in unit_map.units if unit.target_window == "hot-window"]
    warm_candidates = [unit for unit in unit_map.units if unit.target_window == "warm-window"]

    hot_selected, hot_bytes = _add_if_fits(hot_candidates, 0, manifest.working_window_bytes)
    warm_selected, warm_bytes = _add_if_fits(warm_candidates, 0, manifest.prefetch_window_bytes)

    remaining_ids = {unit.unit_id for unit in hot_selected + warm_selected}
    overflow = [unit for unit in unit_map.units if unit.unit_id not in remaining_ids]

    if not hot_selected:
        blockers.append("No shard unit fits inside the current hot-window budget.")

    return WindowSchedule(
        model_id=model_id,
        schedule_path=schedule_path,
        hot_window_target_bytes=manifest.working_window_bytes,
        warm_window_target_bytes=manifest.prefetch_window_bytes,
        hot_window_bytes=hot_bytes,
        warm_window_bytes=warm_bytes,
        hot_window_units=[_scheduled_unit(unit) for unit in hot_selected],
        warm_window_units=[_scheduled_unit(unit) for unit in warm_selected],
        overflow_units=[_scheduled_unit(unit) for unit in overflow],
        blockers=blockers,
        ready=not blockers,
    )


def load_window_schedule(model_id: str) -> WindowSchedule:
    """Load the persisted staged streaming hot/warm schedule for one model."""
    schedule_path = streaming_model_root(model_id) / "schedule.json"
    if not schedule_path.exists():
        return WindowSchedule(
            model_id=model_id,
            schedule_path=schedule_path,
            hot_window_target_bytes=0,
            warm_window_target_bytes=0,
            hot_window_bytes=0,
            warm_window_bytes=0,
            blockers=["Window schedule does not exist yet."],
            ready=False,
        )

    payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    payload["schedule_path"] = str(schedule_path)
    return _schedule_from_payload(payload)


def write_window_schedule(model_id: str, model_dir: Path) -> Path:
    """Persist the current window schedule to disk."""
    schedule = build_window_schedule(model_id, model_dir)
    schedule.schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.schedule_path.write_text(json.dumps(schedule.to_dict(), indent=2), encoding="utf-8")
    return schedule.schedule_path
