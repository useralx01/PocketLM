"""Rotate staged-streaming hot and warm cache windows forward."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.runtime.streaming_materialize import StreamingMaterializationResult, _materialize_from_schedule
from pcketlm.core.runtime.streaming_reader import load_streaming_manifest
from pcketlm.core.runtime.streaming_residency import load_streaming_residency, write_streaming_residency
from pcketlm.core.runtime.streaming_units import build_streaming_unit_map
from pcketlm.core.runtime.window_schedule import ScheduledUnit, WindowSchedule, load_window_schedule
from pcketlm.core.storage.paths import streaming_model_root


@dataclass(slots=True)
class StreamingRotationResult:
    """Result of rotating the staged streaming windows forward."""

    model_id: str
    schedule_path: Path
    consumed_hot_units: list[str] = field(default_factory=list)
    promoted_warm_units: list[str] = field(default_factory=list)
    new_warm_units: list[str] = field(default_factory=list)
    overflow_remaining_units: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False
    materialization: StreamingMaterializationResult | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "schedule_path": str(self.schedule_path),
            "consumed_hot_units": list(self.consumed_hot_units),
            "promoted_warm_units": list(self.promoted_warm_units),
            "new_warm_units": list(self.new_warm_units),
            "overflow_remaining_units": list(self.overflow_remaining_units),
            "blockers": list(self.blockers),
            "ready": self.ready,
            "materialization": self.materialization.to_dict() if self.materialization else None,
        }


def _write_schedule(schedule: WindowSchedule) -> Path:
    schedule.schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.schedule_path.write_text(json.dumps(schedule.to_dict(), indent=2), encoding="utf-8")
    return schedule.schedule_path


def _sum_segment_bytes(unit_ids: list[str], unit_lookup: dict[str, object]) -> int:
    return sum(unit_lookup[unit_id].segment_bytes for unit_id in unit_ids if unit_id in unit_lookup)


def _fill_window(unit_ids: list[str], target_bytes: int, unit_lookup: dict[str, object]) -> list[str]:
    selected: list[str] = []
    current_bytes = 0
    for unit_id in unit_ids:
        unit = unit_lookup[unit_id]
        if current_bytes + unit.segment_bytes <= target_bytes:
            selected.append(unit_id)
            current_bytes += unit.segment_bytes
        else:
            break
    return selected


def _to_scheduled_units(unit_ids: list[str], unit_lookup: dict[str, object]) -> list[ScheduledUnit]:
    return [
        ScheduledUnit(
            unit_id=unit_lookup[unit_id].unit_id,
            shard_name=unit_lookup[unit_id].shard_name,
            segment_gb=unit_lookup[unit_id].segment_gb,
            segment_index=unit_lookup[unit_id].segment_index,
            total_segments=unit_lookup[unit_id].total_segments,
        )
        for unit_id in unit_ids
    ]


def rotate_window_schedule(model_id: str, model_dir: Path) -> StreamingRotationResult:
    """Rotate the current staged streaming windows forward and re-materialize cache files."""
    manifest = load_streaming_manifest(model_id)
    unit_map = build_streaming_unit_map(model_id, model_dir)
    current_schedule = load_window_schedule(model_id)
    schedule_path = streaming_model_root(model_id) / "schedule.json"
    blockers = list(manifest.blockers) + list(unit_map.blockers) + list(current_schedule.blockers)

    if not manifest.ready or not unit_map.ready or not current_schedule.ready:
        return StreamingRotationResult(
            model_id=model_id,
            schedule_path=schedule_path,
            blockers=blockers or ["Streaming rotation is blocked because required state is not ready."],
            ready=False,
        )

    unit_lookup = {unit.unit_id: unit for unit in unit_map.units}
    current_hot_ids = [unit.unit_id for unit in current_schedule.hot_window_units]
    current_warm_ids = [unit.unit_id for unit in current_schedule.warm_window_units]
    current_overflow_ids = [unit.unit_id for unit in current_schedule.overflow_units]

    if current_warm_ids:
        new_hot_ids = list(current_warm_ids)
    else:
        new_hot_ids = _fill_window(current_overflow_ids, current_schedule.hot_window_target_bytes, unit_lookup)

    remaining_overflow_source = [unit_id for unit_id in current_overflow_ids if unit_id not in new_hot_ids]
    new_warm_ids = _fill_window(remaining_overflow_source, current_schedule.warm_window_target_bytes, unit_lookup)
    new_overflow_ids = [unit_id for unit_id in remaining_overflow_source if unit_id not in new_warm_ids]

    if not new_hot_ids:
        blockers.append("No prefetched or overflow segment is available to rotate into the hot window.")

    rotated_schedule = WindowSchedule(
        model_id=model_id,
        schedule_path=schedule_path,
        hot_window_target_bytes=current_schedule.hot_window_target_bytes,
        warm_window_target_bytes=current_schedule.warm_window_target_bytes,
        hot_window_bytes=_sum_segment_bytes(new_hot_ids, unit_lookup),
        warm_window_bytes=_sum_segment_bytes(new_warm_ids, unit_lookup),
        hot_window_units=_to_scheduled_units(new_hot_ids, unit_lookup),
        warm_window_units=_to_scheduled_units(new_warm_ids, unit_lookup),
        overflow_units=_to_scheduled_units(new_overflow_ids, unit_lookup),
        blockers=blockers,
        ready=not blockers and bool(new_hot_ids),
    )
    _write_schedule(rotated_schedule)

    materialization = _materialize_from_schedule(
        model_id,
        manifest,
        rotated_schedule,
        unit_map,
        write_residency=False,
    )
    blockers = list(dict.fromkeys(list(rotated_schedule.blockers) + list(materialization.blockers)))
    previous_state = load_streaming_residency(model_id)
    write_streaming_residency(
        model_id,
        rotated_schedule,
        last_event="rotation",
        consumed_unit_ids=current_hot_ids,
        previous_state=previous_state,
        blockers=blockers,
        refill_count_delta=len(new_warm_ids),
    )

    return StreamingRotationResult(
        model_id=model_id,
        schedule_path=schedule_path,
        consumed_hot_units=current_hot_ids,
        promoted_warm_units=new_hot_ids,
        new_warm_units=new_warm_ids,
        overflow_remaining_units=new_overflow_ids,
        blockers=blockers,
        ready=not blockers and materialization.ready,
        materialization=materialization,
    )
