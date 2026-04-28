"""Map actual model weight files into staged streaming units."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.runtime.streaming_reader import load_streaming_manifest
from pcketlm.core.storage.paths import streaming_model_root


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(slots=True)
class StreamableWeightUnit:
    """One streamable source-weight unit."""

    unit_id: str
    shard_name: str
    shard_path: Path
    shard_bytes: int
    segment_index: int
    segment_offset_bytes: int
    segment_bytes: int
    segment_gb: float
    total_segments: int
    target_window: str

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "shard_name": self.shard_name,
            "shard_path": str(self.shard_path),
            "shard_bytes": self.shard_bytes,
            "segment_index": self.segment_index,
            "segment_offset_bytes": self.segment_offset_bytes,
            "segment_bytes": self.segment_bytes,
            "segment_gb": self.segment_gb,
            "total_segments": self.total_segments,
            "target_window": self.target_window,
        }


@dataclass(slots=True)
class StreamingUnitMap:
    """Runtime-usable mapping from source shard files to streaming units."""

    model_id: str
    model_dir: Path
    manifest_path: Path
    unit_map_path: Path
    units: list[StreamableWeightUnit] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "manifest_path": str(self.manifest_path),
            "unit_map_path": str(self.unit_map_path),
            "units": [unit.to_dict() for unit in self.units],
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


def _to_gb(value: int) -> float:
    return round(value / (1024**3), 2)


def _window_for_index(index: int) -> str:
    return "hot-window" if index == 0 else "warm-window"


def _segment_bytes(total_bytes: int, segment_index: int, total_segments: int) -> tuple[int, int]:
    base = total_bytes // total_segments
    remainder = total_bytes % total_segments
    segment_size = base + (1 if segment_index < remainder else 0)
    offset = 0
    for idx in range(segment_index):
        offset += base + (1 if idx < remainder else 0)
    return offset, segment_size


def build_streaming_unit_map(model_id: str, model_dir: Path) -> StreamingUnitMap:
    """Build a streamable unit map from the actual source shard files."""
    manifest = load_streaming_manifest(model_id)
    unit_map_path = streaming_model_root(model_id) / "units.json"
    blockers = list(manifest.blockers)

    if not manifest.ready:
        return StreamingUnitMap(
            model_id=model_id,
            model_dir=model_dir,
            manifest_path=manifest.manifest_path,
            unit_map_path=unit_map_path,
            units=[],
            blockers=blockers or ["Streaming manifest is not ready yet."],
            ready=False,
        )

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        blockers.append("Model weight index is missing.")
        return StreamingUnitMap(
            model_id=model_id,
            model_dir=model_dir,
            manifest_path=manifest.manifest_path,
            unit_map_path=unit_map_path,
            units=[],
            blockers=blockers,
            ready=False,
        )

    payload = _read_json(index_path)
    weight_map = payload.get("weight_map") or {}
    shard_names = sorted(set(weight_map.values()))
    units: list[StreamableWeightUnit] = []

    for index, shard_name in enumerate(shard_names):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            blockers.append(f"Shard file is missing: {shard_name}")
            continue

        shard_bytes = shard_path.stat().st_size
        segment_budget_bytes = max(1, manifest.segment_budget_bytes)
        total_segments = max(1, -(-shard_bytes // segment_budget_bytes))
        for segment_index in range(total_segments):
            offset, segment_bytes = _segment_bytes(shard_bytes, segment_index, total_segments)
            units.append(
                StreamableWeightUnit(
                    unit_id=f"unit-{index + 1:04d}-seg-{segment_index + 1:03d}",
                    shard_name=shard_name,
                    shard_path=shard_path,
                    shard_bytes=shard_bytes,
                    segment_index=segment_index,
                    segment_offset_bytes=offset,
                    segment_bytes=segment_bytes,
                    segment_gb=_to_gb(segment_bytes),
                    total_segments=total_segments,
                    target_window=_window_for_index(index if segment_index == 0 else 1),
                )
            )

    return StreamingUnitMap(
        model_id=model_id,
        model_dir=model_dir,
        manifest_path=manifest.manifest_path,
        unit_map_path=unit_map_path,
        units=units,
        blockers=blockers,
        ready=not blockers and bool(units),
    )


def write_streaming_unit_map(model_id: str, model_dir: Path) -> Path:
    """Persist the streamable unit map to disk."""
    unit_map = build_streaming_unit_map(model_id, model_dir)
    unit_map.unit_map_path.parent.mkdir(parents=True, exist_ok=True)
    unit_map.unit_map_path.write_text(json.dumps(unit_map.to_dict(), indent=2), encoding="utf-8")
    return unit_map.unit_map_path
