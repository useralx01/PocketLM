"""Materialize scheduled streaming segments into cache state."""

from __future__ import annotations

import json
from hashlib import sha256
from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.runtime.streaming_reader import load_streaming_manifest
from pcketlm.core.runtime.streaming_residency import load_streaming_residency, write_streaming_residency
from pcketlm.core.runtime.streaming_units import build_streaming_unit_map
from pcketlm.core.runtime.window_schedule import WindowSchedule, build_window_schedule, load_window_schedule


@dataclass(slots=True)
class MaterializedSegment:
    """One cached segment written from a source shard."""

    unit_id: str
    window: str
    cache_path: Path
    shard_name: str
    segment_index: int
    bytes_written: int
    checksum_sha256: str
    verified: bool = True

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "window": self.window,
            "cache_path": str(self.cache_path),
            "shard_name": self.shard_name,
            "segment_index": self.segment_index,
            "bytes_written": self.bytes_written,
            "checksum_sha256": self.checksum_sha256,
            "verified": self.verified,
        }


@dataclass(slots=True)
class StreamingMaterializationResult:
    """One cache materialization pass for the current window schedule."""

    model_id: str
    cache_root: Path
    cache_index_path: Path
    hot_window_dir: Path
    warm_window_dir: Path
    hot_segments: list[MaterializedSegment] = field(default_factory=list)
    warm_segments: list[MaterializedSegment] = field(default_factory=list)
    verification_mode: str = "write-boundary"
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "cache_root": str(self.cache_root),
            "cache_index_path": str(self.cache_index_path),
            "hot_window_dir": str(self.hot_window_dir),
            "warm_window_dir": str(self.warm_window_dir),
            "hot_segments": [segment.to_dict() for segment in self.hot_segments],
            "warm_segments": [segment.to_dict() for segment in self.warm_segments],
            "verification_mode": self.verification_mode,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


def _cache_file_path(window_dir: Path, unit_id: str) -> Path:
    return window_dir / f"{unit_id}.bin"


def _clear_window_dir(window_dir: Path) -> None:
    window_dir.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    for path in window_dir.iterdir():
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                blockers.append(f"Unable to clear cached segment {path.name}: {exc}")
    return blockers


def _materialize_unit(unit, window: str, window_dir: Path) -> MaterializedSegment:
    cache_path = _cache_file_path(window_dir, unit.unit_id)

    with unit.shard_path.open("rb") as handle:
        handle.seek(unit.segment_offset_bytes)
        payload = handle.read(unit.segment_bytes)

    if len(payload) != unit.segment_bytes:
        raise ValueError(
            f"Scheduled segment {unit.unit_id} expected {unit.segment_bytes} bytes "
            f"but only read {len(payload)} bytes from {unit.shard_name}."
        )

    cache_path.write_bytes(payload)
    checksum_sha256 = sha256(payload).hexdigest()
    return MaterializedSegment(
        unit_id=unit.unit_id,
        window=window,
        cache_path=cache_path,
        shard_name=unit.shard_name,
        segment_index=unit.segment_index,
        bytes_written=len(payload),
        checksum_sha256=checksum_sha256,
        verified=True,
    )


def _write_cache_index(result: StreamingMaterializationResult) -> Path:
    result.cache_root.mkdir(parents=True, exist_ok=True)
    result.cache_index_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result.cache_index_path


def _materialize_from_schedule(
    model_id: str,
    manifest,
    schedule: WindowSchedule,
    unit_map,
    *,
    write_residency: bool = True,
) -> StreamingMaterializationResult:
    cache_root = manifest.cache_root
    cache_index_path = cache_root / "cache-index.json"
    blockers = list(manifest.blockers) + list(schedule.blockers) + list(unit_map.blockers)

    if not manifest.ready or not schedule.ready or not unit_map.ready:
        result = StreamingMaterializationResult(
            model_id=model_id,
            cache_root=cache_root,
            cache_index_path=cache_index_path,
            hot_window_dir=manifest.hot_window_dir,
            warm_window_dir=manifest.warm_window_dir,
            blockers=blockers or ["Streaming materialization is blocked because required state is not ready."],
            ready=False,
        )
        _write_cache_index(result)
        return result

    unit_lookup = {unit.unit_id: unit for unit in unit_map.units}

    blockers.extend(_clear_window_dir(manifest.hot_window_dir))
    blockers.extend(_clear_window_dir(manifest.warm_window_dir))
    if blockers:
        result = StreamingMaterializationResult(
            model_id=model_id,
            cache_root=cache_root,
            cache_index_path=cache_index_path,
            hot_window_dir=manifest.hot_window_dir,
            warm_window_dir=manifest.warm_window_dir,
            blockers=blockers,
            ready=False,
        )
        _write_cache_index(result)
        return result

    hot_segments: list[MaterializedSegment] = []
    warm_segments: list[MaterializedSegment] = []

    for scheduled in schedule.hot_window_units:
        unit = unit_lookup.get(scheduled.unit_id)
        if unit is None:
            blockers.append(f"Scheduled hot-window unit is missing from the unit map: {scheduled.unit_id}")
            continue
        try:
            hot_segments.append(_materialize_unit(unit, "hot-window", manifest.hot_window_dir))
        except (OSError, ValueError) as exc:
            blockers.append(str(exc))

    for scheduled in schedule.warm_window_units:
        unit = unit_lookup.get(scheduled.unit_id)
        if unit is None:
            blockers.append(f"Scheduled warm-window unit is missing from the unit map: {scheduled.unit_id}")
            continue
        try:
            warm_segments.append(_materialize_unit(unit, "warm-window", manifest.warm_window_dir))
        except (OSError, ValueError) as exc:
            blockers.append(str(exc))

    result = StreamingMaterializationResult(
        model_id=model_id,
        cache_root=cache_root,
        cache_index_path=cache_index_path,
        hot_window_dir=manifest.hot_window_dir,
        warm_window_dir=manifest.warm_window_dir,
        hot_segments=hot_segments,
        warm_segments=warm_segments,
        blockers=blockers,
        ready=not blockers and bool(hot_segments),
    )
    _write_cache_index(result)
    if write_residency:
        write_streaming_residency(
            model_id,
            schedule,
            last_event="materialization",
            consumed_unit_ids=[],
            blockers=blockers,
        )
    return result


def verify_materialized_cache(model_id: str) -> StreamingMaterializationResult:
    """Verify cached segments recorded in the materialization index."""
    manifest = load_streaming_manifest(model_id)
    cache_root = manifest.cache_root
    cache_index_path = cache_root / "cache-index.json"
    blockers = list(manifest.blockers)

    if not cache_index_path.exists():
        return StreamingMaterializationResult(
            model_id=model_id,
            cache_root=cache_root,
            cache_index_path=cache_index_path,
            hot_window_dir=manifest.hot_window_dir,
            warm_window_dir=manifest.warm_window_dir,
            verification_mode="index-check",
            blockers=["Cache index does not exist yet."],
            ready=False,
        )

    payload = json.loads(cache_index_path.read_text(encoding="utf-8"))
    hot_segments: list[MaterializedSegment] = []
    warm_segments: list[MaterializedSegment] = []
    hit_count = 0
    miss_count = 0

    for window_key, target in (("hot_segments", hot_segments), ("warm_segments", warm_segments)):
        for item in payload.get(window_key, []):
            cache_path = Path(item["cache_path"])
            verified = True
            if not cache_path.exists():
                blockers.append(f"Cached segment file is missing: {cache_path.name}")
                verified = False
                actual_bytes = 0
                checksum_value = str(item.get("checksum_sha256", ""))
            else:
                data = cache_path.read_bytes()
                actual_bytes = len(data)
                checksum_value = sha256(data).hexdigest()
                if actual_bytes != int(item.get("bytes_written", 0)):
                    blockers.append(f"Cached segment {cache_path.name} has {actual_bytes} bytes, expected {item.get('bytes_written', 0)}.")
                    verified = False
                if checksum_value != str(item.get("checksum_sha256", "")):
                    blockers.append(f"Cached segment {cache_path.name} checksum does not match the cache index.")
                    verified = False

            if verified:
                hit_count += 1
            else:
                miss_count += 1

            target.append(
                MaterializedSegment(
                    unit_id=str(item.get("unit_id", "")),
                    window=str(item.get("window", "")),
                    cache_path=cache_path,
                    shard_name=str(item.get("shard_name", "")),
                    segment_index=int(item.get("segment_index", 0)),
                    bytes_written=actual_bytes,
                    checksum_sha256=checksum_value,
                    verified=verified,
                )
            )

    result = StreamingMaterializationResult(
        model_id=model_id,
        cache_root=cache_root,
        cache_index_path=cache_index_path,
        hot_window_dir=manifest.hot_window_dir,
        warm_window_dir=manifest.warm_window_dir,
        hot_segments=hot_segments,
        warm_segments=warm_segments,
        verification_mode="index-check",
        blockers=blockers,
        ready=not blockers and bool(hot_segments),
    )
    schedule = load_window_schedule(model_id)
    previous_state = load_streaming_residency(model_id)
    if schedule.ready:
        write_streaming_residency(
            model_id,
            schedule,
            last_event="verification",
            consumed_unit_ids=previous_state.consumed_unit_ids,
            previous_state=previous_state,
            blockers=blockers,
            cache_hits_delta=hit_count,
            cache_misses_delta=miss_count,
        )
    return result


def materialize_window_schedule(model_id: str, model_dir: Path) -> StreamingMaterializationResult:
    """Materialize the current hot/warm schedule into cache files."""
    manifest = load_streaming_manifest(model_id)
    unit_map = build_streaming_unit_map(model_id, model_dir)
    persisted_schedule = load_window_schedule(model_id)
    schedule = persisted_schedule if persisted_schedule.ready else build_window_schedule(model_id, model_dir)
    return _materialize_from_schedule(model_id, manifest, schedule, unit_map)
