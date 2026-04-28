import json
from pathlib import Path

from pcketlm.core.runtime.streaming_rotation import rotate_window_schedule
from pcketlm.core.runtime.window_schedule import load_window_schedule


def test_rotate_window_schedule_promotes_warm_and_refills_warm(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage
    from pcketlm.core.runtime.load_attempt import MemorySnapshot
    from pcketlm.core.runtime.streaming import StreamingCacheLayout, StreamingChunkPlan, StreamingPlan
    from pcketlm.core.runtime.streaming_state import write_streaming_manifest
    from pcketlm.core.runtime.streaming_units import write_streaming_unit_map
    from pcketlm.core.runtime.window_schedule import write_window_schedule

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    model_id = "qwen-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)
    shard_path = model_dir / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"abcdefghijklmnopqrstuvwxyz")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shard_path.name}}),
        encoding="utf-8",
    )

    cache_root = tmp_path / "state" / "streaming" / model_id
    hot_dir = cache_root / "hot-window"
    warm_dir = cache_root / "warm-window"
    hot_dir.mkdir(parents=True)
    warm_dir.mkdir(parents=True)

    plan = StreamingPlan(
        model_id=model_id,
        model_dir=model_dir,
        memory=MemorySnapshot(total_bytes=32, free_bytes=16),
        model_bytes=26,
        model_gb=0.0,
        working_window_bytes=12,
        working_window_gb=0.0,
        prefetch_window_bytes=6,
        prefetch_window_gb=0.0,
        segment_budget_bytes=6,
        segment_budget_gb=0.0,
        estimated_chunks=5,
        cache_layout=StreamingCacheLayout(
            cache_root=cache_root,
            manifest_path=cache_root / "manifest.json",
            hot_window_dir=hot_dir,
            warm_window_dir=warm_dir,
        ),
        chunk_plan=[
            StreamingChunkPlan(chunk_id="hot-window", target_bytes=12, target_gb=0.0, purpose="active"),
            StreamingChunkPlan(chunk_id="warm-prefetch", target_bytes=6, target_gb=0.0, purpose="prefetch"),
        ],
        viability="viable",
        summary="test plan",
        blockers=[],
    )
    write_streaming_manifest(plan)
    write_streaming_unit_map(model_id, model_dir)
    write_window_schedule(model_id, model_dir)

    rotation = rotate_window_schedule(model_id, model_dir)

    assert rotation.ready is True
    assert rotation.consumed_hot_units == ["unit-0001-seg-001"]
    assert rotation.promoted_warm_units == ["unit-0001-seg-002"]
    assert rotation.new_warm_units == ["unit-0001-seg-003"]

    rotated_schedule = load_window_schedule(model_id)
    assert rotated_schedule.hot_window_units[0].unit_id == "unit-0001-seg-002"
    assert rotated_schedule.warm_window_units[0].unit_id == "unit-0001-seg-003"
    assert (hot_dir / "unit-0001-seg-002.bin").read_bytes() == b"ghijk"
    assert (warm_dir / "unit-0001-seg-003.bin").read_bytes() == b"lmnop"
