import json
from pathlib import Path

from pcketlm.core.runtime.streaming_residency import load_streaming_residency
from pcketlm.core.runtime.streaming_rotation import rotate_window_schedule


def test_materialization_writes_residency_state(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage
    from pcketlm.core.runtime.load_attempt import MemorySnapshot
    from pcketlm.core.runtime.streaming import StreamingCacheLayout, StreamingChunkPlan, StreamingPlan
    from pcketlm.core.runtime.streaming_materialize import materialize_window_schedule
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

    materialize_window_schedule(model_id, model_dir)
    residency = load_streaming_residency(model_id)

    assert residency.ready is True
    assert residency.last_event == "materialization"
    assert residency.current_hot_unit_ids == ["unit-0001-seg-001"]
    assert residency.current_warm_unit_ids == ["unit-0001-seg-002"]
    assert residency.overflow_head_unit_id == "unit-0001-seg-003"


def test_rotation_updates_residency_state(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage
    from pcketlm.core.runtime.load_attempt import MemorySnapshot
    from pcketlm.core.runtime.streaming import StreamingCacheLayout, StreamingChunkPlan, StreamingPlan
    from pcketlm.core.runtime.streaming_materialize import materialize_window_schedule
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

    materialize_window_schedule(model_id, model_dir)
    rotate_window_schedule(model_id, model_dir)
    residency = load_streaming_residency(model_id)

    assert residency.ready is True
    assert residency.last_event == "rotation"
    assert residency.rotation_step == 1
    assert residency.current_hot_unit_ids == ["unit-0001-seg-002"]
    assert residency.current_warm_unit_ids == ["unit-0001-seg-003"]
    assert residency.overflow_head_unit_id == "unit-0001-seg-004"
    assert residency.consumed_unit_ids == ["unit-0001-seg-001"]
    assert residency.refill_count >= 1


def test_verification_preserves_rotation_state_and_only_adds_cache_hits(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage
    from pcketlm.core.runtime.load_attempt import MemorySnapshot
    from pcketlm.core.runtime.streaming import StreamingCacheLayout, StreamingChunkPlan, StreamingPlan
    from pcketlm.core.runtime.streaming_materialize import materialize_window_schedule, verify_materialized_cache
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

    materialize_window_schedule(model_id, model_dir)
    rotate_window_schedule(model_id, model_dir)
    before_verify = load_streaming_residency(model_id)

    verify_materialized_cache(model_id)
    after_verify = load_streaming_residency(model_id)

    assert before_verify.ready is True
    assert before_verify.last_event == "rotation"
    assert before_verify.rotation_step == 1
    assert after_verify.ready is True
    assert after_verify.last_event == "verification"
    assert after_verify.rotation_step == before_verify.rotation_step
    assert after_verify.refill_count == before_verify.refill_count
    assert after_verify.current_hot_unit_ids == before_verify.current_hot_unit_ids
    assert after_verify.current_warm_unit_ids == before_verify.current_warm_unit_ids
    assert after_verify.consumed_unit_ids == before_verify.consumed_unit_ids
    assert after_verify.cache_hit_count == before_verify.cache_hit_count + 2
    assert after_verify.cache_miss_count == before_verify.cache_miss_count
