import json
from pathlib import Path

from pcketlm.core.runtime.streaming_control import (
    advance_streaming_runtime,
    advance_streaming_runtime_safely,
    build_streaming_control_state,
)
from pcketlm.core.runtime.streaming_materialize import materialize_window_schedule, verify_materialized_cache
from pcketlm.core.runtime.streaming_residency import load_streaming_residency
from pcketlm.core.runtime.streaming_rotation import rotate_window_schedule


def _bootstrap_streaming_fixture(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
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
    return model_id, model_dir


def test_streaming_control_recommends_rotation_for_healthy_cache(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _bootstrap_streaming_fixture(tmp_path, monkeypatch)

    materialize_window_schedule(model_id, model_dir)
    state = build_streaming_control_state(model_id)

    assert state.ready is True
    assert state.action_key == "rotate-forward"
    assert state.current_hot_unit_ids == ["unit-0001-seg-001"]
    assert state.current_warm_unit_ids == ["unit-0001-seg-002"]


def test_streaming_control_recommends_repair_after_verification_miss(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _bootstrap_streaming_fixture(tmp_path, monkeypatch)

    materialize_window_schedule(model_id, model_dir)
    hot_cache = tmp_path / "state" / "streaming" / model_id / "hot-window" / "unit-0001-seg-001.bin"
    hot_cache.write_bytes(b"broken")
    verify_materialized_cache(model_id)

    state = build_streaming_control_state(model_id)
    assert state.ready is True
    assert state.action_key == "repair-cache-window"
    assert state.last_cache_miss_delta == 1

    result = advance_streaming_runtime(model_id, model_dir)
    repaired = load_streaming_residency(model_id)

    assert result.executed is True
    assert result.action_key == "repair-cache-window"
    assert repaired.last_event == "materialization"
    assert repaired.last_cache_miss_delta == 0


def test_streaming_control_advance_rotates_when_cache_is_healthy(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _bootstrap_streaming_fixture(tmp_path, monkeypatch)

    materialize_window_schedule(model_id, model_dir)
    result = advance_streaming_runtime(model_id, model_dir)
    residency = load_streaming_residency(model_id)

    assert result.executed is True
    assert result.action_key == "rotate-forward"
    assert residency.rotation_step == 1
    assert residency.current_hot_unit_ids == ["unit-0001-seg-002"]
    assert residency.current_warm_unit_ids == ["unit-0001-seg-003"]


def test_safe_advance_repairs_cache_before_rotating(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _bootstrap_streaming_fixture(tmp_path, monkeypatch)

    materialize_window_schedule(model_id, model_dir)
    hot_cache = tmp_path / "state" / "streaming" / model_id / "hot-window" / "unit-0001-seg-001.bin"
    hot_cache.write_bytes(b"broken")

    result = advance_streaming_runtime_safely(model_id, model_dir)
    residency = load_streaming_residency(model_id)

    assert result.executed is True
    assert result.action_key == "safe-repair-cache-window"
    assert result.verification is not None
    assert result.materialization is not None
    assert result.rotation is None
    assert residency.last_event == "materialization"
    assert residency.rotation_step == 0
    assert residency.last_cache_miss_delta == 0


def test_safe_advance_verifies_then_rotates_when_cache_is_healthy(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _bootstrap_streaming_fixture(tmp_path, monkeypatch)

    materialize_window_schedule(model_id, model_dir)
    result = advance_streaming_runtime_safely(model_id, model_dir)
    residency = load_streaming_residency(model_id)

    assert result.executed is True
    assert result.action_key == "safe-rotate-forward"
    assert result.verification is not None
    assert result.rotation is not None
    assert residency.rotation_step == 1
    assert residency.current_hot_unit_ids == ["unit-0001-seg-002"]
