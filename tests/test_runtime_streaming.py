from pathlib import Path

from pcketlm.core.runtime.streaming import plan_staged_disk_streaming


def test_plan_staged_disk_streaming_builds_windows_and_cache_layout(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime
    from pcketlm.core.runtime.load_attempt import MemorySnapshot

    monkeypatch.setattr(
        runtime.streaming,
        "build_acquisition_snapshot",
        lambda model_dir: type(
            "Acq",
            (),
            {"expected_bytes": 24 * 1024**3, "bytes_on_disk": 24 * 1024**3},
        )(),
    )
    monkeypatch.setattr(
        runtime.streaming,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=3 * 1024**3),
    )

    plan = plan_staged_disk_streaming("qwen-test", tmp_path)

    assert plan.working_window_bytes > 0
    assert plan.prefetch_window_bytes > 0
    assert plan.estimated_chunks >= 1
    assert plan.cache_layout.cache_root.name == "qwen-test"
    assert plan.chunk_plan[0].chunk_id == "hot-window"
