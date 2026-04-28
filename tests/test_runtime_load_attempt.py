from pathlib import Path

from pcketlm.core.runtime.load_attempt import MemorySnapshot, attempt_real_model_load


def test_attempt_real_model_load_blocks_when_memory_too_low(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime

    monkeypatch.setattr(
        runtime.load_attempt,
        "build_runtime_bootstrap",
        lambda model_id, model_dir: type(
            "Bootstrap",
            (),
            {"blockers": [], "warnings": []},
        )(),
    )
    monkeypatch.setattr(
        runtime.load_attempt,
        "build_acquisition_snapshot",
        lambda model_dir: type(
            "Acq",
            (),
            {"expected_bytes": 10 * 1024**3, "bytes_on_disk": 10 * 1024**3},
        )(),
    )
    monkeypatch.setattr(
        runtime.load_attempt,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=1 * 1024**3),
    )

    result = attempt_real_model_load("qwen-test", tmp_path)

    assert result.load_attempted is False
    assert result.load_succeeded is False
    assert result.blocker_category == "memory"
    assert result.blocker_severity == "high"
    assert result.recommended_action is not None
    assert any("Free RAM" in blocker for blocker in result.blockers)
