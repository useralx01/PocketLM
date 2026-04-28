from pathlib import Path

from pcketlm.core.benchmark.readiness import build_benchmark_readiness


def test_build_benchmark_readiness_reports_ready_when_streaming_or_runtime_ready(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import benchmark

    monkeypatch.setattr(
        benchmark.readiness,
        "build_runtime_bootstrap",
        lambda model_id, model_dir: type(
            "Bootstrap",
            (),
            {
                "to_dict": lambda self: {
                    "source": {"ready": True},
                    "can_attempt_load": False,
                    "blockers": [],
                    "warnings": [],
                }
            },
        )(),
    )
    monkeypatch.setattr(
        benchmark.readiness,
        "attempt_real_model_load",
        lambda model_id, model_dir: type(
            "Load",
            (),
            {"to_dict": lambda self: {"blockers": ["Plain CPU load needs more RAM."], "warnings": []}},
        )(),
    )
    monkeypatch.setattr(
        benchmark.readiness,
        "build_streaming_control_state",
        lambda model_id: type(
            "Streaming",
            (),
            {"ready": True, "blockers": [], "warnings": []},
        )(),
    )
    monkeypatch.setattr(
        benchmark.readiness,
        "plan_reduced_memory_strategy",
        lambda model_id, model_dir: type("Strategy", (), {"recommended_strategy_id": "staged-disk-streaming"})(),
    )

    result = build_benchmark_readiness("qwen-test", tmp_path)

    assert result.ready is True
    assert result.status == "Ready"
    assert "staged-disk-streaming" in result.summary
    assert "Short prompt quality check" in result.first_checks
    assert result.blockers == ["Plain CPU load needs more RAM."]


def test_build_benchmark_readiness_reports_blocked_when_source_is_not_ready(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import benchmark

    monkeypatch.setattr(
        benchmark.readiness,
        "build_runtime_bootstrap",
        lambda model_id, model_dir: type(
            "Bootstrap",
            (),
            {
                "to_dict": lambda self: {
                    "source": {"ready": False},
                    "can_attempt_load": False,
                    "blockers": ["Missing tokenizer."],
                    "warnings": [],
                }
            },
        )(),
    )
    monkeypatch.setattr(
        benchmark.readiness,
        "attempt_real_model_load",
        lambda model_id, model_dir: type("Load", (), {"to_dict": lambda self: {"blockers": [], "warnings": []}})(),
    )
    monkeypatch.setattr(
        benchmark.readiness,
        "build_streaming_control_state",
        lambda model_id: type("Streaming", (), {"ready": False, "blockers": [], "warnings": []})(),
    )
    monkeypatch.setattr(
        benchmark.readiness,
        "plan_reduced_memory_strategy",
        lambda model_id, model_dir: type("Strategy", (), {"recommended_strategy_id": "staged-disk-streaming"})(),
    )

    result = build_benchmark_readiness("qwen-test", tmp_path)

    assert result.ready is False
    assert result.status == "Blocked"
    assert "Missing tokenizer." in result.summary
    assert result.blockers == ["Missing tokenizer."]
