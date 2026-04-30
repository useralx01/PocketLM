import json
from pathlib import Path
from types import SimpleNamespace

from pcketlm.core.benchmark.runs import (
    build_backend_comparison_record,
    build_measured_benchmark_history,
    run_lightweight_benchmark,
    run_gguf_measured_benchmark,
    run_measured_benchmark,
)


def test_build_backend_comparison_record_tags_measured_winners() -> None:
    run = {
        "cases": [
            {
                "label": "GGUF 4",
                "backend": "llama-cpp-gguf-server",
                "ready": True,
                "elapsed_seconds": 2.0,
                "max_new_tokens": 4,
                "generated_text": "fast",
                "peak_working_set_mb": 8600,
            },
            {
                "label": "Quality",
                "backend": "direct-cpu",
                "ready": True,
                "elapsed_seconds": 80.0,
                "max_new_tokens": 4,
                "generated_text": "quality",
                "peak_working_set_mb": 1200,
            },
            {
                "label": "Direct Boosted",
                "backend": "direct-cpu",
                "ready": True,
                "elapsed_seconds": 70.0,
                "max_new_tokens": 4,
                "generated_text": "boosted",
                "peak_working_set_mb": 1500,
            },
        ]
    }

    comparison = build_backend_comparison_record(run, recommended_backend_id="llama-cpp-gguf")

    tags = {item["tag"]: item["backend_id"] for item in comparison["tags"]}
    assert tags["fastest"] == "gguf"
    assert tags["best quality"] == "direct_boosted"
    assert tags["lowest RAM"] == "direct_standard"
    assert tags["recommended"] == "gguf"
    gguf = next(row for row in comparison["rows"] if row["backend_id"] == "gguf")
    assert gguf["seconds_per_token"] == 0.5


def test_build_backend_comparison_record_marks_missing_boosted_honestly() -> None:
    run = {
        "cases": [
            {
                "label": "Quality",
                "backend": "direct-cpu",
                "ready": True,
                "elapsed_seconds": 80.0,
                "max_new_tokens": 4,
                "generated_text": "quality",
            }
        ]
    }

    comparison = build_backend_comparison_record(run, recommended_backend_id="direct-cpu")

    boosted = next(row for row in comparison["rows"] if row["backend_id"] == "direct_boosted")
    tags = {item["tag"]: item["backend_id"] for item in comparison["tags"]}
    assert boosted["ready"] is False
    assert boosted["status"] == "needs-benchmark"
    assert tags["recommended"] == "direct_standard"


def test_run_lightweight_benchmark_persists_run_and_latest(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import benchmark, storage
    from pcketlm.core.benchmark.readiness import BenchmarkReadiness

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        benchmark.runs,
        "build_benchmark_readiness",
        lambda model_id, model_dir: BenchmarkReadiness(
            model_id=model_id,
            model_dir=model_dir,
            ready=True,
            status="Ready",
            summary="Benchmark setup is ready.",
            first_checks=["Load/readiness check"],
            blockers=[],
            warnings=["Alpha benchmark only."],
        ),
    )

    result = run_lightweight_benchmark("qwen-test", tmp_path / "model")

    assert result.ready is True
    assert result.runtime_settings["math_dtype"] == "bfloat16"
    assert result.runtime_settings["lm_head_chunk_rows"] == 8192
    assert set(result.tensor_residency) == {
        "hits",
        "misses",
        "stores",
        "evictions",
        "skips",
        "resident_bytes",
        "resident_count",
    }
    assert result.benchmark_path.exists()
    assert result.benchmark_path.name.endswith(".lightweight-benchmark.json")
    latest = tmp_path / "models" / "qwen-test" / "benchmarks" / "latest.lightweight-benchmark.json"
    assert latest.exists()
    latest_text = latest.read_text(encoding="utf-8")
    assert "Alpha benchmark only." in latest_text
    assert "tensor_residency" in latest_text
    assert "runtime_settings" in latest_text


def test_run_measured_benchmark_persists_timed_mode_cases(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import benchmark, storage
    from pcketlm.core.benchmark.readiness import BenchmarkReadiness

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        benchmark.runs,
        "build_benchmark_readiness",
        lambda model_id, model_dir: BenchmarkReadiness(
            model_id=model_id,
            model_dir=model_dir,
            ready=True,
            status="Ready",
            summary="Benchmark setup is ready.",
            first_checks=["Load/readiness check"],
            blockers=[],
            warnings=[],
        ),
    )

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        layer_count = kwargs.get("layer_count")
        if layer_count is None and kwargs.get("max_new_tokens") == 1:
            label = "Quick"
        elif layer_count is None and kwargs.get("max_new_tokens") == 2:
            label = "Agent"
        else:
            label = "Quality" if layer_count is None else ("Balanced" if layer_count == 32 else "Fast")
        return SimpleNamespace(
            ready=True,
            generated_text=f"{label} output",
            generated_token_ids=[1, 2],
            stop_reason="step-limit",
            blockers=[],
            timings={
                "total": 20.0,
                "prefill_stack": 9.0,
                "continuation_steps": 8.0,
                "prefill_decode_tail": 1.0,
                "continuation_decode_tail": 1.0,
                "prefill_stack_op_load_tensors": 3.0,
                "continuation_stack_op_load_tensors": 2.0,
            },
        )

    monkeypatch.setattr(benchmark.runs, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(benchmark.runs, "_run_gguf_benchmark_cases", lambda model_id: [])

    result = run_measured_benchmark("qwen-test", tmp_path / "model", max_new_tokens=4)

    assert result.ready is True
    assert [case.label for case in result.cases] == ["Quick", "Agent", "Fast", "Balanced", "Quality"]
    assert [case.layer_count for case in result.cases] == [None, None, 8, 32, None]
    assert result.cases[1].max_new_tokens == 2
    assert result.cases[4].generated_text == "Quality output"
    assert result.cases[4].timing_summary["stack_seconds"] == 17.0
    assert result.cases[4].timing_summary["tensor_load_seconds"] == 5.0
    assert result.cases[4].timing_summary["bottleneck"] == "prefill stack"
    assert result.benchmark_path.name.endswith(".measured-benchmark.json")
    latest = tmp_path / "models" / "qwen-test" / "benchmarks" / "latest.measured-benchmark.json"
    assert latest.exists()
    latest_text = latest.read_text(encoding="utf-8")
    assert "Fastest mode" in latest_text
    assert "Quality output" in latest_text


def test_run_measured_benchmark_treats_non_blocking_readiness_notes_as_warnings(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import benchmark, storage
    from pcketlm.core.benchmark.readiness import BenchmarkReadiness

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        benchmark.runs,
        "build_benchmark_readiness",
        lambda model_id, model_dir: BenchmarkReadiness(
            model_id=model_id,
            model_dir=model_dir,
            ready=True,
            status="Ready",
            summary="Streaming benchmark path is ready.",
            first_checks=[],
            blockers=["Plain CPU load needs more RAM."],
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        benchmark.runs,
        "run_prompt_decode_loop",
        lambda *args, **kwargs: SimpleNamespace(
            ready=True,
            generated_text="ok",
            generated_token_ids=[1],
            stop_reason="step-limit",
            blockers=[],
            timings={},
        ),
    )
    monkeypatch.setattr(benchmark.runs, "_run_gguf_benchmark_cases", lambda model_id: [])

    result = run_measured_benchmark("qwen-test", tmp_path / "model", max_new_tokens=4)

    assert result.ready is True
    assert result.blockers == []
    assert result.warnings == ["Readiness note: Plain CPU load needs more RAM."]


def test_run_measured_benchmark_includes_gguf_cases(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import benchmark, storage
    from pcketlm.core.benchmark.readiness import BenchmarkReadiness

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        benchmark.runs,
        "build_benchmark_readiness",
        lambda model_id, model_dir: BenchmarkReadiness(
            model_id=model_id,
            model_dir=model_dir,
            ready=True,
            status="Ready",
            summary="Benchmark setup is ready.",
            first_checks=[],
            blockers=[],
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        benchmark.runs,
        "run_prompt_decode_loop",
        lambda *args, **kwargs: SimpleNamespace(
            ready=True,
            generated_text="direct",
            generated_token_ids=[1],
            stop_reason="step-limit",
            blockers=[],
            timings={},
        ),
    )
    monkeypatch.setattr(
        benchmark.runs,
        "_run_gguf_benchmark_cases",
        lambda model_id: [
            benchmark.runs.MeasuredBenchmarkCase(
                label="GGUF 1",
                layer_count=None,
                elapsed_seconds=0.5,
                ready=True,
                generated_text="OK",
                backend="llama-cpp-gguf-server",
                prompt_kind="instruction",
                max_new_tokens=1,
            )
        ],
    )

    result = run_measured_benchmark("qwen-test", tmp_path / "model")

    assert [case.label for case in result.cases] == ["Quick", "Agent", "Fast", "Balanced", "Quality", "GGUF 1"]
    assert result.cases[-1].backend == "llama-cpp-gguf-server"
    assert result.cases[-1].prompt_kind == "instruction"


def test_run_gguf_measured_benchmark_persists_only_gguf_cases(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import benchmark, storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        benchmark.runs,
        "_run_gguf_benchmark_cases",
        lambda model_id: [
            benchmark.runs.MeasuredBenchmarkCase(
                label="GGUF Logic",
                layer_count=None,
                elapsed_seconds=0.8,
                ready=True,
                generated_text="YES",
                backend="llama-cpp-gguf-server",
                prompt_kind="logic",
                max_new_tokens=4,
            )
        ],
    )

    result = run_gguf_measured_benchmark("qwen-test", tmp_path / "model")

    assert result.ready is True
    assert result.runtime_settings["benchmark_scope"] == "gguf"
    assert result.cases[0].label == "GGUF Logic"
    assert result.cases[0].generated_text == "YES"
    assert (tmp_path / "models" / "qwen-test" / "benchmarks" / "latest.measured-benchmark.json").exists()


def test_build_measured_benchmark_history_summarizes_saved_runs(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    root = tmp_path / "models" / "qwen-test" / "benchmarks"
    root.mkdir(parents=True)
    for index, quality_seconds in enumerate([50.0, 40.0]):
        payload = {
            "run_id": f"run-{index}",
            "created_at": f"2026-04-28T00:00:0{index}+00:00",
            "ready": True,
            "summary": "ok",
            "cases": [
                {"label": "Fast", "elapsed_seconds": 10 + index, "ready": True, "generated_text": "fast"},
                {
                    "label": "Quality",
                    "elapsed_seconds": quality_seconds,
                    "ready": True,
                    "generated_text": "hello",
                    "timing_summary": {
                        "stack_seconds": quality_seconds - 2,
                        "tensor_load_seconds": 12 + index,
                        "decode_tail_seconds": 2,
                    },
                },
            ],
        }
        (root / f"run-{index}.measured-benchmark.json").write_text(json.dumps(payload), encoding="utf-8")

    history = build_measured_benchmark_history("qwen-test")

    quality = next(item for item in history["labels"] if item["label"] == "Quality")
    assert history["run_count"] == 2
    assert quality["best_seconds"] == 40.0
    assert quality["average_seconds"] == 45.0
    assert quality["worst_seconds"] == 50.0
    assert quality["average_stack_seconds"] == 43.0
    assert quality["average_tensor_load_seconds"] == 12.5
    assert quality["average_decode_tail_seconds"] == 2.0
