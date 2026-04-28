from pathlib import Path

from pcketlm.core.runtime.engine_selector import build_runtime_backend_report, select_runtime_engine
from pcketlm.core.runtime.strategy import plan_reduced_memory_strategy


def test_plan_reduced_memory_strategy_prefers_streaming_when_plain_cpu_is_not_viable(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime
    from pcketlm.core.runtime.load_attempt import MemorySnapshot

    monkeypatch.setattr(
        runtime.strategy,
        "build_acquisition_snapshot",
        lambda model_dir: type(
            "Acq",
            (),
            {"expected_bytes": 20 * 1024**3, "bytes_on_disk": 20 * 1024**3},
        )(),
    )
    monkeypatch.setattr(
        runtime.strategy,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=3 * 1024**3),
    )
    monkeypatch.setattr(runtime.strategy, "_disk_free_bytes", lambda path: 100 * 1024**3)

    plan = plan_reduced_memory_strategy("qwen-test", tmp_path)

    assert plan.recommended_strategy_id == "staged-disk-streaming"
    assert plan.options[0].viable_now is False
    assert plan.options[1].viable_now is True


def test_runtime_backend_report_recommends_gguf_when_no_gpu_packages(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage
    from pcketlm.core.runtime.load_attempt import MemorySnapshot

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(runtime.engine_selector, "_memory_snapshot", lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3))
    monkeypatch.setattr(runtime.engine_selector, "_cuda_devices", lambda: [])
    monkeypatch.setattr(runtime.engine_selector, "_module_available", lambda _name: False)
    monkeypatch.setattr(runtime.engine_selector, "_disk_free_gb", lambda _path: 100.0)

    report = build_runtime_backend_report("qwen-test")

    assert report.recommended_backend_id == "llama-cpp-gguf"
    by_id = {candidate.backend_id: candidate for candidate in report.candidates}
    assert by_id["direct-cpu"].implemented_now is True
    assert by_id["llama-cpp-gguf"].requires_model_conversion is True
    assert by_id["llama-cpp-gguf"].recommended is True


def test_runtime_backend_report_recommends_cuda_when_large_cuda_device_exists(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage
    from pcketlm.core.runtime.load_attempt import MemorySnapshot

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(runtime.engine_selector, "_memory_snapshot", lambda: MemorySnapshot(total_bytes=32 * 1024**3, free_bytes=20 * 1024**3))
    monkeypatch.setattr(
        runtime.engine_selector,
        "_cuda_devices",
        lambda: [{"index": 0, "name": "Test GPU", "total_vram_gb": 12.0, "backend": "cuda"}],
    )
    monkeypatch.setattr(runtime.engine_selector, "_module_available", lambda _name: False)

    report = build_runtime_backend_report("qwen-test")
    decision = select_runtime_engine("qwen-test")

    assert report.recommended_backend_id == "cuda"
    assert decision.selected_engine == "planned-gpu"
    assert decision.recommended_backend_id == "cuda"
