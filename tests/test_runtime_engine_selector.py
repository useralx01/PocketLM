from types import SimpleNamespace

from pcketlm.core.runtime import engine_selector
from pcketlm.core.runtime.engine_selector import build_runtime_backend_report, select_runtime_engine


def test_select_runtime_engine_uses_cpu_when_cuda_is_not_available(monkeypatch) -> None:
    monkeypatch.setattr(engine_selector.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(engine_selector, "_memory_snapshot", lambda: SimpleNamespace(free_gb=6.5))

    decision = select_runtime_engine("qwen-test")

    assert decision.selected_engine == "direct-cpu"
    assert decision.selected_backend == "torch-cpu"
    assert decision.hardware_acceleration_available is False
    assert decision.system_free_gb == 6.5
    assert "No CUDA" in decision.blockers[0]


def test_select_runtime_engine_reports_visible_but_unimplemented_cuda(monkeypatch) -> None:
    class FakeProps:
        name = "Test GPU"
        total_memory = 12 * 1024**3

    monkeypatch.setattr(engine_selector.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine_selector.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(engine_selector.torch.cuda, "get_device_properties", lambda index: FakeProps())
    monkeypatch.setattr(engine_selector, "_memory_snapshot", lambda: SimpleNamespace(free_gb=10.0))

    decision = select_runtime_engine("qwen-test")

    assert decision.selected_engine == "planned-gpu"
    assert decision.selected_backend == "cuda"
    assert decision.hardware_acceleration_available is True
    assert decision.cuda_devices[0]["total_vram_gb"] == 12.0
    assert "not implemented" in decision.blockers[0]


def test_backend_report_recommends_ready_gguf_before_unimplemented_gpu_paths(monkeypatch, tmp_path) -> None:
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(engine_selector.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(engine_selector, "_memory_snapshot", lambda: SimpleNamespace(free_gb=6.5))
    monkeypatch.setattr(engine_selector, "_module_available", lambda module_name: False)
    monkeypatch.setattr(engine_selector, "_disk_free_gb", lambda path: 100.0)
    monkeypatch.setattr(
        engine_selector,
        "build_gguf_backend_status",
        lambda model_id: SimpleNamespace(
            ready=True,
            package_available=True,
            model_files=[SimpleNamespace(path=gguf_file)],
        ),
    )

    report = build_runtime_backend_report("qwen-test")

    gguf = next(candidate for candidate in report.candidates if candidate.backend_id == "llama-cpp-gguf")
    assert report.recommended_backend_id == "llama-cpp-gguf"
    assert gguf.status == "ready"
    assert gguf.available_now is True
    assert gguf.implemented_now is True
    assert "practical speed path" in report.recommended_summary
