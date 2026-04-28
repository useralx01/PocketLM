from types import SimpleNamespace

from pcketlm.core.runtime import engine_selector
from pcketlm.core.runtime.engine_selector import select_runtime_engine


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
