from __future__ import annotations

from pcketlm.core.runtime.gpu_smoke import run_gpu_smoke


def test_gpu_smoke_cpu_fallback_is_deterministic() -> None:
    result = run_gpu_smoke(device="cpu")

    assert result.passed is True
    assert result.device == "cpu"
    assert result.output_shape == [1, 64]
    assert result.max_abs_repeat_diff == 0.0


def test_gpu_smoke_require_cuda_reports_missing_gpu() -> None:
    result = run_gpu_smoke(require_cuda=True, device=None)
    if result.cuda_available:
        assert result.passed is True
        assert result.device.startswith("cuda")
    else:
        assert result.passed is False
        assert result.device == "cpu"
        assert "CUDA is required" in result.note
