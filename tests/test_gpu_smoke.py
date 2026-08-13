from __future__ import annotations

from pcketlm.core.runtime.gpu_smoke import run_deepseek_gpu_probe, run_gpu_smoke


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


def test_deepseek_gpu_probe_cpu_shape_and_projection() -> None:
    result = run_deepseek_gpu_probe(
        device="cpu",
        hidden_size=64,
        intermediate_size=64,
        routed_experts=4,
        top_k=2,
        cache_len=16,
        benchmark_layers=2,
        iterations=1,
    )

    assert result.passed is True
    assert result.device == "cpu"
    assert result.layers_executed == 62
    assert result.output_shape == [1, 64]
    assert result.projected_62_layer_seconds_per_token > 0.0
    assert result.projected_k64_effective_seconds_per_position > 0.0


def test_deepseek_gpu_probe_require_cuda_reports_missing_gpu() -> None:
    result = run_deepseek_gpu_probe(require_cuda=True, device=None)
    if result.cuda_available:
        assert result.passed is True
        assert result.device.startswith("cuda")
    else:
        assert result.passed is False
        assert result.device == "cpu"
        assert "CUDA is required" in result.note
