from __future__ import annotations

import pytest
import torch

from pcketlm.core.runtime.gpu_smoke import run_gpu_smoke


def test_fp8_gpu_smoke_runs_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine; cloud notebook runs this test on GPU.")

    result = run_gpu_smoke(require_cuda=True)

    assert result.passed is True
    assert result.device.startswith("cuda")
    assert result.output_shape == [1, 64]
    assert result.max_abs_repeat_diff == 0.0
