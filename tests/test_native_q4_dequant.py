import importlib

import pytest
import torch

from pcketlm.core.runtime.tensor_loader import _dequantize_q4_tensor, _python_dequantize_q4_tensor
from tools.quantize_to_q4 import quantize_tensor_to_q4


def test_native_dequant_loads() -> None:
    native = importlib.import_module("pcketlm.native")

    assert native.native_q4_available() is True


def test_native_dequant_matches_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_Q4", raising=False)
    shapes = [
        (),
        (1,),
        (7,),
        (1, 1),
        (1, 17),
        (2, 5),
        (3, 4),
        (8, 8),
        (5, 3, 2),
        (4, 7, 3),
    ]
    generator = torch.Generator().manual_seed(1234)
    cases = []
    for shape in shapes:
        cases.append(torch.randn(shape, generator=generator, dtype=torch.float16))
    for index in range(90):
        rows = 1 + (index % 9)
        cols = 1 + ((index * 7) % 37)
        cases.append(torch.randn((rows, cols), generator=generator, dtype=torch.float16))

    for tensor in cases:
        packed, scales, metadata = quantize_tensor_to_q4(tensor)
        native = _dequantize_q4_tensor(packed, scales, metadata["shape"], "F16")
        python = _python_dequantize_q4_tensor(packed, scales, metadata["shape"])

        assert torch.equal(native, python)


def test_native_dequant_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    tensor = torch.randn((4, 9), generator=torch.Generator().manual_seed(7), dtype=torch.float16)
    packed, scales, metadata = quantize_tensor_to_q4(tensor)
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_Q4", "1")

    loaded = _dequantize_q4_tensor(packed, scales, metadata["shape"], "F16")
    expected = _python_dequantize_q4_tensor(packed, scales, metadata["shape"])

    assert torch.equal(loaded, expected)


def test_native_dequant_handles_missing_dll(monkeypatch: pytest.MonkeyPatch) -> None:
    import pcketlm.native as native

    tensor = torch.randn((3, 11), generator=torch.Generator().manual_seed(9), dtype=torch.float16)
    packed, scales, metadata = quantize_tensor_to_q4(tensor)

    def fail_native(*args, **kwargs):
        raise RuntimeError("missing dll")

    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_Q4", raising=False)
    monkeypatch.setattr(native, "q4_dequant_to_fp16", fail_native)
    loaded = _dequantize_q4_tensor(packed, scales, metadata["shape"], "F16")
    expected = _python_dequantize_q4_tensor(packed, scales, metadata["shape"])

    assert torch.equal(loaded, expected)
