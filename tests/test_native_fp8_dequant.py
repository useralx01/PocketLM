import os

import torch

import pcketlm.native as native
from pcketlm.core.runtime.fp8_source import dequantize_fp8_block_scaled


def _fp8_bytes(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()


def test_native_fp8_dequant_matches_python_reference() -> None:
    values = torch.linspace(-4.0, 4.0, steps=16 * 16, dtype=torch.float32).reshape(16, 16)
    fp8 = _fp8_bytes(values)
    scale = torch.tensor([[0.75]], dtype=torch.float32)

    native_out = native.fp8_e4m3_dequant_to_fp16(fp8, scale)

    os.environ["PCKETLM_DISABLE_NATIVE_FP8_DEQUANT"] = "1"
    try:
        python_out = dequantize_fp8_block_scaled(fp8, scale, dtype=torch.float16)
    finally:
        os.environ.pop("PCKETLM_DISABLE_NATIVE_FP8_DEQUANT", None)

    assert native.native_fp8_dequant_available() is True
    assert native_out.dtype == torch.float16
    assert torch.equal(native_out.view(torch.uint16), python_out.view(torch.uint16))


def test_native_fp8_dequant_bf16_matches_python_reference() -> None:
    values = torch.linspace(-8.0, 8.0, steps=8 * 32, dtype=torch.float32).reshape(8, 32)
    fp8 = _fp8_bytes(values)
    scale = torch.tensor([[1.25]], dtype=torch.float32)

    native_out = native.fp8_e4m3_dequant_to_bf16(fp8, scale)

    os.environ["PCKETLM_DISABLE_NATIVE_FP8_DEQUANT"] = "1"
    try:
        python_out = dequantize_fp8_block_scaled(fp8, scale, dtype=torch.bfloat16)
    finally:
        os.environ.pop("PCKETLM_DISABLE_NATIVE_FP8_DEQUANT", None)

    assert native_out.dtype == torch.bfloat16
    assert torch.equal(native_out.view(torch.uint16), python_out.view(torch.uint16))


def test_native_fp8_dequant_matches_python_for_all_fp8_bytes() -> None:
    fp8 = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    scale = torch.tensor([[1.0]], dtype=torch.float32)

    os.environ["PCKETLM_DISABLE_NATIVE_FP8_DEQUANT"] = "1"
    try:
        python_fp16 = dequantize_fp8_block_scaled(fp8, scale, dtype=torch.float16)
        python_bf16 = dequantize_fp8_block_scaled(fp8, scale, dtype=torch.bfloat16)
    finally:
        os.environ.pop("PCKETLM_DISABLE_NATIVE_FP8_DEQUANT", None)

    native_fp16 = native.fp8_e4m3_dequant_to_fp16(fp8, scale)
    native_bf16 = native.fp8_e4m3_dequant_to_bf16(fp8, scale)

    assert torch.equal(native_fp16.view(torch.uint16), python_fp16.view(torch.uint16))
    assert torch.equal(native_bf16.view(torch.uint16), python_bf16.view(torch.uint16))


def test_runtime_fp8_dequant_kill_switch_falls_back_to_python(monkeypatch) -> None:
    values = torch.tensor([[0.5, -1.0, 2.0, -3.0], [4.0, -5.0, 6.0, -7.0]], dtype=torch.float32)
    fp8 = _fp8_bytes(values)
    scale = torch.tensor([[1.25]], dtype=torch.float32)

    enabled = dequantize_fp8_block_scaled(fp8, scale, dtype=torch.float16)
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_FP8_DEQUANT", "1")
    disabled = dequantize_fp8_block_scaled(fp8, scale, dtype=torch.float16)

    assert torch.equal(enabled.view(torch.uint16), disabled.view(torch.uint16))


def test_native_fp8_dequant_rejects_bad_scale_shape() -> None:
    fp8 = torch.zeros((256, 256), dtype=torch.uint8)
    bad_scale = torch.ones((1, 1), dtype=torch.float32)

    try:
        native.fp8_e4m3_dequant_to_fp16(fp8, bad_scale)
    except ValueError as exc:
        assert "128x128 block layout" in str(exc)
    else:
        raise AssertionError("bad scale shape was accepted")
