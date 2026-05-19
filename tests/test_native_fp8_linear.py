import os

import torch
import torch.nn.functional as F

import pcketlm.native as native
from pcketlm.core.runtime.fp8_source import dequantize_fp8_block_scaled


def _fp8_bytes(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()


def test_native_fp8_linear_matches_python_reference() -> None:
    weight_values = torch.linspace(-1.5, 1.5, steps=32 * 64, dtype=torch.float32).reshape(32, 64)
    fp8_weight = _fp8_bytes(weight_values)
    scale = torch.tensor([[0.875]], dtype=torch.float32)
    hidden = torch.linspace(-0.5, 0.5, steps=2 * 64, dtype=torch.float32).reshape(2, 64)

    native_out = native.fp8_e4m3_block_linear_f32(fp8_weight, scale, hidden)

    os.environ["PCKETLM_DISABLE_NATIVE_FP8_DEQUANT"] = "1"
    try:
        dequantized = dequantize_fp8_block_scaled(fp8_weight, scale, dtype=torch.float32)
    finally:
        os.environ.pop("PCKETLM_DISABLE_NATIVE_FP8_DEQUANT", None)
    python_out = F.linear(hidden, dequantized.float())

    assert torch.allclose(native_out, python_out, atol=1e-5, rtol=1e-5)


def test_native_fp8_mlp_matches_python_reference() -> None:
    hidden_cols = 64
    intermediate = 32
    hidden = torch.linspace(-0.5, 0.5, steps=hidden_cols, dtype=torch.float32).reshape(1, hidden_cols)
    gate = _fp8_bytes(torch.linspace(-1.0, 1.0, steps=intermediate * hidden_cols).reshape(intermediate, hidden_cols))
    up = _fp8_bytes(torch.linspace(1.0, -1.0, steps=intermediate * hidden_cols).reshape(intermediate, hidden_cols))
    down = _fp8_bytes(torch.linspace(-0.75, 0.75, steps=hidden_cols * intermediate).reshape(hidden_cols, intermediate))
    gate_scale = torch.tensor([[1.0]], dtype=torch.float32)
    up_scale = torch.tensor([[0.75]], dtype=torch.float32)
    down_scale = torch.tensor([[1.25]], dtype=torch.float32)

    native_out = native.fp8_e4m3_block_mlp_f32(gate, gate_scale, up, up_scale, down, down_scale, hidden)

    os.environ["PCKETLM_DISABLE_NATIVE_FP8_DEQUANT"] = "1"
    try:
        gate_ref = dequantize_fp8_block_scaled(gate, gate_scale, dtype=torch.float32)
        up_ref = dequantize_fp8_block_scaled(up, up_scale, dtype=torch.float32)
        down_ref = dequantize_fp8_block_scaled(down, down_scale, dtype=torch.float32)
    finally:
        os.environ.pop("PCKETLM_DISABLE_NATIVE_FP8_DEQUANT", None)
    activation = F.silu(F.linear(hidden, gate_ref.float())) * F.linear(hidden, up_ref.float())
    python_out = F.linear(activation, down_ref.float())

    assert torch.allclose(native_out, python_out, atol=1e-5, rtol=1e-5)


def test_native_fp8_linear_kill_switch_reports_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_FP8_LINEAR", "1")

    assert native.native_fp8_linear_available() is False

