import os

import pytest
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


def test_native_fp8_linear_batch_reuse_is_bit_exact_to_single_rows() -> None:
    weight_values = torch.linspace(-1.5, 1.5, steps=64 * 128, dtype=torch.float32).reshape(64, 128)
    fp8_weight = _fp8_bytes(weight_values)
    scale = torch.full((1, 1), 0.875, dtype=torch.float32)
    hidden = torch.linspace(-0.5, 0.5, steps=5 * 128, dtype=torch.float32).reshape(5, 128)

    os.environ["PCKETLM_ENABLE_NATIVE_FP8_BATCH_REUSE"] = "1"
    try:
        batch_out = native.fp8_e4m3_block_linear_f32(fp8_weight, scale, hidden)
        split_out = torch.cat(
            [
                native.fp8_e4m3_block_linear_f32(fp8_weight, scale, hidden[index : index + 1])
                for index in range(hidden.shape[0])
            ],
            dim=0,
        )
    finally:
        os.environ.pop("PCKETLM_ENABLE_NATIVE_FP8_BATCH_REUSE", None)

    assert torch.equal(batch_out, split_out)


def test_native_fp8_dual_linear_batch_reuse_is_bit_exact_to_single_rows() -> None:
    hidden_cols = 128
    out_rows = 64
    gate = _fp8_bytes(torch.linspace(-1.0, 1.0, steps=out_rows * hidden_cols).reshape(out_rows, hidden_cols))
    up = _fp8_bytes(torch.linspace(1.0, -1.0, steps=out_rows * hidden_cols).reshape(out_rows, hidden_cols))
    gate_scale = torch.full((1, 1), 0.75, dtype=torch.float32)
    up_scale = torch.full((1, 1), 1.25, dtype=torch.float32)
    hidden = torch.linspace(-0.5, 0.5, steps=4 * hidden_cols, dtype=torch.float32).reshape(4, hidden_cols)

    os.environ["PCKETLM_ENABLE_NATIVE_FP8_BATCH_REUSE"] = "1"
    try:
        batch_a, batch_b = native.fp8_e4m3_block_dual_linear_f32(gate, gate_scale, up, up_scale, hidden)
        split_pairs = [
            native.fp8_e4m3_block_dual_linear_f32(gate, gate_scale, up, up_scale, hidden[index : index + 1])
            for index in range(hidden.shape[0])
        ]
    finally:
        os.environ.pop("PCKETLM_ENABLE_NATIVE_FP8_BATCH_REUSE", None)
    split_a = torch.cat([item[0] for item in split_pairs], dim=0)
    split_b = torch.cat([item[1] for item in split_pairs], dim=0)

    assert torch.equal(batch_a, split_a)
    assert torch.equal(batch_b, split_b)


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


def test_native_fp8_mlp_batch_reuse_is_bit_exact_to_single_rows() -> None:
    hidden_cols = 128
    intermediate = 128
    generator = torch.Generator().manual_seed(777)
    hidden = torch.randn((4, hidden_cols), generator=generator, dtype=torch.float32) * 0.25
    gate = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
    up = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
    down = _fp8_bytes(torch.randn((hidden_cols, intermediate), generator=generator) * 0.25)
    gate_scale = torch.full((1, 1), 0.875, dtype=torch.float32)
    up_scale = torch.full((1, 1), 1.125, dtype=torch.float32)
    down_scale = torch.full((1, 1), 0.75, dtype=torch.float32)

    os.environ["PCKETLM_ENABLE_NATIVE_FP8_BATCH_REUSE"] = "1"
    try:
        batch_out = native.fp8_e4m3_block_mlp_f32(gate, gate_scale, up, up_scale, down, down_scale, hidden)
        split_out = torch.cat(
            [
                native.fp8_e4m3_block_mlp_f32(
                    gate,
                    gate_scale,
                    up,
                    up_scale,
                    down,
                    down_scale,
                    hidden[index : index + 1],
                )
                for index in range(hidden.shape[0])
            ],
            dim=0,
        )
    finally:
        os.environ.pop("PCKETLM_ENABLE_NATIVE_FP8_BATCH_REUSE", None)

    assert torch.equal(batch_out, split_out)


def test_native_fp8_weighted_many_mlp_matches_unweighted_combine() -> None:
    hidden_cols = 64
    intermediate = 32
    hidden = torch.linspace(-0.5, 0.5, steps=hidden_cols, dtype=torch.float32).reshape(1, hidden_cols)
    items = []
    for seed in (1, 2, 3):
        generator = torch.Generator().manual_seed(seed)
        gate = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
        up = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
        down = _fp8_bytes(torch.randn((hidden_cols, intermediate), generator=generator) * 0.25)
        gate_scale = torch.ones((1, 1), dtype=torch.float32)
        up_scale = torch.ones((1, 1), dtype=torch.float32)
        down_scale = torch.ones((1, 1), dtype=torch.float32)
        items.append((gate, gate_scale, up, up_scale, down, down_scale))
    route_weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)

    weighted = native.fp8_e4m3_block_mlp_many_weighted_f32(items, hidden, route_weights)
    unweighted = native.fp8_e4m3_block_mlp_many_f32(items, hidden)
    expected = native._python_ds_moe_layer_forward(unweighted, route_weights)

    assert torch.allclose(weighted, expected, atol=1e-5, rtol=1e-5)


def test_native_fp8_row_weighted_many_mlp_matches_unweighted_combine() -> None:
    if not native.native_fp8_mlp_many_row_weighted_available():
        pytest.skip(f"FP8 row-weighted many-MLP unavailable: {native.native_fp8_linear_error()}")
    hidden_cols = 64
    intermediate = 32
    hidden = torch.linspace(-0.5, 0.5, steps=3 * hidden_cols, dtype=torch.float32).reshape(3, hidden_cols)
    items = []
    for seed in (5, 6, 7, 8):
        generator = torch.Generator().manual_seed(seed)
        gate = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
        up = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
        down = _fp8_bytes(torch.randn((hidden_cols, intermediate), generator=generator) * 0.25)
        gate_scale = torch.ones((1, 1), dtype=torch.float32)
        up_scale = torch.ones((1, 1), dtype=torch.float32)
        down_scale = torch.ones((1, 1), dtype=torch.float32)
        items.append((gate, gate_scale, up, up_scale, down, down_scale))
    route_weights = torch.tensor(
        [
            [0.6, 0.0, 0.4],
            [0.4, 0.5, 0.0],
            [0.0, 0.5, 0.35],
            [0.0, 0.0, 0.25],
        ],
        dtype=torch.float32,
    )

    os.environ["PCKETLM_ENABLE_NATIVE_FP8_BATCH_REUSE"] = "1"
    try:
        weighted = native.fp8_e4m3_block_mlp_many_row_weighted_f32(items, hidden, route_weights)
        unweighted = native.fp8_e4m3_block_mlp_many_f32(items, hidden)
    finally:
        os.environ.pop("PCKETLM_ENABLE_NATIVE_FP8_BATCH_REUSE", None)
    expected = (unweighted * route_weights.view(len(items), hidden.shape[0], 1)).sum(dim=0)

    assert torch.allclose(weighted, expected, atol=1e-5, rtol=1e-5)


def test_native_fp8_pair_weighted_many_mlp_matches_row_weighted() -> None:
    if not native.native_fp8_mlp_many_pair_weighted_available():
        pytest.skip(f"FP8 pair-weighted many-MLP unavailable: {native.native_fp8_linear_error()}")
    hidden_cols = 64
    intermediate = 32
    hidden = torch.linspace(-0.5, 0.5, steps=3 * hidden_cols, dtype=torch.float32).reshape(3, hidden_cols)
    items = []
    for seed in (5, 6, 7, 8):
        generator = torch.Generator().manual_seed(seed)
        gate = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
        up = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
        down = _fp8_bytes(torch.randn((hidden_cols, intermediate), generator=generator) * 0.25)
        gate_scale = torch.ones((1, 1), dtype=torch.float32)
        up_scale = torch.ones((1, 1), dtype=torch.float32)
        down_scale = torch.ones((1, 1), dtype=torch.float32)
        items.append((gate, gate_scale, up, up_scale, down, down_scale))
    route_weights = torch.tensor(
        [
            [0.6, 0.0, 0.4],
            [0.4, 0.5, 0.0],
            [0.0, 0.5, 0.35],
            [0.0, 0.0, 0.25],
        ],
        dtype=torch.float32,
    )
    pair_items = []
    pair_rows = []
    pair_weights = []
    for item_index in range(route_weights.shape[0]):
        for row_index in range(route_weights.shape[1]):
            weight = float(route_weights[item_index, row_index].item())
            if weight != 0.0:
                pair_items.append(item_index)
                pair_rows.append(row_index)
                pair_weights.append(weight)

    expected = native.fp8_e4m3_block_mlp_many_row_weighted_f32(items, hidden, route_weights)
    pair = native.fp8_e4m3_block_mlp_many_pair_weighted_f32(
        items,
        hidden,
        torch.tensor(pair_items, dtype=torch.int64),
        torch.tensor(pair_rows, dtype=torch.int64),
        torch.tensor(pair_weights, dtype=torch.float32),
    )

    assert torch.equal(pair, expected)


def test_native_fp8_avx512_weighted_many_mlp_is_bit_exact_to_scalar() -> None:
    if not native.native_fp8_mlp_many_weighted_avx512_available():
        pytest.skip(f"AVX-512 FP8 weighted many-MLP unavailable: {native.native_fp8_linear_avx512_error()}")
    hidden_cols = 128
    intermediate = 128
    hidden = torch.linspace(-0.5, 0.5, steps=hidden_cols, dtype=torch.float32).reshape(1, hidden_cols)
    items = []
    for seed in (11, 22, 33, 44):
        generator = torch.Generator().manual_seed(seed)
        gate = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
        up = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
        down = _fp8_bytes(torch.randn((hidden_cols, intermediate), generator=generator) * 0.25)
        gate_scale = torch.full((1, 1), 0.875, dtype=torch.float32)
        up_scale = torch.full((1, 1), 1.125, dtype=torch.float32)
        down_scale = torch.full((1, 1), 0.75, dtype=torch.float32)
        items.append((gate, gate_scale, up, up_scale, down, down_scale))
    route_weights = torch.tensor([0.31, 0.17, 0.29, 0.23], dtype=torch.float32)

    scalar = native.fp8_e4m3_block_mlp_many_weighted_f32(items, hidden, route_weights)
    avx512 = native.fp8_e4m3_block_mlp_many_weighted_avx512_f32(items, hidden, route_weights)

    assert torch.equal(avx512, scalar)


def test_native_fp8_avx512_mlp_is_bit_exact_to_scalar() -> None:
    if not native.native_fp8_mlp_avx512_available():
        pytest.skip(f"AVX-512 FP8 MLP unavailable: {native.native_fp8_linear_avx512_error()}")
    hidden_cols = 256
    intermediate = 384
    generator = torch.Generator().manual_seed(555)
    hidden = torch.randn((2, hidden_cols), generator=generator, dtype=torch.float32) * 0.25
    gate = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
    up = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
    down = _fp8_bytes(torch.randn((hidden_cols, intermediate), generator=generator) * 0.25)
    gate_scale = torch.full((3, 2), 0.875, dtype=torch.float32)
    up_scale = torch.full((3, 2), 1.125, dtype=torch.float32)
    down_scale = torch.full((2, 3), 0.75, dtype=torch.float32)

    scalar = native.fp8_e4m3_block_mlp_f32(gate, gate_scale, up, up_scale, down, down_scale, hidden)
    avx512 = native.fp8_e4m3_block_mlp_avx512_f32(gate, gate_scale, up, up_scale, down, down_scale, hidden)

    assert torch.equal(avx512, scalar)


def test_native_fp8_linear_kill_switch_reports_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_FP8_LINEAR", "1")

    assert native.native_fp8_linear_available() is False
