import torch

import pcketlm.native as native


def test_ds_moe_layer_forward_combines_synthetic_top2_experts() -> None:
    expert_outputs = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[10.0, 20.0, 30.0, 40.0]]], dtype=torch.float32)
    route_weights = torch.tensor([0.75, 0.25], dtype=torch.float32)

    with native.DeepSeekNativeSession(num_layers=2, hidden_dim=4, num_experts=4, top_k=2) as session:
        actual = native.ds_moe_layer_forward(session, expert_outputs, route_weights)
        expected = native._python_ds_moe_layer_forward(expert_outputs, route_weights)

        assert torch.equal(actual, expected)
        assert session.monolithic_call_count() == 1
        assert session.expert_invocation_count() == 2


def test_ds_moe_layer_forward_fp8_uses_native_many_mlp_on_synthetic_experts() -> None:
    hidden = torch.tensor([[0.5, -0.25, 0.75, 1.0]], dtype=torch.float32)
    route_weights = torch.tensor([0.6, 0.4], dtype=torch.float32)
    items = [
        _fp8_mlp_item(seed=1, hidden_size=4, intermediate_size=2),
        _fp8_mlp_item(seed=2, hidden_size=4, intermediate_size=2),
    ]

    expected_experts = native.fp8_e4m3_block_mlp_many_f32(items, hidden)
    expected = native._python_ds_moe_layer_forward(expected_experts, route_weights)
    with native.DeepSeekNativeSession(num_layers=2, hidden_dim=4, num_experts=4, top_k=2) as session:
        actual = native.ds_moe_layer_forward_fp8(session, items, hidden, route_weights)

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
        assert session.monolithic_call_count() == 1
        assert session.expert_invocation_count() == 2


def test_ds_moe_layer_forward_kill_switch_falls_back_to_python(monkeypatch) -> None:
    expert_outputs = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    route_weights = torch.tensor([0.25, 0.75], dtype=torch.float32)
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_DS_MOE", "1")

    with native.DeepSeekNativeSession(num_layers=1, hidden_dim=2, num_experts=2, top_k=2) as session:
        actual = native.ds_moe_layer_forward(session, expert_outputs, route_weights)
        expected = native._python_ds_moe_layer_forward(expert_outputs, route_weights)

        assert native.native_ds_moe_available() is False
        assert torch.equal(actual, expected)
        assert session.monolithic_call_count() == 0


def test_ds_moe_layer_forward_matches_real_deepseek_layer3_routed_output() -> None:
    from pcketlm.core.runtime.fp8_source import load_fp8_weight_pair, run_fp8_router

    hidden = torch.linspace(-0.25, 0.25, 7168, dtype=torch.float32).reshape(1, 7168).to(torch.bfloat16)
    router = run_fp8_router("deepseek-v3", 3, hidden)
    assert router.ready is True
    assert router.indices_tensor is not None
    assert router.weights_tensor is not None
    expert_ids = [int(value) for value in router.indices_tensor.reshape(-1).tolist()]
    route_weights = router.weights_tensor.reshape(-1).float()

    items = []
    for expert_id in expert_ids:
        prefix = f"model.layers.3.mlp.experts.{expert_id}"
        gate = load_fp8_weight_pair("deepseek-v3", f"{prefix}.gate_proj.weight")
        up = load_fp8_weight_pair("deepseek-v3", f"{prefix}.up_proj.weight")
        down = load_fp8_weight_pair("deepseek-v3", f"{prefix}.down_proj.weight")
        assert gate.ready and up.ready and down.ready
        assert gate.fp8_bytes is not None and gate.scale_tensor is not None
        assert up.fp8_bytes is not None and up.scale_tensor is not None
        assert down.fp8_bytes is not None and down.scale_tensor is not None
        items.append((gate.fp8_bytes, gate.scale_tensor, up.fp8_bytes, up.scale_tensor, down.fp8_bytes, down.scale_tensor))

    expected_experts = native.fp8_e4m3_block_mlp_many_f32(items, hidden.float())
    expected = native._python_ds_moe_layer_forward(expected_experts, route_weights)
    with native.DeepSeekNativeSession(num_layers=62, hidden_dim=7168, num_experts=256, top_k=8) as session:
        actual = native.ds_moe_layer_forward_fp8(session, items, hidden.float(), route_weights)

        max_abs = float((actual - expected).abs().max().item())
        assert max_abs <= 1e-5
        assert session.monolithic_call_count() == 1
        assert session.expert_invocation_count() == len(expert_ids)


def _fp8_mlp_item(
    *,
    seed: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    gate = torch.randn((intermediate_size, hidden_size), generator=generator, dtype=torch.float32).mul(0.25)
    up = torch.randn((intermediate_size, hidden_size), generator=generator, dtype=torch.float32).mul(0.25)
    down = torch.randn((hidden_size, intermediate_size), generator=generator, dtype=torch.float32).mul(0.25)
    gate_fp8 = gate.to(torch.float8_e4m3fn).view(torch.uint8)
    up_fp8 = up.to(torch.float8_e4m3fn).view(torch.uint8)
    down_fp8 = down.to(torch.float8_e4m3fn).view(torch.uint8)
    gate_scale = torch.ones(((intermediate_size + 127) // 128, (hidden_size + 127) // 128), dtype=torch.float32)
    up_scale = torch.ones_like(gate_scale)
    down_scale = torch.ones(((hidden_size + 127) // 128, (intermediate_size + 127) // 128), dtype=torch.float32)
    return gate_fp8, gate_scale, up_fp8, up_scale, down_fp8, down_scale
