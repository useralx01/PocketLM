import torch

import pcketlm.native as native


def test_ds_attention_bridge_matches_direct_python_callback() -> None:
    hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)

    def attention_fn(layer_idx: int, incoming: torch.Tensor) -> torch.Tensor:
        return incoming + float(layer_idx + 1)

    with native.DeepSeekNativeSession(num_layers=2, hidden_dim=4, num_experts=0, top_k=0) as session:
        actual = native.ds_attention_layer_forward(session, 1, hidden, attention_fn)
        expected = attention_fn(1, hidden)

        assert torch.equal(actual, expected)
        assert session.monolithic_call_count() == 1
        assert session.attention_invocation_count() == 1
        assert session.callback_invocation_count() == 1


def test_ds_attention_bridge_kill_switch_falls_back_to_direct_python(monkeypatch) -> None:
    hidden = torch.ones((1, 1, 4), dtype=torch.float32)
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_DS_ATTENTION_BRIDGE", "1")

    def attention_fn(_layer_idx: int, incoming: torch.Tensor) -> torch.Tensor:
        return incoming * 3.0

    with native.DeepSeekNativeSession(num_layers=1, hidden_dim=4, num_experts=0, top_k=0) as session:
        actual = native.ds_attention_layer_forward(session, 0, hidden, attention_fn)

        assert native.native_ds_attention_bridge_available() is False
        assert torch.equal(actual, attention_fn(0, hidden))
        assert session.monolithic_call_count() == 0
        assert session.attention_invocation_count() == 0


def test_ds_attention_bridge_matches_real_deepseek_layer3_attention() -> None:
    import pcketlm.core.runtime.fp8_source as fp8_source

    hidden = torch.linspace(-0.1, 0.1, 7168, dtype=torch.float32).reshape(1, 1, 7168).to(torch.bfloat16)
    direct = fp8_source.run_fp8_single_token_attention("deepseek-v3", 3, hidden, dtype=torch.float32)
    assert direct.ready is True
    assert direct.output_tensor is not None

    def attention_fn(layer_idx: int, incoming: torch.Tensor) -> torch.Tensor:
        result = fp8_source.run_fp8_single_token_attention(
            "deepseek-v3",
            layer_idx,
            incoming.to(torch.bfloat16),
            dtype=torch.float32,
        )
        assert result.ready is True
        assert result.output_tensor is not None
        return result.output_tensor.float()

    with native.DeepSeekNativeSession(num_layers=62, hidden_dim=7168, num_experts=256, top_k=8) as session:
        bridged = native.ds_attention_layer_forward(session, 3, hidden.float(), attention_fn)

        max_abs = float((bridged - direct.output_tensor.float()).abs().max().item())
        assert max_abs <= 1e-6
        assert session.monolithic_call_count() == 1
        assert session.attention_invocation_count() == 1
