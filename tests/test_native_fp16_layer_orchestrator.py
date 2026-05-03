import torch
import torch.nn.functional as F

from tests.test_native_fp16_kv_cache import _reference_decode


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return hidden * torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def test_native_dense_layer_decode_matches_python_reference() -> None:
    from pcketlm.native import NativeKvSession

    torch.manual_seed(5678)
    hidden_size = 8
    intermediate_size = 16
    num_heads = 2
    num_kv_heads = 1
    kv_width = hidden_size // num_heads * num_kv_heads
    eps = 1e-5
    session = NativeKvSession(layer_count=1, max_seq_len=8, kv_width=kv_width)
    prefix_k = torch.randn((2, kv_width), dtype=torch.float16)
    prefix_v = torch.randn((2, kv_width), dtype=torch.float16)
    session.append_committed(0, prefix_k, prefix_v, count=2)

    hidden = torch.randn((hidden_size,), dtype=torch.float16)
    input_norm = torch.rand((hidden_size,), dtype=torch.float16) + 0.5
    post_norm = torch.rand((hidden_size,), dtype=torch.float16) + 0.5
    q_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)
    k_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)
    v_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)
    o_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)
    gate_weight = (torch.randn((intermediate_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)
    up_weight = (torch.randn((intermediate_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)
    down_weight = (torch.randn((hidden_size, intermediate_size), dtype=torch.float32) * 0.15).to(torch.float16)

    native = session.dense_layer_decode_fp16(
        0,
        hidden,
        input_norm,
        post_norm,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        gate_weight,
        up_weight,
        down_weight,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rms_eps=eps,
        rope_theta=10000.0,
    )

    normed = _rms_norm(hidden.float(), input_norm.float(), eps).to(torch.float16)
    attn_out, _k_new, _v_new = _reference_decode(
        normed,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        prefix_k,
        prefix_v,
        position=2,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=10000.0,
    )
    residual = hidden.float() + attn_out.float()
    post = _rms_norm(residual, post_norm.float(), eps)
    gate = F.linear(post.reshape(1, -1), gate_weight.float())
    up = F.linear(post.reshape(1, -1), up_weight.float())
    mlp = F.linear(F.silu(gate) * up, down_weight.float()).reshape(-1)
    expected = (residual + mlp).to(torch.float16)

    assert torch.allclose(native.float(), expected.float(), atol=2e-3, rtol=2e-3)
    assert session.tentative_length(0) == 1
    session.close()


def test_native_dense_layer_decode_supports_bf16_bias_and_qk_norm() -> None:
    from pcketlm.native import NativeKvSession

    torch.manual_seed(9753)
    hidden_size = 16
    intermediate_size = 24
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    kv_width = head_dim * num_kv_heads
    eps = 1e-6
    session = NativeKvSession(layer_count=1, max_seq_len=8, kv_width=kv_width, dtype=torch.bfloat16)
    prefix_k = torch.randn((2, kv_width), dtype=torch.float32).to(torch.bfloat16)
    prefix_v = torch.randn((2, kv_width), dtype=torch.float32).to(torch.bfloat16)
    session.append_committed(0, prefix_k, prefix_v, count=2)

    hidden = torch.randn((hidden_size,), dtype=torch.float32).to(torch.bfloat16)
    input_norm = (torch.rand((hidden_size,), dtype=torch.float32) + 0.5).to(torch.bfloat16)
    post_norm = (torch.rand((hidden_size,), dtype=torch.float32) + 0.5).to(torch.bfloat16)
    q_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.bfloat16)
    k_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.15).to(torch.bfloat16)
    v_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.15).to(torch.bfloat16)
    o_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.bfloat16)
    q_bias = (torch.randn((hidden_size,), dtype=torch.float32) * 0.03).to(torch.bfloat16)
    k_bias = (torch.randn((kv_width,), dtype=torch.float32) * 0.03).to(torch.bfloat16)
    v_bias = (torch.randn((kv_width,), dtype=torch.float32) * 0.03).to(torch.bfloat16)
    q_norm = (torch.rand((head_dim,), dtype=torch.float32) + 0.5).to(torch.bfloat16)
    k_norm = (torch.rand((head_dim,), dtype=torch.float32) + 0.5).to(torch.bfloat16)
    gate_weight = (torch.randn((intermediate_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.bfloat16)
    up_weight = (torch.randn((intermediate_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.bfloat16)
    down_weight = (torch.randn((hidden_size, intermediate_size), dtype=torch.float32) * 0.15).to(torch.bfloat16)

    native = session.dense_layer_decode_fp16(
        0,
        hidden,
        input_norm,
        post_norm,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        gate_weight,
        up_weight,
        down_weight,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rms_eps=eps,
        rope_theta=10000.0,
        q_bias=q_bias,
        k_bias=k_bias,
        v_bias=v_bias,
        q_norm_weight=q_norm,
        k_norm_weight=k_norm,
    )

    normed = _rms_norm(hidden.float(), input_norm.float(), eps).to(torch.bfloat16)
    attn_out, _k_new, _v_new = _reference_decode(
        normed,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        prefix_k,
        prefix_v,
        position=2,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=10000.0,
        q_bias=q_bias,
        k_bias=k_bias,
        v_bias=v_bias,
        q_norm_weight=q_norm,
        k_norm_weight=k_norm,
        rms_eps=eps,
    )
    residual = hidden.float() + attn_out.float()
    post = _rms_norm(residual, post_norm.float(), eps)
    gate = F.linear(post.reshape(1, -1), gate_weight.float())
    up = F.linear(post.reshape(1, -1), up_weight.float())
    mlp = F.linear(F.silu(gate) * up, down_weight.float()).reshape(-1)
    expected = (residual + mlp).to(torch.bfloat16)

    assert native.dtype == torch.bfloat16
    assert torch.allclose(native.float(), expected.float(), atol=4e-2, rtol=4e-2)
    session.close()
