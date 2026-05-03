import math

import torch
import torch.nn.functional as F


def _apply_rope(values: torch.Tensor, position_offset: int, rope_theta: float) -> torch.Tensor:
    seq_len, head_count, head_dim = values.shape
    out = values.clone()
    for token in range(seq_len):
        position = float(position_offset + token)
        half_dim = head_dim // 2
        for dim in range(half_dim):
            inv_freq = rope_theta ** (-float(dim) / float(head_dim))
            angle = position * inv_freq
            c = math.cos(angle)
            s = math.sin(angle)
            x0 = out[token, :, dim].clone()
            x1 = out[token, :, dim + half_dim].clone()
            out[token, :, dim] = x0 * c - x1 * s
            out[token, :, dim + half_dim] = x1 * c + x0 * s
    return out


def _reference_attention(
    hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    position_offset: int,
    rope_theta: float,
) -> torch.Tensor:
    seq_len, hidden_size = hidden.shape
    head_dim = hidden_size // num_attention_heads
    kv_repeat = num_attention_heads // num_key_value_heads
    hidden_f = hidden.float()
    q = F.linear(hidden_f, q_weight.float()).view(seq_len, num_attention_heads, head_dim)
    k = F.linear(hidden_f, k_weight.float()).view(seq_len, num_key_value_heads, head_dim)
    v = F.linear(hidden_f, v_weight.float()).view(seq_len, num_key_value_heads, head_dim)
    q = _apply_rope(q, position_offset, rope_theta)
    k = _apply_rope(k, position_offset, rope_theta)
    context = torch.zeros((seq_len, num_attention_heads, head_dim), dtype=torch.float32)
    for token in range(seq_len):
        for head in range(num_attention_heads):
            kv_head = head // kv_repeat
            scores = torch.matmul(q[token, head], k[: token + 1, kv_head].T) / math.sqrt(head_dim)
            probs = torch.softmax(scores, dim=-1)
            context[token, head] = torch.matmul(probs, v[: token + 1, kv_head])
    return F.linear(context.reshape(seq_len, hidden_size), o_weight.float()).to(torch.float16)


def test_native_attention_prefill_matches_python_reference() -> None:
    from pcketlm.native import attention_prefill_fp16, native_fp16_attention_available

    assert native_fp16_attention_available() is True
    torch.manual_seed(321)
    seq_len = 4
    hidden_size = 16
    num_heads = 4
    num_kv_heads = 2
    kv_hidden = hidden_size // num_heads * num_kv_heads
    hidden = (torch.randn((seq_len, hidden_size), dtype=torch.float32) * 0.2).to(torch.float16)
    q_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)
    k_weight = (torch.randn((kv_hidden, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)
    v_weight = (torch.randn((kv_hidden, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)
    o_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)

    native = attention_prefill_fp16(
        hidden,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        position_offset=3,
        rope_theta=10000.0,
    )
    expected = _reference_attention(
        hidden,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        num_heads,
        num_kv_heads,
        3,
        10000.0,
    )

    assert torch.allclose(native.float(), expected.float(), atol=1e-3, rtol=1e-3)


def test_native_attention_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_ATTENTION", "1")
    from pcketlm.native import attention_prefill_fp16

    tensor = torch.ones((1, 4), dtype=torch.float16)
    try:
        attention_prefill_fp16(
            tensor,
            torch.ones((4, 4), dtype=torch.float16),
            torch.ones((4, 4), dtype=torch.float16),
            torch.ones((4, 4), dtype=torch.float16),
            torch.ones((4, 4), dtype=torch.float16),
            num_attention_heads=1,
            num_key_value_heads=1,
        )
    except RuntimeError as exc:
        assert "unavailable" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("PCKETLM_DISABLE_NATIVE_ATTENTION did not disable attention")
