import torch
import torch.nn.functional as F


def _rope_one(values: torch.Tensor, position: int, head_count: int, head_dim: int, rope_theta: float) -> torch.Tensor:
    out = values.float().reshape(head_count, head_dim).clone()
    for dim in range(0, head_dim, 2):
        inv_freq = rope_theta ** (-float(dim) / float(head_dim))
        angle = float(position) * inv_freq
        c = torch.tensor(torch.cos(torch.tensor(angle)).item(), dtype=torch.float32)
        s = torch.tensor(torch.sin(torch.tensor(angle)).item(), dtype=torch.float32)
        x0 = out[:, dim].clone()
        x1 = out[:, dim + 1].clone()
        out[:, dim] = x0 * c - x1 * s
        out[:, dim + 1] = x1 * c + x0 * s
    return out


def _reference_decode(
    hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    position: int,
    num_heads: int,
    num_kv_heads: int,
    rope_theta: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    v_bias: torch.Tensor | None = None,
    q_norm_weight: torch.Tensor | None = None,
    k_norm_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_size = hidden.numel()
    head_dim = hidden_size // num_heads
    kv_repeat = num_heads // num_kv_heads
    q = F.linear(hidden.float().reshape(1, -1), q_weight.float(), None if q_bias is None else q_bias.float()).reshape(num_heads, head_dim)
    k_new = F.linear(hidden.float().reshape(1, -1), k_weight.float(), None if k_bias is None else k_bias.float()).reshape(num_kv_heads, head_dim)
    v_new = F.linear(hidden.float().reshape(1, -1), v_weight.float(), None if v_bias is None else v_bias.float()).reshape(num_kv_heads, head_dim)
    if q_norm_weight is not None:
        q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + rms_eps) * q_norm_weight.float().view(1, -1)
    if k_norm_weight is not None:
        k_new = k_new * torch.rsqrt(k_new.pow(2).mean(dim=-1, keepdim=True) + rms_eps) * k_norm_weight.float().view(1, -1)
    q = _rope_one(q, position, num_heads, head_dim, rope_theta)
    k_new = _rope_one(k_new, position, num_kv_heads, head_dim, rope_theta)
    all_k = torch.cat([prefix_k.float().reshape(-1, num_kv_heads, head_dim), k_new.reshape(1, num_kv_heads, head_dim)])
    all_v = torch.cat([prefix_v.float().reshape(-1, num_kv_heads, head_dim), v_new.reshape(1, num_kv_heads, head_dim)])
    context = torch.zeros((num_heads, head_dim), dtype=torch.float32)
    for head in range(num_heads):
        kv_head = head // kv_repeat
        scores = torch.matmul(q[head], all_k[:, kv_head].T) / (head_dim ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        context[head] = torch.matmul(probs, all_v[:, kv_head])
    out = F.linear(context.reshape(1, hidden_size), o_weight.float()).reshape(-1).to(torch.float16)
    return out, k_new.reshape(1, -1).to(torch.float16), v_new.reshape(1, -1).to(torch.float16)


def _project_rotated_kv(
    hidden: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    position_offset: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = hidden.shape[-1]
    head_dim = hidden_size // num_heads
    k = F.linear(hidden.float(), k_weight.float()).reshape(-1, num_kv_heads, head_dim)
    v = F.linear(hidden.float(), v_weight.float()).reshape(-1, num_kv_heads, head_dim)
    rotated = []
    for idx in range(k.shape[0]):
        rotated.append(_rope_one(k[idx], position_offset + idx, num_kv_heads, head_dim, rope_theta))
    return torch.stack(rotated, dim=0).reshape(k.shape[0], -1).to(torch.float16), v.reshape(v.shape[0], -1).to(torch.float16)


def test_native_kv_commit_and_rollback_roundtrip() -> None:
    from pcketlm.native import NativeKvSession, native_fp16_kv_available

    assert native_fp16_kv_available() is True
    session = NativeKvSession(layer_count=2, max_seq_len=16, kv_width=4)
    committed_k = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    committed_v = committed_k + 100
    tentative_k = torch.arange(8, dtype=torch.float16).reshape(2, 4) + 1000
    tentative_v = tentative_k + 100

    session.append_committed(0, committed_k, committed_v, count=3)
    session.append_tentative(0, tentative_k, tentative_v, count=2)
    assert session.committed_length(0) == 3
    assert session.tentative_length(0) == 2

    k_all, v_all = session.copy_layer(0)
    assert torch.equal(k_all, torch.cat([committed_k, tentative_k], dim=0))
    assert torch.equal(v_all, torch.cat([committed_v, tentative_v], dim=0))

    session.commit(1)
    assert session.committed_length(0) == 4
    assert session.tentative_length(0) == 1
    k_after_commit, _ = session.copy_layer(0)
    assert torch.equal(k_after_commit, torch.cat([committed_k, tentative_k], dim=0))

    session.rollback()
    assert session.committed_length(0) == 4
    assert session.tentative_length(0) == 0
    k_committed, v_committed = session.copy_layer(0)
    assert torch.equal(k_committed, torch.cat([committed_k, tentative_k[:1]], dim=0))
    assert torch.equal(v_committed, torch.cat([committed_v, tentative_v[:1]], dim=0))
    session.close()


def test_native_kv_rollback_replaces_tentative_suffix() -> None:
    from pcketlm.native import NativeKvSession

    session = NativeKvSession(layer_count=1, max_seq_len=16, kv_width=2)
    prefix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16)
    first = torch.tensor([[10, 11], [12, 13]], dtype=torch.float16)
    second = torch.tensor([[20, 21], [22, 23]], dtype=torch.float16)
    session.append_committed(0, prefix, prefix + 100, count=2)
    session.append_tentative(0, first, first + 100, count=2)
    session.rollback()
    session.append_tentative(0, second, second + 100, count=2)

    k_all, v_all = session.copy_layer(0)
    assert torch.equal(k_all, torch.cat([prefix, second], dim=0))
    assert torch.equal(v_all, torch.cat([prefix + 100, second + 100], dim=0))
    session.close()


def test_native_kv_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_KV", "1")
    from pcketlm.native import NativeKvSession

    try:
        NativeKvSession(layer_count=1, max_seq_len=4, kv_width=2)
    except RuntimeError as exc:
        assert "unavailable" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("PCKETLM_DISABLE_NATIVE_KV did not disable KV sessions")


def test_native_kv_cache_accepts_bfloat16_storage() -> None:
    from pcketlm.native import NativeKvSession

    session = NativeKvSession(layer_count=1, max_seq_len=8, kv_width=4, dtype=torch.bfloat16)
    committed_k = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16)
    committed_v = (committed_k.float() + 100).to(torch.bfloat16)
    tentative_k = (torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1000).to(torch.bfloat16)
    tentative_v = (tentative_k.float() + 100).to(torch.bfloat16)

    session.append_committed(0, committed_k, committed_v, count=3)
    session.append_tentative(0, tentative_k, tentative_v, count=2)
    k_all, v_all = session.copy_layer(0)
    assert k_all.dtype == torch.bfloat16
    assert v_all.dtype == torch.bfloat16
    assert torch.equal(k_all, torch.cat([committed_k, tentative_k], dim=0))
    assert torch.equal(v_all, torch.cat([committed_v, tentative_v], dim=0))
    session.rollback()
    assert session.tentative_length(0) == 0
    session.close()


def test_native_attention_decode_uses_c_owned_kv_and_tentative_append() -> None:
    from pcketlm.native import NativeKvSession

    torch.manual_seed(1234)
    hidden_size = 8
    num_heads = 2
    num_kv_heads = 1
    kv_width = hidden_size // num_heads * num_kv_heads
    session = NativeKvSession(layer_count=1, max_seq_len=8, kv_width=kv_width)
    prefix_k = torch.randn((2, kv_width), dtype=torch.float16)
    prefix_v = torch.randn((2, kv_width), dtype=torch.float16)
    hidden = torch.randn((hidden_size,), dtype=torch.float16)
    q_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.2).to(torch.float16)
    k_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.2).to(torch.float16)
    v_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.2).to(torch.float16)
    o_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.2).to(torch.float16)
    session.append_committed(0, prefix_k, prefix_v, count=2)

    native = session.attention_decode_fp16(
        0,
        hidden,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rope_theta=10000.0,
    )
    expected, k_new, v_new = _reference_decode(
        hidden,
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

    assert session.committed_length(0) == 2
    assert session.tentative_length(0) == 1
    k_all, v_all = session.copy_layer(0)
    assert torch.allclose(k_all[-1].float(), k_new.reshape(-1).float(), atol=1e-3, rtol=1e-3)
    assert torch.allclose(v_all[-1].float(), v_new.reshape(-1).float(), atol=1e-3, rtol=1e-3)
    assert torch.allclose(native.float(), expected.float(), atol=1e-3, rtol=1e-3)
    session.close()


def test_native_attention_decode_matches_python_with_projection_biases() -> None:
    from pcketlm.native import NativeKvSession

    torch.manual_seed(1357)
    hidden_size = 8
    num_heads = 2
    num_kv_heads = 1
    kv_width = hidden_size // num_heads * num_kv_heads
    session = NativeKvSession(layer_count=1, max_seq_len=8, kv_width=kv_width, dtype=torch.bfloat16)
    prefix_k = torch.randn((2, kv_width), dtype=torch.float32).to(torch.bfloat16)
    prefix_v = torch.randn((2, kv_width), dtype=torch.float32).to(torch.bfloat16)
    hidden = torch.randn((hidden_size,), dtype=torch.float32).to(torch.bfloat16)
    q_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    k_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    v_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    o_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    q_bias = (torch.randn((hidden_size,), dtype=torch.float32) * 0.03).to(torch.bfloat16)
    k_bias = (torch.randn((kv_width,), dtype=torch.float32) * 0.03).to(torch.bfloat16)
    v_bias = (torch.randn((kv_width,), dtype=torch.float32) * 0.03).to(torch.bfloat16)
    session.append_committed(0, prefix_k, prefix_v, count=2)

    native = session.attention_decode_fp16(
        0,
        hidden,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rope_theta=10000.0,
        q_bias=q_bias,
        k_bias=k_bias,
        v_bias=v_bias,
    )
    expected, k_new, v_new = _reference_decode(
        hidden,
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
    )
    k_all, v_all = session.copy_layer(0)
    assert native.dtype == torch.bfloat16
    assert torch.allclose(k_all[-1].float(), k_new.reshape(-1).float(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(v_all[-1].float(), v_new.reshape(-1).float(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(native.float(), expected.float(), atol=3e-2, rtol=3e-2)
    session.close()


def test_native_attention_decode_matches_python_with_qk_norm() -> None:
    from pcketlm.native import NativeKvSession

    torch.manual_seed(8642)
    hidden_size = 16
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    kv_width = head_dim * num_kv_heads
    session = NativeKvSession(layer_count=1, max_seq_len=8, kv_width=kv_width, dtype=torch.bfloat16)
    prefix_k = torch.randn((2, kv_width), dtype=torch.float32).to(torch.bfloat16)
    prefix_v = torch.randn((2, kv_width), dtype=torch.float32).to(torch.bfloat16)
    hidden = torch.randn((hidden_size,), dtype=torch.float32).to(torch.bfloat16)
    q_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    k_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    v_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    o_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    q_norm = (torch.rand((head_dim,), dtype=torch.float32) + 0.5).to(torch.bfloat16)
    k_norm = (torch.rand((head_dim,), dtype=torch.float32) + 0.5).to(torch.bfloat16)
    session.append_committed(0, prefix_k, prefix_v, count=2)

    native = session.attention_decode_fp16(
        0,
        hidden,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rope_theta=10000.0,
        q_norm_weight=q_norm,
        k_norm_weight=k_norm,
        rms_eps=1e-6,
    )
    expected, k_new, v_new = _reference_decode(
        hidden,
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
        q_norm_weight=q_norm,
        k_norm_weight=k_norm,
        rms_eps=1e-6,
    )
    k_all, v_all = session.copy_layer(0)
    assert torch.allclose(k_all[-1].float(), k_new.reshape(-1).float(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(v_all[-1].float(), v_new.reshape(-1).float(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(native.float(), expected.float(), atol=3e-2, rtol=3e-2)
    session.close()


def test_native_attention_decode_accepts_bfloat16_model_tensors() -> None:
    from pcketlm.native import NativeKvSession

    torch.manual_seed(4321)
    hidden_size = 8
    num_heads = 2
    num_kv_heads = 1
    kv_width = hidden_size // num_heads * num_kv_heads
    session = NativeKvSession(layer_count=1, max_seq_len=8, kv_width=kv_width, dtype=torch.bfloat16)
    prefix_k = torch.randn((2, kv_width), dtype=torch.float32).to(torch.bfloat16)
    prefix_v = torch.randn((2, kv_width), dtype=torch.float32).to(torch.bfloat16)
    hidden = torch.randn((hidden_size,), dtype=torch.float32).to(torch.bfloat16)
    q_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    k_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    v_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    o_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    session.append_committed(0, prefix_k, prefix_v, count=2)

    native = session.attention_decode_fp16(
        0,
        hidden,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rope_theta=10000.0,
    )
    expected, k_new, v_new = _reference_decode(
        hidden.to(torch.float16),
        q_weight.to(torch.float16),
        k_weight.to(torch.float16),
        v_weight.to(torch.float16),
        o_weight.to(torch.float16),
        prefix_k.to(torch.float16),
        prefix_v.to(torch.float16),
        position=2,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=10000.0,
    )

    assert native.dtype == torch.bfloat16
    assert session.tentative_length(0) == 1
    k_all, v_all = session.copy_layer(0)
    assert k_all.dtype == torch.bfloat16
    assert v_all.dtype == torch.bfloat16
    assert torch.allclose(k_all[-1].float(), k_new.reshape(-1).float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(v_all[-1].float(), v_new.reshape(-1).float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(native.float(), expected.float(), atol=2e-2, rtol=2e-2)
    session.close()


def test_native_kv_prefill_decode_commit_decode_matches_python_full_context_attention() -> None:
    from pcketlm.native import NativeKvSession

    torch.manual_seed(2468)
    hidden_size = 8
    num_heads = 2
    num_kv_heads = 1
    kv_width = hidden_size // num_heads * num_kv_heads
    session = NativeKvSession(layer_count=1, max_seq_len=16, kv_width=kv_width)
    hidden = torch.randn((7, hidden_size), dtype=torch.float16)
    q_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.2).to(torch.float16)
    k_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.2).to(torch.float16)
    v_weight = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.2).to(torch.float16)
    o_weight = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.2).to(torch.float16)

    prefix_k, prefix_v = _project_rotated_kv(hidden[:5], k_weight, v_weight, num_heads, num_kv_heads, 0, 10000.0)
    session.append_committed(0, prefix_k, prefix_v, count=5)

    native_6 = session.attention_decode_fp16(
        0,
        hidden[5],
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rope_theta=10000.0,
    )
    expected_6, _k6, _v6 = _reference_decode(
        hidden[5],
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        prefix_k,
        prefix_v,
        position=5,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=10000.0,
    )
    assert torch.allclose(native_6.float(), expected_6.float(), atol=1e-3, rtol=1e-3)
    session.commit(1)

    committed_k, committed_v = session.copy_layer(0, include_tentative=False)
    native_7 = session.attention_decode_fp16(
        0,
        hidden[6],
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rope_theta=10000.0,
    )
    expected_7, _k7, _v7 = _reference_decode(
        hidden[6],
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        committed_k,
        committed_v,
        position=6,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=10000.0,
    )
    assert torch.allclose(native_7.float(), expected_7.float(), atol=1e-3, rtol=1e-3)
    assert session.committed_length(0) == 6
    assert session.tentative_length(0) == 1
    session.close()
