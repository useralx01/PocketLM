from __future__ import annotations

import time

import pytest
import torch
import torch.nn.functional as F

from pcketlm import native


def _rms_norm(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def _rope(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] == 0:
        return values
    pair = values.float().reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
    x0 = pair[..., 0]
    x1 = pair[..., 1]
    rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return rotated.flatten(-2)


def _python_fused_reference(
    hidden: torch.Tensor,
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    kv_a: torch.Tensor,
    kv_b: torch.Tensor,
    o_proj: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    previous_kv: torch.Tensor | None,
    previous_pe: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    qk_nope: int,
    qk_rope: int,
    v_head_dim: int,
    kv_lora_rank: int,
    rms_eps: float,
    softmax_scale: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    q_low = F.linear(hidden.reshape(1, -1).float(), q_a.float())
    q_low = _rms_norm(q_low, q_norm.float(), rms_eps)
    q = F.linear(q_low, q_b.float()).view(num_heads, qk_nope + qk_rope)
    q_nope, q_pe = torch.split(q, [qk_nope, qk_rope], dim=-1)
    q_pe = _rope(q_pe, cos, sin)
    kv = F.linear(hidden.reshape(1, -1).float(), kv_a.float()).reshape(-1)
    kv_latent = _rms_norm(kv[:kv_lora_rank], kv_norm.float(), rms_eps)
    k_pe = _rope(kv[kv_lora_rank:].reshape(1, qk_rope), cos, sin).reshape(-1)
    if previous_kv is None:
        kv_cache = kv_latent.reshape(1, kv_lora_rank)
        pe_cache = k_pe.reshape(1, qk_rope)
    else:
        assert previous_pe is not None
        kv_cache = torch.cat([previous_kv.float(), kv_latent.reshape(1, kv_lora_rank)], dim=0)
        pe_cache = torch.cat([previous_pe.float(), k_pe.reshape(1, qk_rope)], dim=0)
    wkv_b = kv_b.float().view(num_heads, qk_nope + v_head_dim, kv_lora_rank)
    q_abs = torch.einsum("hd,hdc->hc", q_nope, wkv_b[:, :qk_nope])
    scores = (torch.einsum("hc,tc->ht", q_abs, kv_cache) + torch.einsum("hr,tr->ht", q_pe, pe_cache)) * softmax_scale
    probs = scores.softmax(dim=-1, dtype=torch.float32)
    latent = torch.einsum("ht,tc->hc", probs, kv_cache)
    heads = torch.einsum("hc,hdc->hd", latent, wkv_b[:, qk_nope:])
    return F.linear(heads.reshape(1, -1), o_proj.float()).reshape(-1), (kv_cache, pe_cache)


def test_native_fused_attention_block_matches_python_reference() -> None:
    torch.manual_seed(12)
    hidden_dim = 16
    q_rank = 6
    kv_rank = 5
    heads = 4
    qk_nope = 3
    qk_rope = 4
    v_dim = 2
    hidden = torch.randn(hidden_dim)
    q_a = torch.randn(q_rank, hidden_dim) * 0.02
    q_b = torch.randn(heads * (qk_nope + qk_rope), q_rank) * 0.02
    kv_a = torch.randn(kv_rank + qk_rope, hidden_dim) * 0.02
    kv_b = torch.randn(heads * (qk_nope + v_dim), kv_rank) * 0.02
    o_proj = torch.randn(hidden_dim, heads * v_dim) * 0.02
    q_norm = torch.ones(q_rank)
    kv_norm = torch.ones(kv_rank)
    previous_kv = torch.randn(3, kv_rank) * 0.02
    previous_pe = torch.randn(3, qk_rope) * 0.02
    angles = torch.arange(qk_rope // 2, dtype=torch.float32) * 0.1
    cos = angles.cos()
    sin = angles.sin()
    kwargs = dict(
        q_lora_rank=q_rank,
        kv_lora_rank=kv_rank,
        num_heads=heads,
        qk_nope_dim=qk_nope,
        qk_rope_dim=qk_rope,
        v_head_dim=v_dim,
        rms_eps=1e-6,
        softmax_scale=0.25,
    )

    expected, expected_cache = _python_fused_reference(
        hidden,
        q_a,
        q_b,
        kv_a,
        kv_b,
        o_proj,
        q_norm,
        kv_norm,
        previous_kv,
        previous_pe,
        cos,
        sin,
        num_heads=heads,
        qk_nope=qk_nope,
        qk_rope=qk_rope,
        v_head_dim=v_dim,
        kv_lora_rank=kv_rank,
        rms_eps=1e-6,
        softmax_scale=0.25,
    )
    with native.DeepSeekNativeSession(num_layers=3, hidden_dim=hidden_dim, num_experts=0, top_k=0) as session:
        actual, actual_cache = native.ds_attention_block_forward(
            session,
            hidden,
            q_a,
            q_b,
            kv_a,
            kv_b,
            o_proj,
            q_norm,
            kv_norm,
            previous_kv_cache=previous_kv,
            previous_pe_cache=previous_pe,
            rope_cos=cos,
            rope_sin=sin,
            **kwargs,
        )
        assert session.fused_attention_invocation_count() == 1

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_cache[0], expected_cache[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_cache[1], expected_cache[1], atol=1e-6, rtol=1e-6)


def test_native_fused_attention_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_FUSED_DS_ATTENTION", "1")
    assert native.native_fused_ds_attention_available() is False


def test_native_fused_attention_microbench_smoke() -> None:
    torch.manual_seed(13)
    hidden_dim = 64
    q_rank = 16
    kv_rank = 16
    heads = 4
    qk_nope = 8
    qk_rope = 8
    v_dim = 8
    hidden = torch.randn(hidden_dim)
    q_a = torch.randn(q_rank, hidden_dim) * 0.01
    q_b = torch.randn(heads * (qk_nope + qk_rope), q_rank) * 0.01
    kv_a = torch.randn(kv_rank + qk_rope, hidden_dim) * 0.01
    kv_b = torch.randn(heads * (qk_nope + v_dim), kv_rank) * 0.01
    o_proj = torch.randn(hidden_dim, heads * v_dim) * 0.01
    q_norm = torch.ones(q_rank)
    kv_norm = torch.ones(kv_rank)
    previous_kv = torch.randn(32, kv_rank) * 0.01
    previous_pe = torch.randn(32, qk_rope) * 0.01
    cos = torch.ones(qk_rope // 2)
    sin = torch.zeros(qk_rope // 2)

    with native.DeepSeekNativeSession(num_layers=3, hidden_dim=hidden_dim, num_experts=0, top_k=0) as session:
        start = time.perf_counter()
        for _ in range(3):
            native.ds_attention_block_forward(
                session,
                hidden,
                q_a,
                q_b,
                kv_a,
                kv_b,
                o_proj,
                q_norm,
                kv_norm,
                previous_kv_cache=previous_kv,
                previous_pe_cache=previous_pe,
                rope_cos=cos,
                rope_sin=sin,
                q_lora_rank=q_rank,
                kv_lora_rank=kv_rank,
                num_heads=heads,
                qk_nope_dim=qk_nope,
                qk_rope_dim=qk_rope,
                v_head_dim=v_dim,
                rms_eps=1e-6,
                softmax_scale=0.125,
            )
        elapsed = time.perf_counter() - start

    assert elapsed < 5.0
