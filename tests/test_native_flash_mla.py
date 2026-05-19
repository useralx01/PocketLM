import time

import torch

import pcketlm.native as native


def _make_mla_inputs(
    *,
    heads: int,
    cache_len: int,
    qk_nope: int,
    qk_rope: int,
    rank: int,
    v_head_dim: int,
    seed: int = 1234,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    q_nope = torch.randn((heads, qk_nope), generator=generator, dtype=torch.float32) * 0.05
    q_pe = torch.randn((heads, qk_rope), generator=generator, dtype=torch.float32) * 0.05
    kv_cache = torch.randn((cache_len, rank), generator=generator, dtype=torch.float32) * 0.05
    pe_cache = torch.randn((cache_len, qk_rope), generator=generator, dtype=torch.float32) * 0.05
    wkv_b = torch.randn((heads, qk_nope + v_head_dim, rank), generator=generator, dtype=torch.float32) * 0.05
    return q_nope, q_pe, kv_cache, pe_cache, wkv_b


def test_flash_mla_matches_python_reference_on_synthetic_fixture() -> None:
    q_nope, q_pe, kv_cache, pe_cache, wkv_b = _make_mla_inputs(
        heads=4,
        cache_len=7,
        qk_nope=3,
        qk_rope=2,
        rank=5,
        v_head_dim=3,
    )

    actual = native.ds_mla_attention_flash_forward(
        q_nope,
        q_pe,
        kv_cache,
        pe_cache,
        wkv_b,
        softmax_scale=0.25,
    )
    expected = native._python_mla_attention_flash_reference(
        q_nope,
        q_pe,
        kv_cache,
        pe_cache,
        wkv_b,
        softmax_scale=0.25,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_flash_mla_kill_switch_falls_back_to_python(monkeypatch) -> None:
    q_nope, q_pe, kv_cache, pe_cache, wkv_b = _make_mla_inputs(
        heads=2,
        cache_len=4,
        qk_nope=2,
        qk_rope=1,
        rank=4,
        v_head_dim=2,
    )
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_FLASH_MLA", "1")

    actual = native.ds_mla_attention_flash_forward(
        q_nope,
        q_pe,
        kv_cache,
        pe_cache,
        wkv_b,
        softmax_scale=0.5,
    )
    expected = native._python_mla_attention_flash_reference(
        q_nope,
        q_pe,
        kv_cache,
        pe_cache,
        wkv_b,
        softmax_scale=0.5,
    )

    assert native.native_flash_mla_available() is False
    assert torch.equal(actual, expected)


def test_flash_mla_microbench_smoke_on_deepseek_core_dims() -> None:
    q_nope, q_pe, kv_cache, pe_cache, wkv_b = _make_mla_inputs(
        heads=16,
        cache_len=64,
        qk_nope=32,
        qk_rope=16,
        rank=64,
        v_head_dim=32,
    )

    native.ds_mla_attention_flash_forward(q_nope, q_pe, kv_cache, pe_cache, wkv_b, softmax_scale=0.125)
    started = time.perf_counter()
    actual = native.ds_mla_attention_flash_forward(q_nope, q_pe, kv_cache, pe_cache, wkv_b, softmax_scale=0.125)
    native_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    expected = native._python_mla_attention_flash_reference(
        q_nope, q_pe, kv_cache, pe_cache, wkv_b, softmax_scale=0.125
    )
    python_ms = (time.perf_counter() - started) * 1000.0

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert native_ms > 0.0
    assert python_ms > 0.0
