import torch


def test_cached_pack_weight_rows8_reuses_packed_tensor(monkeypatch) -> None:
    from pcketlm.native import cached_pack_weight_rows8, packed_gemv_cache_stats, reset_packed_gemv_cache

    reset_packed_gemv_cache()
    monkeypatch.setenv("PCKETLM_PACKED_GEMV_CACHE_MB", "1")
    weight = torch.arange(64, dtype=torch.float16).reshape(8, 8)

    first = cached_pack_weight_rows8("layer0.down", weight)
    second = cached_pack_weight_rows8("layer0.down", weight.clone())

    assert first.data_ptr() == second.data_ptr()
    stats = packed_gemv_cache_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["stores"] == 1
    reset_packed_gemv_cache()


def test_cached_pack_weight_rows8_lru_evicts_under_budget(monkeypatch) -> None:
    from pcketlm.native import cached_pack_weight_rows8, packed_gemv_cache_stats, reset_packed_gemv_cache

    reset_packed_gemv_cache()
    monkeypatch.setenv("PCKETLM_PACKED_GEMV_CACHE_MB", "1")
    a = torch.ones((256, 1024), dtype=torch.float16)
    b = torch.ones((256, 1024), dtype=torch.float16)
    c = torch.ones((256, 1024), dtype=torch.float16)

    cached_pack_weight_rows8("a", a)
    cached_pack_weight_rows8("b", b)
    cached_pack_weight_rows8("c", c)

    stats = packed_gemv_cache_stats()
    assert stats["stores"] == 3
    assert stats["evictions"] >= 1
    assert stats["resident_bytes"] <= stats["budget_bytes"]
    reset_packed_gemv_cache()


def test_cached_pack_weight_rows8_kill_switch(monkeypatch) -> None:
    from pcketlm.native import cached_pack_weight_rows8, packed_gemv_cache_stats, reset_packed_gemv_cache

    reset_packed_gemv_cache()
    monkeypatch.setenv("PCKETLM_PACKED_GEMV_CACHE_MB", "1")
    monkeypatch.setenv("PCKETLM_DISABLE_PACKED_GEMV_CACHE", "1")
    weight = torch.ones((8, 8), dtype=torch.float16)

    first = cached_pack_weight_rows8("same", weight)
    second = cached_pack_weight_rows8("same", weight)

    assert first.data_ptr() != second.data_ptr()
    stats = packed_gemv_cache_stats()
    assert stats["resident_count"] == 0
    assert stats["misses"] == 2
    reset_packed_gemv_cache()


def test_dense_layer_decode_packed_rows8_matches_unpacked(monkeypatch) -> None:
    from pcketlm.native import NativeKvSession, cached_pack_weight_rows8, reset_packed_gemv_cache

    reset_packed_gemv_cache()
    monkeypatch.setenv("PCKETLM_PACKED_GEMV_CACHE_MB", "4")
    torch.manual_seed(777)
    hidden_size = 16
    intermediate_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 4
    kv_width = num_kv_heads * head_dim
    hidden = (torch.randn((hidden_size,), dtype=torch.float32) * 0.1).to(torch.float16)
    norm = torch.ones((hidden_size,), dtype=torch.float16)
    q = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.1).to(torch.float16)
    k = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.1).to(torch.float16)
    v = (torch.randn((kv_width, hidden_size), dtype=torch.float32) * 0.1).to(torch.float16)
    o = (torch.randn((hidden_size, hidden_size), dtype=torch.float32) * 0.1).to(torch.float16)
    gate = (torch.randn((intermediate_size, hidden_size), dtype=torch.float32) * 0.1).to(torch.float16)
    up = (torch.randn((intermediate_size, hidden_size), dtype=torch.float32) * 0.1).to(torch.float16)
    down = (torch.randn((hidden_size, intermediate_size), dtype=torch.float32) * 0.1).to(torch.float16)

    unpacked_session = NativeKvSession(layer_count=1, max_seq_len=8, kv_width=kv_width)
    packed_session = NativeKvSession(layer_count=1, max_seq_len=8, kv_width=kv_width)
    try:
        unpacked = unpacked_session.dense_layer_decode_fp16(
            0, hidden, norm, norm, q, k, v, o, gate, up, down,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            rms_eps=1e-6,
            rope_theta=10000.0,
        )
        packed = packed_session.dense_layer_decode_packed_rows8(
            0,
            hidden,
            norm,
            norm,
            cached_pack_weight_rows8("q", q),
            cached_pack_weight_rows8("k", k),
            cached_pack_weight_rows8("v", v),
            cached_pack_weight_rows8("o", o),
            cached_pack_weight_rows8("gate", gate),
            cached_pack_weight_rows8("up", up),
            cached_pack_weight_rows8("down", down),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_eps=1e-6,
            rope_theta=10000.0,
        )
    finally:
        unpacked_session.close()
        packed_session.close()
        reset_packed_gemv_cache()
    assert torch.equal(packed, unpacked)
