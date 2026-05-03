import torch


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
