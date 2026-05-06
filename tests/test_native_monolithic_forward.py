import torch


def _config() -> dict:
    return {
        "num_hidden_layers": 3,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 64,
        "max_position_embeddings": 32,
    }


def _argmax(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits).item())


def test_monolithic_session_prefill_decode_and_layer_count() -> None:
    from pcketlm.native import MonolithicForwardSession, native_monolithic_available

    assert native_monolithic_available() is True
    with MonolithicForwardSession(_config(), "synthetic") as session:
        prefill_logits = session.prefill([1, 2, 3])
        assert _argmax(prefill_logits) == 10
        assert session.committed_length() == 3
        assert session.layers_executed() == 9

        decode_logits = session.decode(10)
        assert _argmax(decode_logits) == 18
        assert session.committed_length() == 4
        assert session.layers_executed() == 12


def test_monolithic_verify_commit_and_rollback_are_stateful() -> None:
    from pcketlm.native import MonolithicForwardSession

    with MonolithicForwardSession(_config(), "synthetic") as session:
        assert _argmax(session.prefill([1, 2, 3])) == 10
        assert _argmax(session.decode(10)) == 18

        first_verify = session.verify([18, 19, 20])
        assert [_argmax(row) for row in first_verify] == [18, 28, 31, 34]
        assert session.tentative_length() == 3
        assert session.committed_length() == 4

        session.rollback()
        assert session.tentative_length() == 0
        second_verify = session.verify([18, 28, 31])
        assert [_argmax(row) for row in second_verify] == [18, 28, 40, 45]
        session.commit(2)
        assert session.committed_length() == 6
        assert session.tentative_length() == 1


def test_monolithic_session_owns_registered_weight_storage() -> None:
    from pcketlm.native import MonolithicForwardSession

    with MonolithicForwardSession(_config(), "synthetic") as session:
        weight = torch.arange(12, dtype=torch.float16).reshape(3, 4)
        session.register_u16_tensor("model.layers.0.self_attn.q_proj.weight", weight)
        weight.zero_()

        assert session.tensor_count() == 1
        assert session.tensor_nitems("model.layers.0.self_attn.q_proj.weight") == 12
        assert session.tensor_nitems("missing") == -1


def test_monolithic_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_MONOLITHIC", "1")
    from pcketlm.native import MonolithicForwardSession

    try:
        MonolithicForwardSession(_config(), "synthetic")
    except RuntimeError as exc:
        assert "unavailable" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("PCKETLM_DISABLE_MONOLITHIC did not disable the native monolithic session")
