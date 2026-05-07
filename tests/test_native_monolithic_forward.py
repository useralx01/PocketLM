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


def _dense_config() -> dict:
    return {
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1.0e-6,
        "rope_theta": 10000.0,
    }


def _tiny_dense_weights(dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260507)
    cfg = _dense_config()
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    weights: dict[str, torch.Tensor] = {
        "embed": torch.randn(vocab, hidden, generator=generator, dtype=torch.float32).mul(0.05).to(dtype),
        "final_norm": torch.ones(hidden, dtype=dtype),
        "lm_head": torch.randn(vocab, hidden, generator=generator, dtype=torch.float32).mul(0.05).to(dtype),
    }
    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"layer{layer}"
        weights[f"{prefix}.input_norm"] = torch.ones(hidden, dtype=dtype)
        weights[f"{prefix}.post_norm"] = torch.ones(hidden, dtype=dtype)
        weights[f"{prefix}.q"] = torch.randn(hidden, hidden, generator=generator, dtype=torch.float32).mul(0.04).to(dtype)
        weights[f"{prefix}.k"] = torch.randn(hidden // 2, hidden, generator=generator, dtype=torch.float32).mul(0.04).to(dtype)
        weights[f"{prefix}.v"] = torch.randn(hidden // 2, hidden, generator=generator, dtype=torch.float32).mul(0.04).to(dtype)
        weights[f"{prefix}.o"] = torch.randn(hidden, hidden, generator=generator, dtype=torch.float32).mul(0.04).to(dtype)
        weights[f"{prefix}.gate"] = torch.randn(inter, hidden, generator=generator, dtype=torch.float32).mul(0.04).to(dtype)
        weights[f"{prefix}.up"] = torch.randn(inter, hidden, generator=generator, dtype=torch.float32).mul(0.04).to(dtype)
        weights[f"{prefix}.down"] = torch.randn(hidden, inter, generator=generator, dtype=torch.float32).mul(0.04).to(dtype)
    return weights


def _register_tiny_dense(session, weights: dict[str, torch.Tensor]) -> None:
    roles = {
        "embed": 0,
        "input_norm": 1,
        "post_norm": 2,
        "q": 3,
        "k": 4,
        "v": 5,
        "o": 6,
        "gate": 7,
        "up": 8,
        "down": 9,
        "final_norm": 10,
        "lm_head": 11,
    }
    session.register_tensor(layer_idx=-1, tensor_role=roles["embed"], tensor=weights["embed"])
    session.register_tensor(layer_idx=-1, tensor_role=roles["final_norm"], tensor=weights["final_norm"])
    session.register_tensor(layer_idx=-1, tensor_role=roles["lm_head"], tensor=weights["lm_head"])
    for layer in range(_dense_config()["num_hidden_layers"]):
        prefix = f"layer{layer}"
        for name in ("input_norm", "post_norm", "q", "k", "v", "o", "gate", "up", "down"):
            session.register_tensor(layer_idx=layer, tensor_role=roles[name], tensor=weights[f"{prefix}.{name}"])


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean() + eps)
    return (normalized * weight.float()).to(hidden.dtype)


def _tiny_dense_python_sequence(weights: dict[str, torch.Tensor], start_token: int, steps: int) -> list[int]:
    from pcketlm.native import NativeKvSession

    cfg = _dense_config()
    session = NativeKvSession(
        layer_count=cfg["num_hidden_layers"],
        max_seq_len=cfg["max_position_embeddings"],
        kv_width=cfg["num_key_value_heads"] * (cfg["hidden_size"] // cfg["num_attention_heads"]),
        dtype=weights["embed"].dtype,
    )
    tokens: list[int] = []
    token = start_token
    for _ in range(steps):
        hidden = weights["embed"][token].contiguous()
        for layer in range(cfg["num_hidden_layers"]):
            prefix = f"layer{layer}"
            hidden = session.dense_layer_decode_fp16(
                layer,
                hidden,
                weights[f"{prefix}.input_norm"],
                weights[f"{prefix}.post_norm"],
                weights[f"{prefix}.q"],
                weights[f"{prefix}.k"],
                weights[f"{prefix}.v"],
                weights[f"{prefix}.o"],
                weights[f"{prefix}.gate"],
                weights[f"{prefix}.up"],
                weights[f"{prefix}.down"],
                intermediate_size=cfg["intermediate_size"],
                num_attention_heads=cfg["num_attention_heads"],
                num_key_value_heads=cfg["num_key_value_heads"],
                rms_eps=cfg["rms_norm_eps"],
                rope_theta=cfg["rope_theta"],
            )
        session.commit(1)
        normed = _rms_norm(hidden, weights["final_norm"], cfg["rms_norm_eps"])
        logits = torch.mv(weights["lm_head"].float(), normed.float()).to(torch.float16)
        token = _argmax(logits)
        tokens.append(token)
    session.close()
    return tokens


def test_monolithic_session_prefill_decode_and_layer_count() -> None:
    from pcketlm.native import (
        MonolithicForwardSession,
        monolithic_call_count,
        native_monolithic_available,
        reset_monolithic_call_counts,
    )

    assert native_monolithic_available() is True
    reset_monolithic_call_counts()
    with MonolithicForwardSession(_config(), "synthetic") as session:
        assert session.kernels_ready() is True, session.kernel_error()
        prefill_logits = session.prefill([1, 2, 3])
        assert _argmax(prefill_logits) == 10
        assert session.committed_length() == 3
        assert session.layers_executed() == 9
        assert session.call_count("prefill") == 1

        decode_logits = session.decode(10)
        assert _argmax(decode_logits) == 18
        assert session.committed_length() == 4
        assert session.layers_executed() == 12
        assert session.call_count("decode") == 1
        assert session.call_count("all") == 2
        assert monolithic_call_count("prefill") == 1
        assert monolithic_call_count("decode") == 1
        assert monolithic_call_count("all") == 2


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


def test_monolithic_session_borrows_registered_weight_storage_with_python_keepalive() -> None:
    from pcketlm.native import MonolithicForwardSession

    with MonolithicForwardSession(_config(), "synthetic") as session:
        weight = torch.arange(12, dtype=torch.float16).reshape(3, 4)
        session.register_u16_tensor("model.layers.0.self_attn.q_proj.weight", weight)
        session.register_tensor(layer_idx=0, tensor_role=1, tensor=weight)
        registered_role_storage = session._tensor_keepalive["role:0:1"]

        assert session.tensor_count() == 2
        assert session.tensor_nitems("model.layers.0.self_attn.q_proj.weight") == 12
        assert session.tensor_nitems("0:1") == 12
        assert session.tensor_data_ptr("0:1") == int(registered_role_storage.data_ptr())
        assert session._tensor_keepalive
        assert session.tensor_nitems("missing") == -1
        session.clear_tensors()
        assert session.tensor_count() == 0
        assert not session._tensor_keepalive


def test_monolithic_dense_decode_matches_tiny_python_sequence() -> None:
    from pcketlm.native import MonolithicForwardSession

    weights = _tiny_dense_weights()
    expected = _tiny_dense_python_sequence(weights, start_token=3, steps=5)
    with MonolithicForwardSession(_dense_config(), "tiny-dense") as session:
        _register_tiny_dense(session, weights)
        token = 3
        actual: list[int] = []
        for _ in range(5):
            logits = session.decode(token)
            token = _argmax(logits)
            actual.append(token)

    assert actual == expected


def test_monolithic_dense_decode_matches_tiny_bf16_python_sequence() -> None:
    from pcketlm.native import MonolithicForwardSession

    cfg = dict(_dense_config())
    cfg["torch_dtype"] = "bfloat16"
    weights = _tiny_dense_weights(dtype=torch.bfloat16)
    expected = _tiny_dense_python_sequence(weights, start_token=3, steps=5)
    with MonolithicForwardSession(cfg, "tiny-dense-bf16") as session:
        _register_tiny_dense(session, weights)
        token = 3
        actual: list[int] = []
        for _ in range(5):
            logits = session.decode(token)
            token = _argmax(logits)
            actual.append(token)

    assert actual == expected


def test_monolithic_dense_prefill_runs_real_layers_and_populates_kv() -> None:
    from pcketlm.native import MonolithicForwardSession

    weights = _tiny_dense_weights()
    prompt_tokens = [3, 15]
    expected_next = _tiny_dense_python_sequence(weights, start_token=3, steps=2)[-1]
    with MonolithicForwardSession(_dense_config(), "tiny-dense-prefill") as prefill_session:
        _register_tiny_dense(prefill_session, weights)
        prefill_logits = prefill_session.prefill(prompt_tokens)
        assert _argmax(prefill_logits) == expected_next
        assert prefill_session.committed_length() == len(prompt_tokens)
        assert prefill_session.call_count("prefill") == 1
        assert prefill_session.layers_executed() == _dense_config()["num_hidden_layers"] * len(prompt_tokens)

        decode_logits = prefill_session.decode(expected_next)

    with MonolithicForwardSession(_dense_config(), "tiny-dense-decode-reference") as decode_session:
        _register_tiny_dense(decode_session, weights)
        token = prompt_tokens[0]
        for _ in prompt_tokens:
            token = _argmax(decode_session.decode(token))
        reference_logits = decode_session.decode(token)

    assert _argmax(decode_logits) == _argmax(reference_logits)


def test_monolithic_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_MONOLITHIC", "1")
    from pcketlm.native import MonolithicForwardSession

    try:
        MonolithicForwardSession(_config(), "synthetic")
    except RuntimeError as exc:
        assert "unavailable" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("PCKETLM_DISABLE_MONOLITHIC did not disable the native monolithic session")
