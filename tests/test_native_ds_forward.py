import torch

import pcketlm.native as native


def _tiny_oracle_logits(token_id: int, *, vocab_size: int = 9) -> torch.Tensor:
    next_token = (int(token_id) + 2) % int(vocab_size)
    logits = torch.full((vocab_size,), -10.0, dtype=torch.float32)
    logits[next_token] = 10.0
    logits[(next_token + 1) % vocab_size] = 2.5
    return logits


def _greedy_sequence(start: int, steps: int, decode_fn) -> list[int]:
    tokens = []
    current = int(start)
    for _ in range(int(steps)):
        logits = decode_fn(current)
        current = int(torch.argmax(logits).item())
        tokens.append(current)
    return tokens


def test_ds_forward_decode_tiny_oracle_matches_python_greedy_sequence() -> None:
    with native.DeepSeekNativeSession(num_layers=2, hidden_dim=4, num_experts=4, top_k=2) as session:
        native_tokens = _greedy_sequence(
            1,
            5,
            lambda token_id: native.ds_forward_decode(session, token_id, _tiny_oracle_logits, vocab_size=9),
        )
        python_tokens = _greedy_sequence(1, 5, _tiny_oracle_logits)

        assert native_tokens == python_tokens == [3, 5, 7, 0, 2]
        assert session.monolithic_call_count() == 5
        assert session.layers_executed_count() == 10
        assert session.callback_invocation_count() == 5


def test_ds_forward_prefill_and_verify_copy_callback_logits() -> None:
    def prefill(tokens: list[int]) -> torch.Tensor:
        return _tiny_oracle_logits(sum(tokens), vocab_size=7)

    def verify(tokens: list[int]) -> torch.Tensor:
        return torch.stack([_tiny_oracle_logits(token, vocab_size=7) for token in tokens])

    with native.DeepSeekNativeSession(num_layers=2, hidden_dim=4, num_experts=4, top_k=2) as session:
        prefill_logits = native.ds_forward_prefill(session, [1, 2, 3], prefill, vocab_size=7)
        verify_logits = native.ds_forward_verify(session, [4, 5], verify, vocab_size=7)

        assert torch.equal(prefill_logits, prefill([1, 2, 3]))
        assert torch.equal(verify_logits, verify([4, 5]))
        assert session.monolithic_call_count() == 2
        assert session.layers_executed_count() == 10
        assert session.callback_invocation_count() == 2


def test_ds_forward_decode_bridge_copies_real_deepseek_topk_logits() -> None:
    from pcketlm.core.runtime.fp8_source import run_fp8_decode_loop

    captured: dict[str, object] = {}

    def deepseek_topk_logits(_token_id: int) -> torch.Tensor:
        result = run_fp8_decode_loop("deepseek-v3", [0, 1], layer_count=8, max_new_tokens=1)
        assert result.ready is True
        assert result.final_top_token_ids
        captured["top_ids"] = list(result.final_top_token_ids)
        captured["top_logits"] = list(result.final_top_logits)
        return torch.tensor(result.final_top_logits, dtype=torch.float32)

    with native.DeepSeekNativeSession(num_layers=62, hidden_dim=7168, num_experts=256, top_k=8) as session:
        bridged_logits = native.ds_forward_decode(session, 1, deepseek_topk_logits, vocab_size=5)

        expected = torch.tensor(captured["top_logits"], dtype=torch.float32)
        assert torch.equal(bridged_logits, expected)
        assert captured["top_ids"] == [0, 20917, 4178, 94986, 43873]
        assert session.monolithic_call_count() == 1
        assert session.layers_executed_count() == 62
