from types import SimpleNamespace

import torch

from pcketlm.core.runtime import speculative
from pcketlm.core.runtime.speculative import CandidateProposal, VerifierBatchResult


def test_propose_candidates_uses_gguf_and_reencodes_for_verifier(monkeypatch) -> None:
    calls = {}

    monkeypatch.setattr(speculative, "start_gguf_server", lambda model_id: calls.setdefault("server", model_id))
    monkeypatch.setattr(speculative, "decode_token_ids_to_text", lambda model_id, token_ids: ("prompt text", []))
    monkeypatch.setattr(speculative, "encode_prompt_text", lambda model_id, text: ([101, 102, 103], []))

    def fake_runner(model_id: str, prompt: str, **kwargs):
        calls.update({"model_id": model_id, "prompt": prompt, "kwargs": kwargs})
        return SimpleNamespace(
            to_dict=lambda: {
                "ready": True,
                "generated_text": "candidate text",
                "blockers": [],
            }
        )

    result = speculative.propose_candidates(
        "qwen2.5-14b-instruct",
        [1, 2, 3],
        2,
        verifier_model_id="qwen3-30b-a3b",
        gguf_runner=fake_runner,
    )

    assert result.ready is True
    assert result.token_ids == [101, 102]
    assert calls["server"] == "qwen2.5-14b-instruct"
    assert calls["model_id"] == "qwen2.5-14b-instruct"
    assert calls["prompt"] == "prompt text"
    assert calls["kwargs"]["max_tokens"] == 2


def test_propose_candidates_falls_back_to_direct_paged_speculator(monkeypatch) -> None:
    calls = {}

    monkeypatch.setattr(speculative, "_has_gguf_artifact", lambda _model_id: False)
    monkeypatch.setattr(speculative, "decode_token_ids_to_text", lambda model_id, token_ids: ("prompt text", []))
    monkeypatch.setattr(speculative, "encode_prompt_text", lambda model_id, text: ([151667, 198, 42], []))

    def fake_direct(model_id: str, prompt: str, **kwargs):
        calls.update({"model_id": model_id, "prompt": prompt, "kwargs": kwargs})
        return SimpleNamespace(
            ready=True,
            generated_text="<think>\n",
            blockers=[],
        )

    monkeypatch.setattr(speculative, "run_prompt_decode_loop", fake_direct)

    result = speculative.propose_candidates(
        "qwen3-1.7b",
        [1, 2, 3],
        2,
        verifier_model_id="qwen3-30b-a3b",
    )

    assert result.ready is True
    assert result.backend == "direct-paged"
    assert result.token_ids == [151667, 198]
    assert calls["model_id"] == "qwen3-1.7b"
    assert calls["prompt"] == "prompt text"
    assert calls["kwargs"]["apply_chat_format"] is False
    assert calls["kwargs"]["max_new_tokens"] == 2


def test_verify_candidates_once_runs_one_stack_pass_and_returns_k_plus_one(monkeypatch) -> None:
    calls = {"stack": 0, "tail_positions": []}

    monkeypatch.setattr(
        speculative,
        "load_layer_bridge_config",
        lambda _model_id: SimpleNamespace(ready=True, blockers=[], num_hidden_layers=2),
    )
    monkeypatch.setattr(
        speculative,
        "load_token_entry_hidden_state",
        lambda _model_id, token_ids: (torch.zeros((1, len(token_ids), 4), dtype=torch.float32), []),
    )

    def fake_stack(*_args, input_hidden, **_kwargs):
        calls["stack"] += 1
        return SimpleNamespace(
            ready=True,
            output_tensor=torch.arange(input_hidden.shape[1] * 4, dtype=torch.float32).view(1, input_hidden.shape[1], 4),
            executed_layers=[0, 1],
            blockers=[],
        )

    def fake_tail(_model_id, hidden_state, **_kwargs):
        position_marker = int(hidden_state.reshape(-1)[0].item() // 4)
        calls["tail_positions"].append(position_marker)
        return SimpleNamespace(ready=True, top_token_ids=[100 + position_marker], blockers=[])

    monkeypatch.setattr(speculative, "run_layer_bridge_stack", fake_stack)
    monkeypatch.setattr(speculative, "run_decode_tail", fake_tail)

    result = speculative.verify_candidates_once("qwen3-30b-a3b", [10, 11, 12], [20, 21, 22])

    assert result.ready is True
    assert calls["stack"] == 1
    assert calls["tail_positions"] == [2, 3, 4, 5]
    assert result.verifier_token_ids == [102, 103, 104, 105]
    assert result.layers_executed == 2
    assert result.expected_layers_executed == 2


def test_speculative_generate_accepts_all_candidates_and_bonus(monkeypatch) -> None:
    monkeypatch.setattr(
        speculative,
        "prepare_prompt_text",
        lambda *_args, **_kwargs: SimpleNamespace(ready=True, blockers=[], prepared_prompt="prompt", token_ids=[10]),
    )
    monkeypatch.setattr(
        speculative,
        "decode_token_ids_to_text",
        lambda _model_id, token_ids: (" ".join(str(token_id) for token_id in token_ids), []),
    )

    def proposer(_speculator, _current_ids, k, **_kwargs):
        return CandidateProposal(
            speculator_model_id="spec",
            prompt_text="prompt",
            generated_text="1 2",
            token_ids=[1, 2][:k],
            ready=True,
        )

    def verifier(_model_id, _current_ids, candidates):
        return VerifierBatchResult(
            model_id="verifier",
            prompt_token_count=1,
            candidate_token_ids=list(candidates),
            verifier_token_ids=list(candidates) + [3],
            logits_count=len(candidates) + 1,
            layers_executed=2,
            expected_layers_executed=2,
            elapsed_seconds=0.01,
            ready=True,
        )

    result = speculative.speculative_generate("verifier", "spec", "prompt", 3, k=2, proposer=proposer, verifier=verifier)

    assert result.ready is True
    assert result.generated_token_ids == [1, 2, 3]
    assert result.verifier_passes == 1
    assert result.accepted_token_count == 2
    assert result.corrected_token_count == 1
    assert result.anti_cheat_passed is True


def test_speculative_generate_rejects_at_first_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        speculative,
        "prepare_prompt_text",
        lambda *_args, **_kwargs: SimpleNamespace(ready=True, blockers=[], prepared_prompt="prompt", token_ids=[10]),
    )
    monkeypatch.setattr(
        speculative,
        "decode_token_ids_to_text",
        lambda _model_id, token_ids: (" ".join(str(token_id) for token_id in token_ids), []),
    )

    def proposer(_speculator, _current_ids, k, **_kwargs):
        return CandidateProposal(
            speculator_model_id="spec",
            prompt_text="prompt",
            generated_text="1 99",
            token_ids=[1, 99][:k],
            ready=True,
        )

    def verifier(_model_id, _current_ids, candidates):
        return VerifierBatchResult(
            model_id="verifier",
            prompt_token_count=1,
            candidate_token_ids=list(candidates),
            verifier_token_ids=[1, 2, 3],
            logits_count=3,
            layers_executed=2,
            expected_layers_executed=2,
            elapsed_seconds=0.01,
            ready=True,
        )

    result = speculative.speculative_generate("verifier", "spec", "prompt", 2, k=2, proposer=proposer, verifier=verifier)

    assert result.ready is True
    assert result.generated_token_ids == [1, 2]
    assert result.accepted_token_count == 1
    assert result.corrected_token_count == 1
