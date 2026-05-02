from types import SimpleNamespace

import torch

from pcketlm.core.runtime import speculative
from pcketlm.core.runtime.speculative import CandidateProposal, VerifierBatchResult


def _kv_length(kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> int:
    if not kv_caches:
        return 0
    first_key = next(iter(kv_caches.values()))[0]
    return int(first_key.shape[-2])


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


def test_propose_candidates_backend_env_can_force_direct(monkeypatch) -> None:
    calls = {}

    monkeypatch.setenv("PCKETLM_SPECULATOR_BACKEND", "direct")
    monkeypatch.setattr(speculative, "_has_gguf_artifact", lambda _model_id: True)
    monkeypatch.setattr(speculative, "decode_token_ids_to_text", lambda model_id, token_ids: ("prompt text", []))
    monkeypatch.setattr(speculative, "encode_prompt_text", lambda model_id, text: ([1], []))
    monkeypatch.setattr(speculative, "start_gguf_server", lambda _model_id: calls.setdefault("gguf_started", True))

    def fake_direct(model_id: str, prompt: str, **kwargs):
        calls["direct_model_id"] = model_id
        return SimpleNamespace(ready=True, generated_text="<think>", blockers=[])

    monkeypatch.setattr(speculative, "run_prompt_decode_loop", fake_direct)

    result = speculative.propose_candidates("qwen3-1.7b", [1, 2], 1, verifier_model_id="qwen3-30b-a3b")

    assert result.ready is True
    assert result.backend == "direct-paged"
    assert calls["direct_model_id"] == "qwen3-1.7b"
    assert "gguf_started" not in calls


def test_session_prefill_then_single_verify_uses_existing_kv(monkeypatch) -> None:
    calls = []
    tail_tokens = iter([100, 101])

    monkeypatch.setattr(
        speculative,
        "load_layer_bridge_config",
        lambda _model_id: SimpleNamespace(ready=True, blockers=[], num_hidden_layers=2),
    )
    monkeypatch.setattr(
        speculative,
        "load_token_entry_hidden_state",
        lambda _model_id, token_ids: (torch.zeros((1, len(token_ids), 4)), []),
    )

    def fake_stack(_model_id, *, input_hidden, past_key_values=None, position_offset=None, **_kwargs):
        past_len = _kv_length({} if past_key_values is None else past_key_values)
        calls.append({"seq_len": int(input_hidden.shape[1]), "past_len": past_len, "position_offset": position_offset})
        total_len = past_len + int(input_hidden.shape[1])
        kv = {
            layer: (
                torch.zeros((1, 1, total_len, 2)),
                torch.zeros((1, 1, total_len, 2)),
            )
            for layer in (0, 1)
        }
        return SimpleNamespace(
            ready=True,
            output_tensor=torch.zeros((1, int(input_hidden.shape[1]), 4)),
            executed_layers=[0, 1],
            next_kv_caches=kv,
            blockers=[],
        )

    monkeypatch.setattr(speculative, "run_layer_bridge_stack", fake_stack)
    monkeypatch.setattr(
        speculative,
        "run_decode_tail",
        lambda *_args, **_kwargs: SimpleNamespace(ready=True, top_token_ids=[next(tail_tokens)], blockers=[]),
    )

    session = speculative.SpeculativeSession("qwen3-30b-a3b")
    prefill = session.prefill([1, 2, 3, 4, 5])
    verified = session.verify_candidates([100])

    assert prefill.ready is True
    assert verified.ready is True
    assert verified.verifier_token_ids == [100, 101]
    assert calls == [
        {"seq_len": 5, "past_len": 0, "position_offset": None},
        {"seq_len": 1, "past_len": 5, "position_offset": 5},
    ]


def test_session_verify_then_commit_advances_state(monkeypatch) -> None:
    tail_tokens = iter([10, 11, 12])

    monkeypatch.setattr(
        speculative,
        "load_layer_bridge_config",
        lambda _model_id: SimpleNamespace(ready=True, blockers=[], num_hidden_layers=2),
    )
    monkeypatch.setattr(
        speculative,
        "load_token_entry_hidden_state",
        lambda _model_id, token_ids: (torch.zeros((1, len(token_ids), 4)), []),
    )
    monkeypatch.setattr(
        speculative,
        "run_decode_tail",
        lambda *_args, **_kwargs: SimpleNamespace(ready=True, top_token_ids=[next(tail_tokens)], blockers=[]),
    )

    def fake_stack(_model_id, *, input_hidden, past_key_values=None, **_kwargs):
        total_len = _kv_length({} if past_key_values is None else past_key_values) + int(input_hidden.shape[1])
        kv = {0: (torch.zeros((1, 1, total_len, 2)), torch.zeros((1, 1, total_len, 2)))}
        return SimpleNamespace(
            ready=True,
            output_tensor=torch.zeros((1, int(input_hidden.shape[1]), 4)),
            executed_layers=[0, 1],
            next_kv_caches=kv,
            blockers=[],
        )

    monkeypatch.setattr(speculative, "run_layer_bridge_stack", fake_stack)

    session = speculative.SpeculativeSession("qwen3-30b-a3b")
    session.prefill([1, 2, 3, 4, 5])
    session.verify_candidates([10, 11])
    session.commit(2)

    assert session.committed_token_ids == [1, 2, 3, 4, 5, 10, 11]
    assert _kv_length(session.kv_caches) == 7
    assert session.next_token_id == 12


def test_session_rollback_undoes_uncommitted_verify(monkeypatch) -> None:
    calls = []
    tail_tokens = iter([50, 51, 52, 53])

    monkeypatch.setattr(
        speculative,
        "load_layer_bridge_config",
        lambda _model_id: SimpleNamespace(ready=True, blockers=[], num_hidden_layers=2),
    )
    monkeypatch.setattr(
        speculative,
        "load_token_entry_hidden_state",
        lambda _model_id, token_ids: (torch.zeros((1, len(token_ids), 4)), []),
    )
    monkeypatch.setattr(
        speculative,
        "run_decode_tail",
        lambda *_args, **_kwargs: SimpleNamespace(ready=True, top_token_ids=[next(tail_tokens)], blockers=[]),
    )

    def fake_stack(_model_id, *, input_hidden, past_key_values=None, position_offset=None, **_kwargs):
        past_len = _kv_length({} if past_key_values is None else past_key_values)
        calls.append({"seq_len": int(input_hidden.shape[1]), "past_len": past_len, "position_offset": position_offset})
        total_len = past_len + int(input_hidden.shape[1])
        kv = {0: (torch.zeros((1, 1, total_len, 2)), torch.zeros((1, 1, total_len, 2)))}
        return SimpleNamespace(
            ready=True,
            output_tensor=torch.zeros((1, int(input_hidden.shape[1]), 4)),
            executed_layers=[0, 1],
            next_kv_caches=kv,
            blockers=[],
        )

    monkeypatch.setattr(speculative, "run_layer_bridge_stack", fake_stack)

    session = speculative.SpeculativeSession("qwen3-30b-a3b")
    session.prefill([1, 2, 3, 4, 5])
    session.verify_candidates([50, 51])
    session.rollback()
    session.verify_candidates([52])

    assert session.committed_token_ids == [1, 2, 3, 4, 5]
    assert _kv_length(session.kv_caches) == 5
    assert calls[-1] == {"seq_len": 1, "past_len": 5, "position_offset": 5}


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


def test_stateful_speculative_uses_fewer_token_positions(monkeypatch) -> None:
    prompt_ids = list(range(34))

    monkeypatch.setattr(
        speculative,
        "prepare_prompt_text",
        lambda *_args, **_kwargs: SimpleNamespace(ready=True, blockers=[], prepared_prompt="prompt", token_ids=prompt_ids),
    )
    monkeypatch.setattr(
        speculative,
        "decode_token_ids_to_text",
        lambda _model_id, token_ids: (" ".join(str(token_id) for token_id in token_ids), []),
    )
    monkeypatch.setattr(speculative, "expert_residency_snapshot", lambda: {})

    class FakeSession:
        def __init__(self, verifier_model_id: str) -> None:
            self.verifier_model_id = verifier_model_id
            self.committed_token_ids = list(prompt_ids)
            self.layers_executed = 0
            self.expected_layers_executed = 0
            self.verifier_token_positions_processed = 0
            self.blockers = []
            self._last_candidates = []

        def prefill(self, token_ids):
            self.verifier_token_positions_processed += len(token_ids)
            self.layers_executed += 2
            self.expected_layers_executed += 2
            return VerifierBatchResult("verifier", len(token_ids), [], [1000], 1, 2, 2, 0.0, ready=True)

        def verify_initial_candidates(self, prompt_token_ids, candidate_token_ids):
            self.committed_token_ids = list(prompt_token_ids)
            self._last_candidates = list(candidate_token_ids)
            self.verifier_token_positions_processed += len(prompt_token_ids) + len(candidate_token_ids)
            self.layers_executed += 2
            self.expected_layers_executed += 2
            return VerifierBatchResult(
                "verifier",
                len(prompt_token_ids),
                list(candidate_token_ids),
                list(candidate_token_ids) + [9999],
                len(candidate_token_ids) + 1,
                2,
                2,
                0.0,
                ready=True,
            )

        def verify_candidates(self, candidate_token_ids):
            self._last_candidates = list(candidate_token_ids)
            self.verifier_token_positions_processed += len(candidate_token_ids)
            self.layers_executed += 2
            self.expected_layers_executed += 2
            return VerifierBatchResult(
                "verifier",
                len(self.committed_token_ids),
                list(candidate_token_ids),
                list(candidate_token_ids) + [9999],
                len(candidate_token_ids) + 1,
                2,
                2,
                0.0,
                ready=True,
            )

        def commit(self, accepted_count):
            self.committed_token_ids.extend(self._last_candidates[:accepted_count])

        def append_token(self, token_id):
            self.committed_token_ids.append(token_id)
            return VerifierBatchResult("verifier", len(self.committed_token_ids), [token_id], [9999], 1, 2, 2, 0.0, ready=True)

        def rollback(self):
            self._last_candidates = []

    def proposer(_speculator, _current_ids, k, **_kwargs):
        return CandidateProposal(
            speculator_model_id="spec",
            prompt_text="prompt",
            generated_text="ok",
            token_ids=list(range(100, 100 + k)),
            ready=True,
        )

    monkeypatch.setattr(speculative, "SpeculativeSession", FakeSession)

    result = speculative.speculative_generate("verifier", "spec", "prompt", 20, k=8, proposer=proposer)

    stateless_token_positions = (34 + 8) + (42 + 8) + (50 + 4)
    reduction = 1.0 - (result.verifier_token_positions_processed / stateless_token_positions)
    assert result.ready is True
    assert result.verifier_token_positions_processed == 54
    assert reduction > 0.60


def test_stateful_speculative_fuses_first_prefill_and_verify(monkeypatch) -> None:
    prompt_ids = [1, 2, 3, 4, 5]
    calls = {"prefill": 0, "initial": 0, "verify": 0}

    monkeypatch.setattr(
        speculative,
        "prepare_prompt_text",
        lambda *_args, **_kwargs: SimpleNamespace(ready=True, blockers=[], prepared_prompt="prompt", token_ids=prompt_ids),
    )
    monkeypatch.setattr(
        speculative,
        "decode_token_ids_to_text",
        lambda _model_id, token_ids: (" ".join(str(token_id) for token_id in token_ids), []),
    )
    monkeypatch.setattr(speculative, "expert_residency_snapshot", lambda: {})

    class FakeSession:
        def __init__(self, verifier_model_id: str) -> None:
            self.verifier_model_id = verifier_model_id
            self.committed_token_ids = list(prompt_ids)
            self.layers_executed = 0
            self.expected_layers_executed = 0
            self.verifier_token_positions_processed = 0
            self.blockers = []
            self._last_candidates = []

        def prefill(self, _token_ids):
            calls["prefill"] += 1
            raise AssertionError("stateful speculative should fuse initial prefill with first verification")

        def verify_initial_candidates(self, prompt_token_ids, candidate_token_ids):
            calls["initial"] += 1
            self.committed_token_ids = list(prompt_token_ids)
            self._last_candidates = list(candidate_token_ids)
            self.verifier_token_positions_processed += len(prompt_token_ids) + len(candidate_token_ids)
            self.layers_executed += 2
            self.expected_layers_executed += 2
            return VerifierBatchResult(
                "verifier",
                len(prompt_token_ids),
                list(candidate_token_ids),
                list(candidate_token_ids) + [999],
                len(candidate_token_ids) + 1,
                2,
                2,
                0.0,
                ready=True,
            )

        def verify_candidates(self, candidate_token_ids):
            calls["verify"] += 1
            self._last_candidates = list(candidate_token_ids)
            self.verifier_token_positions_processed += len(candidate_token_ids)
            self.layers_executed += 2
            self.expected_layers_executed += 2
            return VerifierBatchResult(
                "verifier",
                len(self.committed_token_ids),
                list(candidate_token_ids),
                list(candidate_token_ids) + [999],
                len(candidate_token_ids) + 1,
                2,
                2,
                0.0,
                ready=True,
            )

        def commit(self, accepted_count):
            self.committed_token_ids.extend(self._last_candidates[:accepted_count])

        def append_token(self, token_id):
            self.committed_token_ids.append(token_id)
            return VerifierBatchResult("verifier", len(self.committed_token_ids), [token_id], [999], 1, 2, 2, 0.0, ready=True)

        def rollback(self):
            self._last_candidates = []

    def proposer(_speculator, _current_ids, k, **_kwargs):
        return CandidateProposal(
            speculator_model_id="spec",
            prompt_text="prompt",
            generated_text="ok",
            token_ids=list(range(100, 100 + k)),
            ready=True,
        )

    monkeypatch.setattr(speculative, "SpeculativeSession", FakeSession)

    result = speculative.speculative_generate("verifier", "spec", "prompt", 8, k=8, proposer=proposer)

    assert result.ready is True
    assert calls == {"prefill": 0, "initial": 1, "verify": 0}
    assert result.layers_executed == 2
