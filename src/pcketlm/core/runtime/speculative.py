"""Speculative decoding helpers for paged verifier runtimes."""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from typing import Callable

import torch

from pcketlm.core.runtime.gguf_backend import run_gguf_prompt, start_gguf_server
from pcketlm.core.runtime.layer_bridge import (
    DEFAULT_LM_HEAD_CHUNK_ROWS,
    load_layer_bridge_config,
    load_token_entry_hidden_state,
    run_decode_tail,
    run_layer_bridge_stack,
    run_prompt_decode_loop,
)
from pcketlm.core.runtime.tensor_residency import expert_residency_snapshot
from pcketlm.core.runtime.tokenizer_runtime import (
    decode_token_ids_to_text,
    encode_prompt_text,
    prepare_prompt_text,
)
from pcketlm.core.storage.paths import artifacts_root


DEFAULT_SPECULATOR_MODEL_ID = "qwen3-1.7b"
DEFAULT_VERIFIER_MODEL_ID = "qwen3-30b-a3b"


@dataclass(slots=True)
class CandidateProposal:
    """Candidate token ids proposed by the fast model."""

    speculator_model_id: str
    prompt_text: str
    generated_text: str
    backend: str = "unknown"
    token_ids: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "speculator_model_id": self.speculator_model_id,
            "prompt_text": self.prompt_text,
            "generated_text": self.generated_text,
            "backend": self.backend,
            "token_ids": list(self.token_ids),
            "elapsed_seconds": self.elapsed_seconds,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class VerifierBatchResult:
    """Verifier logits for K candidates plus the bonus next position."""

    model_id: str
    prompt_token_count: int
    candidate_token_ids: list[int]
    verifier_token_ids: list[int]
    logits_count: int
    layers_executed: int
    expected_layers_executed: int
    elapsed_seconds: float
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "prompt_token_count": self.prompt_token_count,
            "candidate_token_ids": list(self.candidate_token_ids),
            "verifier_token_ids": list(self.verifier_token_ids),
            "logits_count": self.logits_count,
            "layers_executed": self.layers_executed,
            "expected_layers_executed": self.expected_layers_executed,
            "elapsed_seconds": self.elapsed_seconds,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class SpeculativeGenerateResult:
    """End-to-end speculative generation result."""

    verifier_model_id: str
    speculator_model_id: str
    prompt: str
    prepared_prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str
    full_text: str
    max_new_tokens: int
    k: int
    verifier_passes: int
    speculator_calls: int
    accepted_token_count: int
    corrected_token_count: int
    average_accepted_per_pass: float
    elapsed_seconds: float
    effective_tokens_per_second: float
    layers_executed: int
    expected_layers_executed: int
    anti_cheat_passed: bool
    verifier_token_positions_processed: int = 0
    expert_telemetry: dict = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "verifier_model_id": self.verifier_model_id,
            "speculator_model_id": self.speculator_model_id,
            "prompt": self.prompt,
            "prepared_prompt": self.prepared_prompt,
            "prompt_token_ids": list(self.prompt_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
            "generated_text": self.generated_text,
            "full_text": self.full_text,
            "max_new_tokens": self.max_new_tokens,
            "k": self.k,
            "verifier_passes": self.verifier_passes,
            "speculator_calls": self.speculator_calls,
            "accepted_token_count": self.accepted_token_count,
            "corrected_token_count": self.corrected_token_count,
            "average_accepted_per_pass": self.average_accepted_per_pass,
            "elapsed_seconds": self.elapsed_seconds,
            "effective_tokens_per_second": self.effective_tokens_per_second,
            "effective_seconds_per_token": None
            if self.effective_tokens_per_second <= 0
            else round(1.0 / self.effective_tokens_per_second, 4),
            "layers_executed": self.layers_executed,
            "expected_layers_executed": self.expected_layers_executed,
            "anti_cheat_passed": self.anti_cheat_passed,
            "verifier_token_positions_processed": int(self.verifier_token_positions_processed),
            "expert_telemetry": dict(self.expert_telemetry),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


def propose_candidates(
    speculator_model_id: str,
    prompt_token_ids: list[int],
    k: int,
    *,
    verifier_model_id: str | None = None,
    prompt_text: str | None = None,
    gguf_runner: Callable[..., object] | None = None,
) -> CandidateProposal:
    """Use a fast speculator to propose K candidate tokens, then encode them for the verifier."""
    started = time.perf_counter()
    blockers: list[str] = []
    if k <= 0:
        blockers.append("Speculative candidate count k must be greater than zero.")
    text_model_id = verifier_model_id or speculator_model_id
    if prompt_text is None:
        prompt_text, decode_blockers = decode_token_ids_to_text(text_model_id, prompt_token_ids)
        blockers.extend(decode_blockers)
    if blockers:
        return CandidateProposal(
            speculator_model_id=speculator_model_id,
            prompt_text=prompt_text or "",
            generated_text="",
            backend="none",
            elapsed_seconds=round(time.perf_counter() - started, 4),
            blockers=blockers,
            ready=False,
        )

    backend = "direct-paged"
    generated_text = ""
    backend_preference = os.environ.get("PCKETLM_SPECULATOR_BACKEND", "auto").strip().lower()
    use_gguf = gguf_runner is not None or (
        backend_preference not in {"direct", "paged", "safetensors"} and _has_gguf_artifact(speculator_model_id)
    )
    if use_gguf:
        backend = "gguf"
        start_gguf_server(speculator_model_id)
        runner = gguf_runner or run_gguf_prompt
        result = runner(
            speculator_model_id,
            prompt_text,
            max_tokens=max(k, 1),
            stop_strings=["<|im_end|>"],
            prefer_server=True,
        )
        if hasattr(result, "to_dict"):
            payload = result.to_dict()
        else:
            payload = dict(result)  # type: ignore[arg-type]
        generated_text = str(payload.get("generated_text") or "")
        if not bool(payload.get("ready", False)):
            blockers.extend(str(value) for value in payload.get("blockers", []))
    else:
        result = run_prompt_decode_loop(
            speculator_model_id,
            prompt_text,
            steps=max(k, 1),
            max_new_tokens=max(k, 1),
            selection_policy="greedy",
            apply_chat_format=False,
            stop_strings=["<|im_end|>"],
        )
        generated_text = result.generated_text
        if not result.ready:
            blockers.extend(result.blockers)
    encoded_ids, encode_blockers = encode_prompt_text(text_model_id, generated_text)
    blockers.extend(encode_blockers)
    token_ids = encoded_ids[:k]
    if len(token_ids) < k:
        blockers.append(f"Speculator produced {len(token_ids)} verifier token(s), expected {k}.")
    return CandidateProposal(
        speculator_model_id=speculator_model_id,
        prompt_text=prompt_text,
        generated_text=generated_text,
        backend=backend,
        token_ids=token_ids,
        elapsed_seconds=round(time.perf_counter() - started, 4),
        blockers=blockers,
        ready=not blockers and len(token_ids) == k,
    )


def _has_gguf_artifact(model_id: str) -> bool:
    return any(artifacts_root(model_id).glob("*.gguf"))


class SpeculativeSession:
    """Stateful verifier session with persistent committed KV cache."""

    def __init__(self, verifier_model_id: str, *, lm_head_chunk_rows: int = DEFAULT_LM_HEAD_CHUNK_ROWS) -> None:
        self.verifier_model_id = verifier_model_id
        self.lm_head_chunk_rows = int(lm_head_chunk_rows)
        self.config = load_layer_bridge_config(verifier_model_id)
        self.prompt_token_ids: list[int] = []
        self.committed_token_ids: list[int] = []
        self.kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.next_token_id: int | None = None
        self.blockers: list[str] = list(self.config.blockers)
        self.layers_executed = 0
        self.expected_layers_executed = 0
        self.verifier_token_positions_processed = 0
        self._tentative_candidate_token_ids: list[int] = []
        self._tentative_verifier_token_ids: list[int] = []
        self._tentative_kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None

    @property
    def ready(self) -> bool:
        return self.config.ready and not self.blockers and self.next_token_id is not None

    def prefill(self, prompt_token_ids: list[int]) -> VerifierBatchResult:
        """Run the prompt once, store KV, and cache the first verifier next-token argmax."""
        started = time.perf_counter()
        self.prompt_token_ids = list(prompt_token_ids)
        self.committed_token_ids = list(prompt_token_ids)
        if not prompt_token_ids:
            self.blockers.append("Verifier prompt token ids must not be empty.")
        if not self.config.ready:
            self.blockers.extend(self.config.blockers)
        if self.blockers:
            return self._result([], 0, 0, started)

        hidden_state, entry_blockers = load_token_entry_hidden_state(self.verifier_model_id, prompt_token_ids)
        self.blockers.extend(entry_blockers)
        if self.blockers or hidden_state is None:
            return self._result([], 0, 0, started)

        stack = run_layer_bridge_stack(
            self.verifier_model_id,
            start_layer=0,
            layer_count=int(self.config.num_hidden_layers),
            input_hidden=hidden_state,
            return_kv_cache=True,
            collect_step_summaries=False,
            collect_metrics=False,
        )
        self.layers_executed += len(stack.executed_layers)
        self.expected_layers_executed += int(self.config.num_hidden_layers)
        self.verifier_token_positions_processed += len(prompt_token_ids)
        if not stack.ready or stack.output_tensor is None:
            self.blockers.extend(stack.blockers or ["Verifier prefill stack failed."])
            return self._result([], len(stack.executed_layers), int(self.config.num_hidden_layers), started)

        tail = run_decode_tail(
            self.verifier_model_id,
            stack.output_tensor[:, -1:, :],
            lm_head_chunk_rows=self.lm_head_chunk_rows,
            top_k=1,
            return_logits=False,
        )
        if not tail.ready or not tail.top_token_ids:
            self.blockers.extend(tail.blockers or ["Verifier prefill decode tail failed."])
            return self._result([], len(stack.executed_layers), int(self.config.num_hidden_layers), started)

        self.kv_caches = dict(stack.next_kv_caches)
        self.next_token_id = int(tail.top_token_ids[0])
        return self._result([self.next_token_id], len(stack.executed_layers), int(self.config.num_hidden_layers), started)

    def verify_candidates(self, candidate_token_ids: list[int]) -> VerifierBatchResult:
        """Verify K candidates against committed KV without re-running the prompt."""
        started = time.perf_counter()
        self.rollback()
        if not self.ready:
            return self._result([], 0, 0, started, candidate_token_ids)
        if not candidate_token_ids:
            self.blockers.append("Verifier candidate token ids must not be empty.")
            return self._result([], 0, 0, started, candidate_token_ids)

        hidden_state, entry_blockers = load_token_entry_hidden_state(self.verifier_model_id, candidate_token_ids)
        blockers = list(entry_blockers)
        if blockers or hidden_state is None:
            self.blockers.extend(blockers or ["Verifier candidate token entry failed."])
            return self._result([], 0, 0, started, candidate_token_ids)

        stack = run_layer_bridge_stack(
            self.verifier_model_id,
            start_layer=0,
            layer_count=int(self.config.num_hidden_layers),
            input_hidden=hidden_state,
            past_key_values=self.kv_caches,
            position_offset=len(self.committed_token_ids),
            return_kv_cache=True,
            collect_step_summaries=False,
            collect_metrics=False,
        )
        self.layers_executed += len(stack.executed_layers)
        self.expected_layers_executed += int(self.config.num_hidden_layers)
        self.verifier_token_positions_processed += len(candidate_token_ids)
        if not stack.ready or stack.output_tensor is None:
            self.blockers.extend(stack.blockers or ["Verifier candidate stack failed."])
            return self._result([], len(stack.executed_layers), int(self.config.num_hidden_layers), started, candidate_token_ids)

        verifier_token_ids = [int(self.next_token_id)]
        for position in range(len(candidate_token_ids)):
            tail = run_decode_tail(
                self.verifier_model_id,
                stack.output_tensor[:, position : position + 1, :],
                lm_head_chunk_rows=self.lm_head_chunk_rows,
                top_k=1,
                return_logits=False,
            )
            if not tail.ready or not tail.top_token_ids:
                self.blockers.extend(tail.blockers or [f"Verifier decode tail failed at candidate position {position}."])
                return self._result(verifier_token_ids, len(stack.executed_layers), int(self.config.num_hidden_layers), started, candidate_token_ids)
            verifier_token_ids.append(int(tail.top_token_ids[0]))

        self._tentative_candidate_token_ids = list(candidate_token_ids)
        self._tentative_verifier_token_ids = list(verifier_token_ids)
        self._tentative_kv_caches = dict(stack.next_kv_caches)
        return VerifierBatchResult(
            model_id=self.verifier_model_id,
            prompt_token_count=len(self.committed_token_ids),
            candidate_token_ids=list(candidate_token_ids),
            verifier_token_ids=verifier_token_ids,
            logits_count=len(verifier_token_ids),
            layers_executed=len(stack.executed_layers),
            expected_layers_executed=int(self.config.num_hidden_layers),
            elapsed_seconds=round(time.perf_counter() - started, 4),
            blockers=[],
            ready=True,
        )

    def verify_initial_candidates(self, prompt_token_ids: list[int], candidate_token_ids: list[int]) -> VerifierBatchResult:
        """Run prompt + first candidate batch once, then keep prompt KV for later commits."""
        started = time.perf_counter()
        self.rollback()
        self.prompt_token_ids = list(prompt_token_ids)
        self.committed_token_ids = list(prompt_token_ids)
        self.kv_caches = {}
        self.next_token_id = None
        if not prompt_token_ids:
            self.blockers.append("Verifier prompt token ids must not be empty.")
        if not candidate_token_ids:
            self.blockers.append("Verifier candidate token ids must not be empty.")
        if not self.config.ready:
            self.blockers.extend(self.config.blockers)
        if self.blockers:
            return self._result([], 0, 0, started, candidate_token_ids)

        full_input_ids = list(prompt_token_ids) + list(candidate_token_ids)
        hidden_state, entry_blockers = load_token_entry_hidden_state(self.verifier_model_id, full_input_ids)
        blockers = list(entry_blockers)
        if blockers or hidden_state is None:
            self.blockers.extend(blockers or ["Verifier initial token entry failed."])
            return self._result([], 0, 0, started, candidate_token_ids)

        stack = run_layer_bridge_stack(
            self.verifier_model_id,
            start_layer=0,
            layer_count=int(self.config.num_hidden_layers),
            input_hidden=hidden_state,
            return_kv_cache=True,
            collect_step_summaries=False,
            collect_metrics=False,
        )
        self.layers_executed += len(stack.executed_layers)
        self.expected_layers_executed += int(self.config.num_hidden_layers)
        self.verifier_token_positions_processed += len(full_input_ids)
        if not stack.ready or stack.output_tensor is None:
            self.blockers.extend(stack.blockers or ["Verifier initial candidate stack failed."])
            return self._result([], len(stack.executed_layers), int(self.config.num_hidden_layers), started, candidate_token_ids)

        prediction_positions = range(len(prompt_token_ids) - 1, len(prompt_token_ids) + len(candidate_token_ids))
        verifier_token_ids: list[int] = []
        for position in prediction_positions:
            tail = run_decode_tail(
                self.verifier_model_id,
                stack.output_tensor[:, position : position + 1, :],
                lm_head_chunk_rows=self.lm_head_chunk_rows,
                top_k=1,
                return_logits=False,
            )
            if not tail.ready or not tail.top_token_ids:
                self.blockers.extend(tail.blockers or [f"Verifier initial decode tail failed at position {position}."])
                return self._result(verifier_token_ids, len(stack.executed_layers), int(self.config.num_hidden_layers), started, candidate_token_ids)
            verifier_token_ids.append(int(tail.top_token_ids[0]))

        self.next_token_id = int(verifier_token_ids[0])
        self._tentative_candidate_token_ids = list(candidate_token_ids)
        self._tentative_verifier_token_ids = list(verifier_token_ids)
        self._tentative_kv_caches = dict(stack.next_kv_caches)
        return VerifierBatchResult(
            model_id=self.verifier_model_id,
            prompt_token_count=len(prompt_token_ids),
            candidate_token_ids=list(candidate_token_ids),
            verifier_token_ids=verifier_token_ids,
            logits_count=len(verifier_token_ids),
            layers_executed=len(stack.executed_layers),
            expected_layers_executed=int(self.config.num_hidden_layers),
            elapsed_seconds=round(time.perf_counter() - started, 4),
            blockers=[],
            ready=True,
        )

    def commit(self, accepted_count: int) -> None:
        """Commit the accepted prefix from the last tentative verification."""
        if accepted_count < 0:
            return
        if self._tentative_kv_caches is None:
            if accepted_count > 0:
                self.blockers.append("No tentative KV cache is available to commit.")
            return
        accepted_count = min(int(accepted_count), len(self._tentative_candidate_token_ids))
        target_length = len(self.committed_token_ids) + accepted_count
        self.kv_caches = _slice_kv_caches(self._tentative_kv_caches, target_length)
        self.committed_token_ids.extend(self._tentative_candidate_token_ids[:accepted_count])
        if len(self._tentative_verifier_token_ids) > accepted_count:
            self.next_token_id = int(self._tentative_verifier_token_ids[accepted_count])

    def append_token(self, token_id: int) -> VerifierBatchResult:
        """Append one verifier-chosen token to KV, used for corrections and final state alignment."""
        verification = self.verify_candidates([int(token_id)])
        if verification.ready:
            self.commit(1)
        return verification

    def rollback(self) -> None:
        self._tentative_candidate_token_ids = []
        self._tentative_verifier_token_ids = []
        self._tentative_kv_caches = None

    def _result(
        self,
        verifier_token_ids: list[int],
        layers_executed: int,
        expected_layers: int,
        started: float,
        candidate_token_ids: list[int] | None = None,
    ) -> VerifierBatchResult:
        return VerifierBatchResult(
            model_id=self.verifier_model_id,
            prompt_token_count=len(self.committed_token_ids),
            candidate_token_ids=[] if candidate_token_ids is None else list(candidate_token_ids),
            verifier_token_ids=list(verifier_token_ids),
            logits_count=len(verifier_token_ids),
            layers_executed=layers_executed,
            expected_layers_executed=expected_layers,
            elapsed_seconds=round(time.perf_counter() - started, 4),
            blockers=list(self.blockers),
            ready=not self.blockers and bool(verifier_token_ids),
        )


def _slice_kv_caches(
    kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]],
    target_length: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    sliced: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_index, (key_tensor, value_tensor) in kv_caches.items():
        sliced[layer_index] = (
            key_tensor[:, :, :target_length, :].detach().cpu(),
            value_tensor[:, :, :target_length, :].detach().cpu(),
        )
    return sliced


def verify_candidates_once(
    verifier_model_id: str,
    prompt_token_ids: list[int],
    candidate_token_ids: list[int],
    *,
    lm_head_chunk_rows: int = DEFAULT_LM_HEAD_CHUNK_ROWS,
) -> VerifierBatchResult:
    """Run prompt + candidates through the verifier once and return greedy ids for K+1 positions."""
    started = time.perf_counter()
    blockers: list[str] = []
    if not prompt_token_ids:
        blockers.append("Verifier prompt token ids must not be empty.")
    if not candidate_token_ids:
        blockers.append("Verifier candidate token ids must not be empty.")
    config = load_layer_bridge_config(verifier_model_id)
    blockers.extend(config.blockers)
    if blockers or not config.ready:
        return VerifierBatchResult(
            model_id=verifier_model_id,
            prompt_token_count=len(prompt_token_ids),
            candidate_token_ids=list(candidate_token_ids),
            verifier_token_ids=[],
            logits_count=0,
            layers_executed=0,
            expected_layers_executed=0,
            elapsed_seconds=round(time.perf_counter() - started, 4),
            blockers=blockers,
            ready=False,
        )

    full_input_ids = list(prompt_token_ids) + list(candidate_token_ids)
    hidden_state, entry_blockers = load_token_entry_hidden_state(verifier_model_id, full_input_ids)
    blockers.extend(entry_blockers)
    if blockers or hidden_state is None:
        return VerifierBatchResult(
            model_id=verifier_model_id,
            prompt_token_count=len(prompt_token_ids),
            candidate_token_ids=list(candidate_token_ids),
            verifier_token_ids=[],
            logits_count=0,
            layers_executed=0,
            expected_layers_executed=0,
            elapsed_seconds=round(time.perf_counter() - started, 4),
            blockers=blockers,
            ready=False,
        )

    stack = run_layer_bridge_stack(
        verifier_model_id,
        start_layer=0,
        layer_count=int(config.num_hidden_layers),
        input_hidden=hidden_state,
        return_kv_cache=False,
        collect_step_summaries=False,
        collect_metrics=False,
    )
    expected_layers = int(config.num_hidden_layers)
    if not stack.ready or stack.output_tensor is None:
        blockers.extend(stack.blockers or ["Verifier stack failed."])
    if blockers or stack.output_tensor is None:
        return VerifierBatchResult(
            model_id=verifier_model_id,
            prompt_token_count=len(prompt_token_ids),
            candidate_token_ids=list(candidate_token_ids),
            verifier_token_ids=[],
            logits_count=0,
            layers_executed=len(stack.executed_layers),
            expected_layers_executed=expected_layers,
            elapsed_seconds=round(time.perf_counter() - started, 4),
            blockers=blockers,
            ready=False,
        )

    prediction_positions = range(len(prompt_token_ids) - 1, len(prompt_token_ids) + len(candidate_token_ids))
    verifier_token_ids: list[int] = []
    for position in prediction_positions:
        tail = run_decode_tail(
            verifier_model_id,
            stack.output_tensor[:, position : position + 1, :],
            lm_head_chunk_rows=lm_head_chunk_rows,
            top_k=1,
            return_logits=False,
        )
        if not tail.ready or not tail.top_token_ids:
            blockers.extend(tail.blockers or [f"Verifier decode tail failed at position {position}."])
            continue
        verifier_token_ids.append(int(tail.top_token_ids[0]))

    return VerifierBatchResult(
        model_id=verifier_model_id,
        prompt_token_count=len(prompt_token_ids),
        candidate_token_ids=list(candidate_token_ids),
        verifier_token_ids=verifier_token_ids,
        logits_count=len(verifier_token_ids),
        layers_executed=len(stack.executed_layers),
        expected_layers_executed=expected_layers,
        elapsed_seconds=round(time.perf_counter() - started, 4),
        blockers=blockers,
        ready=not blockers and len(verifier_token_ids) == len(candidate_token_ids) + 1,
    )


def speculative_generate(
    verifier_model_id: str,
    speculator_model_id: str,
    prompt: str,
    max_new_tokens: int,
    *,
    k: int = 4,
    proposer: Callable[..., CandidateProposal] | None = None,
    verifier: Callable[..., VerifierBatchResult] | None = None,
) -> SpeculativeGenerateResult:
    """Greedy speculative generation with a stateless tentative verifier pass."""
    started = time.perf_counter()
    blockers: list[str] = []
    if max_new_tokens <= 0:
        blockers.append("max_new_tokens must be greater than zero.")
    if k <= 0:
        blockers.append("k must be greater than zero.")
    prepared = prepare_prompt_text(verifier_model_id, prompt, apply_chat_format=True)
    blockers.extend(prepared.blockers)
    if blockers or not prepared.ready:
        return SpeculativeGenerateResult(
            verifier_model_id=verifier_model_id,
            speculator_model_id=speculator_model_id,
            prompt=prompt,
            prepared_prompt=prepared.prepared_prompt,
            prompt_token_ids=list(prepared.token_ids),
            generated_token_ids=[],
            generated_text="",
            full_text=prompt,
            max_new_tokens=max_new_tokens,
            k=k,
            verifier_passes=0,
            speculator_calls=0,
            accepted_token_count=0,
            corrected_token_count=0,
            average_accepted_per_pass=0.0,
            elapsed_seconds=round(time.perf_counter() - started, 4),
            effective_tokens_per_second=0.0,
            layers_executed=0,
            expected_layers_executed=0,
            anti_cheat_passed=False,
            blockers=blockers,
            ready=False,
        )

    if verifier is None:
        return _stateful_speculative_generate(
            verifier_model_id=verifier_model_id,
            speculator_model_id=speculator_model_id,
            prompt=prompt,
            prepared_prompt=prepared.prepared_prompt,
            prompt_token_ids=list(prepared.token_ids),
            max_new_tokens=max_new_tokens,
            k=k,
            proposer=proposer or propose_candidates,
            started=started,
        )

    propose_fn = proposer or propose_candidates
    verify_fn = verifier or verify_candidates_once
    current_ids = list(prepared.token_ids)
    generated: list[int] = []
    verifier_passes = 0
    speculator_calls = 0
    accepted_token_count = 0
    corrected_token_count = 0
    layers_executed = 0
    expected_layers_executed = 0

    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        batch_k = min(k, remaining)
        current_text, decode_blockers = decode_token_ids_to_text(verifier_model_id, current_ids)
        if decode_blockers:
            blockers.extend(decode_blockers)
            break
        proposal = propose_fn(
            speculator_model_id,
            current_ids,
            batch_k,
            verifier_model_id=verifier_model_id,
            prompt_text=current_text,
        )
        speculator_calls += 1
        if not proposal.ready:
            blockers.extend(proposal.blockers or ["Speculator failed to produce candidates."])
            break

        verification = verify_fn(verifier_model_id, current_ids, proposal.token_ids)
        verifier_passes += 1
        layers_executed += int(verification.layers_executed)
        expected_layers_executed += int(verification.expected_layers_executed)
        if not verification.ready:
            blockers.extend(verification.blockers or ["Verifier failed to check candidates."])
            break

        produced_this_pass: list[int] = []
        all_accepted = True
        for index, candidate_token_id in enumerate(proposal.token_ids):
            verifier_token_id = int(verification.verifier_token_ids[index])
            if int(candidate_token_id) == verifier_token_id:
                produced_this_pass.append(int(candidate_token_id))
                accepted_token_count += 1
                continue
            produced_this_pass.append(verifier_token_id)
            corrected_token_count += 1
            all_accepted = False
            break
        if all_accepted and len(generated) + len(produced_this_pass) < max_new_tokens:
            produced_this_pass.append(int(verification.verifier_token_ids[len(proposal.token_ids)]))
            corrected_token_count += 1
        if not produced_this_pass:
            blockers.append("Speculative pass produced no tokens.")
            break
        produced_this_pass = produced_this_pass[:remaining]
        generated.extend(produced_this_pass)
        current_ids.extend(produced_this_pass)

    generated_text, generated_blockers = decode_token_ids_to_text(verifier_model_id, generated)
    full_text, full_blockers = decode_token_ids_to_text(verifier_model_id, current_ids)
    blockers.extend(generated_blockers)
    blockers.extend(full_blockers)
    elapsed = round(time.perf_counter() - started, 4)
    tokens_per_second = 0.0 if elapsed <= 0 else round(len(generated) / elapsed, 6)
    average_accepted = 0.0 if verifier_passes <= 0 else round(accepted_token_count / verifier_passes, 4)
    anti_cheat = expected_layers_executed > 0 and layers_executed == expected_layers_executed
    return SpeculativeGenerateResult(
        verifier_model_id=verifier_model_id,
        speculator_model_id=speculator_model_id,
        prompt=prompt,
        prepared_prompt=prepared.prepared_prompt,
        prompt_token_ids=list(prepared.token_ids),
        generated_token_ids=generated,
        generated_text=generated_text,
        full_text=full_text,
        max_new_tokens=max_new_tokens,
        k=k,
        verifier_passes=verifier_passes,
        speculator_calls=speculator_calls,
        accepted_token_count=accepted_token_count,
        corrected_token_count=corrected_token_count,
        average_accepted_per_pass=average_accepted,
        elapsed_seconds=elapsed,
        effective_tokens_per_second=tokens_per_second,
        layers_executed=layers_executed,
        expected_layers_executed=expected_layers_executed,
        anti_cheat_passed=anti_cheat,
        verifier_token_positions_processed=0,
        expert_telemetry=expert_residency_snapshot(),
        blockers=blockers,
        ready=not blockers and len(generated) == max_new_tokens and anti_cheat,
    )


def _stateful_speculative_generate(
    *,
    verifier_model_id: str,
    speculator_model_id: str,
    prompt: str,
    prepared_prompt: str,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    k: int,
    proposer: Callable[..., CandidateProposal],
    started: float,
) -> SpeculativeGenerateResult:
    blockers: list[str] = []
    session = SpeculativeSession(verifier_model_id)
    session.prompt_token_ids = list(prompt_token_ids)
    session.committed_token_ids = list(prompt_token_ids)

    generated: list[int] = []
    verifier_passes = 0
    speculator_calls = 0
    accepted_token_count = 0
    corrected_token_count = 0
    first_verification = True

    while not blockers and len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        batch_k = min(k, remaining)
        current_text, decode_blockers = decode_token_ids_to_text(verifier_model_id, session.committed_token_ids)
        if decode_blockers:
            blockers.extend(decode_blockers)
            break

        proposal = proposer(
            speculator_model_id,
            session.committed_token_ids,
            batch_k,
            verifier_model_id=verifier_model_id,
            prompt_text=current_text,
        )
        speculator_calls += 1
        if not proposal.ready:
            blockers.extend(proposal.blockers or ["Speculator failed to produce candidates."])
            break

        if first_verification:
            verification = session.verify_initial_candidates(prompt_token_ids, proposal.token_ids)
            first_verification = False
        else:
            verification = session.verify_candidates(proposal.token_ids)
        verifier_passes += 1
        if not verification.ready:
            blockers.extend(verification.blockers or ["Stateful verifier failed to check candidates."])
            break

        accepted_this_pass = 0
        rejected_token: int | None = None
        for index, candidate_token_id in enumerate(proposal.token_ids):
            verifier_token_id = int(verification.verifier_token_ids[index])
            if int(candidate_token_id) == verifier_token_id:
                generated.append(int(candidate_token_id))
                accepted_this_pass += 1
                accepted_token_count += 1
                if len(generated) >= max_new_tokens:
                    break
                continue
            rejected_token = verifier_token_id
            break

        session.commit(accepted_this_pass)
        if len(generated) >= max_new_tokens:
            session.rollback()
            break

        if rejected_token is not None:
            generated.append(rejected_token)
            corrected_token_count += 1
            append_result = session.append_token(rejected_token)
            if not append_result.ready:
                blockers.extend(append_result.blockers or ["Stateful verifier failed to append correction."])
                break
        session.rollback()

    generated_text, generated_blockers = decode_token_ids_to_text(verifier_model_id, generated)
    full_text, full_blockers = decode_token_ids_to_text(verifier_model_id, session.committed_token_ids)
    blockers.extend(generated_blockers)
    blockers.extend(full_blockers)
    blockers.extend(session.blockers)
    elapsed = round(time.perf_counter() - started, 4)
    tokens_per_second = 0.0 if elapsed <= 0 else round(len(generated) / elapsed, 6)
    average_accepted = 0.0 if verifier_passes <= 0 else round(accepted_token_count / verifier_passes, 4)
    layers_executed = int(session.layers_executed)
    expected_layers = int(session.expected_layers_executed)
    anti_cheat = expected_layers > 0 and layers_executed == expected_layers
    return SpeculativeGenerateResult(
        verifier_model_id=verifier_model_id,
        speculator_model_id=speculator_model_id,
        prompt=prompt,
        prepared_prompt=prepared_prompt,
        prompt_token_ids=list(prompt_token_ids),
        generated_token_ids=generated,
        generated_text=generated_text,
        full_text=full_text,
        max_new_tokens=max_new_tokens,
        k=k,
        verifier_passes=verifier_passes,
        speculator_calls=speculator_calls,
        accepted_token_count=accepted_token_count,
        corrected_token_count=corrected_token_count,
        average_accepted_per_pass=average_accepted,
        elapsed_seconds=elapsed,
        effective_tokens_per_second=tokens_per_second,
        layers_executed=layers_executed,
        expected_layers_executed=expected_layers,
        anti_cheat_passed=anti_cheat,
        verifier_token_positions_processed=int(session.verifier_token_positions_processed),
        expert_telemetry=expert_residency_snapshot(),
        blockers=blockers,
        ready=not blockers and len(generated) == max_new_tokens and anti_cheat,
    )
