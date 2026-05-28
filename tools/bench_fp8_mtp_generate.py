from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcketlm.core.runtime.fp8_source import (
    _load_deepseek_config,
    run_fp8_decode_tail_topk_batch,
    run_fp8_mtp_draft_step,
    run_fp8_prompt_prefill,
)
from pcketlm.core.runtime.tokenizer_runtime import decode_token_ids_to_text, load_generation_settings, prepare_prompt_text


_PUNCTUATION_TOKEN_IDS = {3, 4, 5, 13, 14, 16, 29, 30, 201, 223}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _visible_token_ids(token_ids: list[int], eos_token_ids: set[int]) -> tuple[list[int], bool]:
    visible: list[int] = []
    for token_id in token_ids:
        if int(token_id) in eos_token_ids:
            return visible, True
        visible.append(int(token_id))
    return visible, False


def _select_mtp_candidate(top_ids: list[int], previous_token_id: int, eos_token_ids: set[int], *, prefer_eos_after_punctuation: bool) -> int:
    if prefer_eos_after_punctuation and int(previous_token_id) in _PUNCTUATION_TOKEN_IDS:
        for token_id in top_ids:
            if int(token_id) in eos_token_ids:
                return int(token_id)
    return int(top_ids[0])


def _trim_kv_caches(
    caches: dict[int, tuple[torch.Tensor, torch.Tensor]],
    keep_len: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    return {
        int(layer): (
            kv[:, :keep_len, :].detach().contiguous(),
            pe[:, :keep_len, :].detach().contiguous(),
        )
        for layer, (kv, pe) in caches.items()
    }


def _decode_visible(model_id: str, token_ids: list[int], eos_token_ids: set[int]) -> tuple[list[int], bool, str, list[str]]:
    visible_ids, stopped = _visible_token_ids(token_ids, eos_token_ids)
    text, blockers = decode_token_ids_to_text(model_id, visible_ids)
    return visible_ids, stopped, text, blockers


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate with DeepSeek MTP proposals and exact FP8 verifier commits.")
    parser.add_argument("model_id")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-visible", type=int, default=10)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--max-passes", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefer-eos-after-punctuation", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    started = time.perf_counter()
    prepared = prepare_prompt_text(args.model_id, args.prompt, apply_chat_format=True)
    eos_token_ids = {int(value) for value in load_generation_settings(args.model_id).eos_token_ids}
    layer_count = int(args.layers) if args.layers is not None else int(_load_deepseek_config(args.model_id)["num_hidden_layers"])
    payload: dict = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "max_visible_tokens": int(args.max_visible),
        "k": int(args.k),
        "layer_count": layer_count,
        "prefer_eos_after_punctuation": bool(args.prefer_eos_after_punctuation),
        "prepared_ready": bool(prepared.ready),
        "prepared_blockers": list(prepared.blockers),
        "passes": [],
        "ready": False,
    }
    _write(out_path, payload)
    if not prepared.ready:
        return 1

    prefill_started = time.perf_counter()
    prefill = run_fp8_prompt_prefill(
        args.model_id,
        list(prepared.token_ids),
        layer_count=layer_count,
        include_tail=True,
    )
    payload.update(
        {
            "prompt_token_count": len(prepared.token_ids),
            "prefill_ready": bool(prefill.ready),
            "prefill_elapsed_seconds": round(time.perf_counter() - prefill_started, 4),
            "prefill_top_token_ids": [] if prefill.tail is None else list(prefill.tail.top_token_ids),
            "prefill_blockers": list(prefill.blockers),
        }
    )
    _write(out_path, payload)
    if not prefill.ready or prefill.output_tensor is None or prefill.tail is None or not prefill.tail.top_token_ids:
        return 1

    generated: list[int] = []
    verifier_kv = dict(prefill.next_kv_caches)
    next_token_id = int(prefill.tail.top_token_ids[0])
    current_hidden = prefill.output_tensor[:, -1:, :].detach().contiguous()
    current_input_token = int(prepared.token_ids[-1])
    current_position = len(prepared.token_ids) - 1
    blockers: list[str] = []
    accepted_total = 0
    forced_total = 0
    verifier_layers = len(prefill.executed_layers)
    expected_layers = layer_count

    for pass_index in range(1, int(args.max_passes) + 1):
        visible_ids, stopped_by_eos, generated_text, text_blockers = _decode_visible(args.model_id, generated, eos_token_ids)
        blockers.extend(text_blockers)
        if blockers or stopped_by_eos or len(visible_ids) >= int(args.max_visible):
            break

        candidates: list[int] = []
        mtp_steps: list[dict] = []
        mtp_hidden = current_hidden
        mtp_input = int(current_input_token)
        mtp_position = int(current_position)
        mtp_cache = None
        for mtp_index in range(max(1, int(args.k))):
            step = run_fp8_mtp_draft_step(
                args.model_id,
                mtp_input,
                mtp_hidden,
                position=mtp_position,
                previous_kv_cache=mtp_cache,
                top_k=12,
            )
            step_payload = step.to_dict()
            mtp_steps.append(step_payload)
            if not step.ready or not step.top_token_ids or step.output_tensor is None:
                blockers.extend(step.blockers or ["MTP draft step did not produce a candidate."])
                break
            proposed = _select_mtp_candidate(
                [int(value) for value in step.top_token_ids],
                mtp_input,
                eos_token_ids,
                prefer_eos_after_punctuation=bool(args.prefer_eos_after_punctuation),
            )
            token_id = proposed
            if mtp_index == 0 and int(token_id) != int(next_token_id):
                token_id = int(next_token_id)
                forced_total += 1
                step_payload["forced_to_verifier_next_token_id"] = token_id
            candidates.append(int(token_id))
            if mtp_index == 0 and int(token_id) != int(proposed):
                break
            mtp_input = int(token_id)
            mtp_hidden = step.output_tensor.detach().contiguous()
            mtp_cache = step.next_kv_cache
            mtp_position += 1
            if int(token_id) in eos_token_ids:
                break

        if not candidates:
            candidates = [int(next_token_id)]

        verify_started = time.perf_counter()
        continuation = run_fp8_prompt_prefill(
            args.model_id,
            candidates,
            layer_count=layer_count,
            include_tail=False,
            start_pos=len(prepared.token_ids) + len(generated),
            previous_kv_caches=verifier_kv,
        )
        verify_blockers = list(continuation.blockers)
        verifier_ids = [int(next_token_id)]
        tail_elapsed = 0.0
        if continuation.ready and continuation.output_tensor is not None and not verify_blockers:
            tail_started = time.perf_counter()
            tail = run_fp8_decode_tail_topk_batch(args.model_id, continuation.output_tensor, top_k=1)
            tail_elapsed = time.perf_counter() - tail_started
            verify_blockers.extend(tail.blockers)
            if tail.ready:
                verifier_ids.extend(int(row[0]) for row in tail.top_token_ids_by_position if row)
        elif not verify_blockers:
            verify_blockers.append("Verifier continuation did not produce hidden states.")
        blockers.extend(verify_blockers)

        accepted = 0
        if not blockers:
            for candidate, verifier in zip(candidates, verifier_ids):
                if int(candidate) != int(verifier):
                    break
                accepted += 1
            if accepted <= 0:
                blockers.append("Exact verifier accepted no candidates.")

        if accepted > 0 and continuation.output_tensor is not None:
            generated.extend(int(value) for value in candidates[:accepted])
            accepted_total += accepted
            keep_len = len(prepared.token_ids) + len(generated)
            verifier_kv = _trim_kv_caches(continuation.next_kv_caches, keep_len)
            current_hidden = continuation.output_tensor[:, accepted - 1 : accepted, :].detach().contiguous()
            current_input_token = int(candidates[accepted - 1])
            current_position = keep_len - 1
            if accepted < len(verifier_ids):
                next_token_id = int(verifier_ids[accepted])

        visible_ids, stopped_by_eos, generated_text, text_blockers = _decode_visible(args.model_id, generated, eos_token_ids)
        blockers.extend(text_blockers)
        raw_text, raw_text_blockers = decode_token_ids_to_text(args.model_id, generated)
        blockers.extend(raw_text_blockers)
        verifier_layers += len(continuation.executed_layers)
        expected_layers += layer_count
        pass_payload = {
            "pass_index": pass_index,
            "candidates": list(candidates),
            "verifier_token_ids": list(verifier_ids),
            "accepted": int(accepted),
            "mtp_steps": mtp_steps,
            "verify_elapsed_seconds": round(time.perf_counter() - verify_started, 4),
            "tail_elapsed_seconds": round(tail_elapsed, 4),
            "executed_layers": len(continuation.executed_layers),
            "expected_layers": layer_count,
            "generated_token_ids": list(generated),
            "visible_token_ids": list(visible_ids),
            "generated_text": generated_text,
            "raw_generated_text": raw_text,
            "stopped_by_eos": bool(stopped_by_eos),
            "blockers": list(verify_blockers),
        }
        payload["passes"].append(pass_payload)
        payload.update(
            {
                "generated_token_ids": list(generated),
                "visible_token_ids": list(visible_ids),
                "generated_text": generated_text,
                "raw_generated_text": raw_text,
                "stopped_by_eos": bool(stopped_by_eos),
                "accepted_token_count": int(accepted_total),
                "forced_first_token_count": int(forced_total),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "seconds_per_visible_token": round((time.perf_counter() - started) / max(1, len(visible_ids)), 4),
                "layers_executed": int(verifier_layers),
                "expected_layers_executed": int(expected_layers),
                "anti_cheat_passed": int(verifier_layers) == int(expected_layers),
                "blockers": list(blockers),
            }
        )
        payload["ready"] = bool(
            not blockers
            and (stopped_by_eos or len(visible_ids) >= int(args.max_visible))
            and int(verifier_layers) == int(expected_layers)
        )
        _write(out_path, payload)
        if blockers or stopped_by_eos:
            break

    return 0 if payload.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
