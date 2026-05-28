from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcketlm.core.runtime.fp8_source import (
    run_fp8_decode_tail_topk_batch,
    run_fp8_mtp_draft_step,
    run_fp8_prompt_prefill,
)
from pcketlm.core.runtime.tokenizer_runtime import decode_token_ids_to_text, prepare_prompt_text


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek FP8 MTP draft plus exact verifier.")
    parser.add_argument("model_id")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    started = time.perf_counter()
    prepared = prepare_prompt_text(args.model_id, args.prompt, apply_chat_format=True)
    payload: dict = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "k": int(args.k),
        "prepared_ready": bool(prepared.ready),
        "prepared_blockers": list(prepared.blockers),
        "ready": False,
    }
    _write(out_path, payload)
    if not prepared.ready:
        return 1

    layer_count = args.layers
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
            "layer_count": int(prefill.layer_count),
            "prefill_ready": bool(prefill.ready),
            "prefill_elapsed_seconds": round(time.perf_counter() - prefill_started, 4),
            "prefill_top_token_ids": [] if prefill.tail is None else list(prefill.tail.top_token_ids),
            "prefill_blockers": list(prefill.blockers),
        }
    )
    _write(out_path, payload)
    if not prefill.ready or prefill.output_tensor is None or prefill.tail is None or not prefill.tail.top_token_ids:
        return 1

    candidates: list[int] = []
    mtp_steps: list[dict] = []
    mtp_cache = None
    previous_hidden = prefill.output_tensor[:, -1:, :]
    input_token = int(prepared.token_ids[-1])
    position = len(prepared.token_ids) - 1
    for _ in range(max(1, int(args.k))):
        step = run_fp8_mtp_draft_step(
            args.model_id,
            input_token,
            previous_hidden,
            position=position,
            previous_kv_cache=mtp_cache,
            top_k=8,
        )
        mtp_steps.append(step.to_dict())
        if not step.ready or not step.top_token_ids or step.output_tensor is None:
            break
        token_id = int(step.top_token_ids[0])
        candidates.append(token_id)
        input_token = token_id
        previous_hidden = step.output_tensor
        mtp_cache = step.next_kv_cache
        position += 1
        payload.update({"mtp_candidates": list(candidates), "mtp_steps": mtp_steps})
        _write(out_path, payload)

    verifier_ids = [int(prefill.tail.top_token_ids[0])]
    verify_started = time.perf_counter()
    verify_blockers: list[str] = []
    if candidates:
        continuation = run_fp8_prompt_prefill(
            args.model_id,
            candidates,
            layer_count=int(prefill.layer_count),
            include_tail=False,
            start_pos=len(prepared.token_ids),
            previous_kv_caches=prefill.next_kv_caches,
        )
        verify_blockers.extend(continuation.blockers)
        if continuation.ready and continuation.output_tensor is not None and not verify_blockers:
            tail = run_fp8_decode_tail_topk_batch(args.model_id, continuation.output_tensor, top_k=1)
            verify_blockers.extend(tail.blockers)
            if tail.ready:
                verifier_ids.extend(int(row[0]) for row in tail.top_token_ids_by_position if row)
        elif not verify_blockers:
            verify_blockers.append("Verifier continuation did not produce hidden states.")

    accepted = 0
    for candidate, verifier in zip(candidates, verifier_ids):
        if int(candidate) != int(verifier):
            break
        accepted += 1
    draft_text, draft_text_blockers = decode_token_ids_to_text(args.model_id, candidates)
    accepted_text, accepted_text_blockers = decode_token_ids_to_text(args.model_id, candidates[:accepted])
    payload.update(
        {
            "mtp_candidates": list(candidates),
            "mtp_draft_text": draft_text,
            "accepted_text": accepted_text,
            "mtp_steps": mtp_steps,
            "verifier_token_ids": verifier_ids,
            "accepted_prefix": int(accepted),
            "verify_elapsed_seconds": round(time.perf_counter() - verify_started, 4),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "blockers": verify_blockers + draft_text_blockers + accepted_text_blockers,
        }
    )
    payload["ready"] = bool(not payload["blockers"] and accepted > 0)
    _write(out_path, payload)
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
