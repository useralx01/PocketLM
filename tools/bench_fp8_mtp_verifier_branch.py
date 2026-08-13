from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcketlm.core.runtime.fp8_source import (  # noqa: E402
    _summarize_fp8_verifier_timing,
    apply_fp8_exact_mtp_default_env,
    run_fp8_decode_tail_topk_batch,
    run_fp8_prompt_prefill,
    run_fp8_prompt_prefill_batch,
)
from pcketlm.core.runtime.tokenizer_runtime import prepare_prompt_text  # noqa: E402
from pcketlm.core.runtime.tensor_loader import scoped_tensor_handle_cache  # noqa: E402


def _parse_token_ids(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).replace(";", ",").split(",") if part.strip()]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark one exact DeepSeek MTP verifier candidate branch.")
    parser.add_argument("model_id")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--candidate-token-ids", required=True)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    env = apply_fp8_exact_mtp_default_env()
    candidate_ids = _parse_token_ids(args.candidate_token_ids)
    prepared = prepare_prompt_text(args.model_id, args.prompt, apply_chat_format=True)
    payload: dict = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "candidate_token_ids": list(candidate_ids),
        "candidate_count": len(candidate_ids),
        "env": env,
        "prepared_ready": bool(prepared.ready),
        "prepared_blockers": list(prepared.blockers),
        "ready": False,
    }
    out_path = Path(args.out)
    _write(out_path, payload)
    if not prepared.ready or not candidate_ids:
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        _write(out_path, payload)
        return 1

    with scoped_tensor_handle_cache():
        prefill_started = time.perf_counter()
        prefill = run_fp8_prompt_prefill(
            args.model_id,
            list(prepared.token_ids),
            layer_count=args.layers,
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
                "prefill_layers_executed": len(prefill.executed_layers),
            }
        )
        _write(out_path, payload)
        if not prefill.ready or prefill.output_tensor is None:
            payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            payload["blockers"] = list(prefill.blockers) or ["Prompt prefill did not produce hidden states."]
            _write(out_path, payload)
            return 1

        verify_started = time.perf_counter()
        continuation = run_fp8_prompt_prefill_batch(
            args.model_id,
            [candidate_ids],
            layer_count=int(prefill.layer_count),
            start_pos=len(prepared.token_ids),
            previous_kv_caches=prefill.next_kv_caches,
        )
        verify_elapsed = time.perf_counter() - verify_started
        tail_started = time.perf_counter()
        tail = (
            run_fp8_decode_tail_topk_batch(args.model_id, continuation.output_tensor, top_k=1)
            if continuation.ready and continuation.output_tensor is not None
            else None
        )
        tail_elapsed = time.perf_counter() - tail_started
        tail_ids = [] if tail is None or not tail.ready else [int(row[0]) for row in tail.top_token_ids_by_position if row]
        verifier_ids = [
            *(list(prefill.tail.top_token_ids[:1]) if prefill.tail is not None else []),
            *tail_ids,
        ]
        accepted = 0
        for candidate, verifier in zip(candidate_ids, verifier_ids):
            if int(candidate) != int(verifier):
                break
            accepted += 1
        blockers = [*list(continuation.blockers), *([] if tail is None else list(tail.blockers))]
        payload.update(
            {
                "verifier_ready": bool(continuation.ready),
                "verifier_elapsed_seconds": round(verify_elapsed, 4),
                "tail_ready": bool(tail is not None and tail.ready),
                "tail_elapsed_seconds": round(tail_elapsed, 4),
                "verifier_token_ids": verifier_ids,
                "accepted": int(accepted),
                "accepted_seconds_per_token": None if accepted <= 0 else round((verify_elapsed + tail_elapsed) / accepted, 4),
                "executed_layers": int(len(prefill.executed_layers) + len(continuation.executed_layers)),
                "expected_layers_executed": int(prefill.layer_count * 2),
                "anti_cheat_passed": int(len(prefill.executed_layers) + len(continuation.executed_layers))
                == int(prefill.layer_count * 2),
                "verifier_timing": _summarize_fp8_verifier_timing(continuation.step_summaries),
                "blockers": blockers,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
        )
        payload["ready"] = bool(not blockers and continuation.ready and tail is not None and tail.ready)
        _write(out_path, payload)
        return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
