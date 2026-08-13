from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcketlm.core.runtime.fp8_source import (  # noqa: E402
    _load_deepseek_config,
    _summarize_fp8_verifier_timing,
    apply_fp8_exact_mtp_default_env,
    run_fp8_decode_tail_topk_batch,
    run_fp8_prompt_prefill,
    run_fp8_prompt_prefill_batch,
)
from pcketlm.core.runtime.tensor_loader import scoped_tensor_handle_cache  # noqa: E402
from pcketlm.core.runtime.tokenizer_runtime import prepare_prompt_text  # noqa: E402


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).replace(";", ",").split(",") if part.strip()]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _candidate_prefix(base_ids: list[int], length: int) -> list[int]:
    if length <= 0:
        return []
    if not base_ids:
        return []
    out: list[int] = []
    while len(out) < length:
        out.extend(base_ids)
    return out[:length]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure exact FP8 MTP verifier cost at multiple branch lengths.")
    parser.add_argument("model_id")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--candidate-token-ids", required=True)
    parser.add_argument("--lengths", default="10,20,40")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    out_path = Path(args.out)
    env = apply_fp8_exact_mtp_default_env()
    config = _load_deepseek_config(args.model_id)
    layer_count = int(args.layers) if args.layers is not None else int(config["num_hidden_layers"])
    prepared = prepare_prompt_text(args.model_id, args.prompt, apply_chat_format=True)
    base_ids = _parse_int_list(args.candidate_token_ids)
    lengths = sorted({max(1, int(value)) for value in _parse_int_list(args.lengths)})
    payload: dict = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "base_candidate_token_ids": list(base_ids),
        "lengths": list(lengths),
        "layer_count_arg": args.layers,
        "layer_count": int(layer_count),
        "env": env,
        "prepared_ready": bool(prepared.ready),
        "prepared_blockers": list(prepared.blockers),
        "measurements": [],
        "blockers": [],
        "ready": False,
        "elapsed_seconds": 0.0,
    }
    _write(out_path, payload)
    if not prepared.ready or not base_ids or not lengths:
        payload["blockers"] = list(prepared.blockers)
        if not base_ids:
            payload["blockers"].append("At least one candidate token id is required.")
        if not lengths:
            payload["blockers"].append("At least one length is required.")
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        _write(out_path, payload)
        return 1

    with scoped_tensor_handle_cache():
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
                "prefill_layers_executed": len(prefill.executed_layers),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
        )
        _write(out_path, payload)
        if not prefill.ready or prefill.output_tensor is None:
            payload["blockers"] = list(prefill.blockers) or ["Prompt prefill did not produce hidden states."]
            payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            _write(out_path, payload)
            return 1

        for length in lengths:
            candidate_ids = _candidate_prefix(base_ids, int(length))
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
            blockers = [*list(continuation.blockers), *([] if tail is None else list(tail.blockers))]
            measurement = {
                "candidate_count": int(length),
                "candidate_token_ids": list(candidate_ids),
                "verifier_ready": bool(continuation.ready),
                "tail_ready": bool(tail is not None and tail.ready),
                "verify_elapsed_seconds": round(verify_elapsed, 4),
                "tail_elapsed_seconds": round(tail_elapsed, 4),
                "total_verify_tail_seconds": round(verify_elapsed + tail_elapsed, 4),
                "seconds_per_candidate": round((verify_elapsed + tail_elapsed) / max(1, int(length)), 4),
                "executed_layers": int(len(continuation.executed_layers)),
                "expected_layers": int(prefill.layer_count),
                "one_weight_sweep": int(len(continuation.executed_layers)) == int(prefill.layer_count),
                "verifier_timing": _summarize_fp8_verifier_timing(continuation.step_summaries),
                "blockers": blockers,
            }
            payload["measurements"].append(measurement)
            payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            payload["ready"] = bool(payload["measurements"] and all(not item["blockers"] for item in payload["measurements"]))
            _write(out_path, payload)
            if blockers:
                break

    payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    payload["ready"] = bool(payload["measurements"] and all(not item["blockers"] for item in payload["measurements"]))
    _write(out_path, payload)
    print(
        json.dumps(
            {
                "ready": payload["ready"],
                "measurements": [
                    {
                        "candidate_count": item["candidate_count"],
                        "seconds_per_candidate": item["seconds_per_candidate"],
                        "total_verify_tail_seconds": item["total_verify_tail_seconds"],
                    }
                    for item in payload["measurements"]
                ],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
