from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcketlm.core.runtime.fp8_source import (
    apply_fp8_exact_mtp_default_env,
    run_fp8_mtp_draft_step,
    run_fp8_prompt_prefill,
)
from pcketlm.core.runtime.tokenizer_runtime import decode_token_ids_to_text, prepare_prompt_text


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _rank(token_id: int, top_ids: list[int]) -> int | None:
    for index, value in enumerate(top_ids):
        if int(value) == int(token_id):
            return int(index + 1)
    return None


def _parse_token_ids(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).replace(";", ",").split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Record MTP top-k ranks while forcing a known token sequence.")
    parser.add_argument("model_id")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--sequence-token-ids", required=True)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--mtp-top-k", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    out_path = Path(args.out)
    env = apply_fp8_exact_mtp_default_env()
    prepared = prepare_prompt_text(args.model_id, args.prompt, apply_chat_format=True)
    target_ids = _parse_token_ids(args.sequence_token_ids)
    target_text, target_text_blockers = decode_token_ids_to_text(args.model_id, target_ids)
    payload: dict = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "prepared_prompt": prepared.prepared_prompt,
        "sequence_token_ids": list(target_ids),
        "sequence_text": target_text,
        "sequence_text_blockers": list(target_text_blockers),
        "mtp_top_k": int(args.mtp_top_k),
        "env": env,
        "prepared_ready": bool(prepared.ready),
        "prepared_blockers": list(prepared.blockers),
        "steps": [],
        "ready": False,
    }
    _write(out_path, payload)
    if not prepared.ready or not target_ids:
        return 1

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
        }
    )
    _write(out_path, payload)
    if not prefill.ready or prefill.output_tensor is None:
        return 1

    previous_hidden = prefill.output_tensor[:, -1:, :]
    mtp_cache = None
    input_token = int(prepared.token_ids[-1])
    position = len(prepared.token_ids) - 1
    blockers: list[str] = []
    for index, target_token_id in enumerate(target_ids):
        step_started = time.perf_counter()
        step = run_fp8_mtp_draft_step(
            args.model_id,
            input_token,
            previous_hidden,
            position=position,
            previous_kv_cache=mtp_cache,
            top_k=max(1, int(args.mtp_top_k)),
        )
        top_ids = [int(value) for value in step.top_token_ids]
        rank = _rank(int(target_token_id), top_ids)
        step_payload = step.to_dict()
        step_payload.update(
            {
                "step_index": int(index + 1),
                "target_token_id": int(target_token_id),
                "target_rank": rank,
                "target_in_top_k": rank is not None,
                "step_wall_seconds": round(time.perf_counter() - step_started, 4),
            }
        )
        payload["steps"].append(step_payload)
        blockers.extend(step.blockers)
        payload.update(
            {
                "missing_from_top_k": sum(1 for item in payload["steps"] if item.get("target_rank") is None),
                "rank_sequence": [item.get("target_rank") for item in payload["steps"]],
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "blockers": list(blockers),
            }
        )
        _write(out_path, payload)
        if not step.ready or step.output_tensor is None:
            break
        input_token = int(target_token_id)
        previous_hidden = step.output_tensor
        mtp_cache = step.next_kv_cache
        position += 1

    payload["ready"] = bool(
        not blockers
        and len(payload["steps"]) == len(target_ids)
        and all(item.get("target_rank") is not None for item in payload["steps"])
    )
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    _write(out_path, payload)
    print(
        json.dumps(
            {
                "ready": payload["ready"],
                "rank_sequence": payload.get("rank_sequence", []),
                "missing_from_top_k": payload.get("missing_from_top_k", 0),
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
