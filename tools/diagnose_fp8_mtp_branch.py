from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcketlm.core.runtime.fp8_source import (  # noqa: E402
    _fp8_mtp_tree_choices,
    _fp8_mtp_tree_step_top_k,
    _fp8_mtp_token_rank,
    _load_deepseek_config,
    apply_fp8_exact_mtp_default_env,
    run_fp8_mtp_draft_step,
    run_fp8_prompt_prefill,
)
from pcketlm.core.runtime.tokenizer_runtime import (  # noqa: E402
    decode_token_ids_to_text,
    load_generation_settings,
    prepare_prompt_text,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally trace one DeepSeek V3 MTP candidate branch.")
    parser.add_argument("model_id")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--tree-depth", type=int, default=10)
    parser.add_argument("--tree-width", type=int, default=8)
    parser.add_argument("--mtp-top-k", type=int, default=2048)
    parser.add_argument(
        "--tree-branch-strategy",
        choices=[
            "topn",
            "punctuation-biased",
            "punctuation-next2",
            "punctuation-forced-next2",
            "punctuation-forced-continuation-top1",
            "punctuation-rank-pattern",
            "punctuation-rank-first-next2",
            "punctuation-rank-first-next3",
            "punctuation-rank-first-continuation-rank3",
            "punctuation-rank-onepass-top2048",
            "punctuation-rank-long-top2048",
            "fixed-rank-sequence",
        ],
        default="punctuation-rank-onepass-top2048",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    out_path = Path(args.out)
    env = apply_fp8_exact_mtp_default_env()
    prepared = prepare_prompt_text(args.model_id, args.prompt, apply_chat_format=True)
    config = _load_deepseek_config(args.model_id)
    layer_count = int(args.layers) if args.layers is not None else int(config["num_hidden_layers"])
    eos_token_ids = {int(value) for value in load_generation_settings(args.model_id).eos_token_ids}
    payload: dict = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "prepared_prompt": prepared.prepared_prompt,
        "prompt_token_ids": list(prepared.token_ids),
        "layer_count": int(layer_count),
        "tree_depth": max(1, int(args.tree_depth)),
        "tree_width": max(1, int(args.tree_width)),
        "mtp_top_k": max(1, int(args.mtp_top_k)),
        "tree_branch_strategy": str(args.tree_branch_strategy),
        "env": env,
        "prepared_ready": bool(prepared.ready),
        "prepared_blockers": list(prepared.blockers),
        "steps": [],
        "candidate_token_ids": [],
        "candidate_text": "",
        "blockers": [],
        "ready": False,
        "elapsed_seconds": 0.0,
    }
    _write(out_path, payload)
    if not prepared.ready:
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        _write(out_path, payload)
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
            "prefill_ready": bool(prefill.ready),
            "prefill_elapsed_seconds": round(time.perf_counter() - prefill_started, 4),
            "prefill_top_token_ids": [] if prefill.tail is None else list(prefill.tail.top_token_ids),
            "prefill_blockers": list(prefill.blockers),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
    )
    _write(out_path, payload)
    if not prefill.ready or prefill.output_tensor is None or prefill.tail is None or not prefill.tail.top_token_ids:
        payload["blockers"] = list(prefill.blockers or ["Verifier prefill did not produce a next-token tail."])
        _write(out_path, payload)
        return 1

    current_hidden = prefill.output_tensor[:, -1:, :].detach().contiguous()
    current_input_token = int(prepared.token_ids[-1])
    current_position = len(prepared.token_ids) - 1
    mtp_cache = None
    next_token_id = int(prefill.tail.top_token_ids[0])
    target_depth = max(1, int(args.tree_depth))
    tree_width = max(1, int(args.tree_width))
    mtp_top_k = max(1, int(args.mtp_top_k))
    blockers: list[str] = []
    candidate_tokens: list[int] = []

    for step_index in range(target_depth):
        step_started = time.perf_counter()
        requested_top_k = (
            max(int(mtp_top_k), int(tree_width), 1)
            if step_index == 0
            else _fp8_mtp_tree_step_top_k(
                previous_token_id=int(current_input_token),
                branch_tokens=list(candidate_tokens),
                continuation_mode=False,
                branch_strategy=str(args.tree_branch_strategy),
                tree_width=tree_width,
                mtp_top_k=mtp_top_k,
            )
        )
        step = run_fp8_mtp_draft_step(
            args.model_id,
            int(current_input_token),
            current_hidden,
            position=int(current_position),
            previous_kv_cache=mtp_cache,
            top_k=requested_top_k,
            include_tail=True,
        )
        step_payload = step.to_dict()
        top_ids = [int(value) for value in step.top_token_ids]
        selected_token_id = None
        selected_rank = None
        forced_to_verifier = False
        if not step.ready or step.output_tensor is None or not top_ids:
            blockers.extend(step.blockers or ["MTP draft step did not produce candidates."])
        elif step_index == 0:
            selected_token_id = int(next_token_id)
            selected_rank = _fp8_mtp_token_rank(selected_token_id, top_ids)
            forced_to_verifier = selected_rank is None
        else:
            choices = _fp8_mtp_tree_choices(
                step,
                eos_token_ids,
                width=tree_width,
                previous_token_id=int(current_input_token),
                branch_tokens=list(candidate_tokens),
                continuation_mode=False,
                prefer_eos_after_punctuation=False,
                branch_strategy=str(args.tree_branch_strategy),
            )
            if choices:
                selected_token_id = int(choices[0])
                selected_rank = _fp8_mtp_token_rank(selected_token_id, top_ids)
            else:
                blockers.append("MTP branch strategy produced no choices.")

        step_payload.update(
            {
                "step_index": int(step_index + 1),
                "requested_top_k": int(requested_top_k),
                "selected_token_id": selected_token_id,
                "selected_rank": selected_rank,
                "forced_to_verifier_next_token_id": bool(forced_to_verifier),
                "step_wall_seconds": round(time.perf_counter() - step_started, 4),
            }
        )
        payload["steps"].append(step_payload)
        if selected_token_id is not None:
            candidate_tokens.append(int(selected_token_id))
        candidate_text, text_blockers = decode_token_ids_to_text(args.model_id, candidate_tokens)
        blockers.extend(text_blockers)
        payload.update(
            {
                "candidate_token_ids": list(candidate_tokens),
                "candidate_text": candidate_text,
                "blockers": list(blockers),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
        )
        _write(out_path, payload)
        if blockers or step.output_tensor is None or selected_token_id is None:
            break
        current_hidden = step.output_tensor.detach().contiguous()
        mtp_cache = step.next_kv_cache
        current_input_token = int(selected_token_id)
        current_position += 1
        if int(selected_token_id) in eos_token_ids:
            break

    payload["ready"] = bool(not blockers and len(candidate_tokens) == target_depth)
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    _write(out_path, payload)
    print(
        json.dumps(
            {
                "ready": payload["ready"],
                "candidate_token_ids": payload["candidate_token_ids"],
                "candidate_text": payload["candidate_text"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "step_count": len(payload["steps"]),
            },
            indent=2,
        )
    )
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
