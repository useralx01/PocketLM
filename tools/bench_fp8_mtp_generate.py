from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcketlm.core.runtime import run_fp8_mtp_batched_generate, warm_fp8_mtp_prompt_prefill_cache


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate with DeepSeek MTP proposals and exact FP8 verifier commits.")
    parser.add_argument("model_id")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-visible", type=int, default=10)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--max-passes", type=int, default=20)
    parser.add_argument("--warmup-visible", type=int, default=0)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--warmup-prefill-only", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefer-eos-after-punctuation", action="store_true")
    parser.add_argument("--tree-width", type=int, default=8)
    parser.add_argument("--tree-depth", type=int, default=10)
    parser.add_argument("--max-tree-nodes", type=int, default=8)
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
    args = parser.parse_args()

    warmup_payload = None
    if args.warmup_prefill_only:
        warmup_payload = warm_fp8_mtp_prompt_prefill_cache(
            args.model_id,
            args.prompt,
            apply_chat_format=True,
            layer_count=args.layers,
        )
    elif int(args.warmup_visible) > 0:
        warmup_payload = run_fp8_mtp_batched_generate(
            args.model_id,
            args.prompt,
            apply_chat_format=True,
            max_visible_tokens=args.warmup_visible,
            k=args.k,
            layer_count=args.layers,
            max_passes=args.warmup_passes,
            prefer_eos_after_punctuation=args.prefer_eos_after_punctuation,
            tree_width=args.tree_width,
            tree_depth=args.tree_depth,
            max_tree_nodes=args.max_tree_nodes,
            mtp_top_k=args.mtp_top_k,
            tree_branch_strategy=args.tree_branch_strategy,
        )

    payload = run_fp8_mtp_batched_generate(
        args.model_id,
        args.prompt,
        apply_chat_format=True,
        max_visible_tokens=args.max_visible,
        k=args.k,
        layer_count=args.layers,
        max_passes=args.max_passes,
        prefer_eos_after_punctuation=args.prefer_eos_after_punctuation,
        tree_width=args.tree_width,
        tree_depth=args.tree_depth,
        max_tree_nodes=args.max_tree_nodes,
        mtp_top_k=args.mtp_top_k,
        tree_branch_strategy=args.tree_branch_strategy,
    )
    if warmup_payload is not None:
        payload["warmup_proof"] = warmup_payload
        payload["warmup_ready"] = bool(warmup_payload.get("ready"))
        payload["warmup_visible_tokens"] = int(args.warmup_visible)
        payload["warmup_passes"] = int(args.warmup_passes)
        payload["warmup_prefill_only"] = bool(args.warmup_prefill_only)
    _write(Path(args.out), payload)
    return 0 if payload.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
