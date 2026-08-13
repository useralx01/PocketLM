"""Benchmark the exact FP8 CPU residency split with explicit telemetry."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcketlm.core.runtime.fp8_pack import clear_fp8_pack_readers
from pcketlm.core.runtime.fp8_source import (
    clear_fp8_attention_weight_cache,
    clear_fp8_lm_head_full_cache,
    clear_fp8_mlp_span_cache,
    fp8_residency_split_snapshot,
    run_fp8_decode_loop,
)
from pcketlm.core.runtime.tokenizer_runtime import decode_token_ids_to_text


def _setdefault_env(name: str, value: str) -> None:
    if not os.environ.get(name, "").strip():
        os.environ[name] = value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="deepseek-v3")
    parser.add_argument("--prompt-token", type=int, action="append", default=None)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--layer-count", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--mlp-cache-mb", type=float, default=2048.0)
    parser.add_argument("--attention-cache-mb", type=float, default=4096.0)
    parser.add_argument("--budget-mb", type=float, default=16 * 1024.0)
    parser.add_argument("--reserve-mb", type=float, default=1024.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    _setdefault_env("PCKETLM_ENABLE_FP8_RESIDENCY_SPLIT", "1")
    _setdefault_env("PCKETLM_FP8_MLP_SPAN_CACHE_MB", str(float(args.mlp_cache_mb)))
    _setdefault_env("PCKETLM_FP8_MLP_SPAN_CACHE_POLICY", "preserve-full")
    _setdefault_env("PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_MB", str(float(args.attention_cache_mb)))
    _setdefault_env("PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_POLICY", "prefix")
    _setdefault_env("PCKETLM_FP8_RESIDENCY_SPLIT_BUDGET_MB", str(float(args.budget_mb)))
    _setdefault_env("PCKETLM_FP8_RESIDENCY_SPLIT_RESERVE_MB", str(float(args.reserve_mb)))

    clear_fp8_pack_readers()
    clear_fp8_attention_weight_cache()
    clear_fp8_mlp_span_cache()
    clear_fp8_lm_head_full_cache()

    prompt = [int(value) for value in (args.prompt_token or [100])]
    started = time.perf_counter()
    result = run_fp8_decode_loop(
        args.model_id,
        prompt,
        start_layer=int(args.start_layer),
        layer_count=int(args.layer_count),
        max_new_tokens=int(args.max_new_tokens),
        dtype=torch.bfloat16,
    )
    wall_seconds = time.perf_counter() - started
    generated_text, text_blockers = decode_token_ids_to_text(args.model_id, list(result.generated_token_ids))
    payload = result.to_dict()
    generated_count = len(result.generated_token_ids)
    payload.update(
        {
            "issue": "PLM-18",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": round(float(wall_seconds), 4),
            "generated_token_count": int(generated_count),
            "seconds_per_generated_token": None
            if generated_count <= 0
            else round(float(wall_seconds) / float(generated_count), 4),
            "generated_text": generated_text,
            "text_blockers": text_blockers,
            "hard_constraints": [
                "cpu_only_no_gpu",
                "deepseek_v3_only_mtp_head_allowed",
                "fp8_native_no_quantization_change",
                "no_dropped_experts_or_layers",
            ],
            "residency_split": fp8_residency_split_snapshot(
                args.model_id,
                start_layer=int(args.start_layer),
                layer_count=int(args.layer_count),
            ),
            "environment": {
                "PCKETLM_ENABLE_FP8_RESIDENCY_SPLIT": os.environ.get("PCKETLM_ENABLE_FP8_RESIDENCY_SPLIT", ""),
                "PCKETLM_FP8_MLP_SPAN_CACHE_MB": os.environ.get("PCKETLM_FP8_MLP_SPAN_CACHE_MB", ""),
                "PCKETLM_FP8_MLP_SPAN_CACHE_POLICY": os.environ.get("PCKETLM_FP8_MLP_SPAN_CACHE_POLICY", ""),
                "PCKETLM_FP8_MLP_SPAN_CACHE_FILTER": os.environ.get("PCKETLM_FP8_MLP_SPAN_CACHE_FILTER", ""),
                "PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_MB": os.environ.get("PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_MB", ""),
                "PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_POLICY": os.environ.get(
                    "PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_POLICY",
                    "",
                ),
                "PCKETLM_FP8_RESIDENCY_SPLIT_BUDGET_MB": os.environ.get(
                    "PCKETLM_FP8_RESIDENCY_SPLIT_BUDGET_MB",
                    "",
                ),
                "PCKETLM_FP8_RESIDENCY_SPLIT_RESERVE_MB": os.environ.get(
                    "PCKETLM_FP8_RESIDENCY_SPLIT_RESERVE_MB",
                    "",
                ),
            },
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if result.ready and not text_blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
