"""Benchmark prompt-cached DeepSeek FP8 candidate verification."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pcketlm.core.runtime.speculative import FP8CachedVerifierSession
from pcketlm.core.runtime.fp8_source import _load_deepseek_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="deepseek-v3")
    parser.add_argument("--prompt-token", type=int, action="append", default=None)
    parser.add_argument("--candidate-token", type=int, default=223)
    parser.add_argument("--candidate-count", type=int, default=96)
    parser.add_argument("--layer-count", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt = list(args.prompt_token or [0, 1])
    candidates = [int(args.candidate_token)] * max(1, int(args.candidate_count))
    started = time.perf_counter()
    layer_count = int(args.layer_count) if args.layer_count is not None else int(_load_deepseek_config(args.model_id)["num_hidden_layers"])
    session = FP8CachedVerifierSession(args.model_id, prompt, layer_count=layer_count)
    prefill_elapsed = time.perf_counter() - started
    verify_started = time.perf_counter()
    result = session.verify(candidates)
    verify_elapsed = time.perf_counter() - verify_started
    payload = {
        "model_id": args.model_id,
        "prompt_token_ids": prompt,
        "candidate_token": int(args.candidate_token),
        "candidate_count": len(candidates),
        "layer_count": int(layer_count),
        "session_ready": bool(session.ready),
        "session_blockers": list(session.blockers),
        "prefill_elapsed_seconds": round(prefill_elapsed, 4),
        "session_prefill_elapsed_seconds": float(session.prefill_elapsed_seconds),
        "next_token_id": session.next_token_id,
        "verify": result.to_dict(),
        "verify_elapsed_seconds": round(verify_elapsed, 4),
        "effective_seconds_per_candidate": None
        if not result.ready
        else round(float(result.elapsed_seconds) / max(1, len(candidates)), 4),
        "total_elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
