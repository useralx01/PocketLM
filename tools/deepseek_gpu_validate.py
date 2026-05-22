"""Run PLM-13 real DeepSeek V3 GPU validation probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcketlm.core.runtime.deepseek_remote_gpu import (
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    run_remote_deepseek_layer_probe,
    run_remote_deepseek_resident_layer_probe,
)
from pcketlm.core.runtime.gpu_smoke import run_deepseek_gpu_probe, run_gpu_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate real DeepSeek V3 FP8 work on a CUDA notebook.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--resident-iterations", type=int, default=8)
    args = parser.parse_args(argv)

    smoke = None
    synthetic = None
    if not args.skip_synthetic:
        smoke = run_gpu_smoke(require_cuda=args.require_cuda, device=args.device)
        synthetic = run_deepseek_gpu_probe(require_cuda=args.require_cuda, device=args.device)
    real = run_remote_deepseek_layer_probe(
        repo_id=args.repo_id,
        revision=args.revision,
        layer_index=args.layer,
        token_id=args.token_id,
        require_cuda=args.require_cuda,
        device=args.device,
        dtype=torch.float16,
    )
    resident = run_remote_deepseek_resident_layer_probe(
        repo_id=args.repo_id,
        revision=args.revision,
        layer_index=args.layer,
        token_id=args.token_id,
        iterations=args.resident_iterations,
        require_cuda=args.require_cuda,
        device=args.device,
        dtype=torch.float16,
    )
    payload = {
        "smoke": None if smoke is None else smoke.to_dict(),
        "deepseek_gpu_probe": None if synthetic is None else synthetic.to_dict(),
        "real_deepseek_remote_layer_probe": real.to_dict(),
        "real_deepseek_resident_layer_probe": resident.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "real DeepSeek layer: "
            f"passed={real.passed} device={real.device} layer={real.layer_index} "
            f"elapsed={real.elapsed_seconds:.3f}s projected={real.projected_config_layers_seconds_per_token:.3f}s; "
            f"resident={resident.seconds_per_resident_layer:.3f}s/layer "
            f"projected={resident.projected_config_layers_seconds_per_token:.3f}s"
        )
    if args.require_cuda and not real.cuda_available:
        return 2
    return (
        0
        if real.passed and resident.passed and (smoke is None or smoke.passed) and (synthetic is None or synthetic.passed)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
