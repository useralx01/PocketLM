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
    run_remote_deepseek_paged_decode_probe,
    run_remote_deepseek_resident_decode_probe,
    run_remote_deepseek_resident_layer_probe,
)
from pcketlm.core.runtime.deepseek_gpu_residency import (
    estimate_deepseek_gpu_residency,
    run_local_deepseek_paged_decode_probe,
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
    parser.add_argument("--skip-remote", action="store_true")
    parser.add_argument("--resident-iterations", type=int, default=8)
    parser.add_argument("--resident-decode-layers", type=int, default=0)
    parser.add_argument("--resident-decode-iterations", type=int, default=3)
    parser.add_argument("--paged-decode-layers", type=int, default=0)
    parser.add_argument("--paged-budget-gb", type=float, default=6.0)
    parser.add_argument("--paged-prefetch-window", type=int, default=0)
    parser.add_argument("--local-model-id", default="")
    parser.add_argument("--local-paged-decode-layers", type=int, default=0)
    parser.add_argument("--local-paged-budget-gb", type=float, default=12.0)
    args = parser.parse_args(argv)

    smoke = None
    synthetic = None
    if not args.skip_synthetic:
        smoke = run_gpu_smoke(require_cuda=args.require_cuda, device=args.device)
        synthetic = run_deepseek_gpu_probe(require_cuda=args.require_cuda, device=args.device)
    real = None
    resident = None
    if not args.skip_remote:
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
        "real_deepseek_remote_layer_probe": None if real is None else real.to_dict(),
        "real_deepseek_resident_layer_probe": None if resident is None else resident.to_dict(),
    }
    if args.resident_decode_layers > 0:
        decode = run_remote_deepseek_resident_decode_probe(
            repo_id=args.repo_id,
            revision=args.revision,
            token_id=args.token_id,
            start_layer=args.layer,
            layer_count=args.resident_decode_layers,
            benchmark_iterations=args.resident_decode_iterations,
            require_cuda=args.require_cuda,
            device=args.device,
            dtype=torch.float16,
        )
        payload["real_deepseek_resident_decode_probe"] = decode.to_dict()
    paged = None
    if args.paged_decode_layers > 0:
        paged = run_remote_deepseek_paged_decode_probe(
            repo_id=args.repo_id,
            revision=args.revision,
            token_id=args.token_id,
            start_layer=args.layer,
            layer_count=args.paged_decode_layers,
            resident_budget_bytes=int(float(args.paged_budget_gb) * 1024**3),
            prefetch_window=args.paged_prefetch_window,
            require_cuda=args.require_cuda,
            device=args.device,
            dtype=torch.float16,
        )
        payload["real_deepseek_paged_decode_probe"] = paged.to_dict()
    local_paged = None
    if args.local_model_id:
        estimate = estimate_deepseek_gpu_residency(
            args.local_model_id,
            start_layer=args.layer,
            layer_count=args.local_paged_decode_layers or args.paged_decode_layers or args.resident_decode_layers or 1,
            resident_budget_bytes=int(float(args.local_paged_budget_gb) * 1024**3),
        )
        payload["local_deepseek_gpu_residency_estimate"] = estimate.to_dict()
        if args.local_paged_decode_layers > 0:
            local_paged = run_local_deepseek_paged_decode_probe(
                args.local_model_id,
                token_id=args.token_id,
                start_layer=args.layer,
                layer_count=args.local_paged_decode_layers,
                resident_budget_bytes=int(float(args.local_paged_budget_gb) * 1024**3),
                require_cuda=args.require_cuda,
                device=args.device,
                dtype=torch.float16,
            )
            payload["local_deepseek_paged_decode_probe"] = local_paged.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if real is not None and resident is not None:
            print(
                "real DeepSeek layer: "
                f"passed={real.passed} device={real.device} layer={real.layer_index} "
                f"elapsed={real.elapsed_seconds:.3f}s projected={real.projected_config_layers_seconds_per_token:.3f}s; "
                f"resident={resident.seconds_per_resident_layer:.3f}s/layer "
                f"projected={resident.projected_config_layers_seconds_per_token:.3f}s"
            )
        elif local_paged is not None:
            print(
                "local DeepSeek pager: "
                f"passed={local_paged.passed} device={local_paged.device} "
                f"layers={local_paged.executed_layers} elapsed={local_paged.elapsed_seconds:.3f}s"
            )
        else:
            print("DeepSeek GPU validation completed with remote probes skipped.")
    if args.require_cuda and real is not None and not real.cuda_available:
        return 2
    return (
        0
        if (real is None or real.passed)
        and (resident is None or resident.passed)
        and (paged is None or paged.passed)
        and (local_paged is None or local_paged.passed)
        and (smoke is None or smoke.passed)
        and (synthetic is None or synthetic.passed)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
