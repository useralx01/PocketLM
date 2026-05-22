"""Submit and poll a real DeepSeek V3 GPU validation notebook on Kaggle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "build" / "kaggle_deepseek_gpu_validate"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.kaggle_gpu_smoke import (
    DEFAULT_STATE_DIR,
    _branch_name,
    _commit_sha,
    _kaggle_command,
    _repo_url,
    download_output,
    kaggle_credentials,
    kernel_id_from_submit_output,
    poll_kernel,
    write_state,
)


def prepare_kernel(
    *,
    username: str,
    slug: str,
    title: str,
    out_dir: Path = DEFAULT_BUILD_DIR,
    resident_layers: int = 3,
    resident_iterations: int = 3,
    paged_layers: int = 0,
    paged_budget_gb: float = 8.0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_gpu_source = (ROOT / "src" / "pcketlm" / "core" / "runtime" / "deepseek_remote_gpu.py").read_text(
        encoding="utf-8"
    )
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": "deepseek_gpu_validate.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source = "\n".join(
        [
            "import json",
            "import sys",
            "from pathlib import Path",
            "",
            f"REMOTE_GPU_SOURCE = {remote_gpu_source!r}",
            "Path('deepseek_remote_gpu_embedded.py').write_text(REMOTE_GPU_SOURCE, encoding='utf-8')",
            "",
            "print('PCKETLM_DEEPSEEK_GPU_VALIDATE_START')",
            "import torch",
            "print('PCKETLM_TORCH_PROBE', json.dumps({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()}))",
            "from deepseek_remote_gpu_embedded import (",
            "    run_remote_deepseek_layer_probe,",
            "    run_remote_deepseek_resident_layer_probe,",
            "    run_remote_deepseek_resident_decode_probe,",
            "    run_remote_deepseek_paged_decode_probe,",
            ")",
            "payload = {}",
            "payload['real_deepseek_remote_layer_probe'] = run_remote_deepseek_layer_probe(require_cuda=True, dtype=torch.float16).to_dict()",
            "payload['real_deepseek_resident_layer_probe'] = run_remote_deepseek_resident_layer_probe(require_cuda=True, iterations=8, dtype=torch.float16).to_dict()",
            f"payload['real_deepseek_resident_decode_probe'] = run_remote_deepseek_resident_decode_probe(require_cuda=True, layer_count={int(resident_layers)}, benchmark_iterations={int(resident_iterations)}, dtype=torch.float16).to_dict()",
            (
                f"payload['real_deepseek_paged_decode_probe'] = run_remote_deepseek_paged_decode_probe(require_cuda=True, layer_count={int(paged_layers)}, resident_budget_bytes=int({float(paged_budget_gb)} * 1024**3), dtype=torch.float16).to_dict()"
                if int(paged_layers) > 0
                else "payload['real_deepseek_paged_decode_probe'] = None"
            ),
            "print('PCKETLM_DEEPSEEK_VALIDATE_OUTPUT_START')",
            "print(json.dumps(payload, indent=2, sort_keys=True))",
            "print('PCKETLM_DEEPSEEK_VALIDATE_OUTPUT_END')",
            "passed = payload['real_deepseek_remote_layer_probe']['passed'] and payload['real_deepseek_resident_layer_probe']['passed'] and payload['real_deepseek_resident_decode_probe']['passed']",
            "if payload['real_deepseek_paged_decode_probe'] is not None:",
            "    passed = passed and payload['real_deepseek_paged_decode_probe']['passed']",
            "raise SystemExit(0 if passed else 1)",
            "",
        ]
    )
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "deepseek-gpu-validate",
                "metadata": {},
                "outputs": [],
                "source": [f"{line}\n" for line in source.splitlines()],
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (out_dir / "deepseek_gpu_validate.ipynb").write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return out_dir


def submit_kernel(kernel_dir: Path, *, accelerator: str) -> str:
    return _kaggle_command(["kernels", "push", "-p", str(kernel_dir), "--accelerator", accelerator]).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real DeepSeek V3 GPU validation on Kaggle.")
    parser.add_argument("--slug", default=f"pocketlm-deepseek-gpu-{int(time.time())}")
    parser.add_argument("--title", default="PocketLM DeepSeek GPU Validate")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--resident-layers", type=int, default=3)
    parser.add_argument("--resident-iterations", type=int, default=3)
    parser.add_argument("--paged-layers", type=int, default=0)
    parser.add_argument("--paged-budget-gb", type=float, default=8.0)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    creds = kaggle_credentials()
    username = creds.username or "missing-kaggle-user"
    kernel_dir = prepare_kernel(
        username=username,
        slug=args.slug,
        title=args.title,
        resident_layers=args.resident_layers,
        resident_iterations=args.resident_iterations,
        paged_layers=args.paged_layers,
        paged_budget_gb=args.paged_budget_gb,
    )
    kernel_id = f"{username}/{args.slug}"
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kernel_id": kernel_id,
        "kernel_dir": str(kernel_dir),
        "credentials_available": creds.available,
        "credentials_source": creds.source,
        "credentials_message": creds.message,
        "source_repo": _repo_url(),
        "source_branch": _branch_name(),
        "source_commit": _commit_sha(),
    }
    if not creds.available:
        payload["status"] = "blocked_missing_kaggle_credentials"
        print(json.dumps(payload, indent=2, sort_keys=True))
        write_state(payload, state_dir=DEFAULT_STATE_DIR / "deepseek_validate")
        return 2
    submit_output = submit_kernel(kernel_dir, accelerator=args.accelerator)
    submitted_kernel_id = kernel_id_from_submit_output(submit_output, kernel_id)
    status_output = poll_kernel(submitted_kernel_id, interval_seconds=args.poll_interval, timeout_seconds=args.timeout)
    output_dir = DEFAULT_STATE_DIR / "deepseek_validate" / "output"
    download_log = download_output(submitted_kernel_id, output_dir)
    payload.update(
        {
            "status": "submitted",
            "requested_kernel_id": kernel_id,
            "kernel_id": submitted_kernel_id,
            "submit_output": submit_output,
            "status_output": status_output,
            "download_log": download_log,
            "output_dir": str(output_dir),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    write_state(payload, state_dir=DEFAULT_STATE_DIR / "deepseek_validate")
    return 0 if "complete" in status_output.lower() else 1


if __name__ == "__main__":
    raise SystemExit(main())
