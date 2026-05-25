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
DEFAULT_DEEPSEEK_MODEL_SOURCE = "deepseek-ai/deepseek-v3/transformers/deepseek-v3/2"
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


def _embedded_runtime_files(remote_gpu_source: str) -> dict[str, str]:
    """Return the minimal pcketlm source tree needed for mounted-model GPU probes."""
    files = {
        "src/pcketlm/__init__.py": "",
        "src/pcketlm/core/__init__.py": "",
        "src/pcketlm/core/runtime/__init__.py": "",
        "src/pcketlm/core/storage/__init__.py": "",
        "src/pcketlm/core/runtime/deepseek_remote_gpu.py": remote_gpu_source,
        "src/pcketlm/core/runtime/deepseek_gpu_residency.py": (
            ROOT / "src" / "pcketlm" / "core" / "runtime" / "deepseek_gpu_residency.py"
        ).read_text(encoding="utf-8"),
        "src/pcketlm/core/runtime/fp8_source.py": (
            ROOT / "src" / "pcketlm" / "core" / "runtime" / "fp8_source.py"
        ).read_text(encoding="utf-8"),
        "src/pcketlm/core/runtime/fp8_pack.py": (
            ROOT / "src" / "pcketlm" / "core" / "runtime" / "fp8_pack.py"
        ).read_text(encoding="utf-8"),
        "src/pcketlm/core/runtime/tensor_catalog.py": (
            ROOT / "src" / "pcketlm" / "core" / "runtime" / "tensor_catalog.py"
        ).read_text(encoding="utf-8"),
        "src/pcketlm/core/storage/paths.py": (
            ROOT / "src" / "pcketlm" / "core" / "storage" / "paths.py"
        ).read_text(encoding="utf-8"),
        "src/pcketlm/core/runtime/source.py": _minimal_runtime_source_module(),
        "src/pcketlm/native/__init__.py": _native_stub_module(),
    }
    return files


def _minimal_runtime_source_module() -> str:
    return '''"""Minimal runtime source descriptor for Kaggle embedded validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RuntimeSourceDescriptor:
    model_dir: Path
    format_name: str
    ready: bool
    config: object | None
    config_path: Path | None
    tokenizer_path: Path | None
    index_path: Path | None
    shard_paths: list[Path] = field(default_factory=list)
    expected_shards: int = 0
    present_shards: int = 0
    missing_runtime_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plain_english_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "model_dir": str(self.model_dir),
            "format_name": self.format_name,
            "ready": self.ready,
            "config": None,
            "config_path": str(self.config_path) if self.config_path else None,
            "tokenizer_path": str(self.tokenizer_path) if self.tokenizer_path else None,
            "index_path": str(self.index_path) if self.index_path else None,
            "shard_paths": [str(path) for path in self.shard_paths],
            "expected_shards": self.expected_shards,
            "present_shards": self.present_shards,
            "missing_runtime_files": list(self.missing_runtime_files),
            "warnings": list(self.warnings),
            "plain_english_summary": self.plain_english_summary,
        }


def describe_runtime_source(model_dir: Path) -> RuntimeSourceDescriptor:
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"
    index_path = model_dir / "model.safetensors.index.json"
    missing = []
    for path in (config_path, index_path):
        if not path.exists():
            missing.append(str(path.name))
    shard_paths: list[Path] = []
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map") or {}
            shard_paths = sorted({model_dir / str(name) for name in weight_map.values()})
        except Exception as exc:
            missing.append(f"invalid index: {type(exc).__name__}: {exc}")
    present = sum(1 for path in shard_paths if path.exists())
    if shard_paths and present != len(shard_paths):
        missing.append(f"{len(shard_paths) - present} shard files are missing")
    ready = not missing and bool(shard_paths)
    return RuntimeSourceDescriptor(
        model_dir=model_dir,
        format_name="safetensors",
        ready=ready,
        config=None,
        config_path=config_path if config_path.exists() else None,
        tokenizer_path=tokenizer_path if tokenizer_path.exists() else None,
        index_path=index_path if index_path.exists() else None,
        shard_paths=shard_paths,
        expected_shards=len(shard_paths),
        present_shards=present,
        missing_runtime_files=missing,
        plain_english_summary="Embedded Kaggle source descriptor.",
    )
'''


def _native_stub_module() -> str:
    return '''"""Native-kernel stubs for Kaggle embedded validation."""

from __future__ import annotations


class DeepSeekNativeSession:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Native DeepSeek session is not available in the embedded Kaggle probe.")


def _unavailable(*args, **kwargs):
    raise RuntimeError("Native pcketlm kernel is not available in the embedded Kaggle probe.")


def _false(*args, **kwargs):
    return False


fp8_e4m3_block_dual_linear_f32 = _unavailable
fp8_e4m3_block_linear_f32 = _unavailable
fp8_e4m3_dequant_to_bf16 = _unavailable
fp8_e4m3_dequant_to_fp16 = _unavailable
fp8_e4m3_block_mlp_f32 = _unavailable
fp8_e4m3_block_mlp_many_f32 = _unavailable
ds_attention_block_forward = _unavailable
ds_mla_attention_flash_forward = _unavailable
lm_head_topk_u16 = _unavailable
native_read_tensor_bytes = _unavailable

native_fp16_loader_available = _false
native_fp16_matmul_available = _false
native_fused_ds_attention_available = _false
native_flash_mla_available = _false
native_fp8_dual_linear_available = _false
native_fp8_dequant_available = _false
native_fp8_linear_available = _false
native_fp8_mlp_available = _false
native_fp8_mlp_many_available = _false
'''


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
    model_source: str = "",
    local_model_id: str = "deepseek-v3-kaggle",
    local_decode_layers: int = 0,
    local_budget_gb: float = 12.0,
    local_prefetch_window: int = 0,
    local_max_new_tokens: int = 0,
    local_tail_chunk_rows: int = 8192,
    repo_url: str | None = None,
    branch: str | None = None,
    commit: str | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_gpu_source = (ROOT / "src" / "pcketlm" / "core" / "runtime" / "deepseek_remote_gpu.py").read_text(
        encoding="utf-8"
    )
    embedded_repo_files = _embedded_runtime_files(remote_gpu_source)
    repo_url = _repo_url() if repo_url is None else str(repo_url)
    branch = _branch_name() if branch is None else str(branch)
    commit = _commit_sha() if commit is None else str(commit)
    model_sources = [str(model_source)] if str(model_source).strip() else []
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
        "model_sources": model_sources,
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source = "\n".join(
        [
            "import json",
            "import os",
            "import subprocess",
            "import sys",
            "from pathlib import Path",
            "",
            f"REPO_URL = {repo_url!r}",
            f"SOURCE_BRANCH = {branch!r}",
            f"SOURCE_COMMIT = {commit!r}",
            f"LOCAL_MODEL_ID = {str(local_model_id)!r}",
            f"LOCAL_DECODE_LAYERS = {int(local_decode_layers)!r}",
            f"LOCAL_BUDGET_GB = {float(local_budget_gb)!r}",
            f"LOCAL_PREFETCH_WINDOW = {int(local_prefetch_window)!r}",
            f"LOCAL_MAX_NEW_TOKENS = {int(local_max_new_tokens)!r}",
            f"LOCAL_TAIL_CHUNK_ROWS = {int(local_tail_chunk_rows)!r}",
            f"MODEL_SOURCE = {str(model_source)!r}",
            f"EMBEDDED_REPO_FILES = {embedded_repo_files!r}",
            "",
            "def _run(command, *, cwd=None):",
            "    print('PCKETLM_RUN', json.dumps({'command': list(command), 'cwd': None if cwd is None else str(cwd)}))",
            "    completed = subprocess.run(list(command), cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)",
            "    if completed.stdout:",
            "        print(completed.stdout)",
            "    completed.check_returncode()",
            "    return completed",
            "",
            "def _checkout_repo():",
            "    repo_dir = Path('/kaggle/working/pcketlm')",
            "    for relative, content in EMBEDDED_REPO_FILES.items():",
            "        target = repo_dir / relative",
            "        target.parent.mkdir(parents=True, exist_ok=True)",
            "        target.write_text(content, encoding='utf-8')",
            "    sys.path.insert(0, str(repo_dir))",
            "    sys.path.insert(0, str(repo_dir / 'src'))",
            "    return repo_dir",
            "",
            "def _looks_like_deepseek_dir(path):",
            "    config = path / 'config.json'",
            "    index = path / 'model.safetensors.index.json'",
            "    if not config.exists() or not index.exists():",
            "        return False",
            "    try:",
            "        payload = json.loads(config.read_text(encoding='utf-8'))",
            "    except Exception:",
            "        return False",
            "    return str(payload.get('model_type', '')).lower() in {'deepseek_v3', 'deepseek-v3'}",
            "",
            "def _find_kaggle_deepseek_model_dir():",
            "    candidates = [",
            "        Path('/kaggle/input/deepseek-v3/transformers/deepseek-v3/2'),",
            "        Path('/kaggle/input/deepseek-v3/transformers/deepseek-v3/1'),",
            "        Path('/kaggle/input/deepseek-v3/deepseek-v3/2'),",
            "        Path('/kaggle/input/deepseek-v3'),",
            "    ]",
            "    for candidate in candidates:",
            "        if _looks_like_deepseek_dir(candidate):",
            "            return candidate",
            "    root = Path('/kaggle/input')",
            "    if root.exists():",
            "        for config in root.rglob('config.json'):",
            "            candidate = config.parent",
            "            if _looks_like_deepseek_dir(candidate):",
            "                return candidate",
            "    raise FileNotFoundError('Could not locate mounted DeepSeek V3 config/index under /kaggle/input')",
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
            "if LOCAL_DECODE_LAYERS > 0:",
            "    repo_dir = _checkout_repo()",
            "    from pcketlm.core.runtime.deepseek_gpu_residency import (",
            "        estimate_deepseek_gpu_residency,",
            "        run_local_deepseek_paged_decode_loop,",
            "        run_local_deepseek_paged_decode_probe,",
            "    )",
            "    from pcketlm.core.runtime.tensor_catalog import build_tensor_catalog",
            "    model_dir = _find_kaggle_deepseek_model_dir()",
            "    payload['kaggle_deepseek_model_source'] = {",
            "        'model_source': MODEL_SOURCE,",
            "        'model_dir': str(model_dir),",
            "        'config_exists': (model_dir / 'config.json').exists(),",
            "        'index_exists': (model_dir / 'model.safetensors.index.json').exists(),",
            "    }",
            "    catalog = build_tensor_catalog(LOCAL_MODEL_ID, model_dir)",
            "    payload['local_deepseek_tensor_catalog'] = {",
            "        'ready': bool(catalog.ready),",
            "        'model_dir': str(catalog.model_dir),",
            "        'tensor_count': int(catalog.tensor_count),",
            "        'shard_count': int(catalog.shard_count),",
            "        'fp8_weight_bytes': int(catalog.fp8_weight_bytes),",
            "        'fp8_scale_bytes': int(catalog.fp8_scale_bytes),",
            "        'blockers': list(catalog.blockers),",
            "    }",
            "    payload['local_deepseek_gpu_residency_estimate'] = estimate_deepseek_gpu_residency(",
            "        LOCAL_MODEL_ID,",
            "        start_layer=3,",
            "        layer_count=LOCAL_DECODE_LAYERS,",
            "        resident_budget_bytes=int(float(LOCAL_BUDGET_GB) * 1024**3),",
            "    ).to_dict()",
            "    if LOCAL_MAX_NEW_TOKENS > 0:",
            "        payload['local_deepseek_paged_decode_loop'] = run_local_deepseek_paged_decode_loop(",
            "            LOCAL_MODEL_ID,",
            "            [0],",
            "            start_layer=3,",
            "            layer_count=LOCAL_DECODE_LAYERS,",
            "            max_new_tokens=LOCAL_MAX_NEW_TOKENS,",
            "            resident_budget_bytes=int(float(LOCAL_BUDGET_GB) * 1024**3),",
            "            prefetch_window=LOCAL_PREFETCH_WINDOW,",
            "            tail_chunk_rows=LOCAL_TAIL_CHUNK_ROWS,",
            "            require_cuda=True,",
            "            device='cuda',",
            "            dtype=torch.float16,",
            "        ).to_dict()",
            "    else:",
            "        payload['local_deepseek_paged_decode_probe'] = run_local_deepseek_paged_decode_probe(",
            "            LOCAL_MODEL_ID,",
            "            token_id=0,",
            "            start_layer=3,",
            "            layer_count=LOCAL_DECODE_LAYERS,",
            "            resident_budget_bytes=int(float(LOCAL_BUDGET_GB) * 1024**3),",
            "            prefetch_window=LOCAL_PREFETCH_WINDOW,",
            "            include_tail=True,",
            "            tail_chunk_rows=LOCAL_TAIL_CHUNK_ROWS,",
            "            require_cuda=True,",
            "            device='cuda',",
            "            dtype=torch.float16,",
            "        ).to_dict()",
            "print('PCKETLM_DEEPSEEK_VALIDATE_OUTPUT_START')",
            "print(json.dumps(payload, indent=2, sort_keys=True))",
            "print('PCKETLM_DEEPSEEK_VALIDATE_OUTPUT_END')",
            "passed = payload['real_deepseek_remote_layer_probe']['passed'] and payload['real_deepseek_resident_layer_probe']['passed'] and payload['real_deepseek_resident_decode_probe']['passed']",
            "if payload['real_deepseek_paged_decode_probe'] is not None:",
            "    passed = passed and payload['real_deepseek_paged_decode_probe']['passed']",
            "if payload.get('local_deepseek_paged_decode_probe') is not None:",
            "    passed = passed and payload['local_deepseek_paged_decode_probe']['passed']",
            "if payload.get('local_deepseek_paged_decode_loop') is not None:",
            "    passed = passed and payload['local_deepseek_paged_decode_loop']['passed']",
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
    parser.add_argument("--model-source", default="")
    parser.add_argument("--use-deepseek-kaggle-model", action="store_true")
    parser.add_argument("--local-model-id", default="deepseek-v3-kaggle")
    parser.add_argument("--local-decode-layers", type=int, default=0)
    parser.add_argument("--local-budget-gb", type=float, default=12.0)
    parser.add_argument("--local-prefetch-window", type=int, default=0)
    parser.add_argument("--local-max-new-tokens", type=int, default=0)
    parser.add_argument("--local-tail-chunk-rows", type=int, default=8192)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    creds = kaggle_credentials()
    username = creds.username or "missing-kaggle-user"
    model_source = args.model_source
    if args.use_deepseek_kaggle_model and not model_source:
        model_source = DEFAULT_DEEPSEEK_MODEL_SOURCE
    kernel_dir = prepare_kernel(
        username=username,
        slug=args.slug,
        title=args.title,
        resident_layers=args.resident_layers,
        resident_iterations=args.resident_iterations,
        paged_layers=args.paged_layers,
        paged_budget_gb=args.paged_budget_gb,
        model_source=model_source,
        local_model_id=args.local_model_id,
        local_decode_layers=args.local_decode_layers,
        local_budget_gb=args.local_budget_gb,
        local_prefetch_window=args.local_prefetch_window,
        local_max_new_tokens=args.local_max_new_tokens,
        local_tail_chunk_rows=args.local_tail_chunk_rows,
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
