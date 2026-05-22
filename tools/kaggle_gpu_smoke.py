"""Submit and poll the PLM-14 GPU smoke on Kaggle.

This keeps the recurring GPU test loop terminal-driven instead of requiring the
operator to open Colab and click through a notebook every time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SMOKE_MODULE = ROOT / "src" / "pcketlm" / "core" / "runtime" / "gpu_smoke.py"
DEFAULT_BUILD_DIR = ROOT / "build" / "kaggle_gpu_smoke"
DEFAULT_STATE_DIR = ROOT / "state" / "kaggle_gpu_smoke"
CUDA_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"
DEFAULT_KERNEL_TYPE = "notebook"


@dataclass(frozen=True)
class KaggleCredentials:
    username: str | None
    available: bool
    source: str
    message: str


def kaggle_credentials() -> KaggleCredentials:
    api_token = os.environ.get("KAGGLE_API_TOKEN")
    if api_token:
        return KaggleCredentials(_kaggle_config_username() or "api-token", True, "environment", "KAGGLE_API_TOKEN is set.")

    access_token = Path.home() / ".kaggle" / "access_token"
    if access_token.exists() and access_token.read_text(encoding="utf-8").strip():
        return KaggleCredentials(
            _kaggle_config_username() or "api-token",
            True,
            str(access_token),
            "Kaggle access_token is present.",
        )

    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return KaggleCredentials(env_user, True, "environment", "KAGGLE_USERNAME/KAGGLE_KEY are set.")

    config = Path.home() / ".kaggle" / "kaggle.json"
    if config.exists():
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return KaggleCredentials(None, False, str(config), f"Invalid kaggle.json: {exc}")
        username = data.get("username")
        key = data.get("key")
        if username and key:
            return KaggleCredentials(str(username), True, str(config), "kaggle.json is present.")
        return KaggleCredentials(None, False, str(config), "kaggle.json is missing username or key.")

    return KaggleCredentials(
        None,
        False,
        str(config),
        "Kaggle credentials are missing. Create an API token on Kaggle and save it as ~/.kaggle/kaggle.json.",
    )


def _run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _git_value(args: list[str], fallback: str) -> str:
    try:
        return _run(["git", *args], cwd=ROOT).stdout.strip() or fallback
    except (subprocess.CalledProcessError, FileNotFoundError):
        return fallback


def _kaggle_executable_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    scripts = (
        Path.home()
        / "AppData"
        / "Roaming"
        / "Python"
        / f"Python{sys.version_info.major}{sys.version_info.minor}"
        / "Scripts"
        / "kaggle.exe"
    )
    if scripts.exists():
        return [str(scripts)]
    return [sys.executable, "-m", "kaggle"]


def _kaggle_config_username() -> str | None:
    try:
        completed = _run([*_kaggle_executable_command(), "config", "view"], check=False)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("- username:"):
            value = stripped.split(":", 1)[1].strip()
            if value and value.lower() != "none":
                return value
    return None


def _repo_url() -> str:
    return _git_value(["config", "--get", "remote.origin.url"], "https://github.com/iamlicht1f1-maker/pcketlm.git")


def _branch_name() -> str:
    return _git_value(["branch", "--show-current"], "plm-14-gpu-testing-pipeline")


def _commit_sha() -> str:
    return _git_value(["rev-parse", "--short", "HEAD"], "unknown")


def _kernel_script(*, repo_url: str, branch: str, commit: str) -> str:
    smoke_lines: list[str] = []
    for line in SMOKE_MODULE.read_text(encoding="utf-8").splitlines():
        if line.strip() == "from __future__ import annotations":
            continue
        if line.strip() == 'if __name__ == "__main__":':
            break
        smoke_lines.append(line)
    smoke_source = "\n".join(smoke_lines)
    torch_prelude = "\n".join(
        [
            "import json as _pcketlm_json",
            "import subprocess as _pcketlm_subprocess",
            "import sys as _pcketlm_sys",
            "",
            "def _pcketlm_cuda_torch_probe():",
            "    code = \"import json, torch; print(json.dumps({'version': torch.__version__, 'cuda_available': torch.cuda.is_available()}))\"",
            "    completed = _pcketlm_subprocess.run([_pcketlm_sys.executable, '-c', code], text=True, stdout=_pcketlm_subprocess.PIPE, stderr=_pcketlm_subprocess.STDOUT)",
            "    print('PCKETLM_TORCH_PROBE', completed.stdout.strip())",
            "    try:",
            "        return _pcketlm_json.loads(completed.stdout.strip().splitlines()[-1])",
            "    except Exception:",
            "        return {'version': 'unknown', 'cuda_available': False}",
            "",
            "_pcketlm_probe = _pcketlm_cuda_torch_probe()",
            "if not _pcketlm_probe.get('cuda_available', False):",
            "    print('PCKETLM_TORCH_REPAIR_START')",
            "    try:",
            f"        _pcketlm_subprocess.check_call([_pcketlm_sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall', 'torch', '--index-url', {CUDA_TORCH_INDEX_URL!r}])",
            "    except Exception as exc:",
            "        print('PCKETLM_TORCH_REPAIR_FAILED', repr(exc))",
            "    else:",
            "        _pcketlm_probe = _pcketlm_cuda_torch_probe()",
            "        print('PCKETLM_TORCH_REPAIR_DONE')",
            "    _pcketlm_probe = _pcketlm_cuda_torch_probe()",
            "",
        ]
    )
    return "\n".join(
        [
            "# Auto-generated by tools/kaggle_gpu_smoke.py.",
            f"SOURCE_REPO_URL = {repo_url!r}",
            f"SOURCE_BRANCH = {branch!r}",
            f"SOURCE_COMMIT = {commit!r}",
            torch_prelude,
            smoke_source,
            "",
            "if __name__ == \"__main__\":",
            "    print(\"PCKETLM_KAGGLE_GPU_SMOKE_START\")",
            "    print(f\"source_repo={SOURCE_REPO_URL}\")",
            "    print(f\"source_branch={SOURCE_BRANCH}\")",
            "    print(f\"source_commit={SOURCE_COMMIT}\")",
            "    result = run_gpu_smoke(require_cuda=True)",
            "    print(\"PCKETLM_GPU_SMOKE_JSON_START\")",
            "    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))",
            "    print(\"PCKETLM_GPU_SMOKE_JSON_END\")",
            "    raise SystemExit(0 if result.passed and result.cuda_available else 1)",
            "",
        ]
    )


def _notebook_payload(source: str) -> dict[str, Any]:
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [f"{line}\n" for line in source.splitlines()],
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def prepare_kernel(
    *,
    username: str,
    slug: str,
    title: str,
    out_dir: Path = DEFAULT_BUILD_DIR,
    repo_url: str | None = None,
    branch: str | None = None,
    commit: str | None = None,
    kernel_type: str = DEFAULT_KERNEL_TYPE,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if kernel_type not in {"notebook", "script"}:
        raise ValueError(f"Unsupported Kaggle kernel type: {kernel_type}")
    code_name = "gpu_smoke_kaggle.ipynb" if kernel_type == "notebook" else "gpu_smoke_kaggle.py"
    repo_url = repo_url or _repo_url()
    branch = branch or _branch_name()
    commit = commit or _commit_sha()

    metadata: dict[str, Any] = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": code_name,
        "language": "python",
        "kernel_type": kernel_type,
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
    source = _kernel_script(repo_url=repo_url, branch=branch, commit=commit)
    if kernel_type == "notebook":
        (out_dir / code_name).write_text(json.dumps(_notebook_payload(source), indent=2) + "\n", encoding="utf-8")
        stale_script = out_dir / "gpu_smoke_kaggle.py"
        if stale_script.exists():
            stale_script.unlink()
    else:
        (out_dir / code_name).write_text(source, encoding="utf-8")
        stale_notebook = out_dir / "gpu_smoke_kaggle.ipynb"
        if stale_notebook.exists():
            stale_notebook.unlink()
    return out_dir


def ensure_kaggle_package() -> None:
    command = _kaggle_executable_command()
    if len(command) == 1 and Path(command[0]).exists():
        return
    _run([sys.executable, "-m", "pip", "install", "--user", "kaggle"])


def _kaggle_command(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    ensure_kaggle_package()
    return _run([*_kaggle_executable_command(), *args], check=check)


def submit_kernel(kernel_dir: Path, *, accelerator: str) -> str:
    return _kaggle_command(["kernels", "push", "-p", str(kernel_dir), "--accelerator", accelerator]).stdout


def poll_kernel(kernel_id: str, *, interval_seconds: int, timeout_seconds: int) -> str:
    deadline = time.time() + timeout_seconds
    last_output = ""
    while time.time() < deadline:
        completed = _kaggle_command(["kernels", "status", kernel_id], check=False)
        last_output = completed.stdout
        lowered = last_output.lower()
        if "complete" in lowered or "error" in lowered or "failed" in lowered or "cancel" in lowered:
            return last_output
        time.sleep(interval_seconds)
    return last_output + f"\nTimed out after {timeout_seconds} seconds."


def download_output(kernel_id: str, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    return _kaggle_command(["kernels", "output", kernel_id, "-p", str(out_dir)], check=False).stdout


def write_state(payload: dict[str, Any], *, state_dir: Path = DEFAULT_STATE_DIR) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "latest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the PLM-14 GPU smoke on Kaggle.")
    parser.add_argument("--slug", default="pocketlm-gpu-smoke", help="Kaggle kernel slug.")
    parser.add_argument("--title", default="PocketLM GPU Smoke", help="Kaggle kernel title.")
    parser.add_argument("--prepare-only", action="store_true", help="Only write the Kaggle kernel folder.")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4", help="Kaggle machine shape, for example NvidiaTeslaT4.")
    parser.add_argument(
        "--kernel-type",
        choices=("notebook", "script"),
        default=DEFAULT_KERNEL_TYPE,
        help="Kaggle execution format. Notebook is the default because browser notebooks keep Kaggle's CUDA torch image.",
    )
    parser.add_argument("--poll-interval", type=int, default=30, help="Poll interval in seconds.")
    parser.add_argument("--timeout", type=int, default=900, help="Poll timeout in seconds.")
    args = parser.parse_args(argv)

    creds = kaggle_credentials()
    username = creds.username or "missing-kaggle-user"
    kernel_dir = prepare_kernel(username=username, slug=args.slug, title=args.title, kernel_type=args.kernel_type)
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

    if args.prepare_only:
        payload["status"] = "prepared"
        print(json.dumps(payload, indent=2, sort_keys=True))
        write_state(payload)
        return 0

    if not creds.available:
        payload["status"] = "blocked_missing_kaggle_credentials"
        print(json.dumps(payload, indent=2, sort_keys=True))
        write_state(payload)
        return 2

    submit_output = submit_kernel(kernel_dir, accelerator=args.accelerator)
    status_output = poll_kernel(kernel_id, interval_seconds=args.poll_interval, timeout_seconds=args.timeout)
    output_dir = DEFAULT_STATE_DIR / "output"
    download_log = download_output(kernel_id, output_dir)
    payload.update(
        {
            "status": "submitted",
            "submit_output": submit_output,
            "status_output": status_output,
            "download_log": download_log,
            "output_dir": str(output_dir),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    write_state(payload)
    return 0 if "complete" in status_output.lower() else 1


if __name__ == "__main__":
    raise SystemExit(main())
