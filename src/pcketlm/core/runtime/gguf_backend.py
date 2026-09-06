"""Optional llama.cpp/GGUF backend adapter."""

from __future__ import annotations

import importlib
import importlib.util
import csv
import io
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pcketlm.core.storage.paths import artifacts_root, original_model_root, project_root, state_root


LLAMA_SERVER_PORT = 8767
LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_START_TIMEOUT_SECONDS = 150
LLAMA_SERVER_STATE_FILE_NAME = "llama-server.json"
GGUF_EXPECTED_RAM_OVERHEAD = 1.05
LLAMA_COLD_LOAD_SECONDS_PER_GB = 6.2


def _gguf_sampling_settings(model_id: str) -> dict:
    # Qwen's published thinking/coding profile; do not force greedy decoding
    # or llama.cpp's generic min_p=0.05 on this reasoning model.
    if "qwen3.6" in model_id.lower():
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 0.0, "repeat_penalty": 1.0}
    return {"temperature": 0}


def _gguf_reasoning_settings(model_id: str, max_tokens: int) -> dict:
    """Reserve final-answer capacity without disabling the thinking phase.

    This is a configurable latency/quality tradeoff, not a lossless model
    optimization. -1 opts into unlimited thinking; zero is never accepted.
    """
    if "qwen3.6" not in model_id.lower() or max_tokens < 64:
        return {}
    default = 128 if max_tokens <= 1024 else 512
    try:
        requested = int(os.environ.get("POCKETLM_THINKING_BUDGET", str(default)))
    except ValueError:
        requested = default
    if requested == -1:
        return {}
    if requested <= 0:
        requested = default
    return {
        "reasoning_budget_tokens": min(requested, max_tokens // 2),
        "reasoning_budget_message": (
            "\n\nThe thinking budget is complete. Now give only the final answer "
            "in the exact format the user requested. Do not continue planning or explaining.\n"
        ),
    }


def _gguf_memory_flags(model_id: str) -> list[str]:
    # This installed 35B model's repacked copy exhausted RAM under the host's
    # normal load. Mapping original GGUF weights passed the bounded RAM checks.
    if model_id == "qwen3.6-35b-a3b" and os.environ.get("POCKETLM_GGUF_REPACK", "0") != "1":
        return ["--no-repack"]
    return []


@dataclass(slots=True)
class GGUFModelFile:
    """One local GGUF model artifact candidate."""

    path: Path
    source: str
    size_bytes: int

    def to_dict(self) -> dict:
        return {
            "name": self.path.name,
            "path": str(self.path),
            "directory": str(self.path.parent),
            "source": self.source,
            "location": self.source,
            "kind": "split-shard" if "-of-" in self.path.name else "complete",
            "state": "ready",
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_bytes / (1024**3), 2),
        }


@dataclass(slots=True)
class GGUFBackendStatus:
    """Readiness for the optional GGUF backend."""

    model_id: str
    package_available: bool
    ready: bool
    main_package_available: bool = False
    sidecar_package_available: bool = False
    sidecar_python_path: Path | None = None
    llama_cli_available: bool = False
    llama_cli_path: Path | None = None
    llama_server_available: bool = False
    llama_server_url: str | None = None
    llama_server: dict = field(default_factory=dict)
    model_files: list[GGUFModelFile] = field(default_factory=list)
    artifact_summary: dict = field(default_factory=dict)
    load_estimate: dict = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "package_available": self.package_available,
            "ready": self.ready,
            "main_package_available": self.main_package_available,
            "sidecar_package_available": self.sidecar_package_available,
            "sidecar_python_path": None if self.sidecar_python_path is None else str(self.sidecar_python_path),
            "llama_cli_available": self.llama_cli_available,
            "llama_cli_path": None if self.llama_cli_path is None else str(self.llama_cli_path),
            "llama_server_available": self.llama_server_available,
            "llama_server_url": self.llama_server_url,
            "llama_server": dict(self.llama_server),
            "model_files": [model_file.to_dict() for model_file in self.model_files],
            "artifact_summary": dict(self.artifact_summary),
            "load_estimate": dict(self.load_estimate),
            "blockers": list(self.blockers),
            "summary": self.summary,
        }


@dataclass(slots=True)
class GGUFPromptResult:
    """Result from a GGUF prompt smoke run."""

    model_id: str
    model_path: Path | None
    ready: bool
    generated_text: str = ""
    elapsed_seconds: float = 0.0
    backend: str = "llama-cpp-gguf"
    blockers: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_path": None if self.model_path is None else str(self.model_path),
            "ready": self.ready,
            "generated_text": self.generated_text,
            "elapsed_seconds": self.elapsed_seconds,
            "backend": self.backend,
            "blockers": list(self.blockers),
            "timings": dict(self.timings),
        }


@dataclass(slots=True)
class GGUFServerStatus:
    """Lifecycle state for the local llama.cpp server managed by Pocket."""

    running: bool
    ready: bool
    pid: int | None = None
    url: str = ""
    model_id: str = ""
    model_path: Path | None = None
    working_set_bytes: int | None = None
    blockers: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "ready": self.ready,
            "pid": self.pid,
            "url": self.url,
            "model_id": self.model_id,
            "model_path": None if self.model_path is None else str(self.model_path),
            "working_set_bytes": self.working_set_bytes,
            "working_set_gb": None if self.working_set_bytes is None else round(self.working_set_bytes / (1024**3), 2),
            "blockers": list(self.blockers),
            "summary": self.summary,
            "stdout_log": str(state_root() / "llama-server.out.log"),
            "stderr_log": str(state_root() / "llama-server.err.log"),
        }


@lru_cache(maxsize=1)
def _llama_cpp_available() -> bool:
    return importlib.util.find_spec("llama_cpp") is not None


def gguf_sidecar_python_path() -> Path:
    return state_root() / "backend-envs" / "gguf-py312" / "Scripts" / "python.exe"


def llama_cpp_runtime_root() -> Path:
    return state_root() / "backend-runtimes" / "llama.cpp"


def llama_cli_path() -> Path:
    return llama_cpp_runtime_root() / "llama-cli.exe"


def llama_server_path() -> Path:
    return llama_cpp_runtime_root() / "llama-server.exe"


def llama_server_url() -> str:
    return f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"


def _llama_server_state_path() -> Path:
    return state_root() / LLAMA_SERVER_STATE_FILE_NAME


def _read_llama_server_state() -> dict:
    path = _llama_server_state_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_llama_server_state(payload: dict) -> None:
    path = _llama_server_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _clear_llama_server_state() -> None:
    try:
        _llama_server_state_path().unlink()
    except OSError:
        pass


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0 and str(pid) in result.stdout and not result.stdout.strip().startswith("INFO:")


def _pid_listening_on_llama_server_port() -> int | None:
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    suffix = f":{LLAMA_SERVER_PORT}"
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local_address, state, pid_text = parts[1], parts[3].upper(), parts[-1]
        if state != "LISTENING" or not local_address.endswith(suffix):
            continue
        try:
            return int(pid_text)
        except ValueError:
            return None
    return None


def _llama_server_pid() -> int | None:
    state = _read_llama_server_state()
    try:
        state_pid = int(state.get("pid") or 0)
    except (TypeError, ValueError):
        state_pid = 0
    port_pid = _pid_listening_on_llama_server_port()
    if port_pid:
        return port_pid
    if state_pid and _process_exists(state_pid):
        return state_pid
    return None


def _process_working_set_bytes(pid: int | None) -> int | None:
    if not pid:
        return None
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip() or result.stdout.strip().startswith("INFO:"):
        return None
    try:
        row = next(csv.reader(io.StringIO(result.stdout)))
    except Exception:
        return None
    if len(row) < 5:
        return None
    digits = "".join(char for char in row[4] if char.isdigit())
    if not digits:
        return None
    return int(digits) * 1024


def build_gguf_server_status(model_id: str = "qwen2.5-14b-instruct") -> GGUFServerStatus:
    """Return the current llama.cpp server lifecycle state."""
    pid = _llama_server_pid()
    ready = _llama_server_is_ready()
    running = pid is not None
    state = _read_llama_server_state()
    model_path = None
    if state.get("model_path"):
        model_path = Path(str(state["model_path"]))
    elif running:
        selected = _select_gguf_model_file(model_id)
        model_path = None if selected is None else selected.path
    state_model_id = str(state.get("model_id") or model_id)
    blockers: list[str] = []
    if running and not ready:
        blockers.append("llama.cpp server process is running, but the health check is not ready yet.")
    summary = (
        "GGUF server is loaded and ready."
        if ready
        else "GGUF server is starting or not loaded yet."
        if running
        else "GGUF server is not loaded."
    )
    return GGUFServerStatus(
        running=running,
        ready=ready,
        pid=pid,
        url=llama_server_url(),
        model_id=state_model_id,
        model_path=model_path,
        working_set_bytes=_process_working_set_bytes(pid),
        blockers=blockers,
        summary=summary,
    )


@lru_cache(maxsize=8)
def _sidecar_llama_cpp_available(python_path: Path | None = None) -> bool:
    path = gguf_sidecar_python_path() if python_path is None else Path(python_path)
    if not path.exists():
        return False
    try:
        result = subprocess.run(
            [
                str(path),
                "-c",
                "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('llama_cpp') else 1)",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def find_gguf_model_files(model_id: str) -> list[GGUFModelFile]:
    """Find local GGUF artifacts for one model."""
    roots = [
        ("original", original_model_root(model_id)),
        ("artifact", artifacts_root(model_id)),
    ]
    found: list[GGUFModelFile] = []
    for source, root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.gguf")):
            try:
                found.append(GGUFModelFile(path=path, source=source, size_bytes=path.stat().st_size))
            except OSError:
                continue
    return found


def summarize_gguf_artifacts(model_id: str) -> dict:
    """Summarize GGUF disk use for product status surfaces."""
    files = find_gguf_model_files(model_id)
    complete_files = [model_file for model_file in files if "-of-" not in model_file.path.name]
    split_files = [model_file for model_file in files if "-of-" in model_file.path.name]
    total_bytes = sum(model_file.size_bytes for model_file in files)
    complete_bytes = sum(model_file.size_bytes for model_file in complete_files)
    split_bytes = sum(model_file.size_bytes for model_file in split_files)
    largest_complete = max(complete_files, key=lambda model_file: model_file.size_bytes, default=None)
    return {
        "model_id": model_id,
        "file_count": len(files),
        "complete_file_count": len(complete_files),
        "split_shard_count": len(split_files),
        "total_size_bytes": total_bytes,
        "total_size_gb": round(total_bytes / (1024**3), 2),
        "complete_size_bytes": complete_bytes,
        "complete_size_gb": round(complete_bytes / (1024**3), 2),
        "split_size_bytes": split_bytes,
        "split_size_gb": round(split_bytes / (1024**3), 2),
        "largest_complete_path": None if largest_complete is None else str(largest_complete.path),
        "summary": (
            f"{len(complete_files)} complete GGUF file(s), {len(split_files)} split shard(s), {round(total_bytes / (1024**3), 2)} GB total."
            if files
            else "No GGUF files found."
        ),
    }


def build_gguf_backend_status(model_id: str) -> GGUFBackendStatus:
    """Return the local GGUF backend readiness status."""
    main_package_available = _llama_cpp_available()
    sidecar_python = gguf_sidecar_python_path()
    cli_path = llama_cli_path()
    llama_cli_available = cli_path.exists()
    server_status = build_gguf_server_status(model_id)
    server_available = server_status.ready
    server_binary_available = llama_server_path().exists()
    sidecar_package_available = False
    if not (main_package_available or llama_cli_available or server_binary_available):
        sidecar_package_available = _sidecar_llama_cpp_available(sidecar_python)
    package_available = main_package_available or sidecar_package_available or llama_cli_available or server_binary_available
    model_files = find_gguf_model_files(model_id)
    artifact_summary = summarize_gguf_artifacts(model_id)
    blockers: list[str] = []
    if not package_available:
        blockers.append("No llama.cpp runtime is available yet.")
    if not model_files:
        blockers.append("No GGUF model file is present for this model.")
    selected_model_file = _select_gguf_model_file(model_id)
    load_estimate = estimate_gguf_load_cost(model_id, selected=selected_model_file, server_status=server_status)
    ready = package_available and bool(model_files)
    summary = (
        "GGUF backend is ready for a local smoke run."
        if ready
        else "GGUF backend needs the runtime package and a GGUF model file before it can run."
    )
    return GGUFBackendStatus(
        model_id=model_id,
        package_available=package_available,
        ready=ready,
        main_package_available=main_package_available,
        sidecar_package_available=sidecar_package_available,
        sidecar_python_path=sidecar_python if sidecar_python.exists() else None,
        llama_cli_available=llama_cli_available,
        llama_cli_path=cli_path if llama_cli_available else None,
        llama_server_available=server_available,
        llama_server_url=llama_server_url() if server_binary_available else None,
        llama_server=server_status.to_dict(),
        model_files=model_files,
        artifact_summary=artifact_summary,
        load_estimate=load_estimate,
        blockers=blockers,
        summary=summary,
    )


def _select_gguf_model_file(model_id: str, model_path: str | Path | None = None) -> GGUFModelFile | None:
    if model_path is not None:
        path = Path(model_path)
        try:
            return GGUFModelFile(path=path, source="explicit", size_bytes=path.stat().st_size)
        except OSError:
            return None
    files = find_gguf_model_files(model_id)
    if not files:
        return None
    complete_files = [model_file for model_file in files if "-of-" not in model_file.path.name]
    candidates = complete_files or files
    return max(candidates, key=lambda model_file: model_file.size_bytes)


def estimate_gguf_load_cost(
    model_id: str,
    model_path: str | Path | None = None,
    *,
    selected: GGUFModelFile | None = None,
    server_status: GGUFServerStatus | None = None,
) -> dict:
    """Estimate the visible cost of loading a GGUF artifact into llama-server."""
    selected_file = selected if selected is not None else _select_gguf_model_file(model_id, model_path)
    if selected_file is None:
        return {
            "model_id": model_id,
            "model_path": None,
            "model_file": None,
            "model_size_gb": None,
            "expected_ram_mb": None,
            "estimated_cold_load_seconds": None,
            "state": "missing",
            "load_action": "unavailable",
            "blockers": ["No GGUF model file is available to load."],
            "basis": {
                "expected_ram_overhead": GGUF_EXPECTED_RAM_OVERHEAD,
                "cold_load_seconds_per_gb": LLAMA_COLD_LOAD_SECONDS_PER_GB,
            },
        }

    server = server_status if server_status is not None else build_gguf_server_status(model_id)
    server_path = None if server.model_path is None else Path(server.model_path)
    same_loaded_file = bool(server.running and server_path is not None and server_path == selected_file.path)
    compatible_loaded_file = bool(server.running and server_path is None and server.model_id == model_id)
    state = "ready"
    blockers: list[str] = []
    if same_loaded_file or compatible_loaded_file:
        state = "loaded" if server.ready else "loading"
    elif server.running:
        blockers.append("A llama.cpp server is already running with a different GGUF file.")

    model_size_gb = selected_file.size_bytes / (1024**3)
    expected_ram_mb = int(math.ceil(selected_file.size_bytes * GGUF_EXPECTED_RAM_OVERHEAD / (1024**2)))
    return {
        "model_id": model_id,
        "model_path": str(selected_file.path),
        "model_file": selected_file.path.name,
        "model_size_gb": round(model_size_gb, 2),
        "expected_ram_mb": expected_ram_mb,
        "estimated_cold_load_seconds": round(model_size_gb * LLAMA_COLD_LOAD_SECONDS_PER_GB, 1),
        "state": state,
        "load_action": "unload" if state in {"loaded", "loading"} else "load",
        "blockers": blockers,
        "basis": {
            "expected_ram_overhead": GGUF_EXPECTED_RAM_OVERHEAD,
            "cold_load_seconds_per_gb": LLAMA_COLD_LOAD_SECONDS_PER_GB,
        },
    }


def run_gguf_prompt(
    model_id: str,
    prompt: str,
    *,
    model_path: str | Path | None = None,
    max_tokens: int = 32,
    n_ctx: int = 2048,
    n_threads: int | None = None,
    stop_strings: list[str] | None = None,
    llama_factory: Any | None = None,
    prefer_server: bool = True,
    chat_messages: list[dict[str, str]] | None = None,
) -> GGUFPromptResult:
    """Run a prompt through a local GGUF model when the optional backend is available."""
    selected = _select_gguf_model_file(model_id, model_path)
    blockers: list[str] = []
    if selected is None:
        blockers.append("No GGUF model file is available for this run.")
    if prefer_server and selected is not None and llama_server_path().exists():
        server_result = _run_gguf_prompt_server(
            model_id,
            prompt,
            selected,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            n_threads=n_threads,
            stop_strings=stop_strings,
            chat_messages=chat_messages,
        )
        if server_result.ready or server_result.blockers:
            return server_result
    if llama_factory is None:
        if not _llama_cpp_available():
            if llama_cli_path().exists():
                return _run_gguf_prompt_cli(
                    model_id,
                    prompt,
                    selected,
                    max_tokens=max_tokens,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    stop_strings=stop_strings,
                )
            if _sidecar_llama_cpp_available():
                return _run_gguf_prompt_sidecar(
                    model_id,
                    prompt,
                    selected,
                    max_tokens=max_tokens,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    stop_strings=stop_strings,
                )
            blockers.append("No llama.cpp runtime is available in Python or standalone CLI form.")
        else:
            llama_factory = getattr(importlib.import_module("llama_cpp"), "Llama")
    if blockers:
        return GGUFPromptResult(model_id=model_id, model_path=None if selected is None else selected.path, ready=False, blockers=blockers)

    started = time.perf_counter()
    try:
        kwargs = {
            "model_path": str(selected.path),
            "n_ctx": int(n_ctx),
            "verbose": False,
        }
        if n_threads is not None:
            kwargs["n_threads"] = int(n_threads)
        llm = llama_factory(**kwargs)
        call_kwargs = {"max_tokens": int(max_tokens), "echo": False}
        if stop_strings:
            call_kwargs["stop"] = list(stop_strings)
        output = llm(str(prompt), **call_kwargs)
        choices = output.get("choices", []) if isinstance(output, dict) else []
        text = _clean_stop_text(str(choices[0].get("text", "")) if choices else "", stop_strings)
    except Exception as exc:  # pragma: no cover - optional native backend guard
        return GGUFPromptResult(
            model_id=model_id,
            model_path=selected.path,
            ready=False,
            elapsed_seconds=round(time.perf_counter() - started, 2),
            blockers=[f"GGUF prompt run failed: {exc}"],
        )
    elapsed = round(time.perf_counter() - started, 2)
    return GGUFPromptResult(
        model_id=model_id,
        model_path=selected.path,
        ready=True,
        generated_text=text,
        elapsed_seconds=elapsed,
        timings={"total": elapsed},
    )


def _llama_server_is_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{llama_server_url()}/health", timeout=2) as response:
            return 200 <= int(response.status) < 500
    except Exception:
        return False


@contextmanager
def _server_start_lock():
    """Serialize independent launcher processes so they cannot load duplicate models."""
    lock_path = state_root() / "llama-server.start.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + LLAMA_SERVER_START_TIMEOUT_SECONDS + 10
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Another Pocket model launch is still in progress.")
                time.sleep(0.1)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _start_llama_server(selected, *, n_ctx, n_threads, model_id="") -> bool:
    try:
        with _server_start_lock():
            return _start_llama_server_unlocked(
                selected, n_ctx=n_ctx, n_threads=n_threads, model_id=model_id
            )
    except OSError:
        return False


def _start_llama_server_unlocked(
    selected: GGUFModelFile,
    *,
    n_ctx: int,
    n_threads: int | None,
    model_id: str = "",
) -> bool:
    if _llama_server_is_ready():
        return True
    # The caller can time out before its child finishes loading. A 503 is not
    # permission to load a second copy of the model into the same machine.
    if _llama_server_pid() is not None:
        return False
    exe = llama_server_path()
    if not exe.exists():
        return False
    stdout_path = state_root() / "llama-server.out.log"
    stderr_path = state_root() / "llama-server.err.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(exe),
        "--model",
        str(selected.path),
        "--host",
        LLAMA_SERVER_HOST,
        "--port",
        str(LLAMA_SERVER_PORT),
        "--ctx-size",
        str(int(n_ctx)),
        "--parallel",
        "1",
        "--no-warmup",
    ]
    if n_threads is not None:
        args.extend(["--threads", str(int(n_threads))])
    args.extend(_gguf_memory_flags(model_id))
    try:
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        process = subprocess.Popen(
            args,
            cwd=str(exe.parent),
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
        )
        _write_llama_server_state(
            {
                "pid": process.pid,
                "model_id": model_id,
                "model_path": str(selected.path),
                "n_ctx": int(n_ctx),
                "n_threads": n_threads,
                "started_at": time.time(),
                "url": llama_server_url(),
            }
        )
    except Exception:
        return False
    deadline = time.monotonic() + LLAMA_SERVER_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _llama_server_is_ready():
            return True
        if process.poll() is not None:
            return False
        time.sleep(1.0)
    if process.poll() is None:
        process.terminate()
    return False


def start_gguf_server(
    model_id: str = "qwen2.5-14b-instruct",
    *,
    model_path: str | Path | None = None,
    n_ctx: int = 2048,
    n_threads: int | None = None,
) -> GGUFServerStatus:
    """Start the persistent llama.cpp server for one GGUF model."""
    selected = _select_gguf_model_file(model_id, model_path)
    if selected is None:
        return GGUFServerStatus(
            running=False,
            ready=False,
            url=llama_server_url(),
            model_id=model_id,
            blockers=["No GGUF model file is available to load."],
            summary="GGUF server could not start because no model file is available.",
        )
    if not llama_server_path().exists():
        return GGUFServerStatus(
            running=False,
            ready=False,
            url=llama_server_url(),
            model_id=model_id,
            model_path=selected.path,
            blockers=["llama-server.exe is not installed."],
            summary="GGUF server could not start because the llama.cpp runtime is missing.",
        )
    _start_llama_server(selected, n_ctx=n_ctx, n_threads=n_threads, model_id=model_id)
    return build_gguf_server_status(model_id)


def stop_gguf_server(model_id: str = "qwen2.5-14b-instruct") -> GGUFServerStatus:
    """Stop the persistent llama.cpp server and free its RAM."""
    pid = _llama_server_pid()
    if pid is None:
        _clear_llama_server_state()
        return build_gguf_server_status(model_id)
    try:
        os.kill(pid, 15)
    except OSError:
        pass
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if _pid_listening_on_llama_server_port() is None:
            _clear_llama_server_state()
            return build_gguf_server_status(model_id)
        time.sleep(0.25)
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True, timeout=10, check=False)
    except Exception:
        pass
    _clear_llama_server_state()
    return build_gguf_server_status(model_id)


def _run_gguf_prompt_server(
    model_id: str,
    prompt: str,
    selected: GGUFModelFile,
    *,
    max_tokens: int,
    n_ctx: int,
    n_threads: int | None,
    stop_strings: list[str] | None,
    chat_messages: list[dict[str, str]] | None = None,
) -> GGUFPromptResult:
    started = time.perf_counter()
    if not _start_llama_server(selected, n_ctx=n_ctx, n_threads=n_threads, model_id=model_id):
        return GGUFPromptResult(
            model_id=model_id,
            model_path=selected.path,
            ready=False,
            elapsed_seconds=round(time.perf_counter() - started, 2),
            blockers=["llama.cpp server is not available and could not be started."],
        )
    if chat_messages:
        payload = {
            "messages": chat_messages,
            "max_tokens": int(max_tokens),
            "temperature": 0,
        }
        endpoint = "/v1/chat/completions"
    else:
        payload = {
            "prompt": prompt,
            "n_predict": int(max_tokens),
            "cache_prompt": True,
            "temperature": 0,
        }
        endpoint = "/completion"
    payload.update(_gguf_sampling_settings(model_id))
    if chat_messages and "qwen3.6" in model_id.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        payload.update(_gguf_reasoning_settings(model_id, int(max_tokens)))
    reasoning_budget = payload.get("reasoning_budget_tokens")
    if stop_strings:
        payload["stop"] = list(stop_strings)
    request_payload = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{llama_server_url()}{endpoint}",
        data=request_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return GGUFPromptResult(
            model_id=model_id,
            model_path=selected.path,
            ready=False,
            elapsed_seconds=round(time.perf_counter() - started, 2),
            blockers=[f"llama.cpp server request failed: {exc}"],
        )
    elapsed = round(time.perf_counter() - started, 2)
    generated_text = str(payload.get("content") or "")
    finish_reason = payload.get("stop_type")
    if chat_messages:
        choices = payload.get("choices") or []
        message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
        generated_text = str(message.get("content") or generated_text)
        finish_reason = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
    generated_text = _clean_stop_text(generated_text, stop_strings)
    blockers = []
    if not generated_text:
        blockers.append("Model returned no final answer; reasoning-only output is not completion.")
    if finish_reason in {"length", "limit"} or payload.get("stopped_limit"):
        blockers.append("Generation reached the token limit; the answer may be incomplete.")
    if payload.get("error"):
        blockers.append("Backend returned an error instead of a completed answer.")
    return GGUFPromptResult(
        model_id=model_id,
        model_path=selected.path,
        ready=not blockers,
        generated_text=generated_text,
        blockers=blockers,
        elapsed_seconds=elapsed,
        backend="llama-cpp-gguf-server",
        timings={"total": elapsed, "server": payload.get("timings", {}), "finish_reason": finish_reason,
                 "reasoning_budget_tokens": reasoning_budget, "usage": payload.get("usage", {})},
    )


def _run_gguf_prompt_sidecar(
    model_id: str,
    prompt: str,
    selected: GGUFModelFile | None,
    *,
    max_tokens: int,
    n_ctx: int,
    n_threads: int | None,
    stop_strings: list[str] | None,
) -> GGUFPromptResult:
    if selected is None:
        return GGUFPromptResult(model_id=model_id, model_path=None, ready=False, blockers=["No GGUF model file is available for this run."])
    sidecar_python = gguf_sidecar_python_path()
    script = project_root() / "src" / "pcketlm" / "core" / "runtime" / "gguf_sidecar_runner.py"
    payload = {
        "model_path": str(selected.path),
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "n_ctx": int(n_ctx),
        "n_threads": n_threads,
        "stop_strings": list(stop_strings or []),
    }
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [str(sidecar_python), str(script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - subprocess guard
        return GGUFPromptResult(model_id=model_id, model_path=selected.path, ready=False, blockers=[f"GGUF sidecar failed: {exc}"])
    elapsed = round(time.perf_counter() - started, 2)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "unknown sidecar error").strip()
        return GGUFPromptResult(
            model_id=model_id,
            model_path=selected.path,
            ready=False,
            elapsed_seconds=elapsed,
            blockers=[f"GGUF sidecar failed: {message}"],
        )
    try:
        payload = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return GGUFPromptResult(
            model_id=model_id,
            model_path=selected.path,
            ready=False,
            elapsed_seconds=elapsed,
            blockers=["GGUF sidecar returned invalid JSON."],
        )
    return GGUFPromptResult(
        model_id=model_id,
        model_path=selected.path,
        ready=bool(payload.get("ready")),
        generated_text=_clean_stop_text(str(payload.get("generated_text") or ""), stop_strings),
        elapsed_seconds=float(payload.get("elapsed_seconds") or elapsed),
        timings={"total": float(payload.get("elapsed_seconds") or elapsed)},
        blockers=list(payload.get("blockers", [])),
    )


def _run_gguf_prompt_cli(
    model_id: str,
    prompt: str,
    selected: GGUFModelFile | None,
    *,
    max_tokens: int,
    n_ctx: int,
    n_threads: int | None,
    stop_strings: list[str] | None,
) -> GGUFPromptResult:
    if selected is None:
        return GGUFPromptResult(model_id=model_id, model_path=None, ready=False, blockers=["No GGUF model file is available for this run."])
    exe = llama_cli_path()
    args = [
        str(exe),
        "--model",
        str(selected.path),
        "--prompt",
        str(prompt),
        "--predict",
        str(int(max_tokens)),
        "--ctx-size",
        str(int(n_ctx)),
        "--no-display-prompt",
        "--no-warmup",
        "--single-turn",
        "--simple-io",
    ]
    if n_threads is not None:
        args.extend(["--threads", str(int(n_threads))])
    for stop_string in stop_strings or []:
        args.extend(["--reverse-prompt", str(stop_string)])
    started = time.perf_counter()
    try:
        result = subprocess.run(
            args,
            cwd=str(exe.parent),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - subprocess guard
        return GGUFPromptResult(model_id=model_id, model_path=selected.path, ready=False, blockers=[f"llama.cpp CLI failed: {exc}"])
    elapsed = round(time.perf_counter() - started, 2)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "unknown llama.cpp CLI error").strip()
        return GGUFPromptResult(
            model_id=model_id,
            model_path=selected.path,
            ready=False,
            elapsed_seconds=elapsed,
            blockers=[f"llama.cpp CLI failed: {message}"],
        )
    return GGUFPromptResult(
        model_id=model_id,
        model_path=selected.path,
        ready=True,
        generated_text=_clean_stop_text(_clean_llama_cli_output(result.stdout, prompt), stop_strings),
        elapsed_seconds=elapsed,
        timings={"total": elapsed},
    )


def _clean_llama_cli_output(output: str, prompt: str) -> str:
    """Extract generated text from llama-cli output that may include banners and timings."""
    lines = [line.strip() for line in str(output or "").splitlines()]
    cleaned: list[str] = []
    prompt_seen = False
    for line in lines:
        if not line:
            continue
        if line.startswith(">"):
            if prompt in line:
                prompt_seen = True
            continue
        if not prompt_seen:
            continue
        if line.startswith("[ Prompt:") or line.startswith("Exiting"):
            break
        if line.startswith("available commands:") or line.startswith("/"):
            continue
        cleaned.append(line)
    if cleaned:
        return "\n".join(cleaned).strip()
    return str(output or "").strip()


def _clean_stop_text(text: str, stop_strings: list[str] | None) -> str:
    cleaned = str(text or "")
    for stop_string in stop_strings or []:
        if stop_string and stop_string in cleaned:
            cleaned = cleaned.split(stop_string, 1)[0]
    return cleaned.strip()
