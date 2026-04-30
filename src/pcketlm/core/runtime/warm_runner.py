"""Conservative warm runner for repeated short local Agent calls."""

from __future__ import annotations

import ctypes
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from ctypes import wintypes
from typing import Any

from pcketlm.core.runtime.layer_bridge import KVDecodeState, run_prompt_decode_loop
from pcketlm.core.runtime.load_attempt import _memory_snapshot
from pcketlm.core.runtime.tensor_residency import clear_tensor_residency_cache, tensor_residency_stats
from pcketlm.core.storage.paths import state_root

WARM_RUNNER_MIN_FREE_MEMORY_MB = 4 * 1024
WARM_RUNNER_AGENT_MAX_NEW_TOKENS = 2
WARM_RUNNER_STATUS_DIR_NAME = "warm-runner"


@dataclass(slots=True)
class WarmRunnerRequestResult:
    """One warm runner prompt response plus runner telemetry."""

    model_id: str
    mode: str
    prompt: str
    ready: bool
    generated_text: str
    elapsed_seconds: float
    max_new_tokens: int
    steps_completed: int
    generated_token_ids: list[int] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    prefix_reuse: dict = field(default_factory=dict)
    performance_summary: dict = field(default_factory=dict)
    memory_before: dict = field(default_factory=dict)
    memory_after: dict = field(default_factory=dict)
    runner_status: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "mode": self.mode,
            "prompt": self.prompt,
            "ready": self.ready,
            "generated_text": self.generated_text,
            "elapsed_seconds": self.elapsed_seconds,
            "max_new_tokens": self.max_new_tokens,
            "steps_completed": self.steps_completed,
            "generated_token_ids": list(self.generated_token_ids),
            "blockers": list(self.blockers),
            "prefix_reuse": dict(self.prefix_reuse),
            "performance_summary": dict(self.performance_summary),
            "memory_before": dict(self.memory_before),
            "memory_after": dict(self.memory_after),
            "runner_status": dict(self.runner_status),
        }


@dataclass(slots=True)
class WarmRunner:
    """In-process warm state for one model/mode pair."""

    model_id: str
    mode: str = "Agent"
    state: str = "idle"
    started_at: float = field(default_factory=time.time)
    last_request_at: float | None = None
    request_count: int = 0
    last_latency_seconds: float | None = None
    total_latency_seconds: float = 0.0
    last_error: str | None = None
    blockers: list[str] = field(default_factory=list)
    reusable_token_ids: list[int] = field(default_factory=list)
    decode_state: KVDecodeState | None = None
    prefix_reuse_available: bool = False
    tensor_residency_warm: bool = False
    memory: dict = field(default_factory=dict)

    def average_latency_seconds(self) -> float | None:
        if self.request_count <= 0:
            return None
        return round(self.total_latency_seconds / self.request_count, 3)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "mode": self.mode,
            "state": self.state,
            "started_at": self.started_at,
            "last_request_at": self.last_request_at,
            "request_count": self.request_count,
            "last_latency_seconds": self.last_latency_seconds,
            "average_latency_seconds": self.average_latency_seconds(),
            "last_error": self.last_error,
            "blockers": list(self.blockers),
            "prefix_reuse_available": self.prefix_reuse_available,
            "reusable_token_count": len(self.reusable_token_ids),
            "tensor_residency_warm": self.tensor_residency_warm,
            "memory": dict(self.memory),
        }


_RUNNERS: dict[str, WarmRunner] = {}
_RUNNERS_LOCK = threading.RLock()


def _status_dir() -> Path:
    path = state_root() / WARM_RUNNER_STATUS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _status_path(model_id: str) -> Path:
    safe_id = model_id.replace("/", "_").replace("\\", "_")
    return _status_dir() / f"{safe_id}.json"


def _bytes_to_mb(value: int | None) -> int | None:
    if value is None:
        return None
    return int(round(value / (1024**2)))


def _process_working_set_bytes() -> int | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return None
    return int(counters.WorkingSetSize)


def warm_runner_memory_snapshot() -> dict:
    """Return system RAM plus current process working set without psutil."""
    memory = _memory_snapshot()
    working_set_bytes = _process_working_set_bytes()
    return {
        "total_ram_mb": _bytes_to_mb(int(memory.total_bytes)),
        "free_ram_mb": _bytes_to_mb(int(memory.free_bytes)),
        "process_working_set_mb": _bytes_to_mb(working_set_bytes),
    }


def _performance_summary(timings: dict[str, Any]) -> dict:
    prefill_load = float(timings.get("prefill_stack_op_load_tensors", 0.0) or 0.0)
    continuation_load = float(timings.get("continuation_stack_op_load_tensors", 0.0) or 0.0)
    prefix_load = float(timings.get("prefix_append_stack_op_load_tensors", 0.0) or 0.0)
    tensor_load_seconds = round(prefill_load + continuation_load + prefix_load, 3)
    total = float(timings.get("total", 0.0) or 0.0)
    decode_tail_seconds = round(
        float(timings.get("prefill_decode_tail", 0.0) or 0.0)
        + float(timings.get("continuation_decode_tail", 0.0) or 0.0),
        3,
    )
    stack_seconds = round(
        float(timings.get("prefill_stack", 0.0) or 0.0)
        + float(timings.get("prefix_append", 0.0) or 0.0)
        + float(timings.get("continuation_steps", 0.0) or 0.0),
        3,
    )
    components = {
        "tensor loading": tensor_load_seconds,
        "layer stack": stack_seconds,
        "decode tail": decode_tail_seconds,
    }
    bottleneck, seconds = max(components.items(), key=lambda item: item[1])
    return {
        "total_seconds": round(total, 3) if total else None,
        "stack_seconds": stack_seconds,
        "tensor_load_seconds": tensor_load_seconds,
        "decode_tail_seconds": decode_tail_seconds,
        "tensor_load_share": round(tensor_load_seconds / total, 2) if total else None,
        "bottleneck": bottleneck if seconds > 0 else None,
        "bottleneck_seconds": seconds if seconds > 0 else None,
    }


def _write_runner_status(runner: WarmRunner) -> None:
    _status_path(runner.model_id).write_text(json.dumps(runner.to_dict(), indent=2), encoding="utf-8")


def _runner_key(model_id: str) -> str:
    return model_id.strip().lower()


def start_warm_runner(model_id: str, mode: str = "Agent") -> dict:
    """Start or return the in-process warm runner for a model."""
    with _RUNNERS_LOCK:
        key = _runner_key(model_id)
        runner = _RUNNERS.get(key)
        if runner is None:
            runner = WarmRunner(model_id=model_id, mode=mode, state="ready")
            _RUNNERS[key] = runner
        else:
            runner.mode = mode
            if runner.state in {"stopped", "failed"}:
                runner.state = "ready"
        runner.memory = warm_runner_memory_snapshot()
        runner.blockers = []
        _write_runner_status(runner)
        return runner.to_dict()


def warm_runner_status(model_id: str) -> dict:
    """Return live in-process status, or the last persisted status."""
    with _RUNNERS_LOCK:
        runner = _RUNNERS.get(_runner_key(model_id))
        if runner is not None:
            runner.memory = warm_runner_memory_snapshot()
            _write_runner_status(runner)
            return runner.to_dict()
    try:
        return json.loads(_status_path(model_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "model_id": model_id,
            "mode": "Agent",
            "state": "stopped",
            "request_count": 0,
            "blockers": ["No warm runner has been started in this process."],
            "prefix_reuse_available": False,
            "tensor_residency_warm": False,
            "memory": warm_runner_memory_snapshot(),
        }


def stop_warm_runner(model_id: str) -> dict:
    """Stop the warm runner and clear warm tensor/session state."""
    with _RUNNERS_LOCK:
        runner = _RUNNERS.pop(_runner_key(model_id), None)
        if runner is None:
            runner = WarmRunner(model_id=model_id, state="stopped")
        runner.state = "stopped"
        runner.decode_state = None
        runner.reusable_token_ids = []
        runner.prefix_reuse_available = False
        runner.tensor_residency_warm = False
        runner.memory = warm_runner_memory_snapshot()
        clear_tensor_residency_cache()
        _write_runner_status(runner)
        return runner.to_dict()


def _blocked_result(runner: WarmRunner, prompt: str, max_new_tokens: int, blockers: list[str], memory_before: dict) -> WarmRunnerRequestResult:
    runner.state = "blocked"
    runner.blockers = list(blockers)
    runner.last_error = blockers[0] if blockers else None
    runner.memory = memory_before
    _write_runner_status(runner)
    return WarmRunnerRequestResult(
        model_id=runner.model_id,
        mode=runner.mode,
        prompt=prompt,
        ready=False,
        generated_text="",
        elapsed_seconds=0.0,
        max_new_tokens=max_new_tokens,
        steps_completed=0,
        blockers=list(blockers),
        memory_before=memory_before,
        memory_after=memory_before,
        runner_status=runner.to_dict(),
    )


def run_warm_agent_prompt(
    model_id: str,
    prompt: str,
    *,
    session_id: str = "default",
    max_new_tokens: int = WARM_RUNNER_AGENT_MAX_NEW_TOKENS,
    mode: str = "Agent",
    min_free_memory_mb: int = WARM_RUNNER_MIN_FREE_MEMORY_MB,
    apply_chat_format: bool = True,
    run_prompt_decode_loop_fn=None,
) -> WarmRunnerRequestResult:
    """Run one short Agent prompt through the conservative in-process warm runner."""
    del session_id  # Reserved for the future service form; one runner currently owns one warm state.
    with _RUNNERS_LOCK:
        runner = _RUNNERS.get(_runner_key(model_id))
        if runner is None:
            start_warm_runner(model_id, mode=mode)
            runner = _RUNNERS[_runner_key(model_id)]

        memory_before = warm_runner_memory_snapshot()
        free_ram_mb = memory_before.get("free_ram_mb")
        if free_ram_mb is not None and int(free_ram_mb) < min_free_memory_mb:
            return _blocked_result(
                runner,
                prompt,
                max_new_tokens,
                [f"Free RAM is below the warm-runner guard of {min_free_memory_mb} MB."],
                memory_before,
            )

        runner.state = "running"
        runner.mode = mode
        runner.memory = memory_before
        runner.blockers = []
        _write_runner_status(runner)

        previous_handle_cache = os.environ.get("PCKETLM_SAFETENSOR_HANDLE_CACHE")
        os.environ["PCKETLM_SAFETENSOR_HANDLE_CACHE"] = "0"
        started = time.perf_counter()
        try:
            runner_fn = run_prompt_decode_loop if run_prompt_decode_loop_fn is None else run_prompt_decode_loop_fn
            result = runner_fn(
                model_id,
                prompt=prompt,
                max_new_tokens=max(1, min(int(max_new_tokens), WARM_RUNNER_AGENT_MAX_NEW_TOKENS)),
                min_new_tokens=1,
                selection_policy="greedy",
                apply_chat_format=apply_chat_format,
                initial_decode_state=runner.decode_state,
                initial_token_ids=runner.reusable_token_ids,
            )
        except Exception as exc:
            elapsed = round(time.perf_counter() - started, 3)
            runner.state = "failed"
            runner.last_error = str(exc)
            runner.last_latency_seconds = elapsed
            runner.memory = warm_runner_memory_snapshot()
            _write_runner_status(runner)
            return WarmRunnerRequestResult(
                model_id=model_id,
                mode=mode,
                prompt=prompt,
                ready=False,
                generated_text="",
                elapsed_seconds=elapsed,
                max_new_tokens=max_new_tokens,
                steps_completed=0,
                blockers=[str(exc)],
                memory_before=memory_before,
                memory_after=dict(runner.memory),
                runner_status=runner.to_dict(),
            )
        finally:
            if previous_handle_cache is None:
                os.environ.pop("PCKETLM_SAFETENSOR_HANDLE_CACHE", None)
            else:
                os.environ["PCKETLM_SAFETENSOR_HANDLE_CACHE"] = previous_handle_cache

        elapsed = round(time.perf_counter() - started, 3)
        memory_after = warm_runner_memory_snapshot()
        runner.request_count += 1
        runner.last_request_at = time.time()
        runner.last_latency_seconds = elapsed
        runner.total_latency_seconds += elapsed
        runner.memory = memory_after
        runner.blockers = list(getattr(result, "blockers", []) or [])
        runner.last_error = None if bool(getattr(result, "ready", False)) else "; ".join(runner.blockers)
        runner.state = "ready" if bool(getattr(result, "ready", False)) else "failed"
        runner.decode_state = getattr(result, "final_decode_state", None) if bool(getattr(result, "ready", False)) else None
        runner.reusable_token_ids = list(getattr(result, "reusable_token_ids", []) or []) if bool(getattr(result, "ready", False)) else []
        runner.prefix_reuse_available = bool(runner.decode_state is not None and runner.reusable_token_ids)
        runner.tensor_residency_warm = tensor_residency_stats().resident_count > 0
        _write_runner_status(runner)

        timings = dict(getattr(result, "timings", {}) or {})
        return WarmRunnerRequestResult(
            model_id=model_id,
            mode=mode,
            prompt=prompt,
            ready=bool(getattr(result, "ready", False)),
            generated_text=str(getattr(result, "generated_text", "") or ""),
            elapsed_seconds=elapsed,
            max_new_tokens=int(getattr(result, "max_new_tokens", max_new_tokens) or max_new_tokens),
            steps_completed=int(getattr(result, "steps_completed", 0) or 0),
            generated_token_ids=list(getattr(result, "generated_token_ids", []) or []),
            blockers=list(getattr(result, "blockers", []) or []),
            prefix_reuse=dict(getattr(result, "prefix_reuse", {}) or {}),
            performance_summary=_performance_summary(timings),
            memory_before=memory_before,
            memory_after=memory_after,
            runner_status=runner.to_dict(),
        )
