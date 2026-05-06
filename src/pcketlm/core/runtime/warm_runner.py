"""Conservative warm runner for repeated short local Agent calls."""

from __future__ import annotations

import ctypes
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from ctypes import wintypes
from typing import Any

from pcketlm.core.runtime.layer_bridge import KVDecodeState, run_prompt_decode_loop, run_prompt_prefill_session
from pcketlm.core.runtime.load_attempt import _memory_snapshot
from pcketlm.core.runtime.tensor_residency import (
    clear_dequantized_tensor_residency_cache,
    clear_tensor_residency_cache,
    tensor_residency_stats,
)
from pcketlm.core.storage.paths import state_root

WARM_RUNNER_MIN_FREE_MEMORY_MB = 4 * 1024
WARM_RUNNER_AGENT_MAX_NEW_TOKENS = 2
Q4_MOE_WARM_AGENT_MAX_NEW_TOKENS = 16
WARM_RUNNER_STATUS_DIR_NAME = "warm-runner"
Q4_MOE_WARM_PACKED_CACHE_MB = 4 * 1024
Q4_MOE_WARM_TENSOR_CACHE_MB = 2 * 1024
Q4_MOE_LOW_FREE_RAM_TRIM_MB = 800
Q4_MOE_PRIME_PROMPT = "The capital of France is"
Q4_MOE_PRIME_MODE = "prefill"


@dataclass(slots=True)
class WarmRunnerRequestResult:
    """One warm runner prompt response plus runner telemetry."""

    model_id: str
    session_id: str
    mode: str
    prompt: str
    ready: bool
    generated_text: str
    elapsed_seconds: float
    max_new_tokens: int
    steps_completed: int
    full_text: str = ""
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
            "session_id": self.session_id,
            "mode": self.mode,
            "prompt": self.prompt,
            "ready": self.ready,
            "generated_text": self.generated_text,
            "full_text": self.full_text,
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
    """In-process warm state for one model/session/mode tuple."""

    model_id: str
    session_id: str = "default"
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
    primed: bool = False
    prime_seconds: float | None = None
    prime_generated_text: str = ""

    def average_latency_seconds(self) -> float | None:
        if self.request_count <= 0:
            return None
        return round(self.total_latency_seconds / self.request_count, 3)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "session_id": self.session_id,
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
            "primed": self.primed,
            "prime_seconds": self.prime_seconds,
            "prime_generated_text": self.prime_generated_text,
        }


_RUNNERS: dict[str, WarmRunner] = {}
_RUNNERS_LOCK = threading.RLock()


def _status_dir() -> Path:
    path = state_root() / WARM_RUNNER_STATUS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_status_id(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")


def _status_path(model_id: str, session_id: str = "default") -> Path:
    safe_id = _safe_status_id(model_id)
    safe_session = _safe_status_id(session_id or "default")
    return _status_dir() / f"{safe_id}__{safe_session}.json"


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
    pending_prefix_load = float(
        timings.get("pending_prefix_continuation_prefix_append_stack_op_load_tensors", 0.0) or 0.0
    ) + float(timings.get("pending_prefix_continuation_continuation_stack_op_load_tensors", 0.0) or 0.0)
    pending_direct_load = float(timings.get("pending_prefix_continuation_stack_op_load_tensors", 0.0) or 0.0)
    tensor_load_seconds = round(prefill_load + continuation_load + prefix_load + pending_prefix_load + pending_direct_load, 3)
    total = float(timings.get("total", 0.0) or 0.0)
    decode_tail_seconds = round(
        float(timings.get("prefill_decode_tail", 0.0) or 0.0)
        + float(timings.get("continuation_decode_tail", 0.0) or 0.0),
        3,
    )
    pending_decode_tail_seconds = round(
        float(timings.get("pending_prefix_continuation_prefill_decode_tail", 0.0) or 0.0)
        + float(timings.get("pending_prefix_continuation_continuation_decode_tail", 0.0) or 0.0),
        3,
    )
    pending_direct_decode_tail_seconds = float(timings.get("pending_prefix_continuation_decode_tail", 0.0) or 0.0)
    stack_seconds = round(
        float(timings.get("prefill_stack", 0.0) or 0.0)
        + float(timings.get("prefix_append", 0.0) or 0.0)
        + float(timings.get("continuation_steps", 0.0) or 0.0),
        3,
    )
    pending_stack_seconds = round(
        float(timings.get("pending_prefix_continuation_prefix_append", 0.0) or 0.0)
        + float(timings.get("pending_prefix_continuation_continuation_steps", 0.0) or 0.0),
        3,
    )
    pending_direct_stack_seconds = float(timings.get("pending_prefix_continuation_stack", 0.0) or 0.0)
    components = {
        "tensor loading": tensor_load_seconds,
        "layer stack": round(stack_seconds + pending_stack_seconds + pending_direct_stack_seconds, 3),
        "decode tail": round(decode_tail_seconds + pending_decode_tail_seconds + pending_direct_decode_tail_seconds, 3),
    }
    bottleneck, seconds = max(components.items(), key=lambda item: item[1])
    return {
        "total_seconds": round(total, 3) if total else None,
        "stack_seconds": components["layer stack"],
        "tensor_load_seconds": tensor_load_seconds,
        "decode_tail_seconds": components["decode tail"],
        "tensor_load_share": round(tensor_load_seconds / total, 2) if total else None,
        "bottleneck": bottleneck if seconds > 0 else None,
        "bottleneck_seconds": seconds if seconds > 0 else None,
    }


def _write_runner_status(runner: WarmRunner) -> None:
    _status_path(runner.model_id, runner.session_id).write_text(
        json.dumps(runner.to_dict(), indent=2), encoding="utf-8"
    )


def _runner_key(model_id: str, session_id: str = "default") -> str:
    return f"{model_id.strip().lower()}::{(session_id or 'default').strip().lower()}"


def _commit_generated_prefix_enabled() -> bool:
    return os.environ.get("PCKETLM_WARM_RUNNER_COMMIT_GENERATED_PREFIX", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _q4_moe_warm_cache_defaults(model_id: str) -> dict[str, str]:
    """Return scoped cache defaults for Q4 MoE warm sessions.

    The warm runner is the repeated short-prompt path. For Q4 MoE, the measured
    win comes from keeping packed Q4 bytes plus all front-layer fp16 attention
    tensors warm between turns. Caller-provided env settings always win.
    """
    tensor_source = os.environ.get("PCKETLM_TENSOR_SOURCE", "auto").strip().lower()
    if tensor_source != "q4":
        return {}
    try:
        from pcketlm.core.runtime.layer_bridge import load_layer_bridge_config

        config = load_layer_bridge_config(model_id)
    except Exception:
        return {}
    if not getattr(config, "ready", False):
        return {}
    if int(getattr(config, "num_experts", 0) or 0) <= 0:
        return {}
    if int(getattr(config, "num_experts_per_tok", 0) or 0) <= 0:
        return {}

    defaults: dict[str, str] = {}
    if not os.environ.get("PCKETLM_Q4_PACKED_CACHE_MB", "").strip():
        defaults["PCKETLM_Q4_PACKED_CACHE_MB"] = str(Q4_MOE_WARM_PACKED_CACHE_MB)
    if not os.environ.get("PCKETLM_TENSOR_CACHE_MB", "").strip():
        defaults["PCKETLM_TENSOR_CACHE_MB"] = str(Q4_MOE_WARM_TENSOR_CACHE_MB)
    if not os.environ.get("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", "").strip():
        defaults["PCKETLM_TENSOR_CACHE_FRONT_LAYERS"] = str(int(getattr(config, "num_hidden_layers", 0) or 0))
    return defaults


def _q4_moe_low_ram_trim_enabled(model_id: str) -> bool:
    if os.environ.get("PCKETLM_DISABLE_Q4_MOE_LOW_RAM_TRIM", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    if os.environ.get("PCKETLM_TENSOR_SOURCE", "auto").strip().lower() != "q4":
        return False
    try:
        from pcketlm.core.runtime.layer_bridge import load_layer_bridge_config

        config = load_layer_bridge_config(model_id)
    except Exception:
        return False
    return (
        bool(getattr(config, "ready", False))
        and int(getattr(config, "num_experts", 0) or 0) > 0
        and int(getattr(config, "num_experts_per_tok", 0) or 0) > 0
    )


def _q4_moe_prime_enabled(model_id: str) -> bool:
    if os.environ.get("PCKETLM_DISABLE_Q4_MOE_PRIME", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    return _q4_moe_low_ram_trim_enabled(model_id)


def _q4_moe_trim_threshold_mb() -> int:
    try:
        return max(0, int(os.environ.get("PCKETLM_Q4_MOE_LOW_RAM_TRIM_MB", str(Q4_MOE_LOW_FREE_RAM_TRIM_MB))))
    except ValueError:
        return Q4_MOE_LOW_FREE_RAM_TRIM_MB


def _q4_moe_min_free_memory_mb() -> int:
    try:
        return max(0, int(os.environ.get("PCKETLM_Q4_MOE_MIN_FREE_RAM_MB", "512")))
    except ValueError:
        return 512


def _q4_moe_prime_prompt() -> str:
    return os.environ.get("PCKETLM_Q4_MOE_PRIME_PROMPT", Q4_MOE_PRIME_PROMPT)


def _q4_moe_prime_mode() -> str:
    requested = os.environ.get("PCKETLM_Q4_MOE_PRIME_MODE", Q4_MOE_PRIME_MODE).strip().lower()
    return requested if requested in {"prefill", "generate"} else Q4_MOE_PRIME_MODE


def _warm_runner_agent_max_new_tokens(model_id: str) -> int:
    if _q4_moe_low_ram_trim_enabled(model_id):
        try:
            return max(
                1,
                int(
                    os.environ.get(
                        "PCKETLM_Q4_MOE_AGENT_MAX_NEW_TOKENS",
                        str(Q4_MOE_WARM_AGENT_MAX_NEW_TOKENS),
                    )
                ),
            )
        except ValueError:
            return Q4_MOE_WARM_AGENT_MAX_NEW_TOKENS
    try:
        return max(1, int(os.environ.get("PCKETLM_WARM_RUNNER_AGENT_MAX_NEW_TOKENS", str(WARM_RUNNER_AGENT_MAX_NEW_TOKENS))))
    except ValueError:
        return WARM_RUNNER_AGENT_MAX_NEW_TOKENS


@contextmanager
def _scoped_environment(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _prime_q4_moe_warm_runner(
    runner: WarmRunner,
    *,
    run_prompt_decode_loop_fn=None,
    run_prompt_prefill_session_fn=None,
    prompt: str | None = None,
) -> None:
    """Warm Q4 MoE packed/tensor caches and optionally prepare reusable prefix KV."""
    if runner.primed:
        return
    scoped_env = {"PCKETLM_SAFETENSOR_HANDLE_CACHE": "0"}
    scoped_env.update(_q4_moe_warm_cache_defaults(runner.model_id))
    started = time.perf_counter()
    try:
        with _scoped_environment(scoped_env):
            if _q4_moe_prime_mode() == "prefill":
                prefill_fn = (
                    run_prompt_prefill_session
                    if run_prompt_prefill_session_fn is None
                    else run_prompt_prefill_session_fn
                )
                result = prefill_fn(
                    runner.model_id,
                    prompt=prompt or _q4_moe_prime_prompt(),
                    apply_chat_format=False,
                )
            else:
                runner_fn = run_prompt_decode_loop if run_prompt_decode_loop_fn is None else run_prompt_decode_loop_fn
                result = runner_fn(
                    runner.model_id,
                    prompt=prompt or _q4_moe_prime_prompt(),
                    max_new_tokens=1,
                    min_new_tokens=1,
                    selection_policy="greedy",
                    apply_chat_format=False,
                    initial_decode_state=None,
                    initial_token_ids=None,
                    commit_generated_prefix=False,
                )
    except Exception as exc:
        runner.blockers = [f"Q4 MoE cache prime failed: {exc}"]
        runner.last_error = runner.blockers[0]
        runner.prime_seconds = round(time.perf_counter() - started, 3)
        return
    runner.prime_seconds = round(time.perf_counter() - started, 3)
    runner.primed = bool(getattr(result, "ready", False))
    runner.prime_generated_text = str(getattr(result, "generated_text", "") or "")
    if not runner.primed:
        runner.blockers = [f"Q4 MoE cache prime did not complete: {'; '.join(getattr(result, 'blockers', []) or [])}"]
        runner.last_error = runner.blockers[0]
    else:
        runner.blockers = []
        runner.last_error = None
        runner.decode_state = getattr(result, "final_decode_state", None)
        runner.reusable_token_ids = list(getattr(result, "reusable_token_ids", []) or [])
        runner.prefix_reuse_available = bool(runner.decode_state is not None and runner.reusable_token_ids)
        runner.tensor_residency_warm = tensor_residency_stats().resident_count > 0
    memory_after = warm_runner_memory_snapshot()
    free_after = memory_after.get("free_ram_mb")
    if free_after is not None and int(free_after) < _q4_moe_trim_threshold_mb():
        clear_dequantized_tensor_residency_cache()
        memory_after = warm_runner_memory_snapshot()
    runner.memory = memory_after


def start_warm_runner(
    model_id: str,
    mode: str = "Agent",
    session_id: str = "default",
    *,
    prime: bool | None = None,
    prime_prompt: str | None = None,
    run_prompt_decode_loop_fn=None,
    run_prompt_prefill_session_fn=None,
) -> dict:
    """Start or return the in-process warm runner for a model."""
    with _RUNNERS_LOCK:
        key = _runner_key(model_id, session_id)
        runner = _RUNNERS.get(key)
        if runner is None:
            runner = WarmRunner(model_id=model_id, session_id=session_id or "default", mode=mode, state="ready")
            _RUNNERS[key] = runner
        else:
            runner.mode = mode
            if runner.state in {"stopped", "failed"}:
                runner.state = "ready"
        runner.memory = warm_runner_memory_snapshot()
        runner.blockers = []
        should_prime = _q4_moe_prime_enabled(model_id) if prime is None else bool(prime)
        if should_prime:
            _prime_q4_moe_warm_runner(
                runner,
                run_prompt_decode_loop_fn=run_prompt_decode_loop_fn,
                run_prompt_prefill_session_fn=run_prompt_prefill_session_fn,
                prompt=prime_prompt,
            )
        _write_runner_status(runner)
        return runner.to_dict()


def warm_runner_status(model_id: str, session_id: str = "default") -> dict:
    """Return live in-process status, or the last persisted status."""
    with _RUNNERS_LOCK:
        runner = _RUNNERS.get(_runner_key(model_id, session_id))
        if runner is not None:
            runner.memory = warm_runner_memory_snapshot()
            _write_runner_status(runner)
            return runner.to_dict()
    try:
        return json.loads(_status_path(model_id, session_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "model_id": model_id,
            "session_id": session_id or "default",
            "mode": "Agent",
            "state": "stopped",
            "request_count": 0,
            "blockers": ["No warm runner has been started in this process."],
            "prefix_reuse_available": False,
            "tensor_residency_warm": False,
            "memory": warm_runner_memory_snapshot(),
        }


def stop_warm_runner(model_id: str, session_id: str = "default") -> dict:
    """Stop the warm runner and clear warm tensor/session state."""
    with _RUNNERS_LOCK:
        runner = _RUNNERS.pop(_runner_key(model_id, session_id), None)
        if runner is None:
            runner = WarmRunner(model_id=model_id, session_id=session_id or "default", state="stopped")
        runner.state = "stopped"
        runner.decode_state = None
        runner.reusable_token_ids = []
        runner.prefix_reuse_available = False
        runner.tensor_residency_warm = False
        runner.memory = warm_runner_memory_snapshot()
        if not any(value.model_id == model_id for value in _RUNNERS.values()):
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
        session_id=runner.session_id,
        mode=runner.mode,
        prompt=prompt,
        ready=False,
        generated_text="",
        full_text=prompt,
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
    with _RUNNERS_LOCK:
        runner_key = _runner_key(model_id, session_id)
        runner = _RUNNERS.get(runner_key)
        if runner is None:
            start_warm_runner(model_id, mode=mode, session_id=session_id, prime=False)
            runner = _RUNNERS[runner_key]

        memory_before = warm_runner_memory_snapshot()
        free_ram_mb = memory_before.get("free_ram_mb")
        q4_moe_trim_enabled = _q4_moe_low_ram_trim_enabled(model_id)
        q4_moe_min_free_memory_mb = _q4_moe_min_free_memory_mb() if q4_moe_trim_enabled else min_free_memory_mb
        if (
            free_ram_mb is not None
            and int(free_ram_mb) < min_free_memory_mb
            and q4_moe_trim_enabled
            and int(free_ram_mb) >= q4_moe_min_free_memory_mb
            and int(free_ram_mb) < _q4_moe_trim_threshold_mb()
        ):
            clear_dequantized_tensor_residency_cache()
            memory_before = warm_runner_memory_snapshot()
            free_ram_mb = memory_before.get("free_ram_mb")
        effective_min_free_memory_mb = q4_moe_min_free_memory_mb if q4_moe_trim_enabled else min_free_memory_mb
        if free_ram_mb is not None and int(free_ram_mb) < effective_min_free_memory_mb:
            return _blocked_result(
                runner,
                prompt,
                max_new_tokens,
                [f"Free RAM is below the warm-runner guard of {effective_min_free_memory_mb} MB."],
                memory_before,
            )

        runner.state = "running"
        runner.mode = mode
        runner.memory = memory_before
        runner.blockers = []
        _write_runner_status(runner)

        scoped_env = {"PCKETLM_SAFETENSOR_HANDLE_CACHE": "0"}
        scoped_env.update(_q4_moe_warm_cache_defaults(model_id))
        started = time.perf_counter()
        try:
            with _scoped_environment(scoped_env):
                runner_fn = run_prompt_decode_loop if run_prompt_decode_loop_fn is None else run_prompt_decode_loop_fn
                result = runner_fn(
                    model_id,
                    prompt=prompt,
                    max_new_tokens=max(1, min(int(max_new_tokens), _warm_runner_agent_max_new_tokens(model_id))),
                    min_new_tokens=1,
                    selection_policy="greedy",
                    apply_chat_format=apply_chat_format,
                    initial_decode_state=runner.decode_state,
                    initial_token_ids=runner.reusable_token_ids,
                    commit_generated_prefix=_commit_generated_prefix_enabled(),
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
                session_id=runner.session_id,
                mode=mode,
                prompt=prompt,
                ready=False,
                generated_text="",
                full_text=prompt,
                elapsed_seconds=elapsed,
                max_new_tokens=max_new_tokens,
                steps_completed=0,
                blockers=[str(exc)],
                memory_before=memory_before,
                memory_after=dict(runner.memory),
                runner_status=runner.to_dict(),
            )
        elapsed = round(time.perf_counter() - started, 3)
        memory_after = warm_runner_memory_snapshot()
        low_ram_trim: dict[str, Any] = {"applied": False}
        if _q4_moe_low_ram_trim_enabled(model_id):
            free_after = memory_after.get("free_ram_mb")
            trim_threshold = _q4_moe_trim_threshold_mb()
            if free_after is not None and int(free_after) < trim_threshold:
                clear_dequantized_tensor_residency_cache()
                trimmed_memory = warm_runner_memory_snapshot()
                low_ram_trim = {
                    "applied": True,
                    "threshold_mb": trim_threshold,
                    "free_before_mb": int(free_after),
                    "free_after_mb": trimmed_memory.get("free_ram_mb"),
                }
                memory_after = trimmed_memory
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
        prefix_reuse_payload = dict(getattr(result, "prefix_reuse", {}) or {})
        if low_ram_trim["applied"]:
            prefix_reuse_payload["q4_moe_low_ram_trim"] = dict(low_ram_trim)
        return WarmRunnerRequestResult(
            model_id=model_id,
            session_id=runner.session_id,
            mode=mode,
            prompt=prompt,
            ready=bool(getattr(result, "ready", False)),
            generated_text=str(getattr(result, "generated_text", "") or ""),
            full_text=str(getattr(result, "full_text", prompt) or prompt),
            elapsed_seconds=elapsed,
            max_new_tokens=int(getattr(result, "max_new_tokens", max_new_tokens) or max_new_tokens),
            steps_completed=int(getattr(result, "steps_completed", 0) or 0),
            generated_token_ids=list(getattr(result, "generated_token_ids", []) or []),
            blockers=list(getattr(result, "blockers", []) or []),
            prefix_reuse=prefix_reuse_payload,
            performance_summary=_performance_summary(timings),
            memory_before=memory_before,
            memory_after=memory_after,
            runner_status=runner.to_dict(),
        )
