"""Local web UI server for Pocket LLM."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import mimetypes
import os
import socket
import threading
import time
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pcketlm.app.desktop.status_screen import build_status_screen_model, list_status_screen_options
from pcketlm.core.benchmark import (
    build_backend_comparison_record,
    build_measured_benchmark_history,
    run_backend_comparison_benchmark,
    run_gguf_measured_benchmark,
    run_measured_benchmark,
)
from pcketlm.core.optimize import latest_optimized_artifact_manifest
from pcketlm.core.profiles import build_profile_compare_summary, get_saved_profile, list_saved_profiles
from pcketlm.core.runtime import (
    DEFAULT_LM_HEAD_CHUNK_ROWS,
    build_gguf_backend_status,
    build_gguf_server_status,
    build_runtime_backend_report,
    load_layer_bridge_config,
    run_gguf_prompt,
    run_prompt_decode_loop,
    run_warm_agent_prompt,
    runtime_math_dtype_name,
    runtime_torch_thread_count,
    select_runtime_engine,
    start_warm_runner,
    start_gguf_server,
    stop_gguf_server,
    stop_warm_runner,
    warm_runner_status,
)
from pcketlm.core.runtime.tensor_loader import runtime_pack_selection_snapshot, tensor_load_stats_snapshot
from pcketlm.core.runtime.tensor_residency import (
    clear_tensor_residency_cache,
    tensor_residency_policy_snapshot,
    tensor_residency_stats,
)
from pcketlm.core.storage.paths import benchmarks_root, original_model_root, project_root, state_root


STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_WEB_SYSTEM_PROMPT = "You are Pocket LLM. Brief English."
CHAT_JOB_TTL_SECONDS = 60 * 30
CHAT_RESPONSE_CACHE_TTL_SECONDS = 60 * 30
CHAT_RESPONSE_CACHE_MAX_ENTRIES = 32
CHAT_SESSION_PREFIX_TTL_SECONDS = 60 * 20
CHAT_SESSION_PREFIX_MAX_ENTRIES = 8
DEFAULT_WEB_PORT = 8765
SINGLE_INSTANCE_LOCK_PORT = 8764
MIN_CHAT_FREE_MEMORY_MB = 4 * 1024
QWEN_14B_MODEL_IDS = {"qwen2.5-14b-instruct", "qwen-2.5-14b-instruct"}
QWEN_32B_MODEL_IDS = {"qwen2.5-32b-instruct", "qwen-2.5-32b-instruct"}
QWEN_14B_PROVEN_MAX_NEW_TOKENS = 8
QWEN_32B_PROVEN_MAX_NEW_TOKENS = 8
QWEN_32B_RECOMMENDED_FREE_MEMORY_MB = 5 * 1024
AGENT_MODE_MAX_NEW_TOKENS = 2
RUNTIME_SETTINGS_FILE_NAME = "runtime-settings.json"
VALID_TENSOR_CACHE_PRESETS = {"standard", "boosted"}
VALID_AGENT_WARM_RUNNER_MODES = {"off", "safe", "experimental"}
DOWNLOAD_STATUS_DIR_NAME = "downloads"

DIRECT_RUNTIME_BASELINES = {
    "qwen2.5-14b-instruct": {
        1: {"seconds": 19.494, "working_set_mb": 577, "tensor_load_seconds": 14.11},
        4: {"seconds": 77.568, "working_set_mb": 1022, "tensor_load_seconds": 65.05},
        8: {"seconds": 155.456, "working_set_mb": 1017, "tensor_load_seconds": 131.07},
    },
    "qwen2.5-32b-instruct": {
        1: {"seconds": 48.254, "working_set_mb": 1410, "tensor_load_seconds": 40.87},
        4: {"seconds": 180.573, "working_set_mb": 1318, "tensor_load_seconds": 156.96},
        8: {"seconds": 346.482, "working_set_mb": 1437, "tensor_load_seconds": 301.71},
    },
}


@dataclass(slots=True)
class ChatJob:
    """Background chat job state for the local web UI."""

    job_id: str
    payload: dict
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: dict | None = None
    error: str | None = None
    cancel_requested: bool = False


@dataclass(slots=True)
class ChatSessionPrefix:
    """In-memory reusable prompt prefix for one active web chat session."""

    session_id: str
    model_id: str
    profile_id: str
    mode: str
    layer_count: int | None
    token_ids: list[int]
    decode_state: Any
    updated_at: float = field(default_factory=time.time)


_CHAT_JOBS: dict[str, ChatJob] = {}
_CHAT_JOBS_LOCK = threading.Lock()
_CHAT_RESPONSE_CACHE: dict[str, dict] = {}
_CHAT_RESPONSE_CACHE_ORDER: list[str] = []
_CHAT_RESPONSE_CACHE_LOCK = threading.RLock()
_CHAT_RESPONSE_CACHE_STATS = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0}
_CHAT_SESSION_PREFIXES: dict[str, ChatSessionPrefix] = {}
_CHAT_SESSION_PREFIX_ORDER: list[str] = []
_CHAT_SESSION_PREFIX_LOCK = threading.RLock()
_CHAT_SESSION_PREFIX_STATS = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0}
_INSTANCE_LOCK_SOCKET: socket.socket | None = None


def _runtime_settings_path() -> Path:
    return state_root() / RUNTIME_SETTINGS_FILE_NAME


def _download_status_root() -> Path:
    return state_root() / DOWNLOAD_STATUS_DIR_NAME


def _download_status_payload() -> dict:
    """Return live model download progress records written by import_cli."""
    root = _download_status_root()
    records: list[dict] = []
    if root.exists():
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            payload.setdefault("model_id", path.stem)
            payload.setdefault("status", "unknown")
            records.append(payload)
    active_statuses = {"starting", "downloading"}
    active = [item for item in records if str(item.get("status") or "").lower() in active_statuses]
    return {
        "records": records,
        "active": active,
        "active_count": len(active),
    }


def _normalize_tensor_cache_preset(value: object) -> str:
    preset = str(value or "standard").strip().lower()
    return "boosted" if preset in {"boost", "boosted", "high-ram", "high_ram"} else "standard"


def _normalize_agent_warm_runner_mode(value: object) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in VALID_AGENT_WARM_RUNNER_MODES else "off"


def _load_saved_runtime_settings() -> dict:
    path = _runtime_settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {
            "tensor_cache_preset": os.environ.get("PCKETLM_TENSOR_CACHE_PRESET", "standard"),
            "agent_warm_runner": "off",
        }
    return {
        "tensor_cache_preset": _normalize_tensor_cache_preset(data.get("tensor_cache_preset")),
        "agent_warm_runner": _normalize_agent_warm_runner_mode(data.get("agent_warm_runner")),
    }


def _save_runtime_settings(settings: dict) -> None:
    path = _runtime_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")


def _apply_runtime_settings(settings: dict) -> None:
    os.environ["PCKETLM_TENSOR_CACHE_PRESET"] = _normalize_tensor_cache_preset(settings.get("tensor_cache_preset"))


def _apply_saved_runtime_settings() -> dict:
    settings = _load_saved_runtime_settings()
    _apply_runtime_settings(settings)
    return settings


def _update_runtime_settings(payload: dict) -> dict:
    settings = {
        "tensor_cache_preset": _normalize_tensor_cache_preset(payload.get("tensor_cache_preset")),
        "agent_warm_runner": _normalize_agent_warm_runner_mode(payload.get("agent_warm_runner")),
    }
    _save_runtime_settings(settings)
    _apply_runtime_settings(settings)
    clear_tensor_residency_cache()
    _clear_session_prefix_cache()
    _clear_chat_response_cache()
    return {"runtime_settings": _runtime_settings_payload()}


def _warm_runner_control_payload(payload: dict) -> dict:
    model_id = str(payload.get("model_id") or "qwen2.5-14b-instruct")
    action = str(payload.get("action") or "status").strip().lower()
    prime_prompt = payload.get("prime_prompt")
    if prime_prompt is not None:
        prime_prompt = str(prime_prompt)
    prime_apply_chat_format = bool(payload.get("prime_apply_chat_format", bool(prime_prompt)))
    if action == "start":
        status = start_warm_runner(
            model_id,
            mode="Agent",
            prime_prompt=prime_prompt,
            prime_apply_chat_format=prime_apply_chat_format,
        )
    elif action in {"stop", "unload"}:
        status = stop_warm_runner(model_id)
        _clear_session_prefix_cache()
        _clear_chat_response_cache()
    else:
        status = warm_runner_status(model_id)
    return {
        "warm_runner": status,
        "runtime_settings": _runtime_settings_payload(),
    }


def _chat_layer_count(mode: str) -> int | None:
    normalized = _chat_mode_key(mode)
    if normalized.startswith("quick") or normalized.startswith("agent"):
        return None
    if normalized.startswith("fast"):
        return 8
    if normalized.startswith("balanced"):
        return 32
    return None


def _chat_layer_count_for_request(model_id: str, mode: str, max_new_tokens: int) -> int | None:
    layer_count = _chat_layer_count(mode)
    proven_max_tokens = _direct_runtime_proven_max_tokens(model_id)
    if _is_gguf_mode(mode) or proven_max_tokens is None or max_new_tokens <= proven_max_tokens:
        return layer_count
    config = load_layer_bridge_config(model_id)
    if config.blockers:
        return layer_count
    return int(config.num_hidden_layers)


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _normalize_chat_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []
    messages: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role == "ai":
            role = "assistant"
        text = str(item.get("text") or item.get("content") or "").strip()
        if role not in {"user", "assistant", "system"}:
            continue
        if not text or text == "Working locally...":
            continue
        messages.append({"role": role, "text": text})
    return messages


def _display_model_name(model_id: str) -> str:
    known = {
        "qwen2.5-14b-instruct": "Qwen2.5-14B-Instruct",
        "qwen2.5-32b-instruct": "Qwen2.5-32B-Instruct",
    }
    return known.get(model_id, model_id)


def _is_qwen_32b_model(model_id: str) -> bool:
    return model_id.strip().lower() in QWEN_32B_MODEL_IDS


def _is_qwen_14b_model(model_id: str) -> bool:
    return model_id.strip().lower() in QWEN_14B_MODEL_IDS


def _bytes_to_mb(value: int) -> int:
    return int(round(value / (1024**2)))


def _direct_runtime_proven_max_tokens(model_id: str) -> int | None:
    if _is_qwen_32b_model(model_id):
        return QWEN_32B_PROVEN_MAX_NEW_TOKENS
    if _is_qwen_14b_model(model_id):
        return QWEN_14B_PROVEN_MAX_NEW_TOKENS
    return None


def _direct_runtime_baseline(model_id: str, requested_max_new_tokens: int | None = None) -> dict | None:
    baseline = DIRECT_RUNTIME_BASELINES.get(model_id.strip().lower())
    if not baseline:
        return None
    requested = int(requested_max_new_tokens or max(baseline))
    nearest_tokens = min(baseline, key=lambda tokens: abs(tokens - requested))
    row = baseline[nearest_tokens]
    seconds = float(row["seconds"])
    tensor_load_seconds = float(row["tensor_load_seconds"])
    return {
        "measured_tokens": nearest_tokens,
        "estimated_seconds": round(seconds * max(1, requested) / nearest_tokens, 1),
        "measured_seconds": seconds,
        "working_set_mb": int(row["working_set_mb"]),
        "tensor_load_seconds": tensor_load_seconds,
        "tensor_load_share": round(tensor_load_seconds / seconds, 2) if seconds else None,
        "bottleneck": "tensor loading",
        "source": "Phase 3 local baseline",
    }


def _runtime_performance_summary(timings: dict) -> dict:
    """Summarize timing keys into a customer-safe speed diagnosis."""
    prefill_load = float(timings.get("prefill_stack_op_load_tensors", 0.0) or 0.0)
    continuation_load = float(timings.get("continuation_stack_op_load_tensors", 0.0) or 0.0)
    prefix_load = float(timings.get("prefix_append_stack_op_load_tensors", 0.0) or 0.0)
    tensor_load_seconds = round(prefill_load + continuation_load + prefix_load, 3)
    prefill_stack = float(timings.get("prefill_stack", 0.0) or timings.get("prefill_stack_total", 0.0) or 0.0)
    continuation_steps = float(timings.get("continuation_steps", 0.0) or 0.0)
    prefix_append = float(timings.get("prefix_append", 0.0) or 0.0)
    decode_tail = round(
        float(timings.get("prefill_decode_tail", 0.0) or 0.0)
        + float(timings.get("continuation_decode_tail", 0.0) or 0.0),
        3,
    )
    total = float(timings.get("total", 0.0) or 0.0)
    stack_seconds = round(prefill_stack + continuation_steps + prefix_append, 3)
    components = {
        "tensor loading": tensor_load_seconds,
        "layer stack": stack_seconds,
        "decode tail": decode_tail,
    }
    bottleneck, bottleneck_seconds = max(components.items(), key=lambda item: item[1])
    return {
        "total_seconds": round(total, 3) if total else None,
        "stack_seconds": stack_seconds,
        "tensor_load_seconds": tensor_load_seconds,
        "decode_tail_seconds": decode_tail,
        "tensor_load_share": round(tensor_load_seconds / total, 2) if total else None,
        "bottleneck": bottleneck if bottleneck_seconds > 0 else None,
        "bottleneck_seconds": bottleneck_seconds if bottleneck_seconds > 0 else None,
        "summary": (
            f"Main bottleneck: {bottleneck}."
            if bottleneck_seconds > 0
            else "No detailed runtime timing was recorded."
        ),
    }


def _gguf_generation_speed_payload(timings: dict) -> dict:
    server = dict(timings.get("server") or {})
    predicted_n = int(server.get("predicted_n") or 0)
    predicted_ms = float(server.get("predicted_ms") or 0.0)
    tokens_per_second = float(server.get("predicted_per_second") or 0.0)
    seconds_per_token = None
    if predicted_n > 0 and predicted_ms > 0:
        seconds_per_token = round((predicted_ms / 1000.0) / predicted_n, 3)
    elif tokens_per_second > 0:
        seconds_per_token = round(1.0 / tokens_per_second, 3)
    target_max = 4.0
    return {
        "backend": "llama-cpp-gguf",
        "generated_tokens": predicted_n,
        "generation_seconds_per_token": seconds_per_token,
        "generation_tokens_per_second": round(tokens_per_second, 3) if tokens_per_second > 0 else None,
        "target_seconds_per_token_max": target_max,
        "target_met": bool(seconds_per_token is not None and seconds_per_token <= target_max),
        "summary": (
            f"GGUF generated at {seconds_per_token}s/token."
            if seconds_per_token is not None
            else "GGUF token speed was not reported by llama.cpp."
        ),
    }


def _direct_model_guardrails(model_id: str, requested_max_new_tokens: int | None = None) -> dict:
    """Describe customer-facing safety bounds for the direct runtime path."""
    policy = tensor_residency_policy_snapshot()
    free_bytes = policy.get("free_memory_bytes")
    free_ram_mb = None if free_bytes is None else _bytes_to_mb(int(free_bytes))
    is_32b = _is_qwen_32b_model(model_id)
    is_14b = _is_qwen_14b_model(model_id)
    proven_max_tokens = _direct_runtime_proven_max_tokens(model_id)
    min_free_ram_mb = MIN_CHAT_FREE_MEMORY_MB
    recommended_free_ram_mb = QWEN_32B_RECOMMENDED_FREE_MEMORY_MB if is_32b else MIN_CHAT_FREE_MEMORY_MB
    requested_tokens = None if requested_max_new_tokens is None else int(requested_max_new_tokens)
    below_min = free_ram_mb is not None and free_ram_mb < min_free_ram_mb
    above_proven = proven_max_tokens is not None and requested_tokens is not None and requested_tokens > proven_max_tokens
    blockers: list[str] = []
    warnings: list[str] = []
    if below_min:
        blockers.append(f"Free RAM is below the direct-runtime guard of {round(min_free_ram_mb / 1024, 1)} GB.")
    if above_proven:
        warnings.append(
            f"{_display_model_name(model_id)} is proven to {proven_max_tokens} new tokens on this machine; longer runs are experimental."
        )

    if not is_32b and not is_14b:
        status = "standard"
        summary = "This model uses the standard direct-runtime guardrails."
    elif below_min:
        status = "blocked-low-ram"
        summary = f"{_display_model_name(model_id)} is available but should not start while free RAM is below the direct-runtime guard."
    elif free_ram_mb is not None and free_ram_mb < recommended_free_ram_mb:
        status = "stable-low-headroom"
        summary = f"{_display_model_name(model_id)} can run, but it is slow and close to the RAM floor; keep replies short."
    elif is_14b:
        status = "stable-slow"
        summary = "Qwen 14B is proven for short direct-runtime replies, but tensor loading dominates longer answers."
    else:
        status = "stable-slow"
        summary = "Qwen 32B is stable for short direct-runtime replies, but it is still a slow power-user path."
    baseline = _direct_runtime_baseline(model_id, requested_tokens)

    return {
        "model_id": model_id,
        "model_label": _display_model_name(model_id),
        "model_size_class": "direct-32b" if is_32b else ("direct-14b" if is_14b else "direct-standard"),
        "status": status,
        "ready": not blockers,
        "summary": summary,
        "free_ram_mb": free_ram_mb,
        "min_free_ram_mb": min_free_ram_mb,
        "recommended_free_ram_mb": recommended_free_ram_mb,
        "requested_max_new_tokens": requested_tokens,
        "proven_max_new_tokens": proven_max_tokens,
        "recommended_mode": "Agent" if is_32b or above_proven else "Quality",
        "token_policy": {
            "agent_mode_max_new_tokens": AGENT_MODE_MAX_NEW_TOKENS,
            "experimental_override": "allow_experimental_direct_tokens",
            "legacy_32b_override": "allow_experimental_32b_tokens",
        },
        "baseline_estimate": baseline,
        "scoped_safetensor_handle_cache": {
            "default_enabled": not is_32b,
            "override_env": "PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE",
            "summary": (
                "Disabled by default for Qwen 32B because the scoped handle path caused native Windows access violations."
                if is_32b
                else "Automatic one-token scoped handle reuse may be used on standard direct models."
            ),
        },
        "blockers": blockers,
        "warnings": warnings,
    }


def _runtime_identity_system_prompt(
    model_id: str,
    mode: str,
    profile: Any | None = None,
    *,
    base_prompt: str | None = None,
) -> str:
    """Build the short context header the local model needs to know where it is running."""
    base = (base_prompt or DEFAULT_WEB_SYSTEM_PROMPT).strip()
    profile_label = "Default" if profile is None else str(getattr(profile, "label", "") or getattr(profile, "profile_id", "Profile"))
    model_name = _display_model_name(model_id)
    runtime_label = "local GGUF / llama.cpp" if _is_gguf_mode(mode) else "local direct"
    return (
        f"{base}\n\n"
        f"Runtime: {runtime_label}. Model: {model_name} ({model_id}). "
        f"Mode: {mode or 'Quality'}. Profile: {profile_label}."
    )


def _runtime_context_payload(model_id: str, mode: str, profile: Any | None = None) -> dict:
    return {
        "model_id": model_id,
        "model_label": _display_model_name(model_id),
        "runtime_mode": mode,
        "profile_id": None if profile is None else profile.profile_id,
        "profile_label": None if profile is None else profile.label,
    }


def _conversation_state_payload(
    turn_count: int,
    preformatted_chat: bool,
    *,
    session_id: str = "",
    prefix_reuse: dict | None = None,
) -> dict:
    """Describe the current lightweight conversation-state strategy."""
    reuse = {} if prefix_reuse is None else dict(prefix_reuse)
    reuse_ready = bool(session_id)
    reuse_used = bool(reuse.get("used"))
    return {
        "state_kind": "session-prefix-kv" if reuse_ready else "bounded-recent-chat",
        "turn_count": turn_count,
        "preformatted_chat": preformatted_chat,
        "session_id": session_id or None,
        "reuse_ready": reuse_ready,
        "reuse_used": reuse_used,
        "prefix_reuse": reuse,
        "summary": (
            reuse.get("summary")
            if reuse
            else (
                "This session can keep an in-memory KV prefix for matching follow-up prompts."
                if reuse_ready
                else "Recent chat turns are carried in the prompt; no session prefix is active."
            )
        ),
    }


def _chat_response_cache_key(payload: dict) -> str:
    cache_payload = {
        "model_id": payload.get("model_id") or "qwen2.5-14b-instruct",
        "profile_id": payload.get("profile_id") or "",
        "mode": payload.get("mode") or "",
        "prompt": str(payload.get("prompt") or "").strip(),
        "messages": _normalize_chat_messages(payload.get("messages")),
        "system_prompt": str(payload.get("system_prompt") or DEFAULT_WEB_SYSTEM_PROMPT),
        "max_new_tokens": payload.get("max_new_tokens"),
        "min_new_tokens": payload.get("min_new_tokens"),
        "repetition_penalty": payload.get("repetition_penalty"),
        "stop_strings": list(payload.get("stop_strings") or ["<|im_end|>", "<|im_start|>"]),
    }
    encoded = json.dumps(cache_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _chat_response_cache_enabled() -> bool:
    return True


def _cache_reuse_payload(hit: bool, *, cache_key: str | None = None, stored_at: float | None = None) -> dict:
    return {
        "enabled": _chat_response_cache_enabled(),
        "hit": hit,
        "kind": "exact-prompt-result",
        "cache_key": None if cache_key is None else cache_key[:12],
        "stored_age_seconds": None if stored_at is None else round(max(0.0, time.time() - stored_at), 2),
        "summary": (
            "Exact prompt result reused without a model run."
            if hit
            else "No exact prompt result was reusable; model generation was required."
        ),
    }


def _get_cached_chat_response(cache_key: str, *, record_miss: bool = True) -> dict | None:
    if not _chat_response_cache_enabled():
        return None
    now = time.time()
    with _CHAT_RESPONSE_CACHE_LOCK:
        entry = _CHAT_RESPONSE_CACHE.get(cache_key)
        if entry is None:
            if record_miss:
                _CHAT_RESPONSE_CACHE_STATS["misses"] += 1
            return None
        if now - float(entry["stored_at"]) > CHAT_RESPONSE_CACHE_TTL_SECONDS:
            _CHAT_RESPONSE_CACHE.pop(cache_key, None)
            if cache_key in _CHAT_RESPONSE_CACHE_ORDER:
                _CHAT_RESPONSE_CACHE_ORDER.remove(cache_key)
            if record_miss:
                _CHAT_RESPONSE_CACHE_STATS["misses"] += 1
            return None
        if cache_key in _CHAT_RESPONSE_CACHE_ORDER:
            _CHAT_RESPONSE_CACHE_ORDER.remove(cache_key)
        _CHAT_RESPONSE_CACHE_ORDER.append(cache_key)
        _CHAT_RESPONSE_CACHE_STATS["hits"] += 1
        response = copy.deepcopy(entry["response"])
        response["elapsed_seconds"] = 0.0
        response["runtime_settings"] = _runtime_settings_payload()
        response["response_reuse"] = _cache_reuse_payload(True, cache_key=cache_key, stored_at=float(entry["stored_at"]))
        response["strategy"] = f"{response.get('strategy', 'unknown')}+response-cache"
        return response


def _store_chat_response(cache_key: str, response: dict) -> None:
    if not _chat_response_cache_enabled() or not response.get("ready"):
        return
    response_to_store = copy.deepcopy(response)
    response_to_store["response_reuse"] = _cache_reuse_payload(False, cache_key=cache_key)
    with _CHAT_RESPONSE_CACHE_LOCK:
        if cache_key in _CHAT_RESPONSE_CACHE_ORDER:
            _CHAT_RESPONSE_CACHE_ORDER.remove(cache_key)
        _CHAT_RESPONSE_CACHE[cache_key] = {
            "stored_at": time.time(),
            "response": response_to_store,
        }
        _CHAT_RESPONSE_CACHE_ORDER.append(cache_key)
        _CHAT_RESPONSE_CACHE_STATS["stores"] += 1
        while len(_CHAT_RESPONSE_CACHE_ORDER) > CHAT_RESPONSE_CACHE_MAX_ENTRIES:
            evicted_key = _CHAT_RESPONSE_CACHE_ORDER.pop(0)
            _CHAT_RESPONSE_CACHE.pop(evicted_key, None)
            _CHAT_RESPONSE_CACHE_STATS["evictions"] += 1


def _chat_response_cache_status() -> dict:
    with _CHAT_RESPONSE_CACHE_LOCK:
        return {
            "enabled": _chat_response_cache_enabled(),
            "kind": "exact-prompt-result",
            "entries": len(_CHAT_RESPONSE_CACHE),
            "max_entries": CHAT_RESPONSE_CACHE_MAX_ENTRIES,
            "ttl_seconds": CHAT_RESPONSE_CACHE_TTL_SECONDS,
            **dict(_CHAT_RESPONSE_CACHE_STATS),
        }


def _clear_chat_response_cache() -> None:
    with _CHAT_RESPONSE_CACHE_LOCK:
        _CHAT_RESPONSE_CACHE.clear()
        _CHAT_RESPONSE_CACHE_ORDER.clear()
        for key in _CHAT_RESPONSE_CACHE_STATS:
            _CHAT_RESPONSE_CACHE_STATS[key] = 0


def _normalize_session_id(raw_session_id: Any) -> str:
    session_id = str(raw_session_id or "").strip()
    if not session_id:
        return ""
    return session_id[:96]


def _session_prefix_cache_status() -> dict:
    with _CHAT_SESSION_PREFIX_LOCK:
        return {
            "enabled": True,
            "kind": "in-memory-kv-prefix",
            "entries": len(_CHAT_SESSION_PREFIXES),
            "max_entries": CHAT_SESSION_PREFIX_MAX_ENTRIES,
            "ttl_seconds": CHAT_SESSION_PREFIX_TTL_SECONDS,
            **dict(_CHAT_SESSION_PREFIX_STATS),
        }


def _cleanup_session_prefixes(now: float | None = None) -> None:
    current = time.time() if now is None else now
    with _CHAT_SESSION_PREFIX_LOCK:
        expired = [
            session_id
            for session_id, prefix in _CHAT_SESSION_PREFIXES.items()
            if current - prefix.updated_at > CHAT_SESSION_PREFIX_TTL_SECONDS
        ]
        for session_id in expired:
            _CHAT_SESSION_PREFIXES.pop(session_id, None)
            if session_id in _CHAT_SESSION_PREFIX_ORDER:
                _CHAT_SESSION_PREFIX_ORDER.remove(session_id)


def _get_session_prefix(
    session_id: str,
    *,
    model_id: str,
    profile_id: str,
    mode: str,
    layer_count: int | None,
) -> ChatSessionPrefix | None:
    if not session_id:
        return None
    _cleanup_session_prefixes()
    with _CHAT_SESSION_PREFIX_LOCK:
        prefix = _CHAT_SESSION_PREFIXES.get(session_id)
        if prefix is None:
            _CHAT_SESSION_PREFIX_STATS["misses"] += 1
            return None
        if (
            prefix.model_id != model_id
            or prefix.profile_id != profile_id
            or prefix.mode != mode
            or prefix.layer_count != layer_count
        ):
            _CHAT_SESSION_PREFIX_STATS["misses"] += 1
            return None
        if session_id in _CHAT_SESSION_PREFIX_ORDER:
            _CHAT_SESSION_PREFIX_ORDER.remove(session_id)
        _CHAT_SESSION_PREFIX_ORDER.append(session_id)
        prefix.updated_at = time.time()
        _CHAT_SESSION_PREFIX_STATS["hits"] += 1
        return prefix


def _store_session_prefix(
    session_id: str,
    *,
    model_id: str,
    profile_id: str,
    mode: str,
    layer_count: int | None,
    token_ids: list[int],
    decode_state: Any,
) -> None:
    if not session_id or decode_state is None or not token_ids:
        return
    if not bool(getattr(decode_state, "ready", False)):
        return
    with _CHAT_SESSION_PREFIX_LOCK:
        if session_id in _CHAT_SESSION_PREFIX_ORDER:
            _CHAT_SESSION_PREFIX_ORDER.remove(session_id)
        _CHAT_SESSION_PREFIXES[session_id] = ChatSessionPrefix(
            session_id=session_id,
            model_id=model_id,
            profile_id=profile_id,
            mode=mode,
            layer_count=layer_count,
            token_ids=[int(value) for value in token_ids],
            decode_state=decode_state,
        )
        _CHAT_SESSION_PREFIX_ORDER.append(session_id)
        _CHAT_SESSION_PREFIX_STATS["stores"] += 1
        while len(_CHAT_SESSION_PREFIX_ORDER) > CHAT_SESSION_PREFIX_MAX_ENTRIES:
            evicted_id = _CHAT_SESSION_PREFIX_ORDER.pop(0)
            _CHAT_SESSION_PREFIXES.pop(evicted_id, None)
            _CHAT_SESSION_PREFIX_STATS["evictions"] += 1


def _clear_session_prefix_cache() -> None:
    with _CHAT_SESSION_PREFIX_LOCK:
        _CHAT_SESSION_PREFIXES.clear()
        _CHAT_SESSION_PREFIX_ORDER.clear()
        for key in _CHAT_SESSION_PREFIX_STATS:
            _CHAT_SESSION_PREFIX_STATS[key] = 0


def _local_runtime_context_answer(prompt: str, model_id: str, mode: str, profile: Any | None = None) -> dict | None:
    """Answer deterministic Pocket LLM runtime facts without spending a model run."""
    normalized = " ".join(prompt.strip().lower().replace("?", " ").split())
    asks_model = "model" in normalized and any(
        marker in normalized
        for marker in [
            "which model",
            "what model",
            "current model",
            "loaded model",
            "model are you",
            "model do you",
            "model you run",
            "model running",
        ]
    )
    if not asks_model:
        return None
    context = _runtime_context_payload(model_id, mode, profile)
    return {
        "ready": True,
        "generated_text": f"{context['model_label']} ({model_id})",
        "full_text": f"{prompt}\n{context['model_label']} ({model_id})",
        "generated_token_ids": [],
        "prompt_token_count": 0,
        "steps_completed": 0,
        "max_new_tokens": 0,
        "stop_reason": "local-runtime-context",
        "strategy": "local-runtime-context-answer",
        "cache_sequence_lengths": {},
        "blockers": [],
        "elapsed_seconds": 0.0,
        "timings": {"total": 0.0},
        "runtime_settings": _runtime_settings_payload(),
        "conversation_turn_count": 0,
        "preformatted_chat": False,
        "profile_id": context["profile_id"],
        "profile_label": context["profile_label"],
        "runtime_context": context,
        "model_guardrails": _direct_model_guardrails(model_id),
    }


def _memory_guard_response(prompt: str, model_id: str, mode: str, profile: Any | None = None) -> dict | None:
    policy = tensor_residency_policy_snapshot()
    free_bytes = policy.get("free_memory_bytes")
    if free_bytes is None or int(free_bytes) >= MIN_CHAT_FREE_MEMORY_MB * 1024 * 1024:
        return None
    context = _runtime_context_payload(model_id, mode, profile)
    free_gb = policy.get("free_memory_gb")
    message = (
        "Pocket LLM paused local generation because system memory is too low. "
        f"Free RAM is about {free_gb} GB; close other apps and retry."
    )
    return {
        "ready": False,
        "generated_text": "",
        "full_text": f"{prompt}\n{message}",
        "generated_token_ids": [],
        "prompt_token_count": 0,
        "steps_completed": 0,
        "max_new_tokens": 0,
        "stop_reason": "memory-guard",
        "strategy": "local-memory-guard",
        "cache_sequence_lengths": {},
        "blockers": [message],
        "elapsed_seconds": 0.0,
        "timings": {"total": 0.0},
        "runtime_settings": _runtime_settings_payload(),
        "conversation_turn_count": 0,
        "preformatted_chat": False,
        "profile_id": context["profile_id"],
        "profile_label": context["profile_label"],
        "runtime_context": context,
        "model_guardrails": _direct_model_guardrails(model_id),
    }


def _gguf_load_required_response(
    prompt: str,
    model_id: str,
    mode: str,
    profile: Any | None,
    *,
    backend_status: dict,
    session_id: str = "",
) -> dict:
    load_estimate = dict(backend_status.get("load_estimate") or {})
    server = dict(backend_status.get("llama_server") or {})
    expected_ram = load_estimate.get("expected_ram_mb")
    cold_load = load_estimate.get("estimated_cold_load_seconds")
    message = (
        "Load the GGUF fast model before chatting. "
        "That keeps normal Qwen 14B replies on the warmed speed path instead of starting a cold load inside chat."
    )
    if expected_ram:
        message += f" Expected RAM: {expected_ram} MB."
    if cold_load:
        message += f" Estimated first load: {cold_load}s."
    context = _runtime_context_payload(model_id, mode, profile)
    return {
        "ready": False,
        "generated_text": "",
        "full_text": f"{prompt}\n{message}",
        "generated_token_ids": [],
        "prompt_token_count": 0,
        "steps_completed": 0,
        "max_new_tokens": 0,
        "stop_reason": "gguf-load-required",
        "strategy": "gguf-load-required",
        "cache_sequence_lengths": {},
        "blockers": [message],
        "elapsed_seconds": 0.0,
        "timings": {"total": 0.0},
        "runtime_settings": _runtime_settings_payload(),
        "conversation_turn_count": 0,
        "preformatted_chat": False,
        "profile_id": context["profile_id"],
        "profile_label": context["profile_label"],
        "runtime_context": context,
        "speed_status": _speed_status_payload(model_id),
        "model_guardrails": _direct_model_guardrails(model_id),
        "gguf_backend": {
            **backend_status,
            "llama_server": server,
            "summary": message,
        },
        "conversation_state": _conversation_state_payload(
            0,
            False,
            session_id=session_id,
            prefix_reuse={"enabled": False, "used": False, "reason": "gguf-load-required"},
        ),
        "response_reuse": _cache_reuse_payload(False),
    }


@lru_cache(maxsize=16)
def _supports_im_chat_tokens(model_id: str) -> bool:
    tokenizer_config_path = original_model_root(model_id) / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return False
    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    added_tokens_decoder = tokenizer_config.get("added_tokens_decoder", {})
    special_contents = {
        entry.get("content")
        for entry in added_tokens_decoder.values()
        if isinstance(entry, dict) and "content" in entry
    }
    return "<|im_start|>" in special_contents and "<|im_end|>" in special_contents


def _truncate_turn_text(text: str, *, max_chars: int = 700) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:].lstrip()


def _formatted_chat_prompt(
    model_id: str,
    raw_messages: Any,
    current_prompt: str,
    *,
    system_prompt: str | None = None,
    max_chars: int = 2600,
) -> tuple[str, int, bool]:
    messages = _normalize_chat_messages(raw_messages)
    if not messages:
        return current_prompt, 0, False
    if not _supports_im_chat_tokens(model_id):
        prompt, turn_count = _conversation_prompt(messages, current_prompt, max_chars=max_chars)
        return prompt, turn_count, False

    system_text = system_prompt or DEFAULT_WEB_SYSTEM_PROMPT
    non_system_messages: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "system":
            system_text = message["text"]
        else:
            non_system_messages.append(message)

    recent_messages = non_system_messages[-8:]

    def build_prompt(turns: list[dict[str, str]], user_prompt: str) -> str:
        lines = [f"<|im_start|>system\n{_truncate_turn_text(system_text)}<|im_end|>"]
        for turn in turns:
            lines.append(
                f"<|im_start|>{turn['role']}\n"
                f"{_truncate_turn_text(turn['text'])}<|im_end|>"
            )
        lines.append(f"<|im_start|>user\n{_truncate_turn_text(user_prompt)}<|im_end|>")
        lines.append("<|im_start|>assistant\n")
        return "\n".join(lines)

    prompt = build_prompt(recent_messages, current_prompt)
    while len(prompt) > max_chars and recent_messages:
        recent_messages = recent_messages[1:]
        prompt = build_prompt(recent_messages, current_prompt)
    return prompt, len(recent_messages), True


def _gguf_chat_prompt(model_id: str, payload: dict, current_prompt: str, system_prompt: str) -> tuple[str, int, bool]:
    """Format GGUF prompts for instruct models instead of sending raw completion text."""
    raw_messages = _normalize_chat_messages(payload.get("messages"))
    if _supports_im_chat_tokens(model_id):
        return _formatted_chat_prompt(
            model_id,
            raw_messages or [{"role": "system", "text": system_prompt}],
            current_prompt,
            system_prompt=system_prompt,
            max_chars=3200,
        )
    prompt, turn_count = _conversation_prompt(raw_messages, current_prompt, max_chars=2600)
    return f"System: {system_prompt}\nUser: {prompt}\nAssistant:", turn_count, False


def _conversation_prompt(raw_messages: Any, current_prompt: str, *, max_chars: int = 2400) -> tuple[str, int]:
    messages = _normalize_chat_messages(raw_messages)
    if not messages:
        return current_prompt, 0

    labels = {"user": "User", "assistant": "Assistant", "system": "System"}
    lines = ["Previous conversation:"]
    for message in messages[-8:]:
        lines.append(f"{labels[message['role']]}: {message['text']}")
    lines.extend(["", "Current user message:", current_prompt])

    prompt = "\n".join(lines)
    if len(prompt) > max_chars:
        prompt = "Previous conversation:\n" + prompt[-max_chars:]
    return prompt, len(messages[-8:])


def _latest_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_measured_benchmark_by_scope(model_id: str, scope: str) -> dict | None:
    root = benchmarks_root(model_id)
    try:
        paths = sorted(root.glob("*.measured-benchmark.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return None
    for path in paths:
        if path.name.startswith("latest."):
            continue
        payload = _latest_json(path)
        if not payload:
            continue
        runtime_settings = dict(payload.get("runtime_settings") or {})
        if runtime_settings.get("benchmark_scope") == scope:
            return payload
    return None


def _cleanup_chat_jobs(now: float | None = None) -> None:
    current = time.time() if now is None else now
    with _CHAT_JOBS_LOCK:
        expired = [
            job_id
            for job_id, job in _CHAT_JOBS.items()
            if job.status in {"completed", "failed", "canceled"} and current - job.updated_at > CHAT_JOB_TTL_SECONDS
        ]
        for job_id in expired:
            _CHAT_JOBS.pop(job_id, None)


def _chat_job_payload(job: ChatJob) -> dict:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "cancel_requested": job.cancel_requested,
        "result": job.result,
        "error": job.error,
    }


def _get_chat_job(job_id: str) -> ChatJob | None:
    with _CHAT_JOBS_LOCK:
        return _CHAT_JOBS.get(job_id)


def _cancel_chat_job(job_id: str) -> ChatJob | None:
    with _CHAT_JOBS_LOCK:
        job = _CHAT_JOBS.get(job_id)
        if job is None:
            return None
        job.cancel_requested = True
        job.updated_at = time.time()
        if job.status == "queued":
            job.status = "canceled"
        return job


def _chat_request_runtime_defaults(payload: dict) -> tuple[str, str, Any | None, str, int, int, float]:
    model_id = str(payload.get("model_id") or "qwen2.5-14b-instruct")
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Prompt is required.")
    profile = get_saved_profile(model_id, str(payload.get("profile_id") or ""))
    profile_settings = {} if profile is None else dict(profile.settings)
    mode = str(payload.get("mode") or profile_settings.get("runtime_mode") or (profile.runtime_mode if profile else "Quality"))
    token_maximum = 64 if _is_gguf_mode(mode) else 16
    max_new_tokens = _clamp_int(
        payload.get("max_new_tokens"),
        default=int(profile_settings.get("default_max_new_tokens") or 4),
        minimum=1,
        maximum=token_maximum,
    )
    mode_key = _chat_mode_key(mode)
    if mode_key.startswith("quick"):
        max_new_tokens = 1
    if mode_key.startswith("agent") and not _is_gguf_mode(mode):
        max_new_tokens = min(max_new_tokens, AGENT_MODE_MAX_NEW_TOKENS)
    proven_max_tokens = _direct_runtime_proven_max_tokens(model_id)
    if (
        proven_max_tokens is not None
        and not _is_gguf_mode(mode)
        and max_new_tokens > proven_max_tokens
        and not bool(payload.get("allow_experimental_direct_tokens"))
        and not bool(payload.get("allow_experimental_32b_tokens"))
    ):
        max_new_tokens = proven_max_tokens
    min_new_tokens = _clamp_int(payload.get("min_new_tokens"), default=1, minimum=1, maximum=max_new_tokens)
    repetition_penalty = float(payload.get("repetition_penalty") or 1.1)
    return model_id, prompt, profile, mode, max_new_tokens, min_new_tokens, repetition_penalty


def _chat_mode_key(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized.startswith("direct "):
        return normalized.removeprefix("direct ").strip()
    return normalized


def _is_gguf_mode(mode: str) -> bool:
    return _chat_mode_key(mode).startswith("gguf")


def _is_agent_mode(mode: str) -> bool:
    return _chat_mode_key(mode).startswith("agent")


def _agent_warm_runner_enabled(mode: str) -> bool:
    if not _is_agent_mode(mode):
        return False
    settings = _apply_saved_runtime_settings()
    return _normalize_agent_warm_runner_mode(settings.get("agent_warm_runner")) in {"safe", "experimental"}


def _chat_cache_key_for_runtime_defaults(
    payload: dict,
    *,
    model_id: str,
    profile: Any | None,
    mode: str,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    repetition_penalty: float,
) -> str:
    return _chat_response_cache_key(
        {
            **payload,
            "model_id": model_id,
            "profile_id": "" if profile is None else profile.profile_id,
            "mode": mode,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
    )


def _instant_chat_response_for_payload(payload: dict) -> dict | None:
    model_id, prompt, profile, mode, max_new_tokens, min_new_tokens, repetition_penalty = _chat_request_runtime_defaults(payload)
    local_answer = _local_runtime_context_answer(prompt, model_id, mode, profile)
    if local_answer is not None:
        local_answer["job_fast_path"] = "local-runtime-context"
        return local_answer
    if not _is_gguf_mode(mode):
        memory_guard = _memory_guard_response(prompt, model_id, mode, profile)
        if memory_guard is not None:
            memory_guard["job_fast_path"] = "memory-guard"
            return memory_guard
    cache_key = _chat_cache_key_for_runtime_defaults(
        payload,
        model_id=model_id,
        profile=profile,
        mode=mode,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    cached_response = _get_cached_chat_response(cache_key, record_miss=False)
    if cached_response is not None:
        cached_response["job_fast_path"] = "response-cache"
        return cached_response
    return None


def _run_chat_payload(payload: dict, should_cancel=None) -> dict:
    model_id, prompt, profile, mode, max_new_tokens, min_new_tokens, repetition_penalty = _chat_request_runtime_defaults(payload)
    session_id = _normalize_session_id(payload.get("session_id"))
    local_answer = _local_runtime_context_answer(prompt, model_id, mode, profile)
    if local_answer is not None:
        return local_answer
    if not _is_gguf_mode(mode):
        memory_guard = _memory_guard_response(prompt, model_id, mode, profile)
        if memory_guard is not None:
            return memory_guard
    cache_key = _chat_cache_key_for_runtime_defaults(
        payload,
        model_id=model_id,
        profile=profile,
        mode=mode,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    cached_response = _get_cached_chat_response(cache_key)
    if cached_response is not None:
        return cached_response
    if _is_gguf_mode(mode):
        backend_status = build_gguf_backend_status(model_id).to_dict()
        server = dict(backend_status.get("llama_server") or {})
        if (
            bool(backend_status.get("ready"))
            and "llama_server" in backend_status
            and not bool(server.get("ready"))
            and not bool(payload.get("allow_cold_gguf_load"))
        ):
            return _gguf_load_required_response(
                prompt,
                model_id,
                mode,
                profile,
                backend_status=backend_status,
                session_id=session_id,
            )
        started = time.perf_counter()
        system_prompt = _runtime_identity_system_prompt(
            model_id,
            mode,
            profile,
            base_prompt=str(payload.get("system_prompt") or DEFAULT_WEB_SYSTEM_PROMPT),
        )
        effective_prompt, conversation_turn_count, preformatted_chat = _gguf_chat_prompt(model_id, payload, prompt, system_prompt)
        gguf_result = run_gguf_prompt(
            model_id,
            effective_prompt,
            max_tokens=max_new_tokens,
            n_ctx=2048,
            n_threads=runtime_torch_thread_count(),
            stop_strings=list(payload.get("stop_strings") or ["<|im_end|>", "<|im_start|>"]),
        )
        metadata_started = time.perf_counter()
        phase_started = time.perf_counter()
        runtime_settings = _runtime_settings_payload()
        runtime_settings_elapsed = round(time.perf_counter() - phase_started, 2)
        phase_started = time.perf_counter()
        speed_status = _speed_status_payload(model_id)
        speed_status_elapsed = round(time.perf_counter() - phase_started, 2)
        server_ready = gguf_result.backend == "llama-cpp-gguf-server"
        gguf_model_path = getattr(gguf_result, "model_path", None)
        server_status = {
            "running": server_ready,
            "ready": server_ready,
            "pid": None,
            "url": None,
            "model_id": model_id,
            "model_path": None if gguf_model_path is None else str(gguf_model_path),
            "working_set_bytes": None,
            "working_set_gb": None,
            "blockers": list(gguf_result.blockers),
            "summary": "GGUF server answered this prompt." if server_ready else "GGUF prompt did not use the persistent server.",
        }
        server_status_elapsed = 0.0
        metadata_elapsed = round(time.perf_counter() - metadata_started, 2)
        timings = dict(gguf_result.timings)
        timings["web_metadata"] = metadata_elapsed
        timings["web_runtime_settings"] = runtime_settings_elapsed
        timings["web_speed_status"] = speed_status_elapsed
        timings["web_server_status"] = server_status_elapsed
        gguf_generation_speed = _gguf_generation_speed_payload(timings)
        response = {
            "ready": bool(gguf_result.ready),
            "generated_text": gguf_result.generated_text,
            "full_text": f"{prompt}\n{gguf_result.generated_text}",
            "generated_token_ids": [],
            "prompt_token_count": 0,
            "steps_completed": 0,
            "max_new_tokens": max_new_tokens,
            "stop_reason": "gguf-complete" if gguf_result.ready else "gguf-blocked",
            "strategy": gguf_result.backend,
            "cache_sequence_lengths": {},
            "blockers": list(gguf_result.blockers),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "timings": timings,
            "performance_summary": _runtime_performance_summary(timings),
            "generation_speed": gguf_generation_speed,
            "prefix_reuse": {"enabled": False, "used": False, "reason": "gguf-backend"},
            "reusable_token_count": 0,
            "runtime_settings": runtime_settings,
            "conversation_turn_count": conversation_turn_count,
            "preformatted_chat": preformatted_chat,
            "profile_id": None if profile is None else profile.profile_id,
            "profile_label": None if profile is None else profile.label,
            "runtime_context": _runtime_context_payload(model_id, mode, profile),
            "speed_status": speed_status,
            "model_guardrails": _direct_model_guardrails(model_id, max_new_tokens),
            "gguf_backend": {
                **backend_status,
                "model_id": model_id,
                "ready": bool(gguf_result.ready),
                "llama_server_available": gguf_result.backend == "llama-cpp-gguf-server",
                "llama_server": server_status,
                "blockers": list(gguf_result.blockers),
                "summary": "GGUF prompt completed through the local backend." if gguf_result.ready else "GGUF prompt was blocked.",
            },
            "conversation_state": _conversation_state_payload(
                conversation_turn_count,
                preformatted_chat,
                session_id=session_id,
                prefix_reuse={"enabled": False, "used": False, "reason": "gguf-backend"},
            ),
            "response_reuse": _cache_reuse_payload(False, cache_key=cache_key),
        }
        _store_chat_response(cache_key, response)
        return response
    system_prompt = _runtime_identity_system_prompt(
        model_id,
        mode,
        profile,
        base_prompt=str(payload.get("system_prompt") or DEFAULT_WEB_SYSTEM_PROMPT),
    )
    effective_prompt, conversation_turn_count, preformatted_chat = _formatted_chat_prompt(
        model_id,
        payload.get("messages"),
        prompt,
        system_prompt=system_prompt,
    )
    layer_count = _chat_layer_count_for_request(model_id, mode, max_new_tokens)
    if _agent_warm_runner_enabled(mode):
        if not preformatted_chat and _supports_im_chat_tokens(model_id):
            effective_prompt, conversation_turn_count, preformatted_chat = _formatted_chat_prompt(
                model_id,
                [{"role": "system", "text": system_prompt}],
                prompt,
                system_prompt=system_prompt,
            )
        started = time.perf_counter()
        warm_result = run_warm_agent_prompt(
            model_id,
            effective_prompt,
            mode=mode,
            session_id=session_id,
            max_new_tokens=max_new_tokens,
            apply_chat_format=not preformatted_chat,
        )
        elapsed_seconds = round(time.perf_counter() - started, 2)
        response = {
            "ready": bool(warm_result.ready),
            "generated_text": warm_result.generated_text,
            "full_text": f"{prompt}\n{warm_result.generated_text}",
            "generated_token_ids": list(warm_result.generated_token_ids),
            "prompt_token_count": 0,
            "steps_completed": warm_result.steps_completed,
            "max_new_tokens": warm_result.max_new_tokens,
            "stop_reason": "warm-agent-complete" if warm_result.ready else "warm-agent-blocked",
            "strategy": "warm-agent-runner",
            "cache_sequence_lengths": {},
            "blockers": list(warm_result.blockers),
            "elapsed_seconds": elapsed_seconds,
            "timings": {},
            "performance_summary": dict(warm_result.performance_summary),
            "prefix_reuse": dict(warm_result.prefix_reuse),
            "reusable_token_count": int(warm_result.runner_status.get("reusable_token_count", 0)),
            "runtime_settings": _runtime_settings_payload(),
            "conversation_turn_count": conversation_turn_count,
            "preformatted_chat": preformatted_chat,
            "profile_id": None if profile is None else profile.profile_id,
            "profile_label": None if profile is None else profile.label,
            "runtime_context": _runtime_context_payload(model_id, mode, profile),
            "speed_status": _speed_status_payload(model_id),
            "model_guardrails": _direct_model_guardrails(model_id, max_new_tokens),
            "warm_runner": dict(warm_result.runner_status),
            "conversation_state": _conversation_state_payload(
                conversation_turn_count,
                preformatted_chat,
                session_id=session_id,
                prefix_reuse=warm_result.prefix_reuse,
            ),
            "response_reuse": _cache_reuse_payload(False, cache_key=cache_key),
        }
        _store_chat_response(cache_key, response)
        return response
    profile_id = "" if profile is None else profile.profile_id
    session_prefix = _get_session_prefix(
        session_id,
        model_id=model_id,
        profile_id=profile_id,
        mode=mode,
        layer_count=layer_count,
    )
    started = time.perf_counter()
    result = run_prompt_decode_loop(
        model_id,
        prompt=effective_prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        layer_count=layer_count,
        repetition_penalty=repetition_penalty,
        system_prompt=system_prompt,
        apply_chat_format=not preformatted_chat,
        stop_strings=list(payload.get("stop_strings") or ["<|im_end|>", "<|im_start|>"]),
        should_cancel=should_cancel,
        initial_decode_state=None if session_prefix is None else session_prefix.decode_state,
        initial_token_ids=None if session_prefix is None else session_prefix.token_ids,
    )
    response = _prompt_result_payload(result, round(time.perf_counter() - started, 2))
    response["conversation_turn_count"] = conversation_turn_count
    response["preformatted_chat"] = preformatted_chat
    response["profile_id"] = None if profile is None else profile.profile_id
    response["profile_label"] = None if profile is None else profile.label
    response["runtime_context"] = _runtime_context_payload(model_id, mode, profile)
    response["speed_status"] = _speed_status_payload(model_id)
    response["model_guardrails"] = _direct_model_guardrails(model_id, max_new_tokens)
    response["conversation_state"] = _conversation_state_payload(
        conversation_turn_count,
        preformatted_chat,
        session_id=session_id,
        prefix_reuse=response.get("prefix_reuse"),
    )
    response["response_reuse"] = _cache_reuse_payload(False, cache_key=cache_key)
    _store_session_prefix(
        session_id,
        model_id=model_id,
        profile_id=profile_id,
        mode=mode,
        layer_count=layer_count,
        token_ids=list(getattr(result, "reusable_token_ids", []) or []),
        decode_state=getattr(result, "final_decode_state", None),
    )
    _store_chat_response(cache_key, response)
    return response


def _chat_job_worker(job_id: str) -> None:
    job = _get_chat_job(job_id)
    if job is None:
        return
    with _CHAT_JOBS_LOCK:
        if job.cancel_requested:
            job.status = "canceled"
            job.updated_at = time.time()
            return
        job.status = "running"
        job.updated_at = time.time()
    try:
        result = _run_chat_payload(job.payload, should_cancel=lambda: job.cancel_requested)
    except Exception as exc:  # pragma: no cover - defensive background path
        with _CHAT_JOBS_LOCK:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = time.time()
        return
    with _CHAT_JOBS_LOCK:
        if job.cancel_requested:
            job.status = "canceled"
            job.result = None
        else:
            job.status = "completed"
            job.result = result
        job.updated_at = time.time()


def _start_chat_job(payload: dict) -> ChatJob:
    _cleanup_chat_jobs()
    job = ChatJob(job_id=uuid.uuid4().hex, payload=dict(payload))
    instant_response = _instant_chat_response_for_payload(payload)
    if instant_response is not None:
        job.status = "completed"
        job.result = instant_response
        job.updated_at = time.time()
        with _CHAT_JOBS_LOCK:
            _CHAT_JOBS[job.job_id] = job
        return job
    with _CHAT_JOBS_LOCK:
        _CHAT_JOBS[job.job_id] = job
    threading.Thread(target=_chat_job_worker, args=(job.job_id,), daemon=True).start()
    return job


def _direct_runtime_state(status: Any, latest_benchmark: dict | None) -> dict:
    benchmark_ready = bool(latest_benchmark and latest_benchmark.get("ready"))
    stream_ready = str(status.streaming_status).strip().lower() in {"advancing", "ready", "active"}
    direct_ready = benchmark_ready or stream_ready
    if direct_ready:
        summary = "Direct Pocket runtime is working. Full RAM preflight can still warn on weak hardware."
        state = "Working"
    else:
        summary = status.runtime_summary
        state = status.runtime_status
    return {
        "status": state,
        "summary": summary,
        "preflight_status": status.runtime_status,
        "preflight_summary": status.runtime_summary,
    }


def _runtime_settings_payload() -> dict:
    saved_settings = _apply_saved_runtime_settings()
    agent_warm_runner = _normalize_agent_warm_runner_mode(saved_settings.get("agent_warm_runner"))
    return {
        "math_dtype": runtime_math_dtype_name(),
        "torch_threads": runtime_torch_thread_count(),
        "lm_head_chunk_rows": DEFAULT_LM_HEAD_CHUNK_ROWS,
        "saved_runtime_settings": saved_settings,
        "agent_warm_runner": agent_warm_runner,
        "tensor_residency_policy": tensor_residency_policy_snapshot(),
        "tensor_residency": tensor_residency_stats().to_dict(),
        "tensor_load": tensor_load_stats_snapshot().to_dict(),
        "response_cache": _chat_response_cache_status(),
        "session_prefix_cache": _session_prefix_cache_status(),
    }


def _speed_status_payload(model_id: str) -> dict:
    _apply_saved_runtime_settings()
    load_stats = tensor_load_stats_snapshot().to_dict()
    residency_stats = tensor_residency_stats().to_dict()
    pack_selection = runtime_pack_selection_snapshot(model_id)
    shard_opens = int(load_stats.get("shard_opens", 0))
    artifact_hits = int(load_stats.get("artifact_tensor_hits", 0))
    resident_hits = int(residency_stats.get("hits", 0))
    if shard_opens > 0 and resident_hits + artifact_hits > shard_opens:
        status = "Improving"
        summary = "Pocket LLM is avoiding repeated disk loads with resident tensors and a runtime pack."
    elif pack_selection.get("selected"):
        status = "Pack ready"
        summary = "A runtime pack is selected, but disk loading is still the main speed bottleneck."
    else:
        status = "Baseline"
        summary = "No runtime pack is active yet, so disk loading will stay slow."
    return {
        "status": status,
        "summary": summary,
        "pack": pack_selection,
        "residency_policy": tensor_residency_policy_snapshot(),
        "shard_opens": shard_opens,
        "artifact_tensor_hits": artifact_hits,
        "resident_tensor_hits": resident_hits,
        "resident_tensor_count": residency_stats.get("resident_count", 0),
        "resident_mb": round(float(residency_stats.get("resident_bytes", 0)) / (1024**2), 2),
        "loaded_mb": load_stats.get("loaded_mb", 0.0),
        "response_cache": _chat_response_cache_status(),
        "session_prefix_cache": _session_prefix_cache_status(),
        "model_guardrails": _direct_model_guardrails(model_id),
    }


def _qwen14b_speed_target_payload(model_id: str) -> dict:
    target = {"min_seconds_per_token": 2, "max_seconds_per_token": 4}
    if not _is_qwen_14b_model(model_id):
        return {
            "applies": False,
            "status": "not-qwen-14b",
            "target": target,
            "summary": "The current speed target is scoped to Qwen 14B only.",
        }
    direct_baseline = _direct_runtime_baseline(model_id, 1) or {}
    gguf = build_gguf_backend_status(model_id).to_dict()
    server = dict(gguf.get("llama_server") or {})
    load_estimate = dict(gguf.get("load_estimate") or {})
    if server.get("ready"):
        status = "target-path-loaded"
        summary = "Fast chat path is loaded. Use GGUF for normal Qwen 14B chat."
    elif gguf.get("ready"):
        status = "load-fast-path"
        summary = "Fast chat path is available but not loaded. First GGUF prompt may cold-load, then normal chat should be much faster than direct runtime."
    else:
        status = "blocked"
        summary = "Fast Qwen 14B path is not ready because the GGUF runtime or artifact is missing."
    return {
        "applies": True,
        "status": status,
        "target": target,
        "default_chat_mode": "GGUF",
        "slow_direct_seconds_per_token": (
            None
            if not direct_baseline
            else round(float(direct_baseline["measured_seconds"]) / max(1, int(direct_baseline["measured_tokens"])), 2)
        ),
        "gguf_server_ready": bool(server.get("ready")),
        "gguf_server_running": bool(server.get("running")),
        "estimated_cold_load_seconds": load_estimate.get("estimated_cold_load_seconds"),
        "expected_ram_mb": load_estimate.get("expected_ram_mb"),
        "summary": summary,
    }


def _status_payload() -> dict:
    options = list_status_screen_options()
    selected = options[0] if options else None
    model_id = selected.model_id if selected else "qwen2.5-14b-instruct"
    model_dir = selected.model_dir if selected else original_model_root(model_id)
    status = build_status_screen_model(model_id=model_id, model_dir=model_dir)
    latest_benchmark = _latest_json(benchmarks_root(model_id) / "latest.measured-benchmark.json")
    latest_backend_comparison = _latest_measured_benchmark_by_scope(model_id, "backend-comparison") or latest_benchmark
    latest_artifact = latest_optimized_artifact_manifest(model_id)
    direct_runtime = _direct_runtime_state(status, latest_benchmark)
    backend_report = build_runtime_backend_report(model_id)
    backend_report_payload = backend_report.to_dict()
    recommended_backend_id = str(
        getattr(backend_report, "recommended_backend_id", None)
        or backend_report_payload.get("recommended_backend_id")
        or "direct-cpu"
    )
    return {
        "project_root": str(project_root()),
        "active_model": {
            "model_id": status.model_id,
            "label": status.model_label,
            "family": status.family_label,
            "runtime_status": status.runtime_status,
            "effective_runtime_status": direct_runtime["status"],
            "streaming_status": status.streaming_status,
            "model_dir": str(status.model_dir),
            "summary": status.runtime_summary,
            "effective_summary": direct_runtime["summary"],
            "blockers": list(status.blockers),
            "warnings": list(status.warnings),
        },
        "direct_runtime": direct_runtime,
        "engine_decision": select_runtime_engine(model_id).to_dict(),
        "backend_report": backend_report_payload,
        "backend_comparison": build_backend_comparison_record(
            latest_backend_comparison,
            recommended_backend_id=recommended_backend_id,
        ),
        "gguf_backend": build_gguf_backend_status(model_id).to_dict(),
        "warm_runner": warm_runner_status(model_id),
        "runtime_settings": _runtime_settings_payload(),
        "models": [
            {
                "model_id": option.model_id,
                "label": option.model_label,
                "family": option.family_label,
                "runtime_status": option.family_runtime_status,
                "source": option.source_label,
                "path": str(option.model_dir),
                "registered": option.registered,
            }
            for option in options
        ],
        "profiles": [profile.to_dict() for profile in list_saved_profiles(model_id)],
        "profile_compare": build_profile_compare_summary(model_id, latest_benchmark),
        "optimized_artifact": None if latest_artifact is None else latest_artifact.to_dict(),
        "speed_status": _speed_status_payload(model_id),
        "qwen14b_speed_target": _qwen14b_speed_target_payload(model_id),
        "model_guardrails": _direct_model_guardrails(model_id),
        "downloads": _download_status_payload(),
        "benchmark": latest_benchmark,
        "benchmark_history": build_measured_benchmark_history(model_id),
    }


def _prompt_result_payload(result: Any, elapsed_seconds: float | None = None) -> dict:
    timings = dict(getattr(result, "timings", {}) or {})
    return {
        "ready": bool(result.ready),
        "generated_text": result.generated_text,
        "full_text": result.full_text,
        "generated_token_ids": list(result.generated_token_ids),
        "prompt_token_count": len(result.prompt_token_ids),
        "steps_completed": result.steps_completed,
        "max_new_tokens": result.max_new_tokens,
        "stop_reason": result.stop_reason,
        "strategy": result.strategy,
        "cache_sequence_lengths": dict(result.cache_sequence_lengths),
        "blockers": list(result.blockers),
        "elapsed_seconds": elapsed_seconds,
        "timings": timings,
        "performance_summary": _runtime_performance_summary(timings),
        "prefix_reuse": dict(getattr(result, "prefix_reuse", {}) or {}),
        "reusable_token_count": len(getattr(result, "reusable_token_ids", []) or []),
        "runtime_settings": _runtime_settings_payload(),
    }


def _json_response(handler: BaseHTTPRequestHandler, status_code: int, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


class PocketLLMRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for static assets and local runtime API routes."""

    server_version = "PocketLLM/0.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            _json_response(self, 200, _status_payload())
            return
        if parsed.path == "/api/chat/status":
            job_id = parse_qs(parsed.query).get("job_id", [""])[0]
            job = _get_chat_job(job_id)
            if job is None:
                _json_response(self, 404, {"error": "Chat job not found."})
                return
            _json_response(self, 200, _chat_job_payload(job))
            return
        self._serve_static()

    def do_POST(self) -> None:
        try:
            payload = _read_json_body(self)
            if self.path == "/api/chat":
                self._handle_chat(payload)
                return
            if self.path == "/api/chat/start":
                self._handle_chat_start(payload)
                return
            if self.path == "/api/chat/cancel":
                self._handle_chat_cancel(payload)
                return
            if self.path == "/api/benchmark":
                self._handle_benchmark(payload)
                return
            if self.path == "/api/benchmark/gguf":
                self._handle_gguf_benchmark(payload)
                return
            if self.path == "/api/benchmark/comparison":
                self._handle_backend_comparison_benchmark(payload)
                return
            if self.path == "/api/settings/runtime":
                self._handle_runtime_settings(payload)
                return
            if self.path == "/api/warm-runner":
                self._handle_warm_runner(payload)
                return
            if self.path == "/api/gguf/server":
                self._handle_gguf_server(payload)
                return
            _json_response(self, 404, {"error": "Unknown API route."})
        except Exception as exc:  # pragma: no cover - guard for UI calls
            _json_response(self, 500, {"error": str(exc)})

    def _handle_chat(self, payload: dict) -> None:
        try:
            response = _run_chat_payload(payload)
        except ValueError as exc:
            _json_response(self, 400, {"error": str(exc)})
            return
        _json_response(self, 200, response)

    def _handle_chat_start(self, payload: dict) -> None:
        if not str(payload.get("prompt") or "").strip():
            _json_response(self, 400, {"error": "Prompt is required."})
            return
        job = _start_chat_job(payload)
        _json_response(self, 202, _chat_job_payload(job))

    def _handle_chat_cancel(self, payload: dict) -> None:
        job_id = str(payload.get("job_id") or "")
        job = _cancel_chat_job(job_id)
        if job is None:
            _json_response(self, 404, {"error": "Chat job not found."})
            return
        _json_response(self, 200, _chat_job_payload(job))

    def _handle_benchmark(self, payload: dict) -> None:
        model_id = str(payload.get("model_id") or "qwen2.5-14b-instruct")
        model_dir = original_model_root(model_id)
        run = run_measured_benchmark(model_id, model_dir)
        _json_response(self, 200, run.to_dict())

    def _handle_gguf_benchmark(self, payload: dict) -> None:
        model_id = str(payload.get("model_id") or "qwen2.5-14b-instruct")
        model_dir = original_model_root(model_id)
        run = run_gguf_measured_benchmark(model_id, model_dir)
        _json_response(self, 200, run.to_dict())

    def _handle_backend_comparison_benchmark(self, payload: dict) -> None:
        model_id = str(payload.get("model_id") or "qwen2.5-14b-instruct")
        model_dir = original_model_root(model_id)
        run = run_backend_comparison_benchmark(model_id, model_dir)
        _json_response(self, 200, run.to_dict())

    def _handle_runtime_settings(self, payload: dict) -> None:
        _json_response(self, 200, _update_runtime_settings(payload))

    def _handle_warm_runner(self, payload: dict) -> None:
        _json_response(self, 200, _warm_runner_control_payload(payload))

    def _handle_gguf_server(self, payload: dict) -> None:
        model_id = str(payload.get("model_id") or "qwen2.5-14b-instruct")
        action = str(payload.get("action") or "status").strip().lower()
        if action == "start":
            status = start_gguf_server(
                model_id,
                n_ctx=_clamp_int(payload.get("n_ctx"), default=2048, minimum=256, maximum=8192),
                n_threads=runtime_torch_thread_count(),
            )
        elif action in {"stop", "unload"}:
            status = stop_gguf_server(model_id)
            _clear_chat_response_cache()
            _clear_session_prefix_cache()
        else:
            status = build_gguf_backend_status(model_id).llama_server
            _json_response(self, 200, {"gguf_server": status, "gguf_backend": build_gguf_backend_status(model_id).to_dict()})
            return
        _json_response(
            self,
            200,
            {
                "gguf_server": status.to_dict(),
                "gguf_backend": build_gguf_backend_status(model_id).to_dict(),
            },
        )

    def _serve_static(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"", "/"}:
            path = "/index.html"
        asset_path = (STATIC_ROOT / path.lstrip("/")).resolve()
        if STATIC_ROOT.resolve() not in asset_path.parents and asset_path != STATIC_ROOT.resolve():
            self.send_error(403)
            return
        if not asset_path.exists() or not asset_path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
        data = asset_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def create_server(port: int | None = None) -> ThreadingHTTPServer:
    """Create the local Pocket LLM web server."""
    return ThreadingHTTPServer(("127.0.0.1", port or _find_free_port()), PocketLLMRequestHandler)


def _existing_server_is_pocket_llm(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return False
    return "active_model" in payload and "project_root" in payload


def _acquire_single_instance_lock() -> bool:
    global _INSTANCE_LOCK_SOCKET
    if _INSTANCE_LOCK_SOCKET is not None:
        return True
    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock_socket.bind(("127.0.0.1", SINGLE_INSTANCE_LOCK_PORT))
        lock_socket.listen(1)
    except OSError:
        lock_socket.close()
        return False
    _INSTANCE_LOCK_SOCKET = lock_socket
    return True


def _release_single_instance_lock() -> None:
    global _INSTANCE_LOCK_SOCKET
    if _INSTANCE_LOCK_SOCKET is not None:
        _INSTANCE_LOCK_SOCKET.close()
        _INSTANCE_LOCK_SOCKET = None


def run() -> None:
    """Launch the local web UI."""
    _apply_saved_runtime_settings()
    if not _acquire_single_instance_lock():
        webbrowser.open(f"http://127.0.0.1:{DEFAULT_WEB_PORT}/")
        return
    try:
        server = create_server(DEFAULT_WEB_PORT)
    except OSError:
        if _existing_server_is_pocket_llm(DEFAULT_WEB_PORT):
            webbrowser.open(f"http://127.0.0.1:{DEFAULT_WEB_PORT}/")
            return
        server = create_server()
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        _release_single_instance_lock()


if __name__ == "__main__":
    run()
