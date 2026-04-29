"""Progressive runtime crash-localization diagnostics."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any, Callable

from pcketlm.core.runtime import (
    load_layer_bridge_config,
    load_tensor_by_name,
    load_token_entry_hidden_state,
    run_prompt_decode_loop,
    run_token_entry_layer_bridge,
)
from pcketlm.core.runtime.load_attempt import _memory_snapshot
from pcketlm.core.storage.paths import original_model_root


VALID_SLICES = {
    "load-config",
    "embedding-only",
    "embed-forward",
    "layer-0",
    "layer-0-1",
    "layer-0-7",
    "layer-0-15",
    "full",
}


class _ProcessMemoryCounters(ctypes.Structure):
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


def _bytes_to_mb(value: int) -> int:
    return int(round(value / (1024**2)))


def _working_set_mb() -> int:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return -1
    return _bytes_to_mb(int(counters.WorkingSetSize))


def _ram_payload() -> dict:
    memory = _memory_snapshot()
    return {
        "free_ram_mb": _bytes_to_mb(int(memory.free_bytes)),
        "total_ram_mb": _bytes_to_mb(int(memory.total_bytes)),
        "process_working_set_mb": _working_set_mb(),
    }


def _emit(event: str, *, model_id: str, slice_name: str, operation: str, started_at: float, **extra: Any) -> None:
    payload = {
        "event": event,
        "model_id": model_id,
        "slice": slice_name,
        "operation": operation,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        **_ram_payload(),
        **extra,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def _run_checkpoint(
    *,
    model_id: str,
    slice_name: str,
    operation: str,
    started_at: float,
    callback: Callable[[], dict],
) -> dict:
    _emit("before", model_id=model_id, slice_name=slice_name, operation=operation, started_at=started_at)
    op_started = time.perf_counter()
    result = callback()
    _emit(
        "after",
        model_id=model_id,
        slice_name=slice_name,
        operation=operation,
        started_at=started_at,
        operation_seconds=round(time.perf_counter() - op_started, 3),
        result=result,
    )
    return result


def _load_tokenizer_token_id(model_id: str, text: str = "hello") -> int:
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(original_model_root(model_id), local_files_only=True)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError(f"Tokenizer produced no token ids for {text!r}.")
    return int(token_ids[0])


def _load_config(model_id: str) -> dict:
    config = load_layer_bridge_config(model_id)
    return {
        "ready": config.ready,
        "blockers": list(config.blockers),
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "source_dtype": config.source_dtype,
    }


def _embedding_only(model_id: str) -> dict:
    loaded = load_tensor_by_name(model_id, "model.embed_tokens.weight")
    return loaded.to_dict()


def _embed_forward(model_id: str) -> dict:
    token_id = _load_tokenizer_token_id(model_id)
    hidden_state, blockers = load_token_entry_hidden_state(model_id, [token_id])
    return {
        "ready": hidden_state is not None and not blockers,
        "token_id": token_id,
        "shape": [] if hidden_state is None else [int(value) for value in hidden_state.shape],
        "dtype": None if hidden_state is None else str(hidden_state.dtype),
        "blockers": list(blockers),
    }


def _layer_forward(model_id: str, layer_count: int) -> dict:
    token_id = _load_tokenizer_token_id(model_id)
    result = run_token_entry_layer_bridge(model_id, token_ids=[token_id], start_layer=0, layer_count=layer_count)
    payload = result.to_dict()
    payload["token_id"] = token_id
    payload.pop("output_tensor", None)
    return payload


def _full_forward(model_id: str) -> dict:
    result = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        max_new_tokens=1,
        selection_policy="greedy",
    )
    payload = result.to_dict()
    payload.pop("final_decode_state", None)
    return payload


def _operation_for_slice(slice_name: str) -> tuple[str, Callable[[str], dict]]:
    if slice_name == "load-config":
        return "load-config", _load_config
    if slice_name == "embedding-only":
        return "load-embedding-tensor", _embedding_only
    if slice_name == "embed-forward":
        return "embedding-lookup-hello", _embed_forward
    if slice_name == "layer-0":
        return "layers-0-through-0", lambda model_id: _layer_forward(model_id, 1)
    if slice_name == "layer-0-1":
        return "layers-0-through-1", lambda model_id: _layer_forward(model_id, 2)
    if slice_name == "layer-0-7":
        return "layers-0-through-7", lambda model_id: _layer_forward(model_id, 8)
    if slice_name == "layer-0-15":
        return "layers-0-through-15", lambda model_id: _layer_forward(model_id, 16)
    if slice_name == "full":
        return "full-prompt-decode", _full_forward
    raise ValueError(f"Unknown slice {slice_name!r}.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Pocket LLM runtime crash slices.")
    parser.add_argument("--model", required=True, help="Model id, for example qwen2.5-32b-instruct")
    parser.add_argument("--slice", required=True, choices=sorted(VALID_SLICES), help="Progressive slice to run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_id = str(args.model)
    slice_name = str(args.slice)
    started_at = time.perf_counter()
    _emit(
        "start",
        model_id=model_id,
        slice_name=slice_name,
        operation="diagnostic",
        started_at=started_at,
        pid=os.getpid(),
        model_dir=str(original_model_root(model_id)),
    )
    operation, callback = _operation_for_slice(slice_name)
    try:
        result = _run_checkpoint(
            model_id=model_id,
            slice_name=slice_name,
            operation=operation,
            started_at=started_at,
            callback=lambda: callback(model_id),
        )
    except Exception as exc:
        _emit(
            "python-error",
            model_id=model_id,
            slice_name=slice_name,
            operation=operation,
            started_at=started_at,
            error=repr(exc),
        )
        raise
    _emit(
        "complete",
        model_id=model_id,
        slice_name=slice_name,
        operation="diagnostic",
        started_at=started_at,
        ready=bool(result.get("ready", False)),
        blockers=list(result.get("blockers", [])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
