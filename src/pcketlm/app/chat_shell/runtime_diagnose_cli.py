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

import torch

from pcketlm.core.runtime import (
    DEFAULT_LM_HEAD_CHUNK_ROWS,
    KVDecodeState,
    load_layer_bridge_config,
    load_tensor_by_name,
    load_token_entry_hidden_state,
    run_decode_tail,
    run_kv_decode_step,
    run_layer_bridge_stack,
    run_prompt_decode_loop,
    run_token_entry_layer_bridge,
)
from pcketlm.core.runtime.layer_bridge import _can_select_from_topk, select_next_token, select_next_token_from_topk
from pcketlm.core.runtime.load_attempt import _memory_snapshot
from pcketlm.core.runtime.tensor_loader import reset_tensor_load_stats, tensor_load_stats_snapshot
from pcketlm.core.runtime.tokenizer_runtime import (
    decode_token_ids_to_text,
    load_generation_settings,
    prepare_prompt_text,
)
from pcketlm.core.storage.paths import original_model_root


VALID_SLICES = {
    "load-config",
    "embedding-only",
    "embed-forward",
    "layer-0",
    "layer-0-1",
    "layer-0-7",
    "layer-0-15",
    "layer-0-31",
    "layer-0-39",
    "all-layers",
    "all-layers-norm",
    "all-layers-norm-lm",
    "prompt-prefill",
    "decode-step-1",
    "decode-step-2",
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


def _process_memory_payload() -> dict:
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
        return {"process_working_set_mb": -1, "process_peak_working_set_mb": -1}
    return {
        "process_working_set_mb": _bytes_to_mb(int(counters.WorkingSetSize)),
        "process_peak_working_set_mb": _bytes_to_mb(int(counters.PeakWorkingSetSize)),
    }


def _working_set_mb() -> int:
    return int(_process_memory_payload()["process_working_set_mb"])


def _ram_payload() -> dict:
    memory = _memory_snapshot()
    process_payload = _process_memory_payload()
    return {
        "free_ram_mb": _bytes_to_mb(int(memory.free_bytes)),
        "total_ram_mb": _bytes_to_mb(int(memory.total_bytes)),
        "process_working_set_mb": _working_set_mb(),
        "process_peak_working_set_mb": int(process_payload["process_peak_working_set_mb"]),
    }


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.nelement() * tensor.element_size())


def _kv_cache_summary(kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] | None) -> dict:
    caches = {} if kv_caches is None else kv_caches
    total_bytes = 0
    layer_payload: dict[str, dict[str, Any]] = {}
    for layer_index, pair in sorted(caches.items()):
        key_tensor, value_tensor = pair
        pair_bytes = _tensor_bytes(key_tensor) + _tensor_bytes(value_tensor)
        total_bytes += pair_bytes
        layer_payload[str(layer_index)] = {
            "key_shape": [int(value) for value in key_tensor.shape],
            "value_shape": [int(value) for value in value_tensor.shape],
            "bytes": int(pair_bytes),
        }
    return {
        "kv_cache_layers": len(caches),
        "kv_cache_total_bytes": int(total_bytes),
        "kv_cache_total_mb": _bytes_to_mb(total_bytes),
        "kv_cache_by_layer": layer_payload,
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


def _load_hello_hidden_checkpoint(model_id: str, slice_name: str, started_at: float) -> tuple[int, torch.Tensor]:
    token_payload = _run_checkpoint(
        model_id=model_id,
        slice_name=slice_name,
        operation="tokenize-hello",
        started_at=started_at,
        callback=lambda: {"ready": True, "token_id": _load_tokenizer_token_id(model_id)},
    )
    token_id = int(token_payload["token_id"])
    embed_payload: dict[str, Any] = {}

    def load_hidden() -> dict:
        hidden, blockers = load_token_entry_hidden_state(model_id, [token_id])
        embed_payload["hidden"] = hidden
        return {
            "ready": hidden is not None and not blockers,
            "token_id": token_id,
            "shape": [] if hidden is None else [int(value) for value in hidden.shape],
            "dtype": None if hidden is None else str(hidden.dtype),
            "blockers": list(blockers),
        }

    result = _run_checkpoint(
        model_id=model_id,
        slice_name=slice_name,
        operation="embedding-lookup-hello",
        started_at=started_at,
        callback=load_hidden,
    )
    hidden_state = embed_payload.get("hidden")
    if hidden_state is None or not result.get("ready", False):
        raise RuntimeError(f"Embedding lookup failed for token id {token_id}: {result.get('blockers', [])}")
    return token_id, hidden_state


def _run_layer_stack_checkpoint(
    model_id: str,
    slice_name: str,
    started_at: float,
    hidden_state: torch.Tensor,
    layer_count: int,
) -> Any:
    stack_holder: dict[str, Any] = {}

    def run_stack() -> dict:
        stack = run_layer_bridge_stack(
            model_id,
            start_layer=0,
            layer_count=layer_count,
            input_hidden=hidden_state,
            return_kv_cache=False,
            collect_step_summaries=False,
            collect_metrics=False,
        )
        stack_holder["stack"] = stack
        payload = stack.to_dict()
        payload.pop("output_tensor", None)
        payload.pop("next_kv_caches", None)
        return payload

    result = _run_checkpoint(
        model_id=model_id,
        slice_name=slice_name,
        operation=f"layers-0-through-{layer_count - 1}",
        started_at=started_at,
        callback=run_stack,
    )
    stack_result = stack_holder.get("stack")
    if stack_result is None or not result.get("ready", False) or stack_result.output_tensor is None:
        raise RuntimeError(f"Layer stack failed: {result.get('blockers', [])}")
    return stack_result


def _final_norm_checkpoint(model_id: str, slice_name: str, started_at: float, hidden_state: torch.Tensor) -> torch.Tensor:
    norm_holder: dict[str, torch.Tensor] = {}

    def run_norm() -> dict:
        config = load_layer_bridge_config(model_id)
        loaded = load_tensor_by_name(model_id, "model.norm.weight")
        blockers = list(config.blockers) + list(loaded.blockers)
        if loaded.tensor is None:
            blockers.append("model.norm.weight did not load.")
            return {"ready": False, "blockers": blockers, "input_shape": [int(value) for value in hidden_state.shape]}
        hidden_float = hidden_state.detach().cpu().float()
        variance = hidden_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_float * torch.rsqrt(variance + config.rms_norm_eps)
        normalized = (normalized * loaded.tensor.float().view(1, 1, -1)).to(dtype=hidden_state.dtype)
        norm_holder["normalized"] = normalized
        return {
            "ready": not blockers,
            "blockers": blockers,
            "input_shape": [int(value) for value in hidden_state.shape],
            "normalized_shape": [int(value) for value in normalized.shape],
            "normalized_dtype": str(normalized.dtype),
        }

    result = _run_checkpoint(
        model_id=model_id,
        slice_name=slice_name,
        operation="final-norm",
        started_at=started_at,
        callback=run_norm,
    )
    normalized = norm_holder.get("normalized")
    if normalized is None or not result.get("ready", False):
        raise RuntimeError(f"Final norm failed: {result.get('blockers', [])}")
    return normalized


def _lm_head_checkpoint(model_id: str, slice_name: str, started_at: float, hidden_state: torch.Tensor) -> dict:
    def run_lm_head() -> dict:
        result = run_decode_tail(
            model_id,
            hidden_state[:, -1:, :],
            return_logits=False,
        )
        payload = result.to_dict()
        payload.pop("logits", None)
        return payload

    return _run_checkpoint(
        model_id=model_id,
        slice_name=slice_name,
        operation="final-norm-and-lm-head",
        started_at=started_at,
        callback=run_lm_head,
    )


def _all_layers_slice(model_id: str, slice_name: str, started_at: float, *, include_norm: bool, include_lm: bool) -> dict:
    config = load_layer_bridge_config(model_id)
    token_id, hidden_state = _load_hello_hidden_checkpoint(model_id, slice_name, started_at)
    stack_result = _run_layer_stack_checkpoint(
        model_id,
        slice_name,
        started_at,
        hidden_state,
        layer_count=int(config.num_hidden_layers),
    )
    payload: dict[str, Any] = {
        "ready": bool(stack_result.ready),
        "token_id": token_id,
        "layer_count": int(config.num_hidden_layers),
        "executed_layers": list(stack_result.executed_layers),
        "output_shape": list(stack_result.output_shape),
        "output_dtype": stack_result.output_dtype,
        "blockers": list(stack_result.blockers),
    }
    if include_norm:
        normalized = _final_norm_checkpoint(model_id, slice_name, started_at, stack_result.output_tensor)
        payload["normalized_shape"] = [int(value) for value in normalized.shape]
        payload["normalized_dtype"] = str(normalized.dtype)
    if include_lm:
        lm_payload = _lm_head_checkpoint(model_id, slice_name, started_at, stack_result.output_tensor)
        payload["lm_head"] = lm_payload
        payload["ready"] = payload["ready"] and bool(lm_payload.get("ready", False))
        payload["blockers"].extend(list(lm_payload.get("blockers", [])))
    return payload


def _prompt_generation_context(model_id: str, slice_name: str, started_at: float) -> dict:
    context_holder: dict[str, Any] = {}

    def prepare() -> dict:
        generation_settings = load_generation_settings(model_id)
        prepared = prepare_prompt_text(model_id, "hello world", apply_chat_format=True)
        config = load_layer_bridge_config(model_id)
        top_k = generation_settings.top_k if generation_settings.ready else 5
        top_p = generation_settings.top_p if generation_settings.ready else 1.0
        temperature = generation_settings.temperature if generation_settings.ready else 1.0
        repetition_penalty = generation_settings.repetition_penalty if generation_settings.ready else 1.0
        context_holder.update(
            {
                "prepared": prepared,
                "config": config,
                "generation_settings": generation_settings,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            }
        )
        return {
            "ready": prepared.ready and config.ready,
            "prepared_prompt_chars": len(prepared.prepared_prompt),
            "prompt_token_count": len(prepared.token_ids),
            "prompt_token_ids": list(prepared.token_ids),
            "generation_settings": generation_settings.to_dict(),
            "num_hidden_layers": config.num_hidden_layers,
            "blockers": list(prepared.blockers) + list(config.blockers),
        }

    payload = _run_checkpoint(
        model_id=model_id,
        slice_name=slice_name,
        operation="prepare-prompt-and-generation",
        started_at=started_at,
        callback=prepare,
    )
    if not payload.get("ready", False):
        raise RuntimeError(f"Prompt setup failed: {payload.get('blockers', [])}")
    return context_holder


def _prompt_prefill_checkpoint(model_id: str, slice_name: str, started_at: float) -> dict:
    context = _prompt_generation_context(model_id, slice_name, started_at)
    prepared = context["prepared"]
    config = context["config"]
    holder: dict[str, Any] = {}

    def load_hidden() -> dict:
        hidden, blockers = load_token_entry_hidden_state(model_id, list(prepared.token_ids))
        holder["prompt_hidden_state"] = hidden
        return {
            "ready": hidden is not None and not blockers,
            "prompt_token_count": len(prepared.token_ids),
            "shape": [] if hidden is None else [int(value) for value in hidden.shape],
            "dtype": None if hidden is None else str(hidden.dtype),
            "blockers": list(blockers),
        }

    hidden_payload = _run_checkpoint(
        model_id=model_id,
        slice_name=slice_name,
        operation="prompt-token-entry",
        started_at=started_at,
        callback=load_hidden,
    )
    prompt_hidden_state = holder.get("prompt_hidden_state")
    if prompt_hidden_state is None or not hidden_payload.get("ready", False):
        raise RuntimeError(f"Prompt token entry failed: {hidden_payload.get('blockers', [])}")

    def run_prefill() -> dict:
        stack = run_layer_bridge_stack(
            model_id,
            start_layer=0,
            layer_count=int(config.num_hidden_layers),
            input_hidden=prompt_hidden_state,
            return_kv_cache=True,
            collect_step_summaries=False,
            collect_metrics=False,
        )
        holder["prefill_stack"] = stack
        payload = stack.to_dict()
        payload.pop("output_tensor", None)
        payload.pop("next_kv_caches", None)
        payload.update(_kv_cache_summary(stack.next_kv_caches))
        return payload

    stack_payload = _run_checkpoint(
        model_id=model_id,
        slice_name=slice_name,
        operation="prompt-prefill-stack-with-kv",
        started_at=started_at,
        callback=run_prefill,
    )
    prefill_stack = holder.get("prefill_stack")
    if prefill_stack is None or not stack_payload.get("ready", False) or prefill_stack.output_tensor is None:
        raise RuntimeError(f"Prompt prefill failed: {stack_payload.get('blockers', [])}")

    return {
        "ready": True,
        "context": context,
        "prefill_stack": prefill_stack,
        "prompt_token_ids": list(prepared.token_ids),
        "prefill_payload": stack_payload,
    }


def _first_decode_step_checkpoint(model_id: str, slice_name: str, started_at: float) -> dict:
    prefill = _prompt_prefill_checkpoint(model_id, slice_name, started_at)
    context = prefill["context"]
    prefill_stack = prefill["prefill_stack"]
    prompt_token_ids = list(prefill["prompt_token_ids"])
    policy = "greedy"
    holder: dict[str, Any] = {}

    def run_tail() -> dict:
        tail = run_decode_tail(
            model_id,
            prefill_stack.output_tensor[:, -1:, :],
            lm_head_chunk_rows=DEFAULT_LM_HEAD_CHUNK_ROWS,
            top_k=int(context["top_k"]),
            return_logits=not _can_select_from_topk(policy),
            recent_token_ids=prompt_token_ids,
            repetition_penalty=float(context["repetition_penalty"]),
        )
        holder["prefill_tail"] = tail
        payload = tail.to_dict()
        payload.pop("logits", None)
        return payload

    tail_payload = _run_checkpoint(
        model_id=model_id,
        slice_name=slice_name,
        operation="prefill-decode-tail",
        started_at=started_at,
        callback=run_tail,
    )
    tail = holder.get("prefill_tail")
    if tail is None or not tail_payload.get("ready", False):
        raise RuntimeError(f"Prefill decode tail failed: {tail_payload.get('blockers', [])}")

    def select_first() -> dict:
        if tail.logits is None and _can_select_from_topk(policy):
            selection = select_next_token_from_topk(
                tail.top_token_ids,
                tail.top_logits,
                policy=policy,
                top_p=float(context["top_p"]),
                temperature=float(context["temperature"]),
            )
        else:
            selection = select_next_token(
                tail.logits,
                policy=policy,
                top_k=int(context["top_k"]),
                top_p=float(context["top_p"]),
                temperature=float(context["temperature"]),
                repetition_penalty=float(context["repetition_penalty"]),
                recent_token_ids=prompt_token_ids,
            )
        holder["first_selection"] = selection
        return selection.to_dict()

    selection_payload = _run_checkpoint(
        model_id=model_id,
        slice_name=slice_name,
        operation="first-token-selection",
        started_at=started_at,
        callback=select_first,
    )
    selection = holder.get("first_selection")
    if selection is None or not selection_payload.get("ready", False) or selection.chosen_token_id is None:
        raise RuntimeError(f"First token selection failed: {selection_payload.get('blockers', [])}")

    first_token_id = int(selection.chosen_token_id)
    generated_chain = prompt_token_ids + [first_token_id]
    decode_state = KVDecodeState(
        model_id=model_id,
        next_token_id=first_token_id,
        next_position=len(prompt_token_ids),
        generated_token_ids=generated_chain,
        cache_sequence_lengths=dict(prefill_stack.cache_sequence_lengths),
        kv_caches=dict(prefill_stack.next_kv_caches),
        ready=True,
        blockers=[],
    )
    generated_text, generated_blockers = decode_token_ids_to_text(model_id, [first_token_id])
    return {
        "ready": True,
        "context": context,
        "decode_state": decode_state,
        "first_token_id": first_token_id,
        "generated_token_ids": [first_token_id],
        "generated_text": generated_text,
        "decode_blockers": generated_blockers,
        "cache_sequence_lengths": dict(decode_state.cache_sequence_lengths),
        **_kv_cache_summary(decode_state.kv_caches),
    }


def _second_decode_step_checkpoint(model_id: str, slice_name: str, started_at: float) -> dict:
    first = _first_decode_step_checkpoint(model_id, slice_name, started_at)
    context = first["context"]
    decode_state = first["decode_state"]
    holder: dict[str, Any] = {}

    def run_second() -> dict:
        step = run_kv_decode_step(
            model_id,
            input_token_id=int(decode_state.next_token_id),
            decode_state=decode_state,
            start_layer=0,
            layer_count=int(context["config"].num_hidden_layers),
            lm_head_chunk_rows=DEFAULT_LM_HEAD_CHUNK_ROWS,
            top_k=int(context["top_k"]),
            top_p=float(context["top_p"]),
            selection_policy="greedy",
            temperature=float(context["temperature"]),
            repetition_penalty=float(context["repetition_penalty"]),
            collect_layer_details=False,
        )
        holder["second_step"] = step
        payload = step.to_dict()
        payload.pop("logits", None)
        payload.pop("next_kv_caches", None)
        payload.pop("next_decode_state", None)
        payload.update(_kv_cache_summary(step.next_kv_caches))
        return payload

    step_payload = _run_checkpoint(
        model_id=model_id,
        slice_name=slice_name,
        operation="kv-decode-step-2",
        started_at=started_at,
        callback=run_second,
    )
    step = holder.get("second_step")
    if step is None or not step_payload.get("ready", False) or step.chosen_token_id is None:
        raise RuntimeError(f"Second decode step failed: {step_payload.get('blockers', [])}")

    token_ids = [int(first["first_token_id"]), int(step.chosen_token_id)]
    generated_text, generated_blockers = decode_token_ids_to_text(model_id, token_ids)
    next_state = step.next_decode_state
    return {
        "ready": True,
        "generated_token_ids": token_ids,
        "generated_text": generated_text,
        "decode_blockers": generated_blockers,
        "cache_sequence_lengths": dict(step.cache_sequence_lengths),
        **_kv_cache_summary(None if next_state is None else next_state.kv_caches),
    }


def _full_forward(model_id: str, max_new_tokens: int = 1) -> dict:
    reset_tensor_load_stats()
    result = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        max_new_tokens=max_new_tokens,
        selection_policy="greedy",
    )
    payload = result.to_dict()
    payload.pop("final_decode_state", None)
    payload["tensor_load_stats"] = tensor_load_stats_snapshot().to_dict()
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
    if slice_name == "layer-0-31":
        return "layers-0-through-31", lambda model_id: _layer_forward(model_id, 32)
    if slice_name == "layer-0-39":
        return "layers-0-through-39", lambda model_id: _layer_forward(model_id, 40)
    if slice_name == "all-layers":
        return "all-layers", lambda model_id: {}
    if slice_name == "all-layers-norm":
        return "all-layers-norm", lambda model_id: {}
    if slice_name == "all-layers-norm-lm":
        return "all-layers-norm-lm", lambda model_id: {}
    if slice_name == "prompt-prefill":
        return "prompt-prefill", lambda model_id: {}
    if slice_name == "decode-step-1":
        return "decode-step-1", lambda model_id: {}
    if slice_name == "decode-step-2":
        return "decode-step-2", lambda model_id: {}
    if slice_name == "full":
        return "full-prompt-decode", _full_forward
    raise ValueError(f"Unknown slice {slice_name!r}.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Pocket LLM runtime crash slices.")
    parser.add_argument("--model", required=True, help="Model id, for example qwen2.5-32b-instruct")
    parser.add_argument("--slice", required=True, choices=sorted(VALID_SLICES), help="Progressive slice to run")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="Token cap for the full prompt slice")
    parser.add_argument("--repeat", type=int, default=1, help="Run the chosen slice repeatedly in one process")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_id = str(args.model)
    slice_name = str(args.slice)
    max_new_tokens = max(1, int(args.max_new_tokens))
    repeat = max(1, int(args.repeat))
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
        if slice_name in {"all-layers", "all-layers-norm", "all-layers-norm-lm"}:
            result = _all_layers_slice(
                model_id,
                slice_name,
                started_at,
                include_norm=slice_name in {"all-layers-norm", "all-layers-norm-lm"},
                include_lm=slice_name == "all-layers-norm-lm",
            )
        elif slice_name == "prompt-prefill":
            result = _prompt_prefill_checkpoint(model_id, slice_name, started_at)
            result = {
                "ready": True,
                "prompt_token_ids": list(result["prompt_token_ids"]),
                "prefill": dict(result["prefill_payload"]),
            }
        elif slice_name == "decode-step-1":
            result = _first_decode_step_checkpoint(model_id, slice_name, started_at)
            result.pop("context", None)
            result.pop("decode_state", None)
        elif slice_name == "decode-step-2":
            result = _second_decode_step_checkpoint(model_id, slice_name, started_at)
        elif repeat > 1:
            repeated: list[dict[str, Any]] = []
            for run_index in range(1, repeat + 1):
                selected_callback = (
                    (lambda _model_id: _full_forward(_model_id, max_new_tokens)) if slice_name == "full" else callback
                )
                run_result = _run_checkpoint(
                    model_id=model_id,
                    slice_name=slice_name,
                    operation=f"{operation}-run-{run_index}",
                    started_at=started_at,
                    callback=lambda selected_callback=selected_callback: selected_callback(model_id),
                )
                repeated.append(
                    {
                        "run": run_index,
                        "ready": bool(run_result.get("ready", False)),
                        "generated_text": run_result.get("generated_text"),
                        "timings": run_result.get("timings", {}),
                        "tensor_load_stats": run_result.get("tensor_load_stats", {}),
                    }
                )
            result = {"ready": all(item["ready"] for item in repeated), "runs": repeated, "blockers": []}
        else:
            selected_callback = (lambda _model_id: _full_forward(_model_id, max_new_tokens)) if slice_name == "full" else callback
            result = _run_checkpoint(
                model_id=model_id,
                slice_name=slice_name,
                operation=operation,
                started_at=started_at,
                callback=lambda: selected_callback(model_id),
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
