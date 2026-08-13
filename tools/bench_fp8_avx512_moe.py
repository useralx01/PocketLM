from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pcketlm.native as native
from pcketlm.core.runtime.fp8_pack import clear_fp8_pack_readers
from pcketlm.core.runtime.fp8_source import (
    _preload_packed_mlp_prefixes,
    clear_fp8_attention_weight_cache,
    clear_fp8_lm_head_full_cache,
    clear_fp8_mlp_span_cache,
    clear_fp8_prefill_cache,
    fp8_source_status,
    run_fp8_router,
    run_fp8_single_token_forward,
)


def _fp8_bytes(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()


def _setdefault_env(name: str, value: str) -> None:
    if not os.environ.get(name, "").strip():
        os.environ[name] = value


@contextmanager
def _env_flag(name: str, value: str | None) -> Iterator[None]:
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _clear_runtime_caches() -> None:
    clear_fp8_pack_readers()
    clear_fp8_attention_weight_cache()
    clear_fp8_mlp_span_cache()
    clear_fp8_prefill_cache()
    clear_fp8_lm_head_full_cache()


def _time_call(fn, repeats: int) -> tuple[object, list[float]]:
    result = None
    times: list[float] = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        result = fn()
        times.append(float(time.perf_counter() - started))
    return result, times


def _avg(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _tensor_compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    left_cpu = left.detach().cpu().contiguous()
    right_cpu = right.detach().cpu().contiguous()
    equal = bool(torch.equal(left_cpu, right_cpu))
    diff = (left_cpu.float() - right_cpu.float()).abs()
    return {
        "bit_exact": equal,
        "shape": [int(value) for value in left_cpu.shape],
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
    }


def _synthetic_items() -> tuple[
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
]:
    hidden_cols = 128
    intermediate = 128
    hidden = torch.linspace(-0.5, 0.5, steps=hidden_cols, dtype=torch.float32).reshape(1, hidden_cols)
    items = []
    for seed in (11, 22, 33, 44):
        generator = torch.Generator().manual_seed(seed)
        gate = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
        up = _fp8_bytes(torch.randn((intermediate, hidden_cols), generator=generator) * 0.25)
        down = _fp8_bytes(torch.randn((hidden_cols, intermediate), generator=generator) * 0.25)
        gate_scale = torch.full((1, 1), 0.875, dtype=torch.float32)
        up_scale = torch.full((1, 1), 1.125, dtype=torch.float32)
        down_scale = torch.full((1, 1), 0.75, dtype=torch.float32)
        items.append((gate, gate_scale, up, up_scale, down, down_scale))
    route_weights = torch.tensor([0.31, 0.17, 0.29, 0.23], dtype=torch.float32)
    return items, hidden, route_weights


def _bench_native_pair(
    items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    hidden: torch.Tensor,
    route_weights: torch.Tensor,
    repeats: int,
) -> dict:
    scalar_result, scalar_times = _time_call(
        lambda: native.fp8_e4m3_block_mlp_many_weighted_f32(items, hidden, route_weights),
        repeats,
    )
    avx_result, avx_times = _time_call(
        lambda: native.fp8_e4m3_block_mlp_many_weighted_avx512_f32(items, hidden, route_weights),
        repeats,
    )
    compare = _tensor_compare(scalar_result, avx_result)
    scalar_avg = _avg(scalar_times)
    avx_avg = _avg(avx_times)
    return {
        **compare,
        "scalar_times_seconds": scalar_times,
        "avx512_times_seconds": avx_times,
        "scalar_avg_seconds": scalar_avg,
        "avx512_avg_seconds": avx_avg,
        "speedup": None if avx_avg <= 0.0 else float(scalar_avg / avx_avg),
    }


def _real_layer_items(model_id: str, layer: int) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
    list[int],
]:
    hidden = torch.linspace(-0.25, 0.25, steps=7168, dtype=torch.float32).reshape(1, 1, 7168).to(torch.bfloat16)
    router = run_fp8_router(model_id, int(layer), hidden)
    if not router.ready or router.indices_tensor is None or router.weights_tensor is None:
        raise RuntimeError(f"Router failed for real layer {layer}: {router.blockers}")
    route_indices = router.indices_tensor
    route_weights = router.weights_tensor.float()
    selected = sorted({int(value) for value in route_indices.reshape(-1).tolist()})
    prefixes = [f"model.layers.{int(layer)}.mlp.experts.{expert}.gate_proj.weight" for expert in selected]
    mlp_prefixes = [prefix.rsplit(".", 2)[0] for prefix in prefixes]
    preloaded = _preload_packed_mlp_prefixes(model_id, mlp_prefixes)
    missing = [prefix for prefix in mlp_prefixes if prefix not in preloaded]
    if missing:
        raise RuntimeError(f"Packed MLP prefixes did not preload: {missing[:3]}")
    ordered_route_weights = torch.tensor(
        [
            float(route_weights[torch.where(route_indices == expert)[0][0], torch.where(route_indices == expert)[1][0]].item())
            for expert in selected
        ],
        dtype=torch.float32,
    )
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).to(torch.bfloat16)
    return [preloaded[prefix] for prefix in mlp_prefixes], flat_hidden, ordered_route_weights, selected


def _summarize_forward(result, elapsed: float) -> dict:
    layer_summaries = [dict(item) for item in result.step_summaries]
    attention = float(sum(float(item.get("attention_elapsed_seconds", 0.0)) for item in layer_summaries))
    ffn = float(sum(float(item.get("ffn_elapsed_seconds", 0.0)) for item in layer_summaries))
    return {
        "ready": bool(result.ready),
        "elapsed_seconds": float(elapsed),
        "executed_layers": list(result.executed_layers),
        "attention_seconds": attention,
        "ffn_seconds": ffn,
        "layer_summaries": layer_summaries,
        "blockers": list(result.blockers),
    }


def _run_bounded_forward(model_id: str, token_id: int, layer_count: int, use_avx512: bool):
    _clear_runtime_caches()
    with _env_flag("PCKETLM_DISABLE_NATIVE_FP8_AVX512", None if use_avx512 else "1"):
        started = time.perf_counter()
        result = run_fp8_single_token_forward(
            model_id,
            int(token_id),
            start_layer=0,
            layer_count=int(layer_count),
            include_tail=False,
            dtype=torch.bfloat16,
            position=0,
        )
        elapsed = time.perf_counter() - started
    return result, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove and time the AVX-512 FP8 weighted MoE kernel.")
    parser.add_argument("--model-id", default="deepseek-v3")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--token-id", type=int, default=100)
    parser.add_argument("--layer-count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("state/plm-19-avx512-fp8-moe.json"))
    args = parser.parse_args()

    _setdefault_env("PCKETLM_ENABLE_FP8_RESIDENCY_SPLIT", "1")
    _setdefault_env("PCKETLM_FP8_MLP_SPAN_CACHE_MB", "2048")
    _setdefault_env("PCKETLM_FP8_MLP_SPAN_CACHE_POLICY", "preserve-full")
    _setdefault_env("PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_MB", "4096")
    _setdefault_env("PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_POLICY", "prefix")

    started = time.perf_counter()
    blockers: list[str] = []
    avx_available = native.native_fp8_mlp_many_weighted_avx512_available()
    if not avx_available:
        blockers.append(f"AVX-512 FP8 weighted many-MLP unavailable: {native.native_fp8_linear_avx512_error()}")

    synthetic = None
    real_layer = None
    bounded = None
    if avx_available:
        items, hidden, weights = _synthetic_items()
        synthetic = _bench_native_pair(items, hidden, weights, args.repeats)
        real_items, real_hidden, real_weights, selected = _real_layer_items(args.model_id, args.layer)
        real_layer = _bench_native_pair(real_items, real_hidden, real_weights, args.repeats)
        real_layer["layer"] = int(args.layer)
        real_layer["selected_experts"] = selected

        scalar_forward, scalar_elapsed = _run_bounded_forward(args.model_id, args.token_id, args.layer_count, False)
        avx_forward, avx_elapsed = _run_bounded_forward(args.model_id, args.token_id, args.layer_count, True)
        if scalar_forward.output_tensor is None or avx_forward.output_tensor is None:
            blockers.append("Bounded forward did not materialize both scalar and AVX-512 outputs.")
            output_compare = {"bit_exact": False, "shape": [], "max_abs_diff": None}
        else:
            output_compare = _tensor_compare(scalar_forward.output_tensor, avx_forward.output_tensor)
        scalar_summary = _summarize_forward(scalar_forward, scalar_elapsed)
        avx_summary = _summarize_forward(avx_forward, avx_elapsed)
        bounded = {
            "token_id": int(args.token_id),
            "layer_count": int(args.layer_count),
            "output_compare": output_compare,
            "scalar": scalar_summary,
            "avx512": avx_summary,
            "attention_seconds_delta": float(scalar_summary["attention_seconds"] - avx_summary["attention_seconds"]),
            "ffn_seconds_delta": float(scalar_summary["ffn_seconds"] - avx_summary["ffn_seconds"]),
            "ffn_speedup": None
            if avx_summary["ffn_seconds"] <= 0.0
            else float(scalar_summary["ffn_seconds"] / avx_summary["ffn_seconds"]),
            "elapsed_speedup": None
            if avx_summary["elapsed_seconds"] <= 0.0
            else float(scalar_summary["elapsed_seconds"] / avx_summary["elapsed_seconds"]),
        }
        if not synthetic["bit_exact"]:
            blockers.append("Synthetic tiny AVX-512 output differs from scalar native output.")
        if not real_layer["bit_exact"]:
            blockers.append("Real DeepSeek layer AVX-512 output differs from scalar native output.")
        if not output_compare["bit_exact"]:
            blockers.append("Bounded forward AVX-512 output differs from scalar native output.")
        if float(bounded["ffn_seconds_delta"]) <= 0.0:
            blockers.append("Bounded per-layer FFN timing did not improve with the AVX-512 path.")

    payload = {
        "model_id": args.model_id,
        "ready": not blockers,
        "blockers": blockers,
        "elapsed_seconds": float(time.perf_counter() - started),
        "kernel": "avx512_bitdecode_exact_scalar_order_weighted_many_mlp",
        "hard_constraints": {
            "no_gpu": True,
            "no_other_model": True,
            "no_quantization_change": True,
            "no_dropped_experts_or_layers": True,
        },
        "native_avx512_available": bool(avx_available),
        "native_avx512_error": None if avx_available else str(native.native_fp8_linear_avx512_error()),
        "env": {
            "PCKETLM_ENABLE_FP8_RESIDENCY_SPLIT": os.environ.get("PCKETLM_ENABLE_FP8_RESIDENCY_SPLIT", ""),
            "PCKETLM_FP8_MLP_SPAN_CACHE_MB": os.environ.get("PCKETLM_FP8_MLP_SPAN_CACHE_MB", ""),
            "PCKETLM_FP8_MLP_SPAN_CACHE_POLICY": os.environ.get("PCKETLM_FP8_MLP_SPAN_CACHE_POLICY", ""),
            "PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_MB": os.environ.get("PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_MB", ""),
            "PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_POLICY": os.environ.get("PCKETLM_FP8_ATTENTION_WEIGHT_CACHE_POLICY", ""),
        },
        "runtime_policy": fp8_source_status(args.model_id).get("runtime_policy", {}),
        "synthetic_tiny_oracle": synthetic,
        "real_deepseek_layer_oracle": real_layer,
        "bounded_forward_timing": bounded,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
