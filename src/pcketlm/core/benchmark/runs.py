"""Persisted lightweight benchmark runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pcketlm.core.benchmark.readiness import BenchmarkReadiness, build_benchmark_readiness
from pcketlm.core.runtime import (
    DEFAULT_LM_HEAD_CHUNK_ROWS,
    build_gguf_backend_status,
    run_gguf_prompt,
    run_prompt_decode_loop,
    runtime_math_dtype_name,
    runtime_torch_thread_count,
)
from pcketlm.core.runtime.tensor_residency import clear_tensor_residency_cache, tensor_residency_stats
from pcketlm.core.storage.paths import benchmarks_root


@dataclass(slots=True)
class LightweightBenchmarkRun:
    """A cheap persisted benchmark run that avoids heavy generation work."""

    model_id: str
    run_id: str
    benchmark_path: Path
    created_at: str
    ready: bool
    status: str
    summary: str
    checks: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tensor_residency: dict = field(default_factory=dict)
    runtime_settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize the benchmark run."""
        return {
            "model_id": self.model_id,
            "run_id": self.run_id,
            "benchmark_path": str(self.benchmark_path),
            "created_at": self.created_at,
            "ready": self.ready,
            "status": self.status,
            "summary": self.summary,
            "checks": list(self.checks),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "tensor_residency": dict(self.tensor_residency),
            "runtime_settings": dict(self.runtime_settings),
        }


@dataclass(slots=True)
class MeasuredBenchmarkCase:
    """One timed local prompt run for a runtime mode."""

    label: str
    layer_count: int | None
    elapsed_seconds: float
    ready: bool
    generated_text: str
    backend: str = "direct-cpu"
    prompt_kind: str = "short"
    max_new_tokens: int | None = None
    generated_token_ids: list[int] = field(default_factory=list)
    stop_reason: str | None = None
    blockers: list[str] = field(default_factory=list)
    tensor_residency: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    timing_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "layer_count": self.layer_count,
            "elapsed_seconds": self.elapsed_seconds,
            "ready": self.ready,
            "generated_text": self.generated_text,
            "backend": self.backend,
            "prompt_kind": self.prompt_kind,
            "max_new_tokens": self.max_new_tokens,
            "generated_token_ids": list(self.generated_token_ids),
            "stop_reason": self.stop_reason,
            "blockers": list(self.blockers),
            "tensor_residency": dict(self.tensor_residency),
            "timings": dict(self.timings),
            "timing_summary": dict(self.timing_summary),
        }


@dataclass(slots=True)
class MeasuredBenchmarkRun:
    """Persisted timed benchmark across the desktop runtime modes."""

    model_id: str
    run_id: str
    benchmark_path: Path
    created_at: str
    prompt: str
    max_new_tokens: int
    min_new_tokens: int
    ready: bool
    status: str
    summary: str
    cases: list[MeasuredBenchmarkCase] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime_settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "run_id": self.run_id,
            "benchmark_path": str(self.benchmark_path),
            "created_at": self.created_at,
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "ready": self.ready,
            "status": self.status,
            "summary": self.summary,
            "cases": [case.to_dict() for case in self.cases],
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "runtime_settings": dict(self.runtime_settings),
        }


def _run_id(created_at: str) -> str:
    return created_at.replace(":", "").replace("-", "").replace(".", "").replace("+", "z")


def _benchmark_path(model_id: str, run_id: str) -> Path:
    return benchmarks_root(model_id) / f"{run_id}.lightweight-benchmark.json"


def _measured_benchmark_path(model_id: str, run_id: str) -> Path:
    return benchmarks_root(model_id) / f"{run_id}.measured-benchmark.json"


def _runtime_settings() -> dict:
    return {
        "math_dtype": runtime_math_dtype_name(),
        "lm_head_chunk_rows": DEFAULT_LM_HEAD_CHUNK_ROWS,
        "torch_threads": runtime_torch_thread_count(),
    }


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _round_seconds(value: object) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _timing_summary(timings: dict) -> dict:
    """Build a compact timing summary for comparing real benchmark runs."""
    prefill_stack = _round_seconds(timings.get("prefill_stack"))
    continuation_steps = _round_seconds(timings.get("continuation_steps"))
    prefill_decode_tail = _round_seconds(timings.get("prefill_decode_tail"))
    continuation_decode_tail = _round_seconds(timings.get("continuation_decode_tail"))
    decode_tail_total = round((prefill_decode_tail or 0.0) + (continuation_decode_tail or 0.0), 2)
    load_tensors = _round_seconds(
        float(timings.get("prefill_stack_op_load_tensors", 0.0) or 0.0)
        + float(timings.get("continuation_stack_op_load_tensors", 0.0) or 0.0)
    )
    stack_total = round((prefill_stack or 0.0) + (continuation_steps or 0.0), 2)
    candidates = {
        "prefill stack": prefill_stack,
        "continuation stack": continuation_steps,
        "decode tail": decode_tail_total if decode_tail_total > 0 else None,
        "tensor loading": load_tensors,
    }
    bottleneck = max(
        ((label, value) for label, value in candidates.items() if value is not None),
        key=lambda item: item[1],
        default=(None, None),
    )
    return {
        "total_seconds": _round_seconds(timings.get("total")),
        "stack_seconds": stack_total if stack_total > 0 else None,
        "prefill_stack_seconds": prefill_stack,
        "continuation_stack_seconds": continuation_steps,
        "decode_tail_seconds": decode_tail_total if decode_tail_total > 0 else None,
        "tensor_load_seconds": load_tensors,
        "bottleneck": bottleneck[0],
        "bottleneck_seconds": bottleneck[1],
    }


def _gguf_instruct_prompt(user_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are Pocket LLM. Brief English.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _gguf_stop_strings() -> list[str]:
    return ["<|im_end|>", "<|im_start|>"]


def _run_gguf_benchmark_case(model_id: str, *, label: str, prompt_kind: str, prompt: str, max_new_tokens: int) -> MeasuredBenchmarkCase:
    started = time.perf_counter()
    result = run_gguf_prompt(
        model_id,
        _gguf_instruct_prompt(prompt),
        max_tokens=max_new_tokens,
        n_ctx=2048,
        n_threads=runtime_torch_thread_count(),
        stop_strings=_gguf_stop_strings(),
    )
    elapsed = round(time.perf_counter() - started, 2)
    timings = dict(getattr(result, "timings", {}) or {})
    timings.setdefault("total", elapsed)
    return MeasuredBenchmarkCase(
        label=label,
        layer_count=None,
        elapsed_seconds=elapsed,
        ready=result.ready,
        generated_text=result.generated_text,
        backend=result.backend,
        prompt_kind=prompt_kind,
        max_new_tokens=max_new_tokens,
        generated_token_ids=[],
        stop_reason="gguf-complete" if result.ready else "gguf-blocked",
        blockers=list(result.blockers),
        tensor_residency={},
        timings=timings,
        timing_summary=_timing_summary(timings),
    )


def _run_gguf_benchmark_cases(model_id: str) -> list[MeasuredBenchmarkCase]:
    status = build_gguf_backend_status(model_id)
    if not status.ready:
        return [
            MeasuredBenchmarkCase(
                label="GGUF",
                layer_count=None,
                elapsed_seconds=0.0,
                ready=False,
                generated_text="",
                backend="llama-cpp-gguf",
                prompt_kind="readiness",
                blockers=list(status.blockers),
                stop_reason="gguf-blocked",
            )
        ]
    return [
        _run_gguf_benchmark_case(
            model_id,
            label="GGUF 1",
            prompt_kind="instruction",
            prompt="Reply with OK only.",
            max_new_tokens=1,
        ),
        _run_gguf_benchmark_case(
            model_id,
            label="GGUF 4",
            prompt_kind="instruction",
            prompt="Reply with exactly four words about local AI.",
            max_new_tokens=4,
        ),
        _run_gguf_benchmark_case(
            model_id,
            label="GGUF 8",
            prompt_kind="instruction",
            prompt="In one short sentence, explain what Pocket LLM does.",
            max_new_tokens=8,
        ),
        _run_gguf_benchmark_case(
            model_id,
            label="GGUF 32",
            prompt_kind="instruction",
            prompt="Explain Pocket LLM in one plain English sentence.",
            max_new_tokens=32,
        ),
        _run_gguf_benchmark_case(
            model_id,
            label="GGUF Logic",
            prompt_kind="logic",
            prompt="If 2 plus 3 equals 5, reply YES only.",
            max_new_tokens=4,
        ),
        _run_gguf_benchmark_case(
            model_id,
            label="GGUF Agent",
            prompt_kind="agent",
            prompt="List the next two safe steps for checking a local model. Use exactly two short numbered steps.",
            max_new_tokens=48,
        ),
    ]


def build_measured_benchmark_history(model_id: str, *, limit: int = 12) -> dict:
    """Summarize recent measured benchmark runs for trend comparison."""
    root = benchmarks_root(model_id)
    paths = sorted(root.glob("*.measured-benchmark.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    runs: list[dict] = []
    by_label: dict[str, list[float]] = {}
    timings_by_label: dict[str, dict[str, list[float]]] = {}
    for path in paths[: max(1, limit)]:
        payload = _read_json(path)
        if not payload or path.name.startswith("latest."):
            continue
        cases = []
        for case in payload.get("cases", []) or []:
            label = str(case.get("label") or "")
            elapsed = case.get("elapsed_seconds")
            if not label or elapsed is None:
                continue
            try:
                elapsed_float = float(elapsed)
            except (TypeError, ValueError):
                continue
            by_label.setdefault(label, []).append(elapsed_float)
            timing_summary = dict(case.get("timing_summary") or {})
            timing_values = timings_by_label.setdefault(label, {})
            for key in ["stack_seconds", "prefill_stack_seconds", "continuation_stack_seconds", "decode_tail_seconds", "tensor_load_seconds"]:
                value = _round_seconds(timing_summary.get(key))
                if value is not None:
                    timing_values.setdefault(key, []).append(value)
            cases.append(
                {
                    "label": label,
                    "elapsed_seconds": elapsed_float,
                    "ready": bool(case.get("ready")),
                    "generated_text": str(case.get("generated_text") or ""),
                    "timing_summary": timing_summary,
                }
            )
        runs.append(
            {
                "run_id": str(payload.get("run_id") or path.stem),
                "created_at": str(payload.get("created_at") or ""),
                "ready": bool(payload.get("ready")),
                "summary": str(payload.get("summary") or ""),
                "cases": cases,
            }
        )

    labels = []
    for label in sorted(by_label):
        values = by_label[label]
        labels.append(
            {
                "label": label,
                "count": len(values),
                "best_seconds": round(min(values), 2),
                "average_seconds": round(sum(values) / len(values), 2),
                "worst_seconds": round(max(values), 2),
                "average_stack_seconds": (
                    round(sum(timings_by_label.get(label, {}).get("stack_seconds", [])) / len(timings_by_label.get(label, {}).get("stack_seconds", [])), 2)
                    if timings_by_label.get(label, {}).get("stack_seconds")
                    else None
                ),
                "average_tensor_load_seconds": (
                    round(sum(timings_by_label.get(label, {}).get("tensor_load_seconds", [])) / len(timings_by_label.get(label, {}).get("tensor_load_seconds", [])), 2)
                    if timings_by_label.get(label, {}).get("tensor_load_seconds")
                    else None
                ),
                "average_decode_tail_seconds": (
                    round(sum(timings_by_label.get(label, {}).get("decode_tail_seconds", [])) / len(timings_by_label.get(label, {}).get("decode_tail_seconds", [])), 2)
                    if timings_by_label.get(label, {}).get("decode_tail_seconds")
                    else None
                ),
            }
        )
    return {
        "model_id": model_id,
        "run_count": len(runs),
        "labels": labels,
        "recent_runs": runs,
    }


def run_lightweight_benchmark(model_id: str, model_dir: Path) -> LightweightBenchmarkRun:
    """Run and persist a lightweight benchmark readiness snapshot."""
    readiness: BenchmarkReadiness = build_benchmark_readiness(model_id, model_dir)
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = _run_id(created_at)
    path = _benchmark_path(model_id, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    run = LightweightBenchmarkRun(
        model_id=model_id,
        run_id=run_id,
        benchmark_path=path,
        created_at=created_at,
        ready=readiness.ready,
        status=readiness.status,
        summary=readiness.summary,
        checks=readiness.first_checks,
        blockers=readiness.blockers,
        warnings=readiness.warnings,
        tensor_residency=tensor_residency_stats().to_dict(),
        runtime_settings=_runtime_settings(),
    )
    path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    (path.parent / "latest.lightweight-benchmark.json").write_text(
        json.dumps(run.to_dict(), indent=2),
        encoding="utf-8",
    )
    return run


def run_measured_benchmark(
    model_id: str,
    model_dir: Path,
    *,
    prompt: str = "hello world",
    max_new_tokens: int = 2,
    min_new_tokens: int = 1,
    repetition_penalty: float = 1.1,
) -> MeasuredBenchmarkRun:
    """Run and persist timed Quick, Agent, Fast, Balanced, and Quality prompt checks."""
    readiness: BenchmarkReadiness = build_benchmark_readiness(model_id, model_dir)
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = _run_id(created_at)
    path = _measured_benchmark_path(model_id, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    cases: list[MeasuredBenchmarkCase] = []
    blockers = [] if readiness.ready else list(readiness.blockers)
    warnings = list(readiness.warnings)
    if readiness.ready and readiness.blockers:
        warnings.extend(f"Readiness note: {blocker}" for blocker in readiness.blockers)
    if readiness.ready:
        for label, layer_count, case_max_new_tokens in [
            ("Quick", None, 1),
            ("Agent", None, min(2, max_new_tokens)),
            ("Fast", 8, max_new_tokens),
            ("Balanced", 32, max_new_tokens),
            ("Quality", None, max_new_tokens),
        ]:
            clear_tensor_residency_cache()
            started = time.perf_counter()
            result = run_prompt_decode_loop(
                model_id,
                prompt=prompt,
                max_new_tokens=case_max_new_tokens,
                min_new_tokens=min(min_new_tokens, case_max_new_tokens),
                layer_count=layer_count,
                repetition_penalty=repetition_penalty,
            )
            elapsed = round(time.perf_counter() - started, 2)
            cases.append(
                MeasuredBenchmarkCase(
                    label=label,
                    layer_count=layer_count,
                    elapsed_seconds=elapsed,
                    ready=result.ready,
                    generated_text=result.generated_text,
                    backend="direct-cpu",
                    prompt_kind="short",
                    max_new_tokens=case_max_new_tokens,
                    generated_token_ids=list(result.generated_token_ids),
                    stop_reason=result.stop_reason,
                    blockers=list(result.blockers),
                    tensor_residency=tensor_residency_stats().to_dict(),
                    timings=dict(getattr(result, "timings", {}) or {}),
                    timing_summary=_timing_summary(dict(getattr(result, "timings", {}) or {})),
                )
            )
            blockers.extend(result.blockers)
        gguf_cases = _run_gguf_benchmark_cases(model_id)
        cases.extend(gguf_cases)
        for case in gguf_cases:
            blockers.extend(case.blockers)

    ready = readiness.ready and bool(cases) and all(case.ready for case in cases) and not blockers
    status = "Measured benchmark ready" if ready else "Measured benchmark blocked"
    if cases:
        fastest = min(cases, key=lambda case: case.elapsed_seconds)
        cleanest = next((case for case in cases if case.label == "Quality"), cases[-1])
        summary = (
            f"Fastest mode: {fastest.label} at {fastest.elapsed_seconds}s. "
            f"Quality output: {cleanest.generated_text or '(empty)'}"
        )
    else:
        summary = readiness.summary

    run = MeasuredBenchmarkRun(
        model_id=model_id,
        run_id=run_id,
        benchmark_path=path,
        created_at=created_at,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        ready=ready,
        status=status,
        summary=summary,
        cases=cases,
        blockers=list(dict.fromkeys(blockers)),
        warnings=list(dict.fromkeys(warnings)),
        runtime_settings=_runtime_settings(),
    )
    path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    (path.parent / "latest.measured-benchmark.json").write_text(
        json.dumps(run.to_dict(), indent=2),
        encoding="utf-8",
    )
    return run


def run_gguf_measured_benchmark(model_id: str, model_dir: Path) -> MeasuredBenchmarkRun:
    """Run and persist the fast GGUF-focused benchmark cases."""
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = _run_id(created_at)
    path = _measured_benchmark_path(model_id, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = _run_gguf_benchmark_cases(model_id)
    blockers = list(dict.fromkeys(blocker for case in cases for blocker in case.blockers))
    ready = bool(cases) and all(case.ready for case in cases) and not blockers
    fastest = min(cases, key=lambda case: case.elapsed_seconds) if cases else None
    logic = next((case for case in cases if case.label == "GGUF Logic"), None)
    summary = (
        f"GGUF benchmark ready. Fastest case: {fastest.label} at {fastest.elapsed_seconds}s. "
        f"Logic check: {logic.generated_text or '(empty)'}."
        if ready and fastest is not None
        else "GGUF benchmark blocked."
    )
    run = MeasuredBenchmarkRun(
        model_id=model_id,
        run_id=run_id,
        benchmark_path=path,
        created_at=created_at,
        prompt="GGUF instruction, logic, and agent checks",
        max_new_tokens=48,
        min_new_tokens=1,
        ready=ready,
        status="GGUF benchmark ready" if ready else "GGUF benchmark blocked",
        summary=summary,
        cases=cases,
        blockers=blockers,
        warnings=[],
        runtime_settings={**_runtime_settings(), "benchmark_scope": "gguf"},
    )
    path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    (path.parent / "latest.measured-benchmark.json").write_text(
        json.dumps(run.to_dict(), indent=2),
        encoding="utf-8",
    )
    return run
