"""Benchmark readiness summaries for model-home flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.runtime import (
    attempt_real_model_load,
    build_runtime_bootstrap,
    build_streaming_control_state,
    plan_reduced_memory_strategy,
)


@dataclass(slots=True)
class BenchmarkReadiness:
    """Plain-English readiness state for the first benchmark flow."""

    model_id: str
    model_dir: Path
    ready: bool
    status: str
    summary: str
    first_checks: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize the readiness state."""
        return {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "ready": self.ready,
            "status": self.status,
            "summary": self.summary,
            "first_checks": list(self.first_checks),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


def build_benchmark_readiness(model_id: str, model_dir: Path) -> BenchmarkReadiness:
    """Build a cheap benchmark readiness summary without running a heavy generation pass."""
    bootstrap = build_runtime_bootstrap(model_id, model_dir).to_dict()
    load_attempt = attempt_real_model_load(model_id, model_dir).to_dict()
    streaming = build_streaming_control_state(model_id)
    strategy = plan_reduced_memory_strategy(model_id, model_dir)

    blockers = list(dict.fromkeys(
        list(bootstrap.get("blockers", []))
        + list(load_attempt.get("blockers", []))
        + list(streaming.blockers)
    ))
    warnings = list(dict.fromkeys(
        list(bootstrap.get("warnings", []))
        + list(load_attempt.get("warnings", []))
        + list(getattr(streaming, "warnings", []))
    ))
    source_ready = bool(bootstrap["source"]["ready"])
    runtime_ready = bool(bootstrap["can_attempt_load"])
    streaming_ready = bool(streaming.ready)
    ready = source_ready and (runtime_ready or streaming_ready)

    first_checks = [
        "Load/readiness check",
        "Short prompt quality check",
        "Memory use check",
        "Generation stop/session check",
        "Default-vs-profile comparison once profiles exist",
    ]
    status = "Ready" if ready else "Blocked"
    if ready:
        summary = (
            "Benchmark setup is ready for the first lightweight Qwen checks. "
            f"Recommended runtime path: {strategy.recommended_strategy_id}."
        )
    else:
        blocker_text = blockers[0] if blockers else "Benchmarking needs a ready source and runtime path first."
        summary = f"Benchmark setup is blocked. {blocker_text}"

    return BenchmarkReadiness(
        model_id=model_id,
        model_dir=model_dir,
        ready=ready,
        status=status,
        summary=summary,
        first_checks=first_checks,
        blockers=blockers,
        warnings=warnings,
    )
