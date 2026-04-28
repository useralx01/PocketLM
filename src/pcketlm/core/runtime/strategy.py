"""Reduced-memory runtime strategy planning."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.runtime.load_attempt import MemorySnapshot, _estimate_required_bytes, _memory_snapshot
from pcketlm.core.acquisition import build_acquisition_snapshot


@dataclass(slots=True)
class RuntimeStrategyOption:
    """One possible runtime strategy for the current machine and model."""

    strategy_id: str
    label: str
    summary: str
    viable_now: bool
    implemented_now: bool
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "label": self.label,
            "summary": self.summary,
            "viable_now": self.viable_now,
            "implemented_now": self.implemented_now,
            "pros": list(self.pros),
            "cons": list(self.cons),
            "blockers": list(self.blockers),
        }


@dataclass(slots=True)
class RuntimeStrategyPlan:
    """Planner output for choosing the next reduced-memory runtime path."""

    model_id: str
    model_dir: Path
    memory: MemorySnapshot
    model_bytes: int
    estimated_plain_load_bytes: int
    disk_free_bytes: int
    recommended_strategy_id: str
    recommended_summary: str
    options: list[RuntimeStrategyOption] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "memory": self.memory.to_dict(),
            "model_bytes": self.model_bytes,
            "model_gb": round(self.model_bytes / (1024**3), 2),
            "estimated_plain_load_bytes": self.estimated_plain_load_bytes,
            "estimated_plain_load_gb": round(self.estimated_plain_load_bytes / (1024**3), 2),
            "disk_free_bytes": self.disk_free_bytes,
            "disk_free_gb": round(self.disk_free_bytes / (1024**3), 2),
            "recommended_strategy_id": self.recommended_strategy_id,
            "recommended_summary": self.recommended_summary,
            "options": [option.to_dict() for option in self.options],
        }


def _disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _plain_cpu_option(memory: MemorySnapshot, estimated_plain_load_bytes: int) -> RuntimeStrategyOption:
    viable = memory.free_bytes >= estimated_plain_load_bytes
    blockers: list[str] = []
    if not viable:
        blockers.append(
            f"Free RAM is only {memory.free_gb} GB, but plain CPU loading is estimated to need about {round(estimated_plain_load_bytes / (1024**3), 2)} GB."
        )
    return RuntimeStrategyOption(
        strategy_id="plain-cpu-load",
        label="Plain CPU Load",
        summary="Load the full model into CPU memory in one straightforward pass.",
        viable_now=viable,
        implemented_now=True,
        pros=[
            "Simplest correctness path",
            "Useful as a baseline when enough RAM exists",
        ],
        cons=[
            "Needs a lot of free RAM",
            "Not realistic on tight-memory laptops for this model size",
        ],
        blockers=blockers,
    )


def _staged_streaming_option(memory: MemorySnapshot, model_bytes: int, disk_free_bytes: int) -> RuntimeStrategyOption:
    viable = memory.free_bytes >= 2 * 1024**3 and disk_free_bytes >= model_bytes
    blockers: list[str] = []
    if memory.free_bytes < 2 * 1024**3:
        blockers.append(
            f"Free RAM is only {memory.free_gb} GB, and the first staged streaming path should keep at least a small 2 GB working window free."
        )
    if disk_free_bytes < model_bytes:
        blockers.append("Not enough free disk space for a safe staged offload/cache path.")
    return RuntimeStrategyOption(
        strategy_id="staged-disk-streaming",
        label="Staged Disk Streaming",
        summary="Keep the model on disk and stream only a small working set into memory as layers are needed.",
        viable_now=viable,
        implemented_now=False,
        pros=[
            "Best fit for pcketlm's long-term mission",
            "Can work even when full CPU loading is impossible",
            "Keeps original model intact on disk",
        ],
        cons=[
            "Much more complex than plain loading",
            "Likely slower until caching improves",
            "Needs careful scheduling and memory-window control",
        ],
        blockers=blockers,
    )


def _quantized_artifact_option(memory: MemorySnapshot, disk_free_bytes: int) -> RuntimeStrategyOption:
    viable = disk_free_bytes >= 8 * 1024**3
    blockers: list[str] = []
    if not viable:
        blockers.append("Not enough free disk space to build a new reduced-memory artifact safely.")
    return RuntimeStrategyOption(
        strategy_id="quantized-artifact-first",
        label="Quantized Artifact First",
        summary="Build a smaller derived artifact first, then load that artifact instead of the original weights.",
        viable_now=viable,
        implemented_now=False,
        pros=[
            "More realistic long-term path for limited hardware",
            "Aligns with pcketlm's reversible artifact design",
        ],
        cons=[
            "Needs the optimization/artifact pipeline first",
            "Changes baseline behavior before raw original loading is proven",
        ],
        blockers=blockers,
    )


def plan_reduced_memory_strategy(model_id: str, model_dir: Path) -> RuntimeStrategyPlan:
    """Plan the best next runtime path for a low-memory machine."""
    acquisition = build_acquisition_snapshot(model_dir)
    memory = _memory_snapshot()
    model_bytes = acquisition.expected_bytes or acquisition.bytes_on_disk
    estimated_plain_load_bytes = _estimate_required_bytes(model_bytes)
    disk_free_bytes = _disk_free_bytes(model_dir)

    options = [
        _plain_cpu_option(memory, estimated_plain_load_bytes),
        _staged_streaming_option(memory, model_bytes, disk_free_bytes),
        _quantized_artifact_option(memory, disk_free_bytes),
    ]

    if options[0].viable_now:
        recommended = options[0]
        summary = "Plain CPU loading is actually viable right now, so we should use it as the cleanest baseline path."
    elif options[1].viable_now:
        recommended = options[1]
        summary = "The best next path is staged disk streaming because the machine cannot hold a plain CPU load, but it may support a streamed working set."
    else:
        recommended = options[2]
        summary = "A reduced-memory artifact path is the next best direction because the machine is too tight even for a small staged working window."

    return RuntimeStrategyPlan(
        model_id=model_id,
        model_dir=model_dir,
        memory=memory,
        model_bytes=model_bytes,
        estimated_plain_load_bytes=estimated_plain_load_bytes,
        disk_free_bytes=disk_free_bytes,
        recommended_strategy_id=recommended.strategy_id,
        recommended_summary=summary,
        options=options,
    )
