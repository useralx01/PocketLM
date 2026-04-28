"""Guarded real-load attempt helpers."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.acquisition import build_acquisition_snapshot
from pcketlm.core.runtime.bootstrap import build_runtime_bootstrap


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


@dataclass(slots=True)
class MemorySnapshot:
    """Simple physical memory snapshot."""

    total_bytes: int
    free_bytes: int

    @property
    def total_gb(self) -> float:
        return round(self.total_bytes / (1024**3), 2)

    @property
    def free_gb(self) -> float:
        return round(self.free_bytes / (1024**3), 2)

    def to_dict(self) -> dict:
        return {
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "total_gb": self.total_gb,
            "free_gb": self.free_gb,
        }


@dataclass(slots=True)
class RuntimeLoadAttemptResult:
    """Result of a guarded first real-load attempt."""

    model_id: str
    model_dir: Path
    load_attempted: bool
    load_succeeded: bool
    blockers: list[str] = field(default_factory=list)
    blocker_category: str | None = None
    blocker_severity: str | None = None
    recommended_action: str | None = None
    warnings: list[str] = field(default_factory=list)
    memory: MemorySnapshot | None = None
    estimated_required_bytes: int | None = None
    plain_english_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "load_attempted": self.load_attempted,
            "load_succeeded": self.load_succeeded,
            "blockers": list(self.blockers),
            "blocker_category": self.blocker_category,
            "blocker_severity": self.blocker_severity,
            "recommended_action": self.recommended_action,
            "warnings": list(self.warnings),
            "memory": self.memory.to_dict() if self.memory else None,
            "estimated_required_bytes": self.estimated_required_bytes,
            "estimated_required_gb": round(self.estimated_required_bytes / (1024**3), 2)
            if self.estimated_required_bytes is not None
            else None,
            "plain_english_summary": self.plain_english_summary,
        }


def _memory_snapshot() -> MemorySnapshot:
    stat = _MemoryStatusEx()
    stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return MemorySnapshot(total_bytes=int(stat.ullTotalPhys), free_bytes=int(stat.ullAvailPhys))


def _estimate_required_bytes(model_bytes: int) -> int:
    # Conservative CPU-side estimate: full weights plus overhead for model objects and loading.
    return int(model_bytes * 1.2)


def _build_summary(load_attempted: bool, load_succeeded: bool, blockers: list[str]) -> str:
    if load_succeeded:
        return "The first real model load succeeded."
    if load_attempted:
        return "The first real model load was attempted but failed: " + "; ".join(blockers)
    if blockers:
        return "The first real model load was not attempted because: " + "; ".join(blockers)
    return "The first real model load was not attempted."


def _classify_blocker(blockers: list[str]) -> tuple[str | None, str | None, str | None]:
    if not blockers:
        return None, None, None

    first = blockers[0].lower()
    if "free ram" in first:
        return (
            "memory",
            "high",
            "Free memory before retrying, or move to a reduced-memory load path instead of a plain CPU load.",
        )
    if "tokenizer" in first or "config" in first:
        return (
            "runtime-preflight",
            "high",
            "Fix the runtime preflight issue first, then retry the load.",
        )
    if "shard" in first:
        return (
            "source-files",
            "high",
            "Make sure the full shard set is present before retrying the runtime load.",
        )
    return (
        "runtime",
        "medium",
        "Review the blocker details, fix the runtime issue, then retry the load.",
    )


def attempt_real_model_load(model_id: str, model_dir: Path) -> RuntimeLoadAttemptResult:
    """Attempt a first real model load only when the system looks safe enough."""
    bootstrap = build_runtime_bootstrap(model_id, model_dir)
    acquisition = build_acquisition_snapshot(model_dir)
    memory = _memory_snapshot()

    blockers = list(bootstrap.blockers)
    warnings = list(bootstrap.warnings)

    expected_bytes = acquisition.expected_bytes or acquisition.bytes_on_disk
    estimated_required_bytes = _estimate_required_bytes(expected_bytes)
    if memory.free_bytes < estimated_required_bytes:
        blockers.append(
            f"Free RAM is only {memory.free_gb} GB, but the first CPU load is estimated to need about {round(estimated_required_bytes / (1024**3), 2)} GB."
        )

    if blockers:
        category, severity, action = _classify_blocker(blockers)
        return RuntimeLoadAttemptResult(
            model_id=model_id,
            model_dir=model_dir,
            load_attempted=False,
            load_succeeded=False,
            blockers=blockers,
            blocker_category=category,
            blocker_severity=severity,
            recommended_action=action,
            warnings=warnings,
            memory=memory,
            estimated_required_bytes=estimated_required_bytes,
            plain_english_summary=_build_summary(False, False, blockers),
        )

    try:
        from transformers import AutoModelForCausalLM  # type: ignore

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype="auto",
        )
        del model
        return RuntimeLoadAttemptResult(
            model_id=model_id,
            model_dir=model_dir,
            load_attempted=True,
            load_succeeded=True,
            blockers=[],
            blocker_category=None,
            blocker_severity=None,
            recommended_action="The model loaded successfully. The next step is tokenizer/model runtime testing.",
            warnings=warnings,
            memory=memory,
            estimated_required_bytes=estimated_required_bytes,
            plain_english_summary=_build_summary(True, True, []),
        )
    except Exception as exc:
        blockers.append(str(exc))
        category, severity, action = _classify_blocker(blockers)
        return RuntimeLoadAttemptResult(
            model_id=model_id,
            model_dir=model_dir,
            load_attempted=True,
            load_succeeded=False,
            blockers=blockers,
            blocker_category=category,
            blocker_severity=severity,
            recommended_action=action,
            warnings=warnings,
            memory=memory,
            estimated_required_bytes=estimated_required_bytes,
            plain_english_summary=_build_summary(True, False, blockers),
        )
