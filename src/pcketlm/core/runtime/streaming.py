"""Staged disk-streaming runtime planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.acquisition import build_acquisition_snapshot
from pcketlm.core.runtime.load_attempt import MemorySnapshot, _memory_snapshot
from pcketlm.core.storage.paths import state_root


def _to_gb(value: int) -> float:
    return round(value / (1024**3), 2)


@dataclass(slots=True)
class StreamingChunkPlan:
    """One chunk or window in the staged disk-streaming plan."""

    chunk_id: str
    target_bytes: int
    target_gb: float
    purpose: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "target_bytes": self.target_bytes,
            "target_gb": self.target_gb,
            "purpose": self.purpose,
        }


@dataclass(slots=True)
class StreamingCacheLayout:
    """Filesystem layout for staged streaming state."""

    cache_root: Path
    manifest_path: Path
    hot_window_dir: Path
    warm_window_dir: Path

    def to_dict(self) -> dict:
        return {
            "cache_root": str(self.cache_root),
            "manifest_path": str(self.manifest_path),
            "hot_window_dir": str(self.hot_window_dir),
            "warm_window_dir": str(self.warm_window_dir),
        }


@dataclass(slots=True)
class StreamingPlan:
    """Adaptive staged disk-streaming plan for one model."""

    model_id: str
    model_dir: Path
    memory: MemorySnapshot
    model_bytes: int
    model_gb: float
    working_window_bytes: int
    working_window_gb: float
    prefetch_window_bytes: int
    prefetch_window_gb: float
    segment_budget_bytes: int
    segment_budget_gb: float
    estimated_chunks: int
    cache_layout: StreamingCacheLayout
    chunk_plan: list[StreamingChunkPlan] = field(default_factory=list)
    viability: str = "unknown"
    summary: str = ""
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "memory": self.memory.to_dict(),
            "model_bytes": self.model_bytes,
            "model_gb": self.model_gb,
            "working_window_bytes": self.working_window_bytes,
            "working_window_gb": self.working_window_gb,
            "prefetch_window_bytes": self.prefetch_window_bytes,
            "prefetch_window_gb": self.prefetch_window_gb,
            "segment_budget_bytes": self.segment_budget_bytes,
            "segment_budget_gb": self.segment_budget_gb,
            "estimated_chunks": self.estimated_chunks,
            "cache_layout": self.cache_layout.to_dict(),
            "chunk_plan": [chunk.to_dict() for chunk in self.chunk_plan],
            "viability": self.viability,
            "summary": self.summary,
            "blockers": list(self.blockers),
        }


def _cache_layout(model_id: str) -> StreamingCacheLayout:
    cache_root = state_root() / "streaming" / model_id
    return StreamingCacheLayout(
        cache_root=cache_root,
        manifest_path=cache_root / "manifest.json",
        hot_window_dir=cache_root / "hot-window",
        warm_window_dir=cache_root / "warm-window",
    )


def _working_window_bytes(memory: MemorySnapshot) -> int:
    # Adaptive but conservative: use roughly one third of free RAM, clamped to 512 MB - 2 GB.
    candidate = int(memory.free_bytes * 0.33)
    lower = 512 * 1024**2
    upper = 2 * 1024**3
    return max(lower, min(candidate, upper))


def _prefetch_window_bytes(working_window_bytes: int) -> int:
    return max(256 * 1024**2, int(working_window_bytes * 0.5))


def _segment_budget_bytes(working_window_bytes: int, prefetch_window_bytes: int) -> int:
    return max(256 * 1024**2, min(working_window_bytes, prefetch_window_bytes))


def plan_staged_disk_streaming(model_id: str, model_dir: Path) -> StreamingPlan:
    """Plan the first adaptive staged disk-streaming layout for one model."""
    acquisition = build_acquisition_snapshot(model_dir)
    memory = _memory_snapshot()
    model_bytes = acquisition.expected_bytes or acquisition.bytes_on_disk
    model_gb = _to_gb(model_bytes)

    working_window_bytes = _working_window_bytes(memory)
    prefetch_window_bytes = _prefetch_window_bytes(working_window_bytes)
    segment_budget_bytes = _segment_budget_bytes(working_window_bytes, prefetch_window_bytes)
    estimated_chunks = max(1, -(-model_bytes // segment_budget_bytes))
    cache_layout = _cache_layout(model_id)

    blockers: list[str] = []
    viability = "viable"
    if memory.free_bytes < 1536 * 1024**2:
        viability = "tight"
        blockers.append(
            f"Free RAM is only {memory.free_gb} GB, so even staged streaming will need an unusually small working window."
        )

    chunk_plan = [
        StreamingChunkPlan(
            chunk_id="hot-window",
            target_bytes=working_window_bytes,
            target_gb=_to_gb(working_window_bytes),
            purpose="Actively loaded weights needed right now.",
        ),
        StreamingChunkPlan(
            chunk_id="warm-prefetch",
            target_bytes=prefetch_window_bytes,
            target_gb=_to_gb(prefetch_window_bytes),
            purpose="Next weights prefetched before they are needed.",
        ),
    ]

    summary = (
        f"Plan a staged disk-streaming runtime with a {round(_to_gb(working_window_bytes), 2)} GB hot window "
        f"and a {round(_to_gb(prefetch_window_bytes), 2)} GB warm prefetch window. "
        f"Use a shared segment budget of {round(_to_gb(segment_budget_bytes), 2)} GB so streamed units fit "
        f"both windows, which would split this "
        f"{model_gb} GB model into about {estimated_chunks} streamed chunks."
    )

    return StreamingPlan(
        model_id=model_id,
        model_dir=model_dir,
        memory=memory,
        model_bytes=model_bytes,
        model_gb=model_gb,
        working_window_bytes=working_window_bytes,
        working_window_gb=_to_gb(working_window_bytes),
        prefetch_window_bytes=prefetch_window_bytes,
        prefetch_window_gb=_to_gb(prefetch_window_bytes),
        segment_budget_bytes=segment_budget_bytes,
        segment_budget_gb=_to_gb(segment_budget_bytes),
        estimated_chunks=estimated_chunks,
        cache_layout=cache_layout,
        chunk_plan=chunk_plan,
        viability=viability,
        summary=summary,
        blockers=blockers,
    )
