"""Streaming manifest and cache bootstrap helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pcketlm.core.runtime.streaming import StreamingPlan, plan_staged_disk_streaming


@dataclass(slots=True)
class StreamingBootstrapResult:
    """Result of creating the first staged-streaming cache state."""

    model_id: str
    manifest_path: Path
    cache_root: Path
    hot_window_dir: Path
    warm_window_dir: Path
    created: bool

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "manifest_path": str(self.manifest_path),
            "cache_root": str(self.cache_root),
            "hot_window_dir": str(self.hot_window_dir),
            "warm_window_dir": str(self.warm_window_dir),
            "created": self.created,
        }


def _manifest_payload(plan: StreamingPlan) -> dict:
    return {
        "model_id": plan.model_id,
        "model_dir": str(plan.model_dir),
        "summary": plan.summary,
        "viability": plan.viability,
        "memory": plan.memory.to_dict(),
        "model_bytes": plan.model_bytes,
        "model_gb": plan.model_gb,
        "working_window_bytes": plan.working_window_bytes,
        "working_window_gb": plan.working_window_gb,
        "prefetch_window_bytes": plan.prefetch_window_bytes,
        "prefetch_window_gb": plan.prefetch_window_gb,
        "segment_budget_bytes": plan.segment_budget_bytes,
        "segment_budget_gb": plan.segment_budget_gb,
        "estimated_chunks": plan.estimated_chunks,
        "chunk_plan": [chunk.to_dict() for chunk in plan.chunk_plan],
        "cache_layout": plan.cache_layout.to_dict(),
        "blockers": list(plan.blockers),
    }


def write_streaming_manifest(plan: StreamingPlan) -> Path:
    """Write the streaming manifest for one model."""
    cache_layout = plan.cache_layout
    cache_layout.cache_root.mkdir(parents=True, exist_ok=True)
    payload = _manifest_payload(plan)
    cache_layout.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cache_layout.manifest_path


def bootstrap_streaming_state(model_id: str, model_dir: Path) -> StreamingBootstrapResult:
    """Create the first streaming cache directories and manifest for one model."""
    plan = plan_staged_disk_streaming(model_id, model_dir)
    cache_layout = plan.cache_layout

    cache_layout.cache_root.mkdir(parents=True, exist_ok=True)
    cache_layout.hot_window_dir.mkdir(parents=True, exist_ok=True)
    cache_layout.warm_window_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_streaming_manifest(plan)

    return StreamingBootstrapResult(
        model_id=model_id,
        manifest_path=manifest_path,
        cache_root=cache_layout.cache_root,
        hot_window_dir=cache_layout.hot_window_dir,
        warm_window_dir=cache_layout.warm_window_dir,
        created=True,
    )
