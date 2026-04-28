"""Streaming manifest reader helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.storage.paths import original_model_root, streaming_model_root


@dataclass(slots=True)
class StreamingManifestChunk:
    """One chunk entry loaded from a streaming manifest."""

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
class StreamingManifestView:
    """Runtime-readable view of a staged streaming manifest."""

    model_id: str
    model_dir: Path
    manifest_path: Path
    cache_root: Path
    hot_window_dir: Path
    warm_window_dir: Path
    viability: str
    summary: str
    working_window_bytes: int
    prefetch_window_bytes: int
    segment_budget_bytes: int
    estimated_chunks: int
    chunks: list[StreamingManifestChunk] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "manifest_path": str(self.manifest_path),
            "cache_root": str(self.cache_root),
            "hot_window_dir": str(self.hot_window_dir),
            "warm_window_dir": str(self.warm_window_dir),
            "viability": self.viability,
            "summary": self.summary,
            "working_window_bytes": self.working_window_bytes,
            "prefetch_window_bytes": self.prefetch_window_bytes,
            "segment_budget_bytes": self.segment_budget_bytes,
            "estimated_chunks": self.estimated_chunks,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_streaming_manifest(model_id: str) -> StreamingManifestView:
    """Load and validate the staged streaming manifest for one model."""
    cache_root = streaming_model_root(model_id)
    manifest_path = cache_root / "manifest.json"
    blockers: list[str] = []

    if not manifest_path.exists():
        blockers.append("Streaming manifest does not exist yet.")
        return StreamingManifestView(
            model_id=model_id,
            model_dir=Path(),
            manifest_path=manifest_path,
            cache_root=cache_root,
            hot_window_dir=cache_root / "hot-window",
            warm_window_dir=cache_root / "warm-window",
            viability="missing",
            summary="Streaming manifest is missing.",
            working_window_bytes=0,
            prefetch_window_bytes=0,
            segment_budget_bytes=0,
            estimated_chunks=0,
            chunks=[],
            blockers=blockers,
            ready=False,
        )

    payload = _read_json(manifest_path)
    cache_layout = payload.get("cache_layout") or {}

    canonical_cache_root = cache_root
    canonical_manifest_path = manifest_path
    canonical_hot_window_dir = cache_root / "hot-window"
    canonical_warm_window_dir = cache_root / "warm-window"
    canonical_model_dir = original_model_root(model_id)

    stored_model_dir = Path(payload.get("model_dir", ""))
    if not stored_model_dir.exists() and canonical_model_dir.exists():
        payload["model_dir"] = str(canonical_model_dir)

    payload["cache_layout"] = {
        "cache_root": str(canonical_cache_root),
        "manifest_path": str(canonical_manifest_path),
        "hot_window_dir": str(canonical_hot_window_dir),
        "warm_window_dir": str(canonical_warm_window_dir),
    }

    if payload != _read_json(manifest_path):
        _write_json(manifest_path, payload)

    hot_window_dir = canonical_hot_window_dir
    warm_window_dir = canonical_warm_window_dir

    if not hot_window_dir.exists():
        blockers.append("Hot-window directory is missing.")
    if not warm_window_dir.exists():
        blockers.append("Warm-window directory is missing.")

    chunks = [
        StreamingManifestChunk(
            chunk_id=str(item.get("chunk_id", "")),
            target_bytes=int(item.get("target_bytes", 0)),
            target_gb=float(item.get("target_gb", 0.0)),
            purpose=str(item.get("purpose", "")),
        )
        for item in payload.get("chunk_plan", [])
    ]

    ready = not blockers and bool(chunks)
    return StreamingManifestView(
        model_id=str(payload.get("model_id", model_id)),
        model_dir=Path(payload.get("model_dir", "")),
        manifest_path=manifest_path,
        cache_root=canonical_cache_root,
        hot_window_dir=hot_window_dir,
        warm_window_dir=warm_window_dir,
        viability=str(payload.get("viability", "unknown")),
        summary=str(payload.get("summary", "")),
        working_window_bytes=int(payload.get("working_window_bytes", 0)),
        prefetch_window_bytes=int(payload.get("prefetch_window_bytes", 0)),
        segment_budget_bytes=int(
            payload.get(
                "segment_budget_bytes",
                min(
                    int(payload.get("working_window_bytes", 0)),
                    int(payload.get("prefetch_window_bytes", 0)),
                ),
            )
        ),
        estimated_chunks=int(payload.get("estimated_chunks", 0)),
        chunks=chunks,
        blockers=blockers,
        ready=ready,
    )
