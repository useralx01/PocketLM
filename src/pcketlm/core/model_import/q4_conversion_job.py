"""Resumable Q4 conversion jobs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pcketlm.core.model_import.q4_plan import plan_model_dir_to_q4
from pcketlm.core.model_import.q4_quantize import quantize_model_dir_to_q4
from pcketlm.core.storage.paths import state_root


def q4_conversion_state_path(model_id: str) -> Path:
    """Return the state file used for one compact Q4 conversion job."""
    safe_model_id = model_id.replace("/", "_").replace("\\", "_")
    return state_root() / "q4_conversions" / f"{safe_model_id}.json"


def run_q4_conversion_job(
    source_model_dir: Path,
    output_dir: Path,
    *,
    model_id: str | None = None,
    resume: bool = True,
    check_disk_space: bool = True,
    max_new_shards: int | None = None,
) -> dict:
    """Run or resume a compact Q4 artifact conversion.

    The job is gated by the header-only Q4 plan, writes progress under
    ``state/q4_conversions``, and resumes at source-shard boundaries.
    """
    source_model_dir = source_model_dir.resolve()
    output_dir = output_dir.resolve()
    model_id = model_id or source_model_dir.name
    state_path = q4_conversion_state_path(model_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    partial_manifest_path = output_dir / "q4_manifest.partial.json"

    previous = _read_json(state_path) if resume else None
    completed_shards = set(str(item) for item in ((previous or {}).get("completed_shards") or []))
    existing_manifest = _read_json(partial_manifest_path) if resume else None
    if existing_manifest is None and resume:
        existing_manifest = _read_json(output_dir / "q4_manifest.json")

    plan = plan_model_dir_to_q4(source_model_dir, output_dir)
    shard_names = _source_shard_names(source_model_dir)
    base_state = {
        "model_id": model_id,
        "source_model_dir": str(source_model_dir),
        "output_dir": str(output_dir),
        "plan": plan,
        "source_shard_count": len(shard_names),
        "completed_shards": sorted(completed_shards),
        "completed_shard_count": len(completed_shards),
        "progress_pct": _progress_pct(len(completed_shards), len(shard_names)),
        "started_at": (previous or {}).get("started_at") or _now(),
        "updated_at": _now(),
    }
    if not plan.get("ready_for_conversion"):
        return _write_state(
            state_path,
            {
                **base_state,
                "status": "blocked",
                "blockers": _plan_blockers(plan),
            },
        )
    if check_disk_space:
        free_bytes = _free_bytes_for(output_dir)
        required_bytes = int(plan.get("disk_required_bytes_with_10pct_headroom", 0))
        if free_bytes < required_bytes:
            return _write_state(
                state_path,
                {
                    **base_state,
                    "status": "blocked",
                    "blockers": [
                        f"Not enough free disk space for compact Q4 artifact: need {required_bytes} bytes with headroom, have {free_bytes} bytes."
                    ],
                    "free_bytes": free_bytes,
                    "required_bytes": required_bytes,
                },
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    def on_progress(event: dict) -> None:
        if event.get("event") in {"shard_complete", "shard_skipped"}:
            completed_shards.add(str(event["shard_name"]))
        manifest = event.get("manifest") or {}
        partial_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _write_state(
            state_path,
            {
                **base_state,
                "status": "running",
                "completed_shards": sorted(completed_shards),
                "completed_shard_count": len(completed_shards),
                "progress_pct": _progress_pct(len(completed_shards), len(shard_names)),
                "total_original_bytes": int(manifest.get("total_original_bytes", 0)),
                "total_q4_bytes": int(manifest.get("total_q4_bytes", 0)),
                "last_event": event.get("event"),
                "current_shard": event.get("shard_name"),
                "updated_at": _now(),
            },
        )

    try:
        manifest = quantize_model_dir_to_q4(
            source_model_dir,
            output_dir,
            completed_shards=completed_shards,
            existing_manifest=existing_manifest,
            progress_callback=on_progress,
            max_new_shards=max_new_shards,
        )
    except Exception as exc:
        return _write_state(
            state_path,
            {
                **base_state,
                "status": "error",
                "completed_shards": sorted(completed_shards),
                "completed_shard_count": len(completed_shards),
                "progress_pct": _progress_pct(len(completed_shards), len(shard_names)),
                "error": str(exc),
                "updated_at": _now(),
            },
        )

    complete = len(completed_shards) >= len(shard_names)
    status = "complete" if complete else "paused"
    if complete and partial_manifest_path.exists():
        partial_manifest_path.unlink()
    return _write_state(
        state_path,
        {
            **base_state,
            "status": status,
            "completed_shards": sorted(completed_shards),
            "completed_shard_count": len(completed_shards),
            "progress_pct": _progress_pct(len(completed_shards), len(shard_names)),
            "total_original_bytes": int(manifest.get("total_original_bytes", 0)),
            "total_q4_bytes": int(manifest.get("total_q4_bytes", 0)),
            "compression_ratio": manifest.get("compression_ratio", 0.0),
            "manifest_path": str(output_dir / "q4_manifest.json") if complete else str(partial_manifest_path),
            "updated_at": _now(),
        },
    )


def load_q4_conversion_state(model_id: str) -> dict | None:
    """Load the latest Q4 conversion state for a model id, if present."""
    return _read_json(q4_conversion_state_path(model_id))


def _source_shard_names(source_model_dir: Path) -> list[str]:
    index_path = source_model_dir / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return sorted({str(value) for value in (payload.get("weight_map") or {}).values()})


def _plan_blockers(plan: dict) -> list[str]:
    blockers: list[str] = []
    missing_shards = plan.get("missing_shards") or []
    if missing_shards:
        blockers.append(f"Missing source shards: {', '.join(str(item) for item in missing_shards)}.")
    if int(plan.get("q4_tensor_count", 0)) <= 0:
        blockers.append("No floating tensors were found for Q4 conversion.")
    return blockers or ["The compact Q4 plan is not ready for conversion."]


def _free_bytes_for(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def _progress_pct(done: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(min(done / total, 1.0) * 100, 2)


def _read_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _write_state(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
