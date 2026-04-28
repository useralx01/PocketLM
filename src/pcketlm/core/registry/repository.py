"""Registry persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pcketlm.core.registry.models import ModelRecord
from pcketlm.core.storage.paths import ensure_base_directories, original_model_root, registry_file


def _relocate_record_if_needed(record: ModelRecord) -> tuple[ModelRecord, bool]:
    """Relocate stale in-project paths after the project root moves."""
    current_original = original_model_root(record.model_id)
    changed = False

    if not record.original_path.exists() and current_original.exists():
        record.original_path = current_original
        changed = True

    if not record.source_path.exists() and current_original.exists():
        record.source_path = current_original
        changed = True

    artifact_paths: list[Path] = []
    for artifact_path in record.artifact_paths:
        if artifact_path.exists():
            artifact_paths.append(artifact_path)
            continue

        candidate = current_original.parent / "artifacts" / artifact_path.name
        if candidate.exists():
            artifact_paths.append(candidate)
            changed = True
        else:
            artifact_paths.append(artifact_path)

    if artifact_paths != record.artifact_paths:
        record.artifact_paths = artifact_paths
        changed = True

    return record, changed


def load_model_registry() -> dict[str, ModelRecord]:
    """Load every registered model record from disk."""
    ensure_base_directories()
    path = registry_file()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {
        item["model_id"]: ModelRecord.from_dict(item)
        for item in payload.get("models", [])
    }
    changed = False
    for model_id, record in list(records.items()):
        records[model_id], relocated = _relocate_record_if_needed(record)
        changed = changed or relocated

    if changed:
        save_model_registry(records)

    return records


def save_model_registry(records: dict[str, ModelRecord]) -> None:
    """Persist the model registry to disk."""
    ensure_base_directories()
    payload = {
        "models": [record.to_dict() for record in sorted(records.values(), key=lambda item: item.model_id)]
    }
    registry_file().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_model_record(record: ModelRecord) -> ModelRecord:
    """Insert or update a model record on disk."""
    records = load_model_registry()
    records[record.model_id] = record
    save_model_registry(records)
    return record


def delete_model_record(model_id: str) -> bool:
    """Delete one model record from the registry."""
    records = load_model_registry()
    if model_id not in records:
        return False
    del records[model_id]
    save_model_registry(records)
    return True
