"""Model import service."""

from dataclasses import dataclass
from pathlib import Path

from pcketlm.core.model_import.inspect import inspect_qwen_source
from pcketlm.core.model_families import normalize_family_key
from pcketlm.core.registry.models import ModelRecord, ValidationSummary
from pcketlm.core.registry.repository import upsert_model_record
from pcketlm.core.storage.paths import (
    artifacts_root,
    benchmarks_root,
    ensure_base_directories,
    original_model_root,
    profiles_root,
)
from pcketlm.core.validation.files import validate_qwen_source


@dataclass(slots=True)
class ImportRequest:
    """Requested model import source."""

    model_id: str
    label: str
    family: str
    source_path: Path
    repo_id: str | None = None
    source_origin: str = "unknown"
    source_kind: str = "local-folder"
    model_type: str = "dense"
    format_name: str = "unknown"


def import_model(request: ImportRequest) -> ModelRecord:
    """Import a model into the local registry."""
    ensure_base_directories()

    family = normalize_family_key(request.family)
    inspection = inspect_qwen_source(request.source_path)
    validation = validate_qwen_source(request.source_path)
    original_path = original_model_root(request.model_id)

    for path in (
        original_path,
        artifacts_root(request.model_id),
        profiles_root(request.model_id),
        benchmarks_root(request.model_id),
    ):
        path.mkdir(parents=True, exist_ok=True)

    record = ModelRecord(
        model_id=request.model_id,
        label=request.label,
        family=family,
        model_type=request.model_type,
        source_path=request.source_path,
        original_path=original_path,
        source_kind=request.source_kind,
        source_origin=request.source_origin,
        repo_id=request.repo_id,
        format_name=inspection.format_name if inspection.format_name != "unknown" else request.format_name,
        config=inspection.config,
        imported=True,
        validated=validation.result == "ok",
        runnable=validation.result == "ok" and inspection.expected_shards > 0,
        validation=ValidationSummary(
            result=validation.result,
            missing_files=validation.missing_files,
            warnings=validation.warnings,
        ),
    )
    return upsert_model_record(record)
