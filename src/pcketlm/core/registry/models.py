"""Registry model records."""

from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.model_import.inspect import ConfigSummary


@dataclass(slots=True)
class ValidationSummary:
    """Validation status for an imported model."""

    result: str
    missing_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize the validation summary."""
        return {
            "result": self.result,
            "missing_files": list(self.missing_files),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ValidationSummary":
        """Build a validation summary from plain data."""
        return cls(
            result=str(payload.get("result", "unknown")),
            missing_files=list(payload.get("missing_files", [])),
            warnings=list(payload.get("warnings", [])),
        )


@dataclass(slots=True)
class ModelRecord:
    """Registered model record."""

    model_id: str
    label: str
    family: str
    model_type: str
    source_path: Path
    original_path: Path
    source_kind: str = "local-folder"
    source_origin: str = "unknown"
    repo_id: str | None = None
    format_name: str = "unknown"
    config: ConfigSummary | None = None
    imported: bool = False
    validated: bool = False
    runnable: bool = False
    validation: ValidationSummary | None = None
    artifact_paths: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize the model record."""
        return {
            "model_id": self.model_id,
            "label": self.label,
            "family": self.family,
            "model_type": self.model_type,
            "source_path": str(self.source_path),
            "original_path": str(self.original_path),
            "source_kind": self.source_kind,
            "source_origin": self.source_origin,
            "repo_id": self.repo_id,
            "format_name": self.format_name,
            "config": self.config.to_dict() if self.config else None,
            "imported": self.imported,
            "validated": self.validated,
            "runnable": self.runnable,
            "validation": self.validation.to_dict() if self.validation else None,
            "artifact_paths": [str(path) for path in self.artifact_paths],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelRecord":
        """Build a model record from plain data."""
        validation_payload = payload.get("validation")
        config_payload = payload.get("config") or {}
        return cls(
            model_id=str(payload["model_id"]),
            label=str(payload["label"]),
            family=str(payload["family"]),
            model_type=str(payload["model_type"]),
            source_path=Path(payload["source_path"]),
            original_path=Path(payload["original_path"]),
            source_kind=str(payload.get("source_kind", "local-folder")),
            source_origin=str(payload.get("source_origin", "unknown")),
            repo_id=payload.get("repo_id"),
            format_name=str(payload.get("format_name", "unknown")),
            config=ConfigSummary(**config_payload) if config_payload else None,
            imported=bool(payload.get("imported", False)),
            validated=bool(payload.get("validated", False)),
            runnable=bool(payload.get("runnable", False)),
            validation=ValidationSummary.from_dict(validation_payload) if validation_payload else None,
            artifact_paths=[Path(path) for path in payload.get("artifact_paths", [])],
        )
