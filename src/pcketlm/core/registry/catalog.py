"""Registry-backed model catalog helpers."""

from __future__ import annotations

from dataclasses import dataclass

from pcketlm.core.model_import.download_state import estimate_download_state
from pcketlm.core.registry.repository import load_model_registry


@dataclass(slots=True)
class CatalogEntry:
    """Combined registry and source-state view for one model."""

    model_id: str
    label: str
    family: str
    model_type: str
    format_name: str
    source_origin: str
    imported: bool
    validated: bool
    runnable: bool
    source_status: str
    bytes_on_disk: int
    progress_pct: float | None
    missing_core_files: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        """Serialize the catalog entry."""
        return {
            "model_id": self.model_id,
            "label": self.label,
            "family": self.family,
            "model_type": self.model_type,
            "format_name": self.format_name,
            "source_origin": self.source_origin,
            "imported": self.imported,
            "validated": self.validated,
            "runnable": self.runnable,
            "source_status": self.source_status,
            "bytes_on_disk": self.bytes_on_disk,
            "progress_pct": self.progress_pct,
            "missing_core_files": list(self.missing_core_files),
            "warnings": list(self.warnings),
        }


def build_model_catalog() -> list[CatalogEntry]:
    """Build a live catalog view from the registry and source folders."""
    records = load_model_registry()
    entries: list[CatalogEntry] = []

    for record in sorted(records.values(), key=lambda item: item.model_id):
        download_state = estimate_download_state(record.source_path)
        warnings = list(record.validation.warnings if record.validation else [])

        entries.append(
            CatalogEntry(
                model_id=record.model_id,
                label=record.label,
                family=record.family,
                model_type=record.model_type,
                format_name=record.format_name,
                source_origin=record.source_origin,
                imported=record.imported,
                validated=record.validated,
                runnable=record.runnable,
                source_status=download_state.status,
                bytes_on_disk=download_state.bytes_on_disk,
                progress_pct=download_state.progress_pct,
                missing_core_files=download_state.missing_core_files,
                warnings=warnings,
            )
        )

    return entries
