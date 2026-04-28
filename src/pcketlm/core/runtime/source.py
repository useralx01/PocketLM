"""Runtime source readiness helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.model_import.inspect import ConfigSummary, inspect_qwen_source
from pcketlm.core.validation.files import validate_qwen_source


@dataclass(slots=True)
class RuntimeSourceDescriptor:
    """Loader-facing description of a source model folder."""

    model_dir: Path
    format_name: str
    ready: bool
    config: ConfigSummary | None
    config_path: Path | None
    tokenizer_path: Path | None
    index_path: Path | None
    shard_paths: list[Path] = field(default_factory=list)
    expected_shards: int = 0
    present_shards: int = 0
    missing_runtime_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plain_english_summary: str = ""

    def to_dict(self) -> dict:
        """Serialize the descriptor."""
        return {
            "model_dir": str(self.model_dir),
            "format_name": self.format_name,
            "ready": self.ready,
            "config": self.config.to_dict() if self.config else None,
            "config_path": str(self.config_path) if self.config_path else None,
            "tokenizer_path": str(self.tokenizer_path) if self.tokenizer_path else None,
            "index_path": str(self.index_path) if self.index_path else None,
            "shard_paths": [str(path) for path in self.shard_paths],
            "expected_shards": self.expected_shards,
            "present_shards": self.present_shards,
            "missing_runtime_files": list(self.missing_runtime_files),
            "warnings": list(self.warnings),
            "plain_english_summary": self.plain_english_summary,
        }


def _build_runtime_summary(ready: bool, missing_runtime_files: list[str], expected_shards: int, present_shards: int) -> str:
    if ready:
        return "The source folder has the core files and shard set the runtime needs for the first loader pass."
    if missing_runtime_files:
        joined = ", ".join(missing_runtime_files)
        return f"The source is not runtime-ready yet because these loader files are still missing: {joined}."
    if expected_shards and present_shards < expected_shards:
        return f"The source has its loader metadata, but only {present_shards} of {expected_shards} expected shard files are present."
    return "The source is not runtime-ready yet."


def describe_runtime_source(model_dir: Path) -> RuntimeSourceDescriptor:
    """Describe how ready a source folder is for the first runtime loader."""
    inspection = inspect_qwen_source(model_dir)
    validation = validate_qwen_source(model_dir)

    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"
    index_path = model_dir / "model.safetensors.index.json"

    shard_paths: list[Path] = []
    if index_path.exists():
        index_inspection = inspect_qwen_source(model_dir)
        if index_inspection.expected_shards:
            # Re-read the index via the existing inspector contract by using file discovery.
            shard_paths = sorted(model_dir.glob("model-*.safetensors"))

    missing_runtime_files = list(validation.missing_files)
    ready = validation.result == "ok" and inspection.expected_shards > 0

    return RuntimeSourceDescriptor(
        model_dir=model_dir,
        format_name=inspection.format_name,
        ready=ready,
        config=inspection.config,
        config_path=config_path if config_path.exists() else None,
        tokenizer_path=tokenizer_path if tokenizer_path.exists() else None,
        index_path=index_path if index_path.exists() else None,
        shard_paths=shard_paths,
        expected_shards=inspection.expected_shards,
        present_shards=inspection.present_shards,
        missing_runtime_files=missing_runtime_files,
        warnings=list(validation.warnings),
        plain_english_summary=_build_runtime_summary(
            ready,
            missing_runtime_files,
            inspection.expected_shards,
            inspection.present_shards,
        ),
    )
