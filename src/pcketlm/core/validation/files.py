"""Basic file validation helpers."""

from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.model_import.inspect import inspect_qwen_source


REQUIRED_QWEN_FILES = (
    "config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
)


@dataclass(slots=True)
class ValidationResult:
    """Result of validating a source model folder."""

    result: str
    missing_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def find_missing_required_files(model_dir: Path) -> list[str]:
    """Return missing required files for a baseline Qwen import."""
    return [name for name in REQUIRED_QWEN_FILES if not (model_dir / name).exists()]


def validate_qwen_source(model_dir: Path) -> ValidationResult:
    """Validate a Qwen source directory for the first import pass."""
    inspection = inspect_qwen_source(model_dir)
    warnings: list[str] = []

    missing_files = find_missing_required_files(model_dir)

    has_vocab_pair = (model_dir / "vocab.json").exists() and (model_dir / "merges.txt").exists()
    if not has_vocab_pair:
        warnings.append("Tokenizer pair files are incomplete or missing.")

    if inspection.index_present and inspection.present_shards < inspection.expected_shards:
        warnings.append(
            f"Only {inspection.present_shards} of {inspection.expected_shards} expected shard files are present."
        )

    if missing_files:
        return ValidationResult(
            result="partial",
            missing_files=missing_files,
            warnings=warnings,
        )

    result = "ok" if inspection.present_shards == inspection.expected_shards else "partial"
    return ValidationResult(result=result, missing_files=[], warnings=warnings)
