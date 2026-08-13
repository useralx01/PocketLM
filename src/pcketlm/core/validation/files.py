"""Basic file validation helpers."""

from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.model_import.inspect import inspect_model_source
from pcketlm.core.model_families import normalize_family_key


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
    return validate_model_source(model_dir, family="qwen")


def validate_model_source(model_dir: Path, *, family: str | None = None) -> ValidationResult:
    """Validate local source structure without loading model weights."""
    inspection = inspect_model_source(model_dir)
    family_key = normalize_family_key(family)
    warnings: list[str] = []

    if inspection.format_name == "gguf":
        return ValidationResult(result="ok")

    missing_files: list[str] = []
    if not (model_dir / "config.json").exists():
        missing_files.append("config.json")
    has_weights = inspection.present_shards > 0
    if not has_weights:
        missing_files.append("model.safetensors.index.json")

    if family_key != "kronos":
        has_tokenizer = any(
            (model_dir / name).exists()
            for name in ("tokenizer.json", "tokenizer.model", "spiece.model")
        )
        if not has_tokenizer:
            missing_files.append("tokenizer.json")

    has_vocab_pair = (model_dir / "vocab.json").exists() and (model_dir / "merges.txt").exists()
    if family_key in {"qwen", "qwen-moe"} and not has_vocab_pair:
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

    result = "ok" if has_weights and inspection.present_shards == inspection.expected_shards else "partial"
    return ValidationResult(result=result, missing_files=[], warnings=warnings)
