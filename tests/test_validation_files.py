from pathlib import Path

from pcketlm.core.validation.files import find_missing_required_files
from pcketlm.core.validation.files import validate_qwen_source


def test_find_missing_required_files(tmp_path: Path) -> None:
    missing = find_missing_required_files(tmp_path)
    assert "config.json" in missing
    assert "tokenizer.json" in missing
    assert "model.safetensors.index.json" in missing


def test_validate_qwen_source_ok(tmp_path: Path) -> None:
    for file_name in ("config.json", "tokenizer.json", "model.safetensors.index.json", "vocab.json", "merges.txt"):
        (tmp_path / file_name).write_text("x", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":1},"weight_map":{"a":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_text("x", encoding="utf-8")

    result = validate_qwen_source(tmp_path)
    assert result.result == "ok"
    assert result.missing_files == []


def test_validate_qwen_source_partial_when_shards_missing(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
    (tmp_path / "merges.txt").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":1},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_text("x", encoding="utf-8")

    result = validate_qwen_source(tmp_path)
    assert result.result == "partial"
    assert any("expected shard files" in warning for warning in result.warnings)
