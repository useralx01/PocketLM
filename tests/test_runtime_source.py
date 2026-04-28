import json
from pathlib import Path

from pcketlm.core.runtime.source import describe_runtime_source


def test_describe_runtime_source_partial(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    descriptor = describe_runtime_source(tmp_path)

    assert descriptor.ready is False
    assert "tokenizer.json" in descriptor.missing_runtime_files
    assert "not runtime-ready yet" in descriptor.plain_english_summary


def test_describe_runtime_source_ready(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
    (tmp_path / "merges.txt").write_text("", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {"a": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_text("x", encoding="utf-8")

    descriptor = describe_runtime_source(tmp_path)

    assert descriptor.ready is True
    assert descriptor.index_path is not None
    assert descriptor.tokenizer_path is not None
    assert descriptor.expected_shards == 1
    assert descriptor.present_shards == 1
    assert descriptor.plain_english_summary.startswith("The source folder has the core files")
