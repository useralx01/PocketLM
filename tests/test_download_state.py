import json
from pathlib import Path

from pcketlm.core.model_import.download_state import estimate_download_state


def test_estimate_download_state_partial(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    state = estimate_download_state(tmp_path)
    assert state.status == "partial"
    assert "tokenizer.json" in state.missing_core_files


def test_estimate_download_state_ready(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
    (tmp_path / "merges.txt").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 10},
                "weight_map": {"a": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_text("1234567890", encoding="utf-8")
    state = estimate_download_state(tmp_path)
    assert state.status == "ready"
    assert state.expected_shards == 1
    assert state.present_shards == 1
