import json
from pathlib import Path

from pcketlm.core.model_import.inspect import inspect_qwen_source


def test_inspect_qwen_source_reads_config_and_index(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 5120,
                "num_hidden_layers": 48,
                "num_attention_heads": 40,
                "max_position_embeddings": 32768,
                "vocab_size": 152064,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1234},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_text("x", encoding="utf-8")

    inspection = inspect_qwen_source(tmp_path)
    assert inspection.config.model_type == "qwen2"
    assert inspection.expected_shards == 2
    assert inspection.present_shards == 1
    assert inspection.format_name == "safetensors-sharded"
