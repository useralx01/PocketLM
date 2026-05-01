import json
from pathlib import Path

import numpy as np


def _fixture_root(name: str) -> Path:
    return Path(__file__).parent / "fixtures" / name


def _load_reference(name: str) -> tuple[dict, Path]:
    root = _fixture_root(name)
    reference_path = root / "reference" / "reference.json"
    return json.loads(reference_path.read_text(encoding="utf-8")), root


def test_tiny_qwen3_moe_oracle_shapes_match_config() -> None:
    reference, root = _load_reference("tiny_moe_qwen3")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))

    assert reference["config"]["model_type"] == "qwen3_moe"
    assert reference["config"]["num_hidden_layers"] == 2
    assert reference["config"]["hidden_size"] == 128
    assert reference["config"]["num_experts"] == 8
    assert reference["config"]["num_experts_per_tok"] == 2
    assert len(reference["generated_token_ids"]) == 10
    assert config["hidden_size"] == 128

    embedding = np.load(root / "reference" / "embedding_first_token.npy")
    layer0_attention = np.load(root / "reference" / "layer0_attention_output.npy")
    router_logits = np.load(root / "reference" / "layer0_router_logits.npy")
    final_hidden = np.load(root / "reference" / "final_hidden_before_lm_head.npy")

    assert embedding.shape == (1, 1, 128)
    assert layer0_attention.shape == (1, 4, 128)
    assert router_logits.shape == (4, 8)
    assert final_hidden.shape == (1, 4, 128)


def test_tiny_mixtral_moe_oracle_shapes_match_config() -> None:
    reference, root = _load_reference("tiny_moe_mixtral")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))

    assert reference["config"]["model_type"] == "mixtral"
    assert reference["config"]["num_hidden_layers"] == 2
    assert reference["config"]["hidden_size"] == 128
    assert reference["config"]["num_experts"] == 8
    assert reference["config"]["num_experts_per_tok"] == 2
    assert len(reference["generated_token_ids"]) == 10
    assert config["hidden_size"] == 128

    embedding = np.load(root / "reference" / "embedding_first_token.npy")
    layer0_attention = np.load(root / "reference" / "layer0_attention_output.npy")
    router_logits = np.load(root / "reference" / "layer0_router_logits.npy")
    final_hidden = np.load(root / "reference" / "final_hidden_before_lm_head.npy")

    assert embedding.shape == (1, 1, 128)
    assert layer0_attention.shape == (1, 4, 128)
    assert router_logits.shape == (4, 8)
    assert final_hidden.shape == (1, 4, 128)
