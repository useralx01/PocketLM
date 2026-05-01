import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from pcketlm.core.runtime.tensor_catalog import build_tensor_catalog, find_tensor_catalog_entry, load_tensor_entry_index


def test_build_tensor_catalog_reads_tensor_headers_and_groups_layers(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    model_id = "qwen-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)

    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "max_position_embeddings": 128,
                "vocab_size": 16,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.model").write_text("", encoding="utf-8")

    shard_1 = model_dir / "model-00001-of-00002.safetensors"
    shard_2 = model_dir / "model-00002-of-00002.safetensors"

    save_file(
        {
            "model.embed_tokens.weight": torch.zeros((16, 8), dtype=torch.bfloat16),
            "model.layers.0.mlp.gate_proj.weight": torch.ones((8, 8), dtype=torch.bfloat16),
            "model.layers.0.input_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
        },
        str(shard_1),
    )
    save_file(
        {
            "model.layers.1.self_attn.q_proj.weight": torch.ones((8, 8), dtype=torch.bfloat16),
            "model.norm.weight": torch.ones((8,), dtype=torch.bfloat16),
            "lm_head.weight": torch.ones((16, 8), dtype=torch.bfloat16),
        },
        str(shard_2),
    )

    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard_1.stat().st_size + shard_2.stat().st_size},
                "weight_map": {
                    "model.embed_tokens.weight": shard_1.name,
                    "model.layers.0.mlp.gate_proj.weight": shard_1.name,
                    "model.layers.0.input_layernorm.weight": shard_1.name,
                    "model.layers.1.self_attn.q_proj.weight": shard_2.name,
                    "model.norm.weight": shard_2.name,
                    "lm_head.weight": shard_2.name,
                },
            }
        ),
        encoding="utf-8",
    )

    catalog = build_tensor_catalog(model_id, model_dir)

    assert catalog.ready is True
    assert catalog.tensor_count == 6
    assert catalog.shard_count == 2
    assert catalog.layer_count == 2
    assert catalog.dtype_counts == {"BF16": 6}
    assert catalog.component_group_counts["embeddings"] == 1
    assert catalog.component_group_counts["mlp"] == 1
    assert catalog.component_group_counts["attention"] == 1
    assert catalog.component_group_counts["layer_norm"] == 1
    assert catalog.component_group_counts["final_norm"] == 1
    assert catalog.component_group_counts["lm_head"] == 1
    assert catalog.catalog_path.exists()
    first_layer_tensor = next(entry for entry in catalog.tensors if entry.tensor_name == "model.layers.0.mlp.gate_proj.weight")
    assert first_layer_tensor.layer_index == 0
    assert first_layer_tensor.data_nbytes > 0

    entry_index = load_tensor_entry_index(model_id)
    assert set(entry_index) == {entry.tensor_name for entry in catalog.tensors}
    assert find_tensor_catalog_entry(model_id, "model.layers.0.mlp.gate_proj.weight") == first_layer_tensor


def test_build_tensor_catalog_blocks_on_incomplete_source(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    model_id = "qwen-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    catalog = build_tensor_catalog(model_id, model_dir)

    assert catalog.ready is False
    assert catalog.tensor_count == 0
    assert catalog.catalog_path.exists() is False
    assert catalog.blockers


def test_build_tensor_catalog_records_qwen32b_config_values(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    model_id = "qwen32b-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)

    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "intermediate_size": 27648,
                "max_position_embeddings": 32768,
                "vocab_size": 152064,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.model").write_text("", encoding="utf-8")

    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.embed_tokens.weight": torch.zeros((2, 1), dtype=torch.bfloat16),
        "lm_head.weight": torch.ones((2, 1), dtype=torch.bfloat16),
    }
    tensors.update({f"model.layers.{index}.input_layernorm.weight": torch.ones((1,), dtype=torch.bfloat16) for index in range(64)})
    save_file(tensors, str(shard))

    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {tensor_name: shard.name for tensor_name in tensors},
            }
        ),
        encoding="utf-8",
    )

    catalog = build_tensor_catalog(model_id, model_dir)

    assert catalog.ready is True
    assert catalog.layer_count == 64
    assert catalog.num_hidden_layers == 64
    assert catalog.hidden_size == 5120
    assert catalog.num_attention_heads == 40
    assert catalog.num_key_value_heads == 8
    assert catalog.intermediate_size == 27648
    assert catalog.vocab_size == 152064


def test_build_tensor_catalog_classifies_moe_router_and_experts(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    model_id = "qwen3-moe-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 16,
                "moe_intermediate_size": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "max_position_embeddings": 128,
                "vocab_size": 16,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.model").write_text("", encoding="utf-8")

    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.embed_tokens.weight": torch.zeros((16, 8), dtype=torch.bfloat16),
        "model.layers.0.mlp.gate.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.ones((8, 4), dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.3.gate_proj.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.layers.1.mlp.gate.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.norm.weight": torch.ones((8,), dtype=torch.bfloat16),
        "lm_head.weight": torch.ones((16, 8), dtype=torch.bfloat16),
    }
    save_file(tensors, str(shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": shard.stat().st_size}, "weight_map": {name: shard.name for name in tensors}}),
        encoding="utf-8",
    )

    catalog = build_tensor_catalog(model_id, model_dir)

    assert catalog.ready is True
    assert catalog.model_type == "qwen3_moe"
    assert catalog.num_experts == 4
    assert catalog.num_experts_per_tok == 2
    assert catalog.moe_intermediate_size == 4
    assert catalog.component_group_counts["router"] == 2
    assert catalog.component_group_counts["expert_mlp"] == 4
    expert = find_tensor_catalog_entry(model_id, "model.layers.0.mlp.experts.3.gate_proj.weight")
    assert expert is not None
    assert expert.layer_index == 0
    assert expert.expert_index == 3
    assert expert.component_group == "expert_mlp"


def test_build_tensor_catalog_classifies_mixtral_router_and_experts(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    model_id = "mixtral-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MixtralForCausalLM"],
                "model_type": "mixtral",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 4,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
                "max_position_embeddings": 128,
                "vocab_size": 16,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.model").write_text("", encoding="utf-8")

    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.embed_tokens.weight": torch.zeros((16, 8), dtype=torch.bfloat16),
        "model.layers.0.block_sparse_moe.gate.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.layers.0.block_sparse_moe.experts.3.w1.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.layers.0.block_sparse_moe.experts.3.w2.weight": torch.ones((8, 4), dtype=torch.bfloat16),
        "model.layers.0.block_sparse_moe.experts.3.w3.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.norm.weight": torch.ones((8,), dtype=torch.bfloat16),
        "lm_head.weight": torch.ones((16, 8), dtype=torch.bfloat16),
    }
    save_file(tensors, str(shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": shard.stat().st_size}, "weight_map": {name: shard.name for name in tensors}}),
        encoding="utf-8",
    )

    catalog = build_tensor_catalog(model_id, model_dir)

    assert catalog.ready is True
    assert catalog.model_type == "mixtral"
    assert catalog.num_experts == 4
    assert catalog.moe_intermediate_size == 4
    assert catalog.component_group_counts["router"] == 1
    assert catalog.component_group_counts["expert_mlp"] == 3
    expert = find_tensor_catalog_entry(model_id, "model.layers.0.block_sparse_moe.experts.3.w2.weight")
    assert expert is not None
    assert expert.layer_index == 0
    assert expert.expert_index == 3
    assert expert.component_group == "expert_mlp"
