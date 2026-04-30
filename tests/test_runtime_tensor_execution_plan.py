import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from pcketlm.core.runtime.tensor_execution_plan import build_tensor_execution_plan


def test_build_tensor_execution_plan_groups_tensors_into_runtime_units(tmp_path: Path, monkeypatch) -> None:
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
    (model_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (model_dir / "merges.txt").write_text("", encoding="utf-8")

    shard_1 = model_dir / "model-00001-of-00002.safetensors"
    shard_2 = model_dir / "model-00002-of-00002.safetensors"

    save_file(
        {
            "model.embed_tokens.weight": torch.zeros((16, 8), dtype=torch.bfloat16),
            "model.layers.0.input_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
            "model.layers.0.self_attn.q_proj.weight": torch.ones((8, 8), dtype=torch.bfloat16),
            "model.layers.0.mlp.gate_proj.weight": torch.ones((8, 8), dtype=torch.bfloat16),
        },
        str(shard_1),
    )
    save_file(
        {
            "model.layers.1.post_attention_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
            "model.layers.1.self_attn.q_proj.weight": torch.ones((8, 8), dtype=torch.bfloat16),
            "model.layers.1.mlp.up_proj.weight": torch.ones((8, 8), dtype=torch.bfloat16),
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
                    "model.layers.0.input_layernorm.weight": shard_1.name,
                    "model.layers.0.self_attn.q_proj.weight": shard_1.name,
                    "model.layers.0.mlp.gate_proj.weight": shard_1.name,
                    "model.layers.1.post_attention_layernorm.weight": shard_2.name,
                    "model.layers.1.self_attn.q_proj.weight": shard_2.name,
                    "model.layers.1.mlp.up_proj.weight": shard_2.name,
                    "model.norm.weight": shard_2.name,
                    "lm_head.weight": shard_2.name,
                },
            }
        ),
        encoding="utf-8",
    )

    plan = build_tensor_execution_plan(model_id, model_dir)

    assert plan.ready is True
    assert plan.unit_count == 9
    assert plan.phases == ["prefill", "layer-entry", "layer-attention", "layer-mlp", "decode-head"]
    assert plan.plan_path.exists()

    embeddings = next(unit for unit in plan.units if unit.unit_id == "embeddings")
    assert embeddings.phase == "prefill"
    assert embeddings.tensor_count == 1

    layer0_attn = next(unit for unit in plan.units if unit.unit_id == "layer-00-attention")
    assert layer0_attn.phase == "layer-attention"
    assert layer0_attn.layer_index == 0

    final_norm = next(unit for unit in plan.units if unit.unit_id == "final_norm")
    lm_head = next(unit for unit in plan.units if unit.unit_id == "lm_head")
    assert final_norm.phase == "decode-head"
    assert lm_head.phase == "decode-head"


def test_build_tensor_execution_plan_blocks_when_catalog_is_not_ready(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    model_id = "qwen-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    plan = build_tensor_execution_plan(model_id, model_dir)

    assert plan.ready is False
    assert plan.unit_count == 0
    assert plan.plan_path.exists() is False
    assert plan.blockers


def test_build_tensor_execution_plan_handles_qwen32b_layer_count(tmp_path: Path, monkeypatch) -> None:
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
    (model_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (model_dir / "merges.txt").write_text("", encoding="utf-8")

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

    plan = build_tensor_execution_plan(model_id, model_dir)

    assert plan.ready is True
    assert len([unit for unit in plan.units if unit.component_group == "layer_norm"]) == 64
    assert plan.units[1].unit_id == "layer-00-layer_norm"
    assert plan.units[-2].unit_id == "layer-63-layer_norm"


def test_build_tensor_execution_plan_groups_moe_experts_separately(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    model_id = "qwen3-moe-plan-test"
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
    (model_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (model_dir / "merges.txt").write_text("", encoding="utf-8")

    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.embed_tokens.weight": torch.zeros((16, 8), dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.ones((8, 8), dtype=torch.bfloat16),
        "model.layers.0.mlp.gate.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.ones((8, 4), dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.3.gate_proj.weight": torch.ones((4, 8), dtype=torch.bfloat16),
        "model.norm.weight": torch.ones((8,), dtype=torch.bfloat16),
        "lm_head.weight": torch.ones((16, 8), dtype=torch.bfloat16),
    }
    save_file(tensors, str(shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": shard.stat().st_size}, "weight_map": {name: shard.name for name in tensors}}),
        encoding="utf-8",
    )

    plan = build_tensor_execution_plan(model_id, model_dir)

    assert plan.ready is True
    assert "layer-router" in plan.phases
    assert "layer-expert" in plan.phases
    router = next(unit for unit in plan.units if unit.unit_id == "layer-00-router")
    expert0 = next(unit for unit in plan.units if unit.unit_id == "layer-00-expert-000")
    expert3 = next(unit for unit in plan.units if unit.unit_id == "layer-00-expert-003")
    assert router.phase == "layer-router"
    assert router.expert_index is None
    assert expert0.phase == "layer-expert"
    assert expert0.expert_index == 0
    assert expert0.tensor_count == 3
    assert expert3.expert_index == 3
