import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from pcketlm.core.runtime.tensor_execution_plan import build_tensor_execution_plan
from pcketlm.core.runtime.tensor_loader import (
    load_tensor_by_name,
    load_tensors_by_name,
    q4_source_status,
    reset_tensor_load_stats,
    tensor_load_stats_snapshot,
)
from pcketlm.core.runtime.tensor_residency import clear_tensor_residency_cache, q4_packed_cache_stats
from tools.quantize_to_q4 import (
    dequantize_q4_tensor,
    pack_int4,
    plan_model_dir_to_q4,
    quantize_model_dir_to_q4,
    quantize_tensor_to_q4,
    unpack_int4,
)


def test_q4_pack_unpack_is_bit_exact() -> None:
    values = torch.tensor([-7, -6, -1, 0, 1, 6, 7], dtype=torch.int8)

    packed = pack_int4(values)
    unpacked = unpack_int4(packed, int(values.numel()))

    assert unpacked.tolist() == values.tolist()


def test_q4_round_trip_grid_tensor_has_low_relative_error() -> None:
    tensor = torch.tensor(
        [
            [-7.0, -3.0, 0.0, 3.0, 7.0],
            [-14.0, -6.0, 0.0, 6.0, 14.0],
        ],
        dtype=torch.float16,
    )

    packed, scales, metadata = quantize_tensor_to_q4(tensor)
    restored = dequantize_q4_tensor(packed, scales, metadata["shape"], dtype=torch.float16)
    rel = ((restored.float() - tensor.float()).abs() / tensor.float().abs().clamp_min(1e-6)).max().item()

    assert rel < 0.02


def test_native_q4_selected_moe_matches_dequantized_path() -> None:
    from pcketlm.native import q4_moe_selected_forward_u16

    torch.manual_seed(123)
    hidden_size = 16
    intermediate_size = 12
    selected_count = 2
    hidden = torch.randn((1, hidden_size), dtype=torch.float16)
    route_weights = torch.tensor([0.7, 0.3], dtype=torch.float32)
    native_experts = []
    python_experts = []
    for _index in range(selected_count):
        gate = torch.randn((intermediate_size, hidden_size), dtype=torch.float16)
        up = torch.randn((intermediate_size, hidden_size), dtype=torch.float16)
        down = torch.randn((hidden_size, intermediate_size), dtype=torch.float16)
        gate_packed, gate_scales, gate_meta = quantize_tensor_to_q4(gate)
        up_packed, up_scales, up_meta = quantize_tensor_to_q4(up)
        down_packed, down_scales, down_meta = quantize_tensor_to_q4(down)
        native_experts.append((gate_packed, gate_scales, up_packed, up_scales, down_packed, down_scales))
        python_experts.append(
            (
                dequantize_q4_tensor(gate_packed, gate_scales, gate_meta["shape"], dtype=torch.float16),
                dequantize_q4_tensor(up_packed, up_scales, up_meta["shape"], dtype=torch.float16),
                dequantize_q4_tensor(down_packed, down_scales, down_meta["shape"], dtype=torch.float16),
            )
        )

    expected = torch.zeros((1, hidden_size), dtype=torch.float32)
    for expert_index, (gate, up, down) in enumerate(python_experts):
        expert_hidden = F.silu(F.linear(hidden.float(), gate.float())) * F.linear(hidden.float(), up.float())
        expected = expected + F.linear(expert_hidden, down.float()) * route_weights[expert_index]

    actual = q4_moe_selected_forward_u16(
        hidden,
        native_experts,
        route_weights,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    assert torch.allclose(actual.float(), expected.to(torch.float16).float(), atol=1e-2, rtol=1e-2)


def test_q4_artifact_size_is_about_quarter_fp16(tmp_path: Path) -> None:
    model_dir = _write_quantizer_fixture(tmp_path, torch.arange(8192, dtype=torch.float16).reshape(64, 128))
    manifest = quantize_model_dir_to_q4(model_dir, tmp_path / "q4")

    assert manifest["compression_ratio"] < 0.27


def test_q4_dry_run_plan_reads_headers_without_writing_artifact(tmp_path: Path) -> None:
    model_dir = _write_quantizer_fixture(tmp_path, torch.arange(8192, dtype=torch.float16).reshape(64, 128))
    output_dir = tmp_path / "q4-plan-only"

    plan = plan_model_dir_to_q4(model_dir, output_dir)

    assert plan["ready_for_conversion"] is True
    assert plan["q4_tensor_count"] == 1
    assert plan["missing_shards"] == []
    assert plan["estimated_total_q4_bytes"] > 0
    assert plan["estimated_total_q4_bytes"] < plan["total_original_bytes"] * 0.27
    assert plan["planned_output_dir"] == str(output_dir.resolve())
    assert not output_dir.exists()


def test_q4_dry_run_plan_splits_expert_and_non_expert_bytes(tmp_path: Path) -> None:
    model_dir = tmp_path / "source"
    model_dir.mkdir(parents=True)
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight": torch.ones((8, 8), dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.ones((8, 8), dtype=torch.bfloat16),
    }
    save_file(tensors, str(shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )

    plan = plan_model_dir_to_q4(model_dir, tmp_path / "q4")

    assert plan["q4_tensor_count"] == 2
    assert plan["estimated_expert_q4_bytes"] > 0
    assert plan["estimated_non_expert_q4_bytes"] > 0
    assert plan["estimated_expert_q4_bytes"] == plan["estimated_non_expert_q4_bytes"]


def test_q4_dry_run_plan_blocks_missing_shards(tmp_path: Path) -> None:
    model_dir = tmp_path / "source"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}}),
        encoding="utf-8",
    )

    plan = plan_model_dir_to_q4(model_dir, tmp_path / "q4")

    assert plan["ready_for_conversion"] is False
    assert plan["missing_shards"] == ["missing.safetensors"]
    assert plan["q4_tensor_count"] == 0


def test_q4_loader_dequantizes_to_runtime_tensor(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    model_id = "q4-loader-test"
    model_dir = _write_runtime_fixture(tmp_path, model_id)
    build_tensor_execution_plan(model_id, model_dir)
    quantize_model_dir_to_q4(model_dir, tmp_path / "models" / model_id / "artifacts" / "q4")
    reset_tensor_load_stats()

    loaded = load_tensor_by_name(model_id, "model.layers.0.self_attn.q_proj.weight")
    stats = tensor_load_stats_snapshot()
    status = q4_source_status(model_id)

    assert loaded.ready is True
    assert loaded.q4_loaded is True
    assert loaded.tensor is not None
    assert loaded.tensor.dtype == torch.float16
    assert torch.allclose(loaded.tensor.float(), torch.full((8, 8), 3.0), atol=0.01)
    assert stats.q4_loads == 1
    assert stats.to_dict()["q4_loaded"] is True
    assert status["ready"] is True


def test_q4_loader_reuses_packed_cache_on_repeated_load(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    clear_tensor_residency_cache()
    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.setenv("PCKETLM_Q4_PACKED_CACHE_MB", "1")
    monkeypatch.delenv("PCKETLM_DISABLE_Q4_PACKED_CACHE", raising=False)
    model_id = "q4-loader-cache-test"
    model_dir = _write_runtime_fixture(tmp_path, model_id)
    build_tensor_execution_plan(model_id, model_dir)
    quantize_model_dir_to_q4(model_dir, tmp_path / "models" / model_id / "artifacts" / "q4")
    reset_tensor_load_stats()

    first = load_tensor_by_name(model_id, "model.layers.0.self_attn.q_proj.weight")
    second = load_tensor_by_name(model_id, "model.layers.0.self_attn.q_proj.weight")
    stats = q4_packed_cache_stats()
    load_stats = tensor_load_stats_snapshot()

    assert first.ready is True
    assert second.ready is True
    assert torch.allclose(first.tensor.float(), second.tensor.float())
    assert stats.misses == 1
    assert stats.hits == 1
    assert stats.disk_reads == 1
    assert load_stats.shard_opens == 2


def test_q4_batch_loader_uses_native_many_dequant(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage
    import pcketlm.native as native_module

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    model_id = "q4-loader-batch-test"
    model_dir = _write_runtime_fixture(tmp_path, model_id)
    build_tensor_execution_plan(model_id, model_dir)
    quantize_model_dir_to_q4(model_dir, tmp_path / "models" / model_id / "artifacts" / "q4")
    calls = {"count": 0, "item_count": 0}

    def fake_many(items):
        calls["count"] += 1
        calls["item_count"] += len(items)
        outputs = []
        for packed, scales, num_channels, channel_size in items:
            outputs.append(
                dequantize_q4_tensor(
                    packed,
                    scales,
                    [int(num_channels), int(channel_size)],
                    dtype=torch.float16,
                ).reshape(-1)
            )
        return outputs

    monkeypatch.setattr(native_module, "q4_dequant_many_to_fp16", fake_many)
    reset_tensor_load_stats()

    loaded = load_tensors_by_name(
        model_id,
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ],
    )
    stats = tensor_load_stats_snapshot()

    assert calls == {"count": 1, "item_count": 2}
    assert loaded["model.layers.0.self_attn.q_proj.weight"].ready is True
    assert loaded["model.layers.0.mlp.gate_proj.weight"].ready is True
    assert stats.q4_loads == 2


def _write_quantizer_fixture(tmp_path: Path, tensor: torch.Tensor) -> Path:
    model_dir = tmp_path / "source"
    model_dir.mkdir(parents=True)
    shard = model_dir / "model-00001-of-00001.safetensors"
    save_file({"weight": tensor}, str(shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": shard.stat().st_size}, "weight_map": {"weight": shard.name}}),
        encoding="utf-8",
    )
    return model_dir


def _write_runtime_fixture(tmp_path: Path, model_id: str) -> Path:
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 8,
                "num_hidden_layers": 1,
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
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.embed_tokens.weight": torch.arange(128, dtype=torch.bfloat16).reshape(16, 8),
        "model.layers.0.input_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.full((8,), 2, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.full((8, 8), 3, dtype=torch.bfloat16),
        "model.layers.0.mlp.gate_proj.weight": torch.full((8, 8), 4, dtype=torch.bfloat16),
        "model.norm.weight": torch.full((8,), 5, dtype=torch.bfloat16),
        "lm_head.weight": torch.full((16, 8), 6, dtype=torch.bfloat16),
    }
    save_file(tensors, str(shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )
    return model_dir
