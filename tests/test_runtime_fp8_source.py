import json
import struct
from pathlib import Path

import torch

from pcketlm.core.runtime.fp8_source import load_fp8_weight_pair, plan_fp8_layer_working_set
from pcketlm.core.runtime.tensor_catalog import build_tensor_catalog, find_tensor_catalog_entry


def test_tensor_catalog_records_fp8_weight_scale_pairs(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)

    catalog = build_tensor_catalog(model_id, model_dir)

    assert catalog.ready is True
    assert catalog.fp8_weight_count == 3
    assert catalog.fp8_scale_count == 3
    assert catalog.fp8_pair_count == 3
    weight = find_tensor_catalog_entry(model_id, "model.layers.0.mlp.experts.1.gate_proj.weight")
    scale = find_tensor_catalog_entry(model_id, "model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv")
    assert weight is not None
    assert scale is not None
    assert weight.dtype == "F8_E4M3"
    assert weight.tensor_role == "weight"
    assert weight.scale_tensor_name == scale.tensor_name
    assert weight.physical_format == "fp8_block_scaled"
    assert scale.tensor_role == "scale_companion"
    assert scale.weight_tensor_name == weight.tensor_name


def test_plan_fp8_layer_working_set_keeps_selected_expert_subset(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    plan = plan_fp8_layer_working_set(model_id, 0, [1])

    assert plan.ready is True
    assert "model.layers.0.self_attn.q_proj.weight" in plan.tensor_names
    assert "model.layers.0.mlp.experts.1.gate_proj.weight" in plan.tensor_names
    assert "model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv" in plan.tensor_names
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in plan.tensor_names
    assert plan.fp8_weight_count == 2
    assert plan.scale_count == 2
    assert plan.total_nbytes == plan.fp8_weight_bytes + plan.scale_bytes + plan.non_fp8_bytes


def test_load_fp8_weight_pair_reads_raw_weight_bytes_and_scale(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    loaded = load_fp8_weight_pair(model_id, "model.layers.0.self_attn.q_proj.weight")

    assert loaded.ready is True
    assert loaded.fp8_bytes is not None
    assert loaded.scale_tensor is not None
    assert loaded.fp8_bytes.dtype == torch.uint8
    assert loaded.fp8_bytes.shape == (2, 4)
    assert loaded.fp8_bytes.flatten().tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert loaded.scale_tensor.dtype == torch.float32
    assert torch.allclose(loaded.scale_tensor, torch.tensor([[0.5]], dtype=torch.float32))


def _write_fp8_runtime_fixture(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "deepseek-fp8-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "intermediate_size": 4,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "vocab_size": 8,
                "quantization_config": {"fmt": "e4m3", "quant_method": "fp8"},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": ("F8_E4M3", [2, 4], bytes([1, 2, 3, 4, 5, 6, 7, 8])),
        "model.layers.0.self_attn.q_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 0.5)),
        "model.layers.0.mlp.experts.0.gate_proj.weight": ("F8_E4M3", [2, 4], bytes([9, 10, 11, 12, 13, 14, 15, 16])),
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.mlp.experts.1.gate_proj.weight": ("F8_E4M3", [2, 4], bytes([17, 18, 19, 20, 21, 22, 23, 24])),
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 2.0)),
        "model.layers.0.input_layernorm.weight": ("BF16", [4], b"\x00\x00" * 4),
    }
    shard = model_dir / "model-00001-of-00001.safetensors"
    _write_safetensors_bytes(shard, tensors)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": shard.stat().st_size}, "weight_map": {name: shard.name for name in tensors}}),
        encoding="utf-8",
    )
    return model_id, model_dir


def _write_safetensors_bytes(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    offset = 0
    data = bytearray()
    header = {}
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(payload)]}
        data.extend(payload)
        offset += len(payload)
    header_payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_payload)) + header_payload + bytes(data))
