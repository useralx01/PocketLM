import json
import struct
from pathlib import Path

from pcketlm.core.model_import.q4_plan import plan_model_dir_to_q4


def test_fp8_planner_classifies_fp8_weights_and_scale_companions(tmp_path: Path) -> None:
    model_dir = tmp_path / "fp8-source"
    model_dir.mkdir()
    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": ("F8_E4M3", [256, 128]),
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": ("F32", [2, 1]),
        "model.layers.0.self_attn.q_proj.weight": ("F8_E4M3", [128, 128]),
        "model.layers.0.self_attn.q_proj.weight_scale_inv": ("F32", [1, 1]),
        "model.norm.weight": ("BF16", [128]),
    }
    _write_header_only_safetensors(model_dir / "model-00001-of-00001.safetensors", tensors)
    _write_index(model_dir, tensors)
    (model_dir / "config.json").write_text(
        json.dumps({"num_hidden_layers": 1, "num_experts_per_tok": 1, "n_routed_experts": 1}),
        encoding="utf-8",
    )

    plan = plan_model_dir_to_q4(model_dir, tmp_path / "q4")

    assert plan["fp8_native"] is True
    assert plan["ready_for_conversion"] is False
    assert plan["ready_for_fp8_runtime_planning"] is True
    assert plan["fp8_tensor_count"] == 2
    assert plan["fp8_scale_tensor_count"] == 2
    assert plan["scale_pair_count"] == 2
    assert plan["fp8_weight_bytes"] == (256 * 128) + (128 * 128)
    assert plan["fp8_scale_bytes"] == (2 * 1 * 4) + (1 * 1 * 4)
    assert plan["non_fp8_non_scale_bytes"] == 128 * 2
    assert plan["estimated_q4_from_source_lossy"] is True
    assert plan["q4_requantization_warning"]
    assert plan["recommended_path"] in {
        "FP8 native, no conversion needed",
        "FP8 native paged",
        "stream FP8 from disk with paged residency",
    }


def test_fp8_planner_does_not_false_positive_bf16_source(tmp_path: Path) -> None:
    model_dir = tmp_path / "bf16-source"
    model_dir.mkdir()
    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": ("BF16", [128, 128]),
        "model.layers.0.self_attn.q_proj.weight": ("BF16", [128, 128]),
    }
    _write_header_only_safetensors(model_dir / "model-00001-of-00001.safetensors", tensors)
    _write_index(model_dir, tensors)

    plan = plan_model_dir_to_q4(model_dir, tmp_path / "q4")

    assert plan["fp8_native"] is False
    assert plan["fp8_tensor_count"] == 0
    assert plan["fp8_scale_tensor_count"] == 0
    assert plan["ready_for_conversion"] is True
    assert plan["recommended_path"] == "compact_q4"


def _write_index(model_dir: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors},
            }
        ),
        encoding="utf-8",
    )


def _write_header_only_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    offset = 0
    header = {}
    for name, (dtype, shape) in tensors.items():
        bytes_per_value = _dtype_bytes(dtype)
        size = bytes_per_value
        for value in shape:
            size *= int(value)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def _dtype_bytes(dtype: str) -> int:
    return {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}[dtype]
