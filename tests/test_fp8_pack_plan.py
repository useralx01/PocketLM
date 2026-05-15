import json
import struct
from pathlib import Path

from pcketlm.core.model_import.fp8_pack_plan import plan_model_dir_to_fp8_pack, write_fp8_pack_plan


def test_fp8_pack_plan_groups_routed_attention_and_persistent_bytes(tmp_path: Path) -> None:
    model_dir = tmp_path / "fp8-source"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00001.safetensors"
    routed = "model.layers.3.mlp.experts.7.gate_proj.weight"
    routed_scale = f"{routed}_scale_inv"
    attention = "model.layers.3.self_attn.q_a_proj.weight"
    attention_scale = f"{attention}_scale_inv"
    shared = "model.layers.3.mlp.shared_experts.down_proj.weight"
    router = "model.layers.3.mlp.gate.weight"
    dense = "model.layers.0.mlp.gate_proj.weight"
    lm_head = "lm_head.weight"
    _write_header_only_safetensors(
        shard,
        {
            routed: ("F8_E4M3", [128, 128]),
            routed_scale: ("F32", [1, 1]),
            attention: ("F8_E4M3", [128, 128]),
            attention_scale: ("F32", [1, 1]),
            shared: ("F8_E4M3", [128, 128]),
            router: ("BF16", [8, 128]),
            dense: ("F8_E4M3", [128, 128]),
            lm_head: ("BF16", [16, 128]),
        },
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    routed: shard.name,
                    routed_scale: shard.name,
                    attention: shard.name,
                    attention_scale: shard.name,
                    shared: shard.name,
                    router: shard.name,
                    dense: shard.name,
                    lm_head: shard.name,
                }
            }
        ),
        encoding="utf-8",
    )

    plan = plan_model_dir_to_fp8_pack(model_dir, tmp_path / "fp8-pack")

    assert plan["ready_for_pack"] is True
    assert plan["lossless"] is True
    assert plan["routed_expert_pack_count"] == 1
    assert plan["attention_pack_count"] == 1
    assert plan["shared_expert_pack_count"] == 1
    assert plan["router_pack_count"] == 1
    assert plan["dense_mlp_pack_count"] == 1
    assert plan["lm_head_bytes"] == 16 * 128 * 2
    assert plan["estimated_packed_bytes"] == plan["total_source_bytes"]
    assert any(unit["kind"] == "routed_expert" and unit["layer"] == 3 for unit in plan["largest_pack_units"])


def test_fp8_pack_plan_blocks_missing_shards(tmp_path: Path) -> None:
    model_dir = tmp_path / "missing-source"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.self_attn.q_a_proj.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )

    plan = plan_model_dir_to_fp8_pack(model_dir)

    assert plan["ready_for_pack"] is False
    assert plan["missing_shards"] == ["missing.safetensors"]


def test_write_fp8_pack_plan_writes_json(tmp_path: Path) -> None:
    model_dir = tmp_path / "fp8-source"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensor = "model.layers.0.self_attn.q_a_proj.weight"
    _write_header_only_safetensors(shard, {tensor: ("F8_E4M3", [128, 128])})
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor: shard.name}}),
        encoding="utf-8",
    )

    plan = write_fp8_pack_plan(model_dir, tmp_path / "planned-pack")

    saved = tmp_path / "planned-pack" / "fp8_pack_plan.json"
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8"))["format"] == plan["format"]


def _write_header_only_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    offset = 0
    header = {}
    for name, (dtype, shape) in tensors.items():
        size = {"F8_E4M3": 1, "F32": 4, "BF16": 2}[dtype]
        for value in shape:
            size *= int(value)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)
