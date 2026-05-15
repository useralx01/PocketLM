"""Header-only FP8 packed artifact planning for FP8-native safetensors."""

from __future__ import annotations

import json
from pathlib import Path

from pcketlm.core.model_import.q4_plan import (
    _canonical_dtype,
    _is_fp8_dtype,
    _read_safetensors_header,
    _scale_partner_name,
    _tensor_layer_and_expert,
    _weight_map,
)


FP8_PACK_FORMAT = "pcketlm-fp8-pack-plan"


def plan_model_dir_to_fp8_pack(model_dir: Path, output_dir: Path | None = None) -> dict:
    """Plan a lossless FP8 repack layout from safetensors headers only."""
    model_dir = model_dir.resolve()
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = _weight_map(index_path)
    all_tensor_names = set(weight_map)
    tensors_by_shard: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        tensors_by_shard.setdefault(shard_name, []).append(tensor_name)

    missing_shards: list[str] = []
    total_source_bytes = 0
    fp8_weight_bytes = 0
    scale_bytes = 0
    non_fp8_bytes = 0
    fp8_weight_count = 0
    scale_count = 0
    non_fp8_count = 0
    routed_expert_groups: dict[tuple[int, int], int] = {}
    shared_expert_groups: dict[int, int] = {}
    router_groups: dict[int, int] = {}
    dense_mlp_groups: dict[int, int] = {}
    attention_groups: dict[int, int] = {}
    layer_other_groups: dict[int, int] = {}
    persistent_bytes = 0
    lm_head_bytes = 0
    dtype_bytes: dict[str, int] = {}

    for shard_name, tensor_names in sorted(tensors_by_shard.items()):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            missing_shards.append(shard_name)
            continue
        header = _read_safetensors_header(shard_path)
        for tensor_name in sorted(tensor_names):
            metadata = header.get(tensor_name)
            if metadata is None:
                continue
            dtype = _canonical_dtype(str(metadata.get("dtype", "")))
            offsets = metadata.get("data_offsets") or [0, 0]
            tensor_bytes = int(offsets[1]) - int(offsets[0])
            total_source_bytes += tensor_bytes
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + tensor_bytes

            partner_name = _scale_partner_name(tensor_name, all_tensor_names)
            if _is_fp8_dtype(dtype) and partner_name is None:
                fp8_weight_count += 1
                fp8_weight_bytes += tensor_bytes
            elif partner_name is not None:
                scale_count += 1
                scale_bytes += tensor_bytes
            else:
                non_fp8_count += 1
                non_fp8_bytes += tensor_bytes

            layer_index, expert_index = _tensor_layer_and_expert(tensor_name)
            lowered = tensor_name.lower()
            if tensor_name == "lm_head.weight":
                lm_head_bytes += tensor_bytes
                persistent_bytes += tensor_bytes
            elif layer_index is None:
                persistent_bytes += tensor_bytes
            elif expert_index is not None:
                key = (int(layer_index), int(expert_index))
                routed_expert_groups[key] = routed_expert_groups.get(key, 0) + tensor_bytes
            elif ".shared_experts." in lowered:
                shared_expert_groups[int(layer_index)] = shared_expert_groups.get(int(layer_index), 0) + tensor_bytes
            elif ".mlp.gate." in lowered:
                router_groups[int(layer_index)] = router_groups.get(int(layer_index), 0) + tensor_bytes
            elif ".self_attn." in lowered:
                attention_groups[int(layer_index)] = attention_groups.get(int(layer_index), 0) + tensor_bytes
            elif ".mlp." in lowered:
                dense_mlp_groups[int(layer_index)] = dense_mlp_groups.get(int(layer_index), 0) + tensor_bytes
            else:
                layer_other_groups[int(layer_index)] = layer_other_groups.get(int(layer_index), 0) + tensor_bytes

    output_dir = output_dir.resolve() if output_dir is not None else model_dir.parent / "artifacts" / "fp8-pack"
    routed_values = list(routed_expert_groups.values())
    grouped_bytes = (
        sum(routed_expert_groups.values())
        + sum(shared_expert_groups.values())
        + sum(router_groups.values())
        + sum(dense_mlp_groups.values())
        + sum(attention_groups.values())
        + sum(layer_other_groups.values())
        + persistent_bytes
    )
    pack_unit_count = (
        len(routed_expert_groups)
        + len(shared_expert_groups)
        + len(router_groups)
        + len(dense_mlp_groups)
        + len(attention_groups)
        + len(layer_other_groups)
        + (1 if persistent_bytes else 0)
    )
    largest_units = [
        *[
            {"kind": "routed_expert", "layer": layer, "expert": expert, "bytes": value}
            for (layer, expert), value in routed_expert_groups.items()
        ],
        *[{"kind": "shared_expert", "layer": layer, "bytes": value} for layer, value in shared_expert_groups.items()],
        *[{"kind": "router", "layer": layer, "bytes": value} for layer, value in router_groups.items()],
        *[{"kind": "dense_mlp", "layer": layer, "bytes": value} for layer, value in dense_mlp_groups.items()],
        *[{"kind": "attention", "layer": layer, "bytes": value} for layer, value in attention_groups.items()],
        *[{"kind": "layer_other", "layer": layer, "bytes": value} for layer, value in layer_other_groups.items()],
        *([{"kind": "persistent", "bytes": persistent_bytes}] if persistent_bytes else []),
    ]
    largest_units.sort(key=lambda item: int(item["bytes"]), reverse=True)
    ready = not missing_shards and fp8_weight_bytes > 0
    return {
        "format": FP8_PACK_FORMAT,
        "source_model_dir": str(model_dir),
        "planned_output_dir": str(output_dir),
        "source_tensor_count": len(weight_map),
        "source_shard_count": len(tensors_by_shard),
        "missing_shards": missing_shards,
        "fp8_native": fp8_weight_bytes > 0,
        "ready_for_pack": ready,
        "lossless": True,
        "total_source_bytes": int(total_source_bytes),
        "estimated_packed_bytes": int(grouped_bytes),
        "estimated_manifest_overhead_bytes": int(max(4096, pack_unit_count * 256)),
        "pack_unit_count": int(pack_unit_count),
        "fp8_weight_count": int(fp8_weight_count),
        "fp8_scale_count": int(scale_count),
        "non_fp8_count": int(non_fp8_count),
        "fp8_weight_bytes": int(fp8_weight_bytes),
        "fp8_scale_bytes": int(scale_bytes),
        "non_fp8_bytes": int(non_fp8_bytes),
        "dtype_bytes": dtype_bytes,
        "persistent_bytes": int(persistent_bytes),
        "lm_head_bytes": int(lm_head_bytes),
        "attention_pack_count": len(attention_groups),
        "attention_pack_bytes": int(sum(attention_groups.values())),
        "routed_expert_pack_count": len(routed_expert_groups),
        "routed_expert_pack_bytes": int(sum(routed_expert_groups.values())),
        "average_routed_expert_pack_bytes": 0 if not routed_values else int(round(sum(routed_values) / len(routed_values))),
        "max_routed_expert_pack_bytes": 0 if not routed_values else int(max(routed_values)),
        "shared_expert_pack_count": len(shared_expert_groups),
        "shared_expert_pack_bytes": int(sum(shared_expert_groups.values())),
        "router_pack_count": len(router_groups),
        "router_pack_bytes": int(sum(router_groups.values())),
        "dense_mlp_pack_count": len(dense_mlp_groups),
        "dense_mlp_pack_bytes": int(sum(dense_mlp_groups.values())),
        "layer_other_pack_count": len(layer_other_groups),
        "layer_other_pack_bytes": int(sum(layer_other_groups.values())),
        "largest_pack_units": largest_units[:10],
        "recommended_layout": [
            "one persistent metadata/embed/norm/lm_head region",
            "one attention region per layer",
            "one router region per MoE layer",
            "one routed-expert region per layer/expert containing gate/up/down FP8 weights and scales",
            "one shared-expert region per layer",
        ],
        "reason": (
            "Lossless FP8 repack keeps DeepSeek quality while making runtime reads more local than scattered source shards."
            if ready
            else "FP8 pack planning requires a complete FP8-native safetensors source."
        ),
    }


def write_fp8_pack_plan(model_dir: Path, output_dir: Path | None = None) -> dict:
    """Write the header-only FP8 pack plan beside the planned artifact output."""
    plan = plan_model_dir_to_fp8_pack(model_dir, output_dir)
    plan_path = Path(plan["planned_output_dir"]) / "fp8_pack_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan
