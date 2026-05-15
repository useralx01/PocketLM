"""Plan a lossless pcketlm FP8 packed artifact from safetensors headers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pcketlm.core.model_import.fp8_pack_plan import plan_model_dir_to_fp8_pack, write_fp8_pack_plan

__all__ = ["plan_model_dir_to_fp8_pack", "write_fp8_pack_plan"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan pcketlm FP8 packed artifacts without reading tensor payloads.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--write", action="store_true", help="Write fp8_pack_plan.json to the output directory.")
    args = parser.parse_args()

    plan = write_fp8_pack_plan(args.model_dir, args.output_dir) if args.write else plan_model_dir_to_fp8_pack(args.model_dir, args.output_dir)
    print(
        json.dumps(
            {
                key: plan[key]
                for key in (
                    "format",
                    "ready_for_pack",
                    "lossless",
                    "source_tensor_count",
                    "source_shard_count",
                    "missing_shards",
                    "total_source_bytes",
                    "estimated_packed_bytes",
                    "pack_unit_count",
                    "fp8_weight_bytes",
                    "fp8_scale_bytes",
                    "non_fp8_bytes",
                    "persistent_bytes",
                    "lm_head_bytes",
                    "attention_pack_count",
                    "attention_pack_bytes",
                    "routed_expert_pack_count",
                    "routed_expert_pack_bytes",
                    "average_routed_expert_pack_bytes",
                    "max_routed_expert_pack_bytes",
                    "shared_expert_pack_count",
                    "shared_expert_pack_bytes",
                    "router_pack_count",
                    "router_pack_bytes",
                    "dense_mlp_pack_count",
                    "dense_mlp_pack_bytes",
                    "largest_pack_units",
                    "recommended_layout",
                    "reason",
                )
            },
            indent=2,
        )
    )
    return 0 if plan["ready_for_pack"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
