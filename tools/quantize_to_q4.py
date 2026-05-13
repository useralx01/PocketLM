"""Offline pcketlm Q4 streaming artifact builder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pcketlm.core.model_import.q4_conversion_job import run_q4_conversion_job
from pcketlm.core.model_import.q4_plan import plan_model_dir_to_q4
from pcketlm.core.model_import.q4_quantize import (
    dequantize_q4_tensor,
    pack_int4,
    quantize_model_dir_to_q4,
    quantize_tensor_to_q4,
    unpack_int4,
)

__all__ = [
    "dequantize_q4_tensor",
    "pack_int4",
    "plan_model_dir_to_q4",
    "quantize_model_dir_to_q4",
    "quantize_tensor_to_q4",
    "unpack_int4",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pcketlm Q4 streaming artifacts.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=None, help="Model id for resumable state tracking")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch instead of reusing conversion state")
    parser.add_argument("--skip-disk-check", action="store_true", help="Skip the free-space gate before conversion")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate the Q4 artifact from safetensors headers without reading tensor payloads or writing files.",
    )
    args = parser.parse_args()
    if args.dry_run:
        plan = plan_model_dir_to_q4(args.model_dir, args.output_dir)
        print(
            json.dumps(
                {
                    key: plan[key]
                    for key in (
                        "format",
                        "scheme",
                        "ready_for_conversion",
                        "fp8_native",
                        "ready_for_fp8_runtime_planning",
                        "source_tensor_count",
                        "source_shard_count",
                        "q4_tensor_count",
                        "fp8_tensor_count",
                        "fp8_scale_tensor_count",
                        "missing_shards",
                        "dtype_bytes",
                        "total_original_bytes",
                        "fp8_weight_bytes",
                        "fp8_scale_bytes",
                        "non_fp8_non_scale_bytes",
                        "estimated_total_q4_bytes",
                        "estimated_q4_from_source_lossy",
                        "estimated_fp8_repacked_artifact_bytes",
                        "estimated_fp8_runtime_bytes",
                        "estimated_bf16_dequantized_runtime_bytes",
                        "max_active_layer_fp8_and_scale_bytes",
                        "avg_active_layer_fp8_and_scale_bytes",
                        "max_active_layer_estimate",
                        "estimated_fp8_paged_working_set_bytes",
                        "estimated_expert_q4_bytes",
                        "estimated_non_expert_q4_bytes",
                        "compression_ratio",
                        "disk_required_bytes_with_10pct_headroom",
                        "moe_top_k",
                        "recommended_path",
                        "recommendation_reason",
                        "q4_requantization_warning",
                        "conversion_blockers",
                    )
                },
                indent=2,
            )
        )
        return 0 if plan["ready_for_conversion"] or plan["ready_for_fp8_runtime_planning"] else 2

    state = run_q4_conversion_job(
        args.model_dir,
        args.output_dir,
        model_id=args.model_id,
        resume=not args.no_resume,
        check_disk_space=not args.skip_disk_check,
    )
    print(
        json.dumps(
            {
                key: state.get(key)
                for key in (
                    "model_id",
                    "status",
                    "progress_pct",
                    "completed_shard_count",
                    "source_shard_count",
                    "total_original_bytes",
                    "total_q4_bytes",
                    "compression_ratio",
                    "manifest_path",
                    "blockers",
                    "error",
                )
                if key in state
            },
            indent=2,
        )
    )
    return 0 if state.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
