"""Show a product-facing acquisition summary for a local model folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.acquisition import AcquisitionSnapshot, build_acquisition_snapshot


def _format_bytes_gb(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.2f} GiB"


def _format_fp8_line(snapshot: AcquisitionSnapshot) -> str | None:
    status = snapshot.fp8_runtime_status
    if not status:
        return None
    policy = status.get("runtime_policy") or {}
    blockers = status.get("blockers") or []
    fp8_gb = int(status.get("fp8_weight_bytes", 0)) / (1024**3)
    scale_gb = int(status.get("fp8_scale_bytes", 0)) / (1024**3)
    ready = "ready" if status.get("ready") else "not ready"
    path = policy.get("weight_residency", "unknown")
    layers = policy.get("layer_count", "unknown")
    pairs = status.get("fp8_pair_count", 0)
    blocker_text = "none" if not blockers else "; ".join(str(item) for item in blockers)
    return (
        f"FP8 runtime: {ready}, path {path}, layers {layers}, pairs {pairs}, "
        f"weights {fp8_gb:.2f} GiB, scales {scale_gb:.2f} GiB, blockers {blocker_text}"
    )


def format_plain_snapshot(snapshot: AcquisitionSnapshot) -> str:
    """Format a short operator-facing acquisition summary."""
    if snapshot.status == "ready":
        size_line = f"Size: {_format_bytes_gb(snapshot.bytes_on_disk_gb)} on disk, shards {snapshot.present_shards}/{snapshot.expected_shards}"
    else:
        expected = _format_bytes_gb(snapshot.expected_bytes_gb)
        progress = "unknown" if snapshot.progress_pct is None else f"{snapshot.progress_pct:.2f}%"
        size_line = (
            f"Size: {_format_bytes_gb(snapshot.bytes_on_disk_gb)} / {expected}, "
            f"{progress}, shards {snapshot.present_shards}/{snapshot.expected_shards}"
        )
    lines = [
        f"Status: {snapshot.status}",
        size_line,
        f"Summary: {snapshot.plain_english_summary}",
        f"Next: {snapshot.recommended_next_step}",
    ]
    fp8_line = _format_fp8_line(snapshot)
    if fp8_line:
        lines.append(fp8_line)
    return "\n".join(lines)


def main() -> int:
    """Run the acquisition status command."""
    parser = argparse.ArgumentParser(description="Show pcketlm acquisition status.")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    parser.add_argument("--plain", action="store_true", help="Print a concise human-readable summary")
    args = parser.parse_args()

    snapshot = build_acquisition_snapshot(args.model_dir)
    if args.plain:
        print(format_plain_snapshot(snapshot))
    else:
        print(json.dumps(snapshot.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
