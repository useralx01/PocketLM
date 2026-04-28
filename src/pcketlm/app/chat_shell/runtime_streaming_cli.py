"""Show the first staged disk-streaming plan for a local model folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.runtime import plan_staged_disk_streaming


def main() -> int:
    """Run the staged disk-streaming planner command."""
    parser = argparse.ArgumentParser(description="Show pcketlm staged disk-streaming runtime plan.")
    parser.add_argument("model_id", help="Logical pcketlm model id")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    args = parser.parse_args()

    plan = plan_staged_disk_streaming(args.model_id, args.model_dir)
    print(json.dumps(plan.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
