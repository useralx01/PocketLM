"""Build and show the staged streaming hot/warm window schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.runtime import build_window_schedule, write_window_schedule


def main() -> int:
    """Run the window scheduling command."""
    parser = argparse.ArgumentParser(description="Build pcketlm staged streaming window schedule.")
    parser.add_argument("model_id", help="Logical pcketlm model id")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    parser.add_argument("--write", action="store_true", help="Persist the schedule to disk")
    args = parser.parse_args()

    schedule = build_window_schedule(args.model_id, args.model_dir)
    payload = schedule.to_dict()
    if args.write:
        write_window_schedule(args.model_id, args.model_dir)
        payload["written"] = True
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
