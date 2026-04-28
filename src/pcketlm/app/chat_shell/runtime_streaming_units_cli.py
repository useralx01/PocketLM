"""Build and show the staged streaming unit map for a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.runtime import build_streaming_unit_map, write_streaming_unit_map


def main() -> int:
    """Run the streaming unit-map command."""
    parser = argparse.ArgumentParser(description="Build pcketlm staged streaming unit map.")
    parser.add_argument("model_id", help="Logical pcketlm model id")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    parser.add_argument("--write", action="store_true", help="Persist the unit map to disk")
    args = parser.parse_args()

    unit_map = build_streaming_unit_map(args.model_id, args.model_dir)
    payload = unit_map.to_dict()
    if args.write:
        write_streaming_unit_map(args.model_id, args.model_dir)
        payload["written"] = True
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
