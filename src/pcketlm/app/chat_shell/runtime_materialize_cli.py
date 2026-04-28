"""Materialize the current staged streaming schedule into cache files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.runtime.streaming_materialize import materialize_window_schedule


def main() -> int:
    """Run the streaming materialization command."""
    parser = argparse.ArgumentParser(description="Materialize pcketlm staged streaming cache.")
    parser.add_argument("model_id", help="Logical pcketlm model id")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    args = parser.parse_args()

    result = materialize_window_schedule(args.model_id, args.model_dir)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
