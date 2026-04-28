"""Create the first staged-streaming cache bootstrap for a local model folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.runtime import bootstrap_streaming_state


def main() -> int:
    """Run the streaming bootstrap command."""
    parser = argparse.ArgumentParser(description="Bootstrap pcketlm staged streaming state.")
    parser.add_argument("model_id", help="Logical pcketlm model id")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    args = parser.parse_args()

    result = bootstrap_streaming_state(args.model_id, args.model_dir)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
