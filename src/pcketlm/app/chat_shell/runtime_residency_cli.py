"""Show the persisted staged streaming residency state."""

from __future__ import annotations

import argparse
import json

from pcketlm.core.runtime.streaming_residency import load_streaming_residency


def main() -> int:
    """Run the streaming residency command."""
    parser = argparse.ArgumentParser(description="Show pcketlm staged streaming residency state.")
    parser.add_argument("model_id", help="Logical pcketlm model id")
    args = parser.parse_args()

    result = load_streaming_residency(args.model_id)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
