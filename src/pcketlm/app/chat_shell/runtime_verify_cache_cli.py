"""Verify the staged streaming cache index and cached segment files."""

from __future__ import annotations

import argparse
import json

from pcketlm.core.runtime.streaming_materialize import verify_materialized_cache


def main() -> int:
    """Run the cache verification command."""
    parser = argparse.ArgumentParser(description="Verify pcketlm staged streaming cache.")
    parser.add_argument("model_id", help="Logical pcketlm model id")
    args = parser.parse_args()

    result = verify_materialized_cache(args.model_id)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
