"""Remove a model record from the local registry."""

from __future__ import annotations

import argparse
import json

from pcketlm.core.registry.repository import delete_model_record


def main() -> int:
    """Run the model removal command."""
    parser = argparse.ArgumentParser(description="Remove a pcketlm model registry entry.")
    parser.add_argument("model_id", help="Registry model id to remove")
    args = parser.parse_args()

    removed = delete_model_record(args.model_id)
    print(json.dumps({"model_id": args.model_id, "removed": removed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
