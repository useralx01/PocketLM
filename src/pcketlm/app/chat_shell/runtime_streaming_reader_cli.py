"""Load and show the staged streaming manifest for a model."""

from __future__ import annotations

import argparse
import json

from pcketlm.core.runtime import load_streaming_manifest


def main() -> int:
    """Run the streaming manifest reader command."""
    parser = argparse.ArgumentParser(description="Load pcketlm staged streaming manifest.")
    parser.add_argument("model_id", help="Logical pcketlm model id")
    args = parser.parse_args()

    manifest = load_streaming_manifest(args.model_id)
    print(json.dumps(manifest.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
