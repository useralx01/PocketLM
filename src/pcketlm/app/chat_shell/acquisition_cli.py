"""Show a product-facing acquisition summary for a local model folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.acquisition import build_acquisition_snapshot


def main() -> int:
    """Run the acquisition status command."""
    parser = argparse.ArgumentParser(description="Show pcketlm acquisition status.")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    args = parser.parse_args()

    snapshot = build_acquisition_snapshot(args.model_dir)
    print(json.dumps(snapshot.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
