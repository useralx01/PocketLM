"""Show runtime source readiness for a local model folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.runtime import describe_runtime_source


def main() -> int:
    """Run the runtime source readiness command."""
    parser = argparse.ArgumentParser(description="Show pcketlm runtime source readiness.")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    args = parser.parse_args()

    descriptor = describe_runtime_source(args.model_dir)
    print(json.dumps(descriptor.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
