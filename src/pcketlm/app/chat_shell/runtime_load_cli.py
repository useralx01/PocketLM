"""Run a guarded first real-load attempt for a local model folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.runtime import attempt_real_model_load


def main() -> int:
    """Run the runtime load attempt command."""
    parser = argparse.ArgumentParser(description="Attempt the first guarded pcketlm model load.")
    parser.add_argument("model_id", help="Logical pcketlm model id")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    args = parser.parse_args()

    result = attempt_real_model_load(args.model_id, args.model_dir)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
