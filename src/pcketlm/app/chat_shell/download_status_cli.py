"""Show source download state for a local model folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.model_import.download_state import estimate_download_state


def main() -> int:
    """Run the download status command."""
    parser = argparse.ArgumentParser(description="Show pcketlm model download status.")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    args = parser.parse_args()

    state = estimate_download_state(args.model_dir)
    print(
        json.dumps(
            {
                "model_dir": str(args.model_dir),
                "status": state.status,
                "bytes_on_disk": state.bytes_on_disk,
                "expected_bytes": state.expected_bytes,
                "progress_pct": state.progress_pct,
                "expected_shards": state.expected_shards,
                "present_shards": state.present_shards,
                "missing_core_files": state.missing_core_files,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
