"""Show source download state for a local model folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.model_import.download_state import estimate_download_state
from pcketlm.core.storage.paths import state_root


def _download_status_path(model_id: str) -> Path:
    safe_model_id = model_id.replace("/", "_").replace("\\", "_")
    return state_root() / "downloads" / f"{safe_model_id}.json"


def main() -> int:
    """Run the download status command."""
    parser = argparse.ArgumentParser(description="Show pcketlm model download status.")
    parser.add_argument("model_dir", nargs="?", type=Path, help="Path to the source model directory")
    parser.add_argument("--model-id", default=None, help="Read live downloader status for this model id")
    args = parser.parse_args()

    live_status = None
    if args.model_id:
        path = _download_status_path(args.model_id)
        if path.exists():
            live_status = json.loads(path.read_text(encoding="utf-8"))

    if args.model_dir is None:
        print(json.dumps({"live_status": live_status}, indent=2))
        return 0

    state = estimate_download_state(args.model_dir)
    print(
        json.dumps(
            {
                "model_dir": str(args.model_dir),
                "live_status": live_status,
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
