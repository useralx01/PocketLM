"""Small status command for model source folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.model_import.inspect import inspect_qwen_source
from pcketlm.core.validation.files import validate_qwen_source


def build_status_payload(model_dir: Path) -> dict:
    """Build a plain JSON-friendly status payload for a source folder."""
    inspection = inspect_qwen_source(model_dir)
    validation = validate_qwen_source(model_dir)
    return {
        "model_dir": str(model_dir),
        "status": validation.result,
        "missing_files": list(validation.missing_files),
        "warnings": list(validation.warnings),
        "inspection": {
            "format_name": inspection.format_name,
            "index_present": inspection.index_present,
            "expected_shards": inspection.expected_shards,
            "present_shards": inspection.present_shards,
            "total_size_bytes": inspection.total_size_bytes,
            "config": inspection.config.to_dict(),
        },
    }


def main() -> int:
    """Run the source status command."""
    parser = argparse.ArgumentParser(description="Show pcketlm source-model status.")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    args = parser.parse_args()

    payload = build_status_payload(args.model_dir)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
