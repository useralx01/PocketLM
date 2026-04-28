"""Small import command for local model folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.model_import.service import ImportRequest, import_model


def build_import_request(args: argparse.Namespace) -> ImportRequest:
    """Build an import request from CLI arguments."""
    return ImportRequest(
        model_id=args.model_id,
        label=args.label or args.model_id,
        family=args.family,
        source_path=args.source_path,
        repo_id=args.repo_id,
        source_origin=args.source_origin,
        source_kind="local-folder",
        model_type=args.model_type,
        format_name=args.format_name,
    )


def main() -> int:
    """Run the local import command."""
    parser = argparse.ArgumentParser(description="Import a local model folder into pcketlm.")
    parser.add_argument("model_id", help="Stable pcketlm model id")
    parser.add_argument("source_path", type=Path, help="Path to the local model folder")
    parser.add_argument("--label", default=None, help="Human-friendly model label")
    parser.add_argument("--family", default="qwen", help="Model family")
    parser.add_argument("--model-type", default="dense", help="Model type")
    parser.add_argument("--repo-id", default=None, help="Original upstream repo id if known")
    parser.add_argument("--source-origin", default="local", help="Origin label for the model source")
    parser.add_argument("--format-name", default="unknown", help="Expected source format name")
    args = parser.parse_args()

    record = import_model(build_import_request(args))
    print(json.dumps(record.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
