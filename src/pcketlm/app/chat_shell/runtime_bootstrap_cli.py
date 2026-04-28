"""Show runtime bootstrap/preflight state for a local model folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.runtime import build_runtime_bootstrap, create_runtime_session


def main() -> int:
    """Run the runtime bootstrap command."""
    parser = argparse.ArgumentParser(description="Show pcketlm runtime bootstrap state.")
    parser.add_argument("model_id", help="Logical pcketlm model id")
    parser.add_argument("model_dir", type=Path, help="Path to the source model directory")
    parser.add_argument("--profile-id", default=None, help="Optional profile id")
    args = parser.parse_args()

    bootstrap = build_runtime_bootstrap(args.model_id, args.model_dir, profile_id=args.profile_id)
    session = create_runtime_session(args.model_id, args.model_dir, profile_id=args.profile_id)

    print(
        json.dumps(
            {
                "bootstrap": bootstrap.to_dict(),
                "session": session.to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
