"""Show the registry-backed model catalog."""

from __future__ import annotations

import json

from pcketlm.core.registry.catalog import build_model_catalog


def main() -> int:
    """Run the model catalog command."""
    payload = [entry.to_dict() for entry in build_model_catalog()]
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
