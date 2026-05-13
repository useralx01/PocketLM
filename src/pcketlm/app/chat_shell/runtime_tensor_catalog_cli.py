"""CLI for building a tensor-aware safetensors catalog."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pcketlm.core.runtime import build_tensor_catalog
from pcketlm.core.storage.paths import original_model_root


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if not args:
        print("Usage: py -m pcketlm.app.chat_shell.runtime_tensor_catalog_cli <model-id> [--model-dir <path>]")
        return 1

    model_id = args[0]
    model_dir = original_model_root(model_id)
    if "--model-dir" in args:
        index = args.index("--model-dir")
        if index + 1 >= len(args):
            print("--model-dir requires a path")
            return 1
        model_dir = Path(args[index + 1])
    catalog = build_tensor_catalog(model_id, model_dir)
    print(json.dumps(catalog.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
