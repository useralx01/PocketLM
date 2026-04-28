"""CLI for building the first tensor-aware execution plan."""

from __future__ import annotations

import json
import sys

from pcketlm.core.runtime import build_tensor_execution_plan
from pcketlm.core.storage.paths import original_model_root


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if not args:
        print("Usage: py -m pcketlm.app.chat_shell.runtime_tensor_execution_plan_cli <model-id>")
        return 1

    model_id = args[0]
    plan = build_tensor_execution_plan(model_id, original_model_root(model_id))
    print(json.dumps(plan.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
