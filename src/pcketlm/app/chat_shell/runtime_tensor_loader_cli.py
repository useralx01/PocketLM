"""CLI for loading one tensor or one execution unit on demand."""

from __future__ import annotations

import json
import sys

from pcketlm.core.runtime import (
    load_execution_unit,
    load_tensor_by_name,
    verify_execution_unit,
    verify_loaded_tensor,
)


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) < 3 or args[1] not in {"--tensor", "--unit", "--verify-tensor", "--verify-unit"}:
        print("Usage: py -m pcketlm.app.chat_shell.runtime_tensor_loader_cli <model-id> --tensor <tensor-name>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_tensor_loader_cli <model-id> --unit <unit-id>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_tensor_loader_cli <model-id> --verify-tensor <tensor-name>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_tensor_loader_cli <model-id> --verify-unit <unit-id>")
        return 1

    model_id = args[0]
    mode = args[1]
    target = args[2]

    if mode == "--tensor":
        result = load_tensor_by_name(model_id, target)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if mode == "--verify-tensor":
        result = verify_loaded_tensor(model_id, target)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if mode == "--verify-unit":
        result = verify_execution_unit(model_id, target)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    result = load_execution_unit(model_id, target)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
