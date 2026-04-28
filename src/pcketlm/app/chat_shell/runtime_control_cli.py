"""CLI for staged-streaming runtime control decisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pcketlm.core.runtime import (
    advance_streaming_runtime,
    advance_streaming_runtime_safely,
    build_streaming_control_state,
)
from pcketlm.core.storage.paths import original_model_root


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if not args:
        print("Usage: py -m pcketlm.app.chat_shell.runtime_control_cli <model-id> [--advance] [--safe]")
        return 1

    model_id = args[0]
    do_advance = "--advance" in args[1:]
    safe_mode = "--safe" in args[1:]

    if do_advance:
        runner = advance_streaming_runtime_safely if safe_mode else advance_streaming_runtime
        result = runner(model_id, original_model_root(model_id))
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    state = build_streaming_control_state(model_id)
    print(json.dumps(state.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
