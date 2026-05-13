"""CLI for FP8-native source readiness and paged working-set probes."""

from __future__ import annotations

import json
import sys

from pcketlm.core.runtime import fp8_source_status, load_fp8_weight_pair, plan_fp8_layer_working_set


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) < 2:
        print("Usage: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --status")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --layer <n> [--experts 1,2]")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --pair <tensor-name> [--no-payload]")
        return 1

    model_id = args[0]
    mode = args[1]
    if mode == "--status":
        print(json.dumps(fp8_source_status(model_id), indent=2))
        return 0
    if mode == "--layer":
        if len(args) < 3:
            print("--layer requires a layer index")
            return 1
        experts: list[int] = []
        if "--experts" in args:
            index = args.index("--experts")
            if index + 1 >= len(args):
                print("--experts requires a comma-separated list")
                return 1
            experts = [int(value) for value in args[index + 1].split(",") if value.strip()]
        result = plan_fp8_layer_working_set(model_id, int(args[2]), experts)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ready else 2
    if mode == "--pair":
        if len(args) < 3:
            print("--pair requires a tensor name")
            return 1
        result = load_fp8_weight_pair(model_id, args[2], load_payload="--no-payload" not in args)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ready else 2

    print(f"Unknown mode: {mode}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
