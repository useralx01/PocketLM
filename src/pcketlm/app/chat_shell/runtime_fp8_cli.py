"""CLI for FP8-native source readiness and paged working-set probes."""

from __future__ import annotations

import json
import sys

import torch

from pcketlm.core.runtime import (
    fp8_source_status,
    load_dequantized_fp8_weight,
    load_fp8_weight_pair,
    plan_fp8_layer_working_set,
    run_fp8_expert_mlp,
)


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) < 2:
        print("Usage: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --status")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --layer <n> [--experts 1,2]")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --pair <tensor-name> [--no-payload]")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --dequant-pair <tensor-name>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --expert <layer> <expert>")
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
    if mode == "--dequant-pair":
        if len(args) < 3:
            print("--dequant-pair requires a tensor name")
            return 1
        result = load_dequantized_fp8_weight(model_id, args[2])
        payload = result.to_dict()
        if result.tensor is not None:
            values = result.tensor.float()
            payload["mean_abs"] = float(values.abs().mean().item())
            payload["max_abs"] = float(values.abs().max().item())
        print(json.dumps(payload, indent=2))
        return 0 if result.ready else 2
    if mode == "--expert":
        if len(args) < 4:
            print("--expert requires a layer index and expert index")
            return 1
        layer_index = int(args[2])
        expert_index = int(args[3])
        hidden_size = _hidden_size_from_status(model_id)
        hidden = torch.ones((1, hidden_size), dtype=torch.bfloat16)
        result = run_fp8_expert_mlp(model_id, layer_index, expert_index, hidden)
        payload = result.to_dict()
        if result.output_tensor is not None:
            values = result.output_tensor.float()
            payload["output_mean_abs"] = float(values.abs().mean().item())
            payload["output_max_abs"] = float(values.abs().max().item())
        print(json.dumps(payload, indent=2))
        return 0 if result.ready else 2

    print(f"Unknown mode: {mode}")
    return 1


def _hidden_size_from_status(model_id: str) -> int:
    from pcketlm.core.runtime.tensor_catalog import load_tensor_catalog

    catalog = load_tensor_catalog(model_id)
    return int(catalog.hidden_size or 7168)


if __name__ == "__main__":
    raise SystemExit(main())
