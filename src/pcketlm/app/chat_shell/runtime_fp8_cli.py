"""CLI for FP8-native source readiness and paged working-set probes."""

from __future__ import annotations

import json
import sys
import time

import torch

from pcketlm.core.runtime import (
    fp8_source_status,
    load_fp8_token_embedding,
    load_dequantized_fp8_weight,
    load_fp8_weight_pair,
    plan_fp8_layer_working_set,
    run_fp8_decode_loop,
    run_fp8_decode_tail_topk,
    run_fp8_dense_mlp,
    run_fp8_expert_mlp,
    run_fp8_moe,
    run_fp8_prompt_prefill,
    run_fp8_router,
    run_fp8_single_token_attention,
    run_fp8_single_token_block,
    run_fp8_single_token_forward,
)
from pcketlm.core.runtime.speculative import fp8_speculative_generate


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) < 2:
        print("Usage: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --status")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --layer <n> [--experts 1,2]")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --pair <tensor-name> [--no-payload]")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --dequant-pair <tensor-name>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --expert <layer> <expert>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --router <layer>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --moe <layer>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --attention <layer>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --block <layer>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --tail")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --embedding <token-id>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --dense <layer>")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --token-forward <token-id> [--start-layer n] [--layers n] [--no-tail]")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --prefill <token-id,...> [--layers n] [--no-tail]")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --decode-loop <token-id,...> [--layers n] [--max-new n]")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --chat <text> [--layers n] [--max-new n] [--max-prompt-tokens n] [--system-prompt text] [--raw-chat]")
        print("   or: py -m pcketlm.app.chat_shell.runtime_fp8_cli <model-id> --spec-chat <text> [--speculator model-id] [--layers n] [--max-new n] [--k n]")
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
    if mode == "--router":
        if len(args) < 3:
            print("--router requires a layer index")
            return 1
        layer_index = int(args[2])
        hidden_size = _hidden_size_from_status(model_id)
        hidden = torch.ones((1, hidden_size), dtype=torch.bfloat16)
        result = run_fp8_router(model_id, layer_index, hidden)
        payload = result.to_dict()
        if result.weights_tensor is not None:
            payload["weights"] = [float(value) for value in result.weights_tensor.reshape(-1).float().tolist()]
        print(json.dumps(payload, indent=2))
        return 0 if result.ready else 2
    if mode == "--moe":
        if len(args) < 3:
            print("--moe requires a layer index")
            return 1
        layer_index = int(args[2])
        hidden_size = _hidden_size_from_status(model_id)
        hidden = torch.ones((1, hidden_size), dtype=torch.bfloat16)
        result = run_fp8_moe(model_id, layer_index, hidden)
        payload = result.to_dict()
        if result.output_tensor is not None:
            values = result.output_tensor.float()
            payload["output_mean_abs"] = float(values.abs().mean().item())
            payload["output_max_abs"] = float(values.abs().max().item())
        print(json.dumps(payload, indent=2))
        return 0 if result.ready else 2
    if mode == "--attention":
        if len(args) < 3:
            print("--attention requires a layer index")
            return 1
        layer_index = int(args[2])
        hidden_size = _hidden_size_from_status(model_id)
        hidden = torch.ones((1, 1, hidden_size), dtype=torch.bfloat16)
        result = run_fp8_single_token_attention(model_id, layer_index, hidden)
        payload = result.to_dict()
        if result.output_tensor is not None:
            values = result.output_tensor.float()
            payload["output_mean_abs"] = float(values.abs().mean().item())
            payload["output_max_abs"] = float(values.abs().max().item())
        print(json.dumps(payload, indent=2))
        return 0 if result.ready else 2
    if mode == "--block":
        if len(args) < 3:
            print("--block requires a layer index")
            return 1
        layer_index = int(args[2])
        hidden_size = _hidden_size_from_status(model_id)
        hidden = torch.ones((1, 1, hidden_size), dtype=torch.bfloat16)
        result = run_fp8_single_token_block(model_id, layer_index, hidden)
        payload = result.to_dict()
        if result.output_tensor is not None:
            values = result.output_tensor.float()
            payload["output_mean_abs"] = float(values.abs().mean().item())
            payload["output_max_abs"] = float(values.abs().max().item())
        print(json.dumps(payload, indent=2))
        return 0 if result.ready else 2
    if mode == "--tail":
        hidden_size = _hidden_size_from_status(model_id)
        hidden = torch.ones((1, 1, hidden_size), dtype=torch.bfloat16)
        result = run_fp8_decode_tail_topk(model_id, hidden)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ready else 2
    if mode == "--embedding":
        if len(args) < 3:
            print("--embedding requires a token id")
            return 1
        result = load_fp8_token_embedding(model_id, int(args[2]))
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ready else 2
    if mode == "--dense":
        if len(args) < 3:
            print("--dense requires a layer index")
            return 1
        layer_index = int(args[2])
        hidden_size = _hidden_size_from_status(model_id)
        hidden = torch.ones((1, 1, hidden_size), dtype=torch.bfloat16)
        result = run_fp8_dense_mlp(model_id, layer_index, hidden)
        payload = result.to_dict()
        if result.output_tensor is not None:
            values = result.output_tensor.float()
            payload["output_mean_abs"] = float(values.abs().mean().item())
            payload["output_max_abs"] = float(values.abs().max().item())
        print(json.dumps(payload, indent=2))
        return 0 if result.ready else 2
    if mode == "--token-forward":
        if len(args) < 3:
            print("--token-forward requires a token id")
            return 1
        start_layer = _int_option(args, "--start-layer", 0)
        layer_count = _int_option(args, "--layers", 1)
        result = run_fp8_single_token_forward(
            model_id,
            int(args[2]),
            start_layer=start_layer,
            layer_count=layer_count,
            include_tail="--no-tail" not in args,
        )
        payload = result.to_dict()
        if result.output_tensor is not None:
            values = result.output_tensor.float()
            payload["hidden_mean_abs"] = float(values.abs().mean().item())
            payload["hidden_max_abs"] = float(values.abs().max().item())
        print(json.dumps(payload, indent=2))
        return 0 if result.ready else 2
    if mode == "--decode-loop":
        if len(args) < 3:
            print("--decode-loop requires comma-separated token ids")
            return 1
        token_ids = [int(value) for value in args[2].split(",") if value.strip()]
        start_layer = _int_option(args, "--start-layer", 0)
        layer_count = _int_option(args, "--layers", 1)
        max_new = _int_option(args, "--max-new", 1)
        started = time.perf_counter()
        result = run_fp8_decode_loop(
            model_id,
            token_ids,
            start_layer=start_layer,
            layer_count=layer_count,
            max_new_tokens=max_new,
        )
        payload = result.to_dict()
        payload["elapsed_seconds"] = float(time.perf_counter() - started)
        print(json.dumps(payload, indent=2))
        return 0 if result.ready else 2
    if mode == "--chat":
        if len(args) < 3:
            print("--chat requires prompt text")
            return 1
        prompt = args[2]
        start_layer = _int_option(args, "--start-layer", 0)
        layer_count = _int_option(args, "--layers", 1)
        max_new = _int_option(args, "--max-new", 1)
        max_prompt_tokens = _int_option(args, "--max-prompt-tokens", 16)
        system_prompt = _str_option(args, "--system-prompt", "")
        if "--raw-chat" in args:
            prepared_prompt = prompt
            token_ids, token_blockers = _encode_with_catalog_tokenizer(model_id, prompt)
            chat_template_used = False
        else:
            prepared_prompt, token_ids, token_blockers = _prepare_chat_with_catalog_tokenizer(
                model_id,
                prompt,
                system_prompt=system_prompt,
            )
            chat_template_used = not token_blockers
        used_token_ids = token_ids[-max(1, int(max_prompt_tokens)):] if token_ids else []
        started = time.perf_counter()
        result = run_fp8_decode_loop(
            model_id,
            used_token_ids,
            start_layer=start_layer,
            layer_count=layer_count,
            max_new_tokens=max_new,
        )
        generated_text, decode_blockers = _decode_with_catalog_tokenizer(model_id, result.generated_token_ids)
        payload = result.to_dict()
        payload.update(
            {
                "prompt": prompt,
                "prepared_prompt": prepared_prompt,
                "prompt_token_ids": token_ids,
                "used_prompt_token_ids": used_token_ids,
                "generated_text": generated_text,
                "chat_template_used": chat_template_used,
                "tokenizer_blockers": token_blockers + decode_blockers,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
        print(json.dumps(payload, indent=2))
        return 0 if result.ready and not token_blockers and not decode_blockers else 2
    if mode == "--spec-chat":
        if len(args) < 3:
            print("--spec-chat requires prompt text")
            return 1
        prompt = args[2]
        speculator_id = _str_option(args, "--speculator", "qwen3-1.7b")
        layer_count = _optional_int_option(args, "--layers")
        max_new = _int_option(args, "--max-new", 1)
        k = _int_option(args, "--k", 96)
        started = time.perf_counter()
        result = fp8_speculative_generate(
            model_id,
            speculator_id,
            prompt,
            max_new_tokens=max_new,
            k=k,
            layer_count=layer_count,
        )
        payload = result.to_dict()
        payload["elapsed_seconds"] = float(time.perf_counter() - started)
        print(json.dumps(payload, indent=2))
        return 0 if result.ready else 2
    if mode == "--prefill":
        if len(args) < 3:
            print("--prefill requires comma-separated token ids")
            return 1
        token_ids = [int(value) for value in args[2].split(",") if value.strip()]
        start_layer = _int_option(args, "--start-layer", 0)
        layer_count = _int_option(args, "--layers", 1)
        started = time.perf_counter()
        result = run_fp8_prompt_prefill(
            model_id,
            token_ids,
            start_layer=start_layer,
            layer_count=layer_count,
            include_tail="--no-tail" not in args,
        )
        payload = {
            "model_id": result.model_id,
            "prompt_token_ids": list(result.prompt_token_ids),
            "start_layer": result.start_layer,
            "layer_count": result.layer_count,
            "executed_layers": list(result.executed_layers),
            "hidden_shape": list(result.hidden_shape),
            "tail_top_token_ids": [] if result.tail is None else list(result.tail.top_token_ids),
            "tail_elapsed_seconds": float(result.tail_elapsed_seconds),
            "cache_sequence_lengths": {
                str(key): int(value[0].shape[1]) for key, value in result.next_kv_caches.items()
            },
            "step_summaries": [dict(item) for item in result.step_summaries],
            "elapsed_seconds": float(time.perf_counter() - started),
            "blockers": list(result.blockers),
            "ready": bool(result.ready),
        }
        print(json.dumps(payload, indent=2))
        return 0 if result.ready else 2

    print(f"Unknown mode: {mode}")
    return 1


def _hidden_size_from_status(model_id: str) -> int:
    from pcketlm.core.runtime.tensor_catalog import load_tensor_catalog

    catalog = load_tensor_catalog(model_id)
    return int(catalog.hidden_size or 7168)


def _int_option(args: list[str], name: str, default: int) -> int:
    if name not in args:
        return int(default)
    index = args.index(name)
    if index + 1 >= len(args):
        return int(default)
    return int(args[index + 1])


def _optional_int_option(args: list[str], name: str) -> int | None:
    if name not in args:
        return None
    index = args.index(name)
    if index + 1 >= len(args):
        return None
    return int(args[index + 1])


def _str_option(args: list[str], name: str, default: str) -> str:
    if name not in args:
        return default
    index = args.index(name)
    if index + 1 >= len(args):
        return default
    return str(args[index + 1])


def _prepare_chat_with_catalog_tokenizer(
    model_id: str,
    prompt: str,
    *,
    system_prompt: str = "",
) -> tuple[str, list[int], list[str]]:
    from transformers import AutoTokenizer
    from pcketlm.core.runtime.tensor_catalog import load_tensor_catalog

    catalog = load_tensor_catalog(model_id)
    if not catalog.model_dir.exists():
        return prompt, [], [f"Model directory does not exist: {catalog.model_dir}."]
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(catalog.model_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
        prepared = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = [int(value) for value in tokenizer.encode(prepared, add_special_tokens=False)]
    except Exception as exc:
        return prompt, [], [f"Chat tokenizer prepare failed: {exc}."]
    if not token_ids:
        return prepared, [], ["Chat prompt encoded to zero tokens."]
    return prepared, token_ids, []


def _encode_with_catalog_tokenizer(model_id: str, prompt: str) -> tuple[list[int], list[str]]:
    from tokenizers import Tokenizer
    from pcketlm.core.runtime.tensor_catalog import load_tensor_catalog

    catalog = load_tensor_catalog(model_id)
    tokenizer_path = catalog.model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        return [], [f"Missing tokenizer at {tokenizer_path}."]
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        encoded = tokenizer.encode(prompt)
    except Exception as exc:
        return [], [f"Tokenizer encode failed: {exc}."]
    token_ids = [int(value) for value in encoded.ids]
    if not token_ids:
        return [], ["Prompt text encoded to zero tokens."]
    return token_ids, []


def _decode_with_catalog_tokenizer(model_id: str, token_ids: list[int]) -> tuple[str, list[str]]:
    from tokenizers import Tokenizer
    from pcketlm.core.runtime.tensor_catalog import load_tensor_catalog

    if not token_ids:
        return "", []
    catalog = load_tensor_catalog(model_id)
    tokenizer_path = catalog.model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        return "", [f"Missing tokenizer at {tokenizer_path}."]
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer.decode([int(value) for value in token_ids], skip_special_tokens=False), []
    except Exception as exc:
        return "", [f"Tokenizer decode failed: {exc}."]


if __name__ == "__main__":
    raise SystemExit(main())
