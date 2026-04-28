"""CLI for the first minimal layer-forward bridge."""

from __future__ import annotations

import json
import sys

from pcketlm.core.runtime import (
    DEFAULT_LM_HEAD_CHUNK_ROWS,
    run_decode_benchmark,
    run_decode_tail,
    run_kv_decode_loop,
    run_layer_bridge_stack,
    run_minimal_layer_forward_bridge,
    run_prompt_decode_loop,
    run_repeated_decode_loop,
    run_token_decode_step,
    run_token_entry_layer_bridge,
)


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if not args:
        print("Usage: py -m pcketlm.app.chat_shell.runtime_layer_bridge_cli <model-id> [--layer <index>] [--layers <count>]")
        return 1

    model_id = args[0]
    layer_index = 0
    layer_count = 1
    layers_explicit = False
    token_id: int | None = None
    decode = False
    loop_steps: int | None = None
    chunk_rows = DEFAULT_LM_HEAD_CHUNK_ROWS
    history_window = 1
    policy = "greedy"
    temperature = 1.0
    top_p: float | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    min_new_tokens = 1
    stop_token_ids: list[int] | None = None
    stop_strings: list[str] | None = None
    sample_seed: int | None = None
    kv_aware = False
    benchmark = False
    prompt: str | None = None
    system_prompt: str | None = None
    raw_prompt = False
    index = 1
    while index < len(args):
        flag = args[index]
        if flag == "--layer" and index + 1 < len(args):
            try:
                layer_index = int(args[index + 1])
            except ValueError:
                print("Layer index must be an integer.")
                return 1
            index += 2
            continue
        if flag == "--layers" and index + 1 < len(args):
            try:
                layer_count = int(args[index + 1])
            except ValueError:
                print("Layer count must be an integer.")
                return 1
            layers_explicit = True
            index += 2
            continue
        if flag == "--token-id" and index + 1 < len(args):
            try:
                token_id = int(args[index + 1])
            except ValueError:
                print("Token id must be an integer.")
                return 1
            index += 2
            continue
        if flag == "--chunk-rows" and index + 1 < len(args):
            try:
                chunk_rows = int(args[index + 1])
            except ValueError:
                print("Chunk rows must be an integer.")
                return 1
            index += 2
            continue
        if flag == "--decode":
            decode = True
            index += 1
            continue
        if flag == "--loop-steps" and index + 1 < len(args):
            try:
                loop_steps = int(args[index + 1])
            except ValueError:
                print("Loop steps must be an integer.")
                return 1
            index += 2
            continue
        if flag == "--history-window" and index + 1 < len(args):
            try:
                history_window = int(args[index + 1])
            except ValueError:
                print("History window must be an integer.")
                return 1
            index += 2
            continue
        if flag == "--policy" and index + 1 < len(args):
            policy = args[index + 1]
            index += 2
            continue
        if flag == "--temperature" and index + 1 < len(args):
            try:
                temperature = float(args[index + 1])
            except ValueError:
                print("Temperature must be a number.")
                return 1
            index += 2
            continue
        if flag == "--top-p" and index + 1 < len(args):
            try:
                top_p = float(args[index + 1])
            except ValueError:
                print("Top-p must be a number.")
                return 1
            index += 2
            continue
        if flag == "--sample-seed" and index + 1 < len(args):
            try:
                sample_seed = int(args[index + 1])
            except ValueError:
                print("Sample seed must be an integer.")
                return 1
            index += 2
            continue
        if flag == "--repetition-penalty" and index + 1 < len(args):
            try:
                repetition_penalty = float(args[index + 1])
            except ValueError:
                print("Repetition penalty must be a number.")
                return 1
            index += 2
            continue
        if flag == "--max-new-tokens" and index + 1 < len(args):
            try:
                max_new_tokens = int(args[index + 1])
            except ValueError:
                print("Max new tokens must be an integer.")
                return 1
            index += 2
            continue
        if flag == "--min-new-tokens" and index + 1 < len(args):
            try:
                min_new_tokens = int(args[index + 1])
            except ValueError:
                print("Min new tokens must be an integer.")
                return 1
            index += 2
            continue
        if flag == "--stop-token-ids" and index + 1 < len(args):
            try:
                stop_token_ids = [int(part.strip()) for part in args[index + 1].split(",") if part.strip()]
            except ValueError:
                print("Stop token ids must be a comma-separated list of integers.")
                return 1
            index += 2
            continue
        if flag == "--stop-strings" and index + 1 < len(args):
            stop_strings = [part for part in args[index + 1].split("|") if part]
            index += 2
            continue
        if flag == "--prompt" and index + 1 < len(args):
            prompt = args[index + 1]
            index += 2
            continue
        if flag == "--system-prompt" and index + 1 < len(args):
            system_prompt = args[index + 1]
            index += 2
            continue
        if flag == "--raw-prompt":
            raw_prompt = True
            index += 1
            continue
        if flag == "--kv-aware":
            kv_aware = True
            index += 1
            continue
        if flag == "--benchmark":
            benchmark = True
            index += 1
            continue
        print("Usage: py -m pcketlm.app.chat_shell.runtime_layer_bridge_cli <model-id> [--layer <index>] [--layers <count>] [--token-id <id>] [--prompt <text>] [--system-prompt <text>] [--raw-prompt] [--decode] [--loop-steps <n>] [--max-new-tokens <n>] [--min-new-tokens <n>] [--stop-token-ids <id,id,...>] [--stop-strings <value|value>] [--chunk-rows <rows>] [--history-window <n>] [--policy <greedy|top-k-sample>] [--temperature <float>] [--top-p <float>] [--repetition-penalty <float>] [--sample-seed <int>] [--kv-aware] [--benchmark]")
        return 1

    if prompt is not None:
        result = run_prompt_decode_loop(
            model_id,
            prompt=prompt,
            steps=1 if loop_steps is None else loop_steps,
            max_new_tokens=max_new_tokens,
            start_layer=layer_index,
            layer_count=layer_count if layers_explicit else None,
            lm_head_chunk_rows=chunk_rows,
            selection_policy=policy,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            system_prompt=system_prompt,
            apply_chat_format=not raw_prompt,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            sample_seed=sample_seed,
        )
    elif token_id is not None and loop_steps is not None and benchmark:
        result = run_decode_benchmark(
            model_id,
            seed_token_id=token_id,
            steps=loop_steps,
            start_layer=layer_index,
            layer_count=layer_count,
            lm_head_chunk_rows=chunk_rows,
            history_window=history_window,
            temperature=temperature,
            sample_seed=sample_seed,
        )
    elif token_id is not None and loop_steps is not None and kv_aware:
        result = run_kv_decode_loop(
            model_id,
            seed_token_id=token_id,
            steps=loop_steps,
            start_layer=layer_index,
            layer_count=layer_count,
            lm_head_chunk_rows=chunk_rows,
            selection_policy=policy,
            temperature=temperature,
            sample_seed=sample_seed,
        )
    elif token_id is not None and loop_steps is not None:
        result = run_repeated_decode_loop(
            model_id,
            seed_token_id=token_id,
            steps=loop_steps,
            start_layer=layer_index,
            layer_count=layer_count,
            lm_head_chunk_rows=chunk_rows,
            history_window=history_window,
            selection_policy=policy,
            temperature=temperature,
            sample_seed=sample_seed,
        )
    elif token_id is not None and decode:
        result = run_token_decode_step(
            model_id,
            token_ids=[token_id],
            start_layer=layer_index,
            layer_count=layer_count,
            lm_head_chunk_rows=chunk_rows,
            history_window=history_window,
            selection_policy=policy,
            temperature=temperature,
            sample_seed=sample_seed,
        )
    elif token_id is not None:
        result = run_token_entry_layer_bridge(
            model_id,
            token_ids=[token_id],
            start_layer=layer_index,
            layer_count=layer_count,
        )
    elif decode:
        stack_result = run_layer_bridge_stack(model_id, start_layer=layer_index, layer_count=layer_count)
        if not stack_result.ready or stack_result.output_tensor is None:
            result = stack_result
        else:
            result = run_decode_tail(model_id, stack_result.output_tensor, lm_head_chunk_rows=chunk_rows)
    elif layer_count > 1:
        result = run_layer_bridge_stack(model_id, start_layer=layer_index, layer_count=layer_count)
    else:
        result = run_minimal_layer_forward_bridge(model_id, layer_index=layer_index)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
