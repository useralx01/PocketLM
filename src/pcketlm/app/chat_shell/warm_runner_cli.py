"""CLI for the conservative in-process warm Agent runner."""

from __future__ import annotations

import json
import sys

from pcketlm.core.runtime import run_warm_agent_prompt, start_warm_runner, stop_warm_runner, warm_runner_status


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _usage() -> str:
    return (
        "Usage: py -m pcketlm.app.chat_shell.warm_runner_cli "
        "<start|run|status|stop|sequence> --model <id> [--session-id <id>] [--prompt <text>] [--second-prompt <text>] "
        "[--prime-prompt <text>] [--max-new-tokens <n>] [--min-free-memory-mb <n>] [--raw] [--independent-second]"
    )


def _parse(args: list[str]) -> tuple[str | None, dict]:
    if not args:
        return None, {}
    command = args[0].strip().lower()
    options: dict = {
        "model": "qwen2.5-14b-instruct",
        "prompt": "hello world",
        "second_prompt": "reply ok only",
        "prime_prompt": None,
        "session_id": "default",
        "max_new_tokens": 2,
        "min_free_memory_mb": 4096,
        "apply_chat_format": True,
        "chain_second": True,
    }
    index = 1
    while index < len(args):
        flag = args[index]
        if flag == "--model" and index + 1 < len(args):
            options["model"] = args[index + 1]
            index += 2
            continue
        if flag == "--prompt" and index + 1 < len(args):
            options["prompt"] = args[index + 1]
            index += 2
            continue
        if flag == "--second-prompt" and index + 1 < len(args):
            options["second_prompt"] = args[index + 1]
            index += 2
            continue
        if flag == "--prime-prompt" and index + 1 < len(args):
            options["prime_prompt"] = args[index + 1]
            index += 2
            continue
        if flag == "--session-id" and index + 1 < len(args):
            options["session_id"] = args[index + 1]
            index += 2
            continue
        if flag == "--max-new-tokens" and index + 1 < len(args):
            options["max_new_tokens"] = max(1, int(args[index + 1]))
            index += 2
            continue
        if flag == "--min-free-memory-mb" and index + 1 < len(args):
            options["min_free_memory_mb"] = max(0, int(args[index + 1]))
            index += 2
            continue
        if flag == "--raw":
            options["apply_chat_format"] = False
            index += 1
            continue
        if flag == "--independent-second":
            options["chain_second"] = False
            index += 1
            continue
        raise ValueError(_usage())
    return command, options


def main(argv: list[str] | None = None) -> int:
    try:
        command, options = _parse(list(sys.argv[1:] if argv is None else argv))
    except ValueError as exc:
        print(str(exc))
        return 1
    if command is None:
        print(_usage())
        return 1

    model_id = str(options["model"])
    session_id = str(options["session_id"])
    if command == "start":
        _print(
            start_warm_runner(
                model_id,
                session_id=session_id,
                prime_prompt=options.get("prime_prompt"),
                prime_apply_chat_format=bool(options["apply_chat_format"]),
            )
        )
        return 0
    if command == "status":
        _print(warm_runner_status(model_id, session_id=session_id))
        return 0
    if command == "stop":
        _print(stop_warm_runner(model_id, session_id=session_id))
        return 0
    if command == "run":
        result = run_warm_agent_prompt(
            model_id,
            str(options["prompt"]),
            session_id=session_id,
            max_new_tokens=int(options["max_new_tokens"]),
            min_free_memory_mb=int(options["min_free_memory_mb"]),
            apply_chat_format=bool(options["apply_chat_format"]),
        )
        _print(result.to_dict())
        return 0 if result.ready else 2
    if command == "sequence":
        start = start_warm_runner(
            model_id,
            session_id=session_id,
            prime_prompt=str(options.get("prime_prompt") or options["prompt"]),
            prime_apply_chat_format=bool(options["apply_chat_format"]),
        )
        first = run_warm_agent_prompt(
            model_id,
            str(options["prompt"]),
            session_id=session_id,
            max_new_tokens=int(options["max_new_tokens"]),
            min_free_memory_mb=int(options["min_free_memory_mb"]),
            apply_chat_format=bool(options["apply_chat_format"]),
        )
        second_prompt = str(options["second_prompt"])
        if bool(options["chain_second"]) and first.ready:
            full_text = str(first.full_text or options["prompt"])
            separator = "" if full_text.endswith(("\n", " ")) else "\n"
            second_prompt = f"{full_text}{separator}{second_prompt}"
        second = run_warm_agent_prompt(
            model_id,
            second_prompt,
            session_id=session_id,
            max_new_tokens=int(options["max_new_tokens"]),
            min_free_memory_mb=int(options["min_free_memory_mb"]),
            apply_chat_format=bool(options["apply_chat_format"]),
        )
        status = warm_runner_status(model_id, session_id=session_id)
        _print(
            {
                "start": start,
                "first": first.to_dict(),
                "second": second.to_dict(),
                "status": status,
                "sequence": {
                    "second_prompt_chained": bool(options["chain_second"]) and first.ready,
                    "second_prompt_token_prefix_expected": bool(options["chain_second"]) and first.ready,
                },
            }
        )
        return 0 if first.ready and second.ready else 2

    print(_usage())
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
