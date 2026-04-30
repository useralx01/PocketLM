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
        "<start|run|status|stop|sequence> --model <id> [--prompt <text>] [--second-prompt <text>] [--max-new-tokens <n>]"
    )


def _parse(args: list[str]) -> tuple[str | None, dict]:
    if not args:
        return None, {}
    command = args[0].strip().lower()
    options: dict = {"model": "qwen2.5-14b-instruct", "prompt": "hello world", "second_prompt": "reply ok only", "max_new_tokens": 2}
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
        if flag == "--max-new-tokens" and index + 1 < len(args):
            options["max_new_tokens"] = max(1, int(args[index + 1]))
            index += 2
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
    if command == "start":
        _print(start_warm_runner(model_id))
        return 0
    if command == "status":
        _print(warm_runner_status(model_id))
        return 0
    if command == "stop":
        _print(stop_warm_runner(model_id))
        return 0
    if command == "run":
        result = run_warm_agent_prompt(
            model_id,
            str(options["prompt"]),
            max_new_tokens=int(options["max_new_tokens"]),
        )
        _print(result.to_dict())
        return 0 if result.ready else 2
    if command == "sequence":
        start = start_warm_runner(model_id)
        first = run_warm_agent_prompt(
            model_id,
            str(options["prompt"]),
            max_new_tokens=int(options["max_new_tokens"]),
        )
        second = run_warm_agent_prompt(
            model_id,
            str(options["second_prompt"]),
            max_new_tokens=int(options["max_new_tokens"]),
        )
        status = warm_runner_status(model_id)
        _print({"start": start, "first": first.to_dict(), "second": second.to_dict(), "status": status})
        return 0 if first.ready and second.ready else 2

    print(_usage())
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
