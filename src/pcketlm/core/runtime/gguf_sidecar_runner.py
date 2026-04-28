"""Subprocess runner for optional llama.cpp/GGUF prompts."""

from __future__ import annotations

import json
import sys
import time

from llama_cpp import Llama


def main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    started = time.perf_counter()
    llm_kwargs = {
        "model_path": str(payload["model_path"]),
        "n_ctx": int(payload.get("n_ctx") or 2048),
        "verbose": False,
    }
    if payload.get("n_threads") is not None:
        llm_kwargs["n_threads"] = int(payload["n_threads"])
    llm = Llama(**llm_kwargs)
    call_kwargs = {"max_tokens": int(payload.get("max_tokens") or 32), "echo": False}
    stop_strings = list(payload.get("stop_strings") or [])
    if stop_strings:
        call_kwargs["stop"] = stop_strings
    output = llm(str(payload.get("prompt") or ""), **call_kwargs)
    choices = output.get("choices", []) if isinstance(output, dict) else []
    text = str(choices[0].get("text", "")) if choices else ""
    for stop_string in stop_strings:
        if stop_string and stop_string in text:
            text = text.split(stop_string, 1)[0]
    print(
        json.dumps(
            {
                "ready": True,
                "generated_text": text,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
