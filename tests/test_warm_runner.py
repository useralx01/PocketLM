from types import SimpleNamespace

from pcketlm.core.runtime.load_attempt import MemorySnapshot
from pcketlm.core.runtime.warm_runner import (
    run_warm_agent_prompt,
    start_warm_runner,
    stop_warm_runner,
    warm_runner_status,
)


def test_warm_runner_start_status_stop(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)

    started = start_warm_runner("qwen-test")
    status = warm_runner_status("qwen-test")
    stopped = stop_warm_runner("qwen-test")

    assert started["state"] == "ready"
    assert status["memory"]["free_ram_mb"] == 8192
    assert status["memory"]["process_working_set_mb"] == 512
    assert stopped["state"] == "stopped"
    assert stopped["prefix_reuse_available"] is False


def test_warm_runner_second_request_uses_prior_decode_state(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage

    calls = []
    first_state = SimpleNamespace(ready=True)
    second_state = SimpleNamespace(ready=True)

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls.append(kwargs)
        state = first_state if len(calls) == 1 else second_state
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text=f"full-{len(calls)}",
            generated_token_ids=[1],
            steps_completed=1,
            max_new_tokens=kwargs["max_new_tokens"],
            blockers=[],
            timings={
                "total": 2.0,
                "prefill_stack_op_load_tensors": 1.0,
            },
            prefix_reuse={"used": bool(kwargs.get("initial_decode_state"))},
            reusable_token_ids=[1, 2, len(calls)],
            final_decode_state=state,
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)

    first = run_warm_agent_prompt("qwen-test", "hello", run_prompt_decode_loop_fn=fake_run_prompt_decode_loop)
    second = run_warm_agent_prompt("qwen-test", "again", run_prompt_decode_loop_fn=fake_run_prompt_decode_loop)

    assert first.ready is True
    assert second.ready is True
    assert first.full_text == "full-1"
    assert first.to_dict()["full_text"] == "full-1"
    assert calls[0]["initial_decode_state"] is None
    assert calls[1]["initial_decode_state"] is first_state
    assert calls[1]["initial_token_ids"] == [1, 2, 1]
    assert calls[1]["apply_chat_format"] is True
    assert second.prefix_reuse["used"] is True
    assert second.performance_summary["bottleneck"] == "tensor loading"


def test_warm_runner_cli_sequence_chains_second_prompt(monkeypatch, capsys) -> None:
    from pcketlm.app.chat_shell import warm_runner_cli

    calls = []

    def fake_start(model_id: str):
        return {"model_id": model_id, "state": "ready"}

    def fake_status(model_id: str):
        return {"model_id": model_id, "state": "ready", "request_count": 2}

    def fake_run(
        model_id: str,
        prompt: str,
        *,
        max_new_tokens: int,
        min_free_memory_mb: int,
        apply_chat_format: bool,
    ):
        calls.append(
            {
                "model_id": model_id,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "min_free_memory_mb": min_free_memory_mb,
                "apply_chat_format": apply_chat_format,
            }
        )
        return SimpleNamespace(
            ready=True,
            full_text="hello OK" if len(calls) == 1 else prompt + " OK",
            to_dict=lambda: {"ready": True, "prompt": prompt, "full_text": "hello OK"},
        )

    monkeypatch.setattr(warm_runner_cli, "start_warm_runner", fake_start)
    monkeypatch.setattr(warm_runner_cli, "warm_runner_status", fake_status)
    monkeypatch.setattr(warm_runner_cli, "run_warm_agent_prompt", fake_run)

    exit_code = warm_runner_cli.main(
        [
            "sequence",
            "--model",
            "qwen-test",
            "--prompt",
            "hello",
            "--second-prompt",
            "again",
            "--max-new-tokens",
            "2",
            "--min-free-memory-mb",
            "1024",
            "--raw",
        ]
    )

    assert exit_code == 0
    assert calls[0]["prompt"] == "hello"
    assert calls[0]["max_new_tokens"] == 2
    assert calls[0]["min_free_memory_mb"] == 1024
    assert calls[0]["apply_chat_format"] is False
    assert calls[1]["prompt"] == "hello OK\nagain"
    assert calls[1]["apply_chat_format"] is False
    payload = capsys.readouterr().out
    assert '"second_prompt_chained": true' in payload


def test_warm_runner_blocks_low_memory_before_generation(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage

    def should_not_run(*args, **kwargs):
        raise AssertionError("warm runner should block before model generation")

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=1 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)

    result = run_warm_agent_prompt("qwen-test", "hello", run_prompt_decode_loop_fn=should_not_run)

    assert result.ready is False
    assert "Free RAM is below" in result.blockers[0]
    assert result.runner_status["state"] == "blocked"
