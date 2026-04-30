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
    assert calls[0]["initial_decode_state"] is None
    assert calls[1]["initial_decode_state"] is first_state
    assert calls[1]["initial_token_ids"] == [1, 2, 1]
    assert calls[1]["apply_chat_format"] is True
    assert second.prefix_reuse["used"] is True
    assert second.performance_summary["bottleneck"] == "tensor loading"


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
