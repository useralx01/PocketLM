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
    assert started["session_id"] == "default"
    assert status["memory"]["free_ram_mb"] == 8192
    assert status["memory"]["process_working_set_mb"] == 512
    assert stopped["state"] == "stopped"
    assert stopped["prefix_reuse_available"] is False


def test_warm_runner_explicit_q4_moe_start_primes_cache(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage
    from pcketlm.core.runtime import layer_bridge

    calls = []

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls.append((model_id, kwargs))
        return SimpleNamespace(
            ready=True,
            generated_text=" Paris",
            full_text="The capital of France is Paris",
            generated_token_ids=[12095],
            steps_completed=1,
            max_new_tokens=1,
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={},
            reusable_token_ids=[1, 2, 3],
            final_decode_state=SimpleNamespace(ready=True),
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)
    monkeypatch.setattr(runtime.warm_runner, "tensor_residency_stats", lambda: SimpleNamespace(resident_count=3))
    monkeypatch.setattr(
        layer_bridge,
        "load_layer_bridge_config",
        lambda model_id: SimpleNamespace(
            ready=True,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=48,
        ),
    )
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.setenv("PCKETLM_DISABLE_Q4_MOE_AUTO_PRIME_REQUEST", "1")
    monkeypatch.setenv("PCKETLM_Q4_MOE_PRIME_MODE", "generate")

    status = start_warm_runner(
        "qwen3-q4-moe-prime-test",
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )

    assert status["primed"] is True
    assert status["prime_generated_text"] == " Paris"
    assert status["request_count"] == 0
    assert status["reusable_token_count"] == 3
    assert status["prefix_reuse_available"] is True
    assert status["tensor_residency_warm"] is True
    assert calls == [
        (
            "qwen3-q4-moe-prime-test",
            {
                "prompt": "The capital of France is",
                "max_new_tokens": 1,
                "min_new_tokens": 1,
                "selection_policy": "greedy",
                "apply_chat_format": False,
                "initial_decode_state": None,
                "initial_token_ids": None,
                "commit_generated_prefix": False,
            },
        )
    ]


def test_warm_runner_default_q4_moe_start_prefills_reusable_prefix(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage
    from pcketlm.core.runtime import layer_bridge

    calls = []
    state = SimpleNamespace(ready=True)

    def fake_run_prompt_prefill_session(model_id: str, **kwargs):
        calls.append((model_id, kwargs))
        return SimpleNamespace(
            ready=True,
            generated_text="",
            full_text="The capital of France is",
            generated_token_ids=[],
            steps_completed=0,
            max_new_tokens=0,
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={"prefill_session": True},
            reusable_token_ids=[1, 2, 3, 4, 5],
            final_decode_state=state,
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)
    monkeypatch.setattr(runtime.warm_runner, "tensor_residency_stats", lambda: SimpleNamespace(resident_count=3))
    monkeypatch.setattr(
        layer_bridge,
        "load_layer_bridge_config",
        lambda model_id: SimpleNamespace(
            ready=True,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=48,
        ),
    )
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.delenv("PCKETLM_Q4_MOE_PRIME_MODE", raising=False)
    monkeypatch.setenv("PCKETLM_Q4_MOE_PRIME_CACHE_TOKENS", "0")

    status = start_warm_runner(
        "qwen3-q4-moe-prefill-prime-test",
        run_prompt_prefill_session_fn=fake_run_prompt_prefill_session,
    )

    assert status["primed"] is True
    assert status["prime_generated_text"] == ""
    assert status["request_count"] == 0
    assert status["reusable_token_count"] == 5
    assert status["prefix_reuse_available"] is True
    assert calls == [
        (
            "qwen3-q4-moe-prefill-prime-test",
            {
                "prompt": "The capital of France is",
                "apply_chat_format": False,
            },
        )
    ]


def test_warm_runner_prefill_prime_can_warm_decode_expert_cache(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage
    from pcketlm.core.runtime import layer_bridge

    state = SimpleNamespace(ready=True, label="prefill")
    calls = []

    def fake_run_prompt_prefill_session(model_id: str, **kwargs):
        calls.append(("prefill", model_id, kwargs))
        return SimpleNamespace(
            ready=True,
            generated_text="",
            full_text="The capital of France is",
            generated_token_ids=[],
            steps_completed=0,
            max_new_tokens=0,
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={"prefill_session": True},
            reusable_token_ids=[1, 2, 3, 4, 5],
            final_decode_state=state,
        )

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls.append(("cache", model_id, kwargs))
        return SimpleNamespace(
            ready=True,
            generated_text=" Paris",
            full_text="The capital of France is Paris",
            generated_token_ids=[12095],
            steps_completed=1,
            max_new_tokens=kwargs["max_new_tokens"],
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={"used": True},
            reusable_token_ids=[1, 2, 3, 4, 5, 12095],
            final_decode_state=SimpleNamespace(ready=True, label="warmed"),
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)
    monkeypatch.setattr(runtime.warm_runner, "tensor_residency_stats", lambda: SimpleNamespace(resident_count=3))
    monkeypatch.setattr(
        layer_bridge,
        "load_layer_bridge_config",
        lambda model_id: SimpleNamespace(
            ready=True,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=48,
        ),
    )
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.delenv("PCKETLM_Q4_MOE_PRIME_MODE", raising=False)
    monkeypatch.setenv("PCKETLM_Q4_MOE_PRIME_CACHE_TOKENS", "3")

    status = start_warm_runner(
        "qwen3-q4-moe-prefill-cache-prime-test",
        run_prompt_prefill_session_fn=fake_run_prompt_prefill_session,
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )

    assert status["primed"] is True
    assert status["prime_generated_text"] == " Paris"
    assert status["reusable_token_count"] == 5
    assert calls[1][2]["max_new_tokens"] == 3
    assert calls[1][2]["initial_decode_state"] is state
    assert calls[1][2]["initial_token_ids"] == [1, 2, 3, 4, 5]


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
    assert calls[1]["commit_generated_prefix"] is False
    assert calls[1]["apply_chat_format"] is True
    assert second.prefix_reuse["used"] is True
    assert second.performance_summary["bottleneck"] == "tensor loading"


def test_warm_runner_can_opt_into_generated_token_prefix_commit(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage

    calls = []

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="hello OK",
            generated_token_ids=[1],
            steps_completed=1,
            max_new_tokens=kwargs["max_new_tokens"],
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={},
            reusable_token_ids=[1, 2],
            final_decode_state=SimpleNamespace(ready=True),
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)
    monkeypatch.setenv("PCKETLM_WARM_RUNNER_COMMIT_GENERATED_PREFIX", "1")

    result = run_warm_agent_prompt(
        "qwen-test-commit",
        "hello",
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )

    assert result.ready is True
    assert calls[0]["commit_generated_prefix"] is True


def test_warm_runner_applies_q4_moe_cache_defaults_during_generation(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage
    from pcketlm.core.runtime import layer_bridge

    captured_env = {}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        del model_id, kwargs
        captured_env.update(
            {
                "PCKETLM_Q4_PACKED_CACHE_MB": __import__("os").environ.get("PCKETLM_Q4_PACKED_CACHE_MB"),
                "PCKETLM_TENSOR_CACHE_MB": __import__("os").environ.get("PCKETLM_TENSOR_CACHE_MB"),
                "PCKETLM_TENSOR_CACHE_FRONT_LAYERS": __import__("os").environ.get("PCKETLM_TENSOR_CACHE_FRONT_LAYERS"),
                "PCKETLM_SAFETENSOR_HANDLE_CACHE": __import__("os").environ.get("PCKETLM_SAFETENSOR_HANDLE_CACHE"),
                "PCKETLM_ENABLE_Q4_MOE_LM_HEAD_FULL_CACHE": __import__("os").environ.get(
                    "PCKETLM_ENABLE_Q4_MOE_LM_HEAD_FULL_CACHE"
                ),
            }
        )
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="hello OK",
            generated_token_ids=[1],
            steps_completed=1,
            max_new_tokens=1,
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={},
            reusable_token_ids=[1, 2],
            final_decode_state=SimpleNamespace(ready=True),
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)
    monkeypatch.setattr(
        layer_bridge,
        "load_layer_bridge_config",
        lambda model_id: SimpleNamespace(
            ready=True,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=48,
        ),
    )
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.setenv("PCKETLM_DISABLE_Q4_MOE_AUTO_PRIME_REQUEST", "1")
    monkeypatch.delenv("PCKETLM_Q4_PACKED_CACHE_MB", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_MB", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", raising=False)
    monkeypatch.delenv("PCKETLM_ENABLE_Q4_MOE_LM_HEAD_FULL_CACHE", raising=False)

    result = run_warm_agent_prompt(
        "qwen3-q4-moe-test",
        "hello",
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )

    assert result.ready is True
    assert captured_env["PCKETLM_Q4_PACKED_CACHE_MB"] == "5120"
    assert captured_env["PCKETLM_TENSOR_CACHE_MB"] == "2048"
    assert captured_env["PCKETLM_TENSOR_CACHE_FRONT_LAYERS"] == "48"
    assert captured_env["PCKETLM_SAFETENSOR_HANDLE_CACHE"] == "0"
    assert captured_env["PCKETLM_ENABLE_Q4_MOE_LM_HEAD_FULL_CACHE"] == "1"
    assert __import__("os").environ.get("PCKETLM_Q4_PACKED_CACHE_MB") is None
    assert __import__("os").environ.get("PCKETLM_TENSOR_CACHE_MB") is None
    assert __import__("os").environ.get("PCKETLM_TENSOR_CACHE_FRONT_LAYERS") is None
    assert __import__("os").environ.get("PCKETLM_ENABLE_Q4_MOE_LM_HEAD_FULL_CACHE") is None


def test_warm_runner_auto_primes_first_q4_moe_request_with_prompt(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage
    from pcketlm.core.runtime import layer_bridge

    state = SimpleNamespace(ready=True, label="prefill")
    calls = []

    def fake_run_prompt_prefill_session(model_id: str, **kwargs):
        calls.append(("prefill", model_id, kwargs))
        return SimpleNamespace(
            ready=True,
            generated_text="",
            full_text=kwargs["prompt"],
            generated_token_ids=[],
            steps_completed=0,
            max_new_tokens=0,
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={"prefill_session": True},
            reusable_token_ids=[10, 11],
            final_decode_state=state,
        )

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls.append(("decode", model_id, kwargs))
        return SimpleNamespace(
            ready=True,
            generated_text=" OK",
            full_text=f"{kwargs['prompt']} OK",
            generated_token_ids=[1],
            steps_completed=1,
            max_new_tokens=kwargs["max_new_tokens"],
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={"used": bool(kwargs.get("initial_decode_state"))},
            reusable_token_ids=[10, 11, 1],
            final_decode_state=SimpleNamespace(ready=True, label="visible"),
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)
    monkeypatch.setattr(runtime.warm_runner, "tensor_residency_stats", lambda: SimpleNamespace(resident_count=3))
    monkeypatch.setattr(
        layer_bridge,
        "load_layer_bridge_config",
        lambda model_id: SimpleNamespace(
            ready=True,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=48,
        ),
    )
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.setenv("PCKETLM_Q4_MOE_PRIME_CACHE_TOKENS", "2")

    result = run_warm_agent_prompt(
        "qwen3-q4-moe-auto-prime-test",
        "use this prompt",
        apply_chat_format=False,
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
        run_prompt_prefill_session_fn=fake_run_prompt_prefill_session,
    )

    assert result.ready is True
    assert result.runner_status["primed"] is True
    assert result.runner_status["auto_primed"] is True
    assert result.runner_status["prime_prompt"] == "use this prompt"
    assert result.runner_status["prime_cache_tokens"] == 2
    assert calls[0][0] == "prefill"
    assert calls[0][2]["prompt"] == "use this prompt"
    assert calls[0][2]["apply_chat_format"] is False
    assert calls[1][0] == "decode"
    assert calls[1][2]["max_new_tokens"] == 2
    assert calls[2][0] == "decode"
    assert calls[2][2]["initial_decode_state"] is state


def test_warm_runner_allows_longer_q4_moe_visible_chunks(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage
    from pcketlm.core.runtime import layer_bridge

    calls = []

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="hello OK",
            generated_token_ids=[1, 2, 3, 4, 5],
            steps_completed=5,
            max_new_tokens=kwargs["max_new_tokens"],
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={},
            reusable_token_ids=[1, 2],
            final_decode_state=SimpleNamespace(ready=True),
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)
    monkeypatch.setattr(
        layer_bridge,
        "load_layer_bridge_config",
        lambda model_id: SimpleNamespace(
            ready=True,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=48,
        ),
    )
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.setenv("PCKETLM_DISABLE_Q4_MOE_AUTO_PRIME_REQUEST", "1")

    result = run_warm_agent_prompt(
        "qwen3-q4-moe-chunk-test",
        "hello",
        max_new_tokens=10,
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )

    assert result.ready is True
    assert calls[0]["max_new_tokens"] == 10
    assert result.max_new_tokens == 10


def test_warm_runner_keeps_legacy_two_token_cap_for_non_q4_moe(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage

    calls = []

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="hello OK",
            generated_token_ids=[1, 2],
            steps_completed=2,
            max_new_tokens=kwargs["max_new_tokens"],
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={},
            reusable_token_ids=[1, 2],
            final_decode_state=SimpleNamespace(ready=True),
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)
    monkeypatch.delenv("PCKETLM_TENSOR_SOURCE", raising=False)

    result = run_warm_agent_prompt(
        "qwen-test-cap",
        "hello",
        max_new_tokens=10,
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )

    assert result.ready is True
    assert calls[0]["max_new_tokens"] == 2
    assert result.max_new_tokens == 2


def test_warm_runner_trims_dequantized_q4_moe_cache_under_low_ram(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage
    from pcketlm.core.runtime import layer_bridge

    snapshots = [
        MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
        MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
        MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=1500 * 1024**2),
        MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=3200 * 1024**2),
    ]
    trims = []

    def fake_memory_snapshot():
        if snapshots:
            return snapshots.pop(0)
        return MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=3200 * 1024**2)

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        del model_id, kwargs
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="hello OK",
            generated_token_ids=[1],
            steps_completed=1,
            max_new_tokens=1,
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={},
            reusable_token_ids=[1, 2],
            final_decode_state=SimpleNamespace(ready=True),
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(runtime.warm_runner, "_memory_snapshot", fake_memory_snapshot)
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)
    monkeypatch.setattr(
        layer_bridge,
        "load_layer_bridge_config",
        lambda model_id: SimpleNamespace(
            ready=True,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=48,
        ),
    )
    monkeypatch.setattr(runtime.warm_runner, "clear_dequantized_tensor_residency_cache", lambda: trims.append(True))
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.setenv("PCKETLM_DISABLE_Q4_MOE_AUTO_PRIME_REQUEST", "1")
    monkeypatch.setenv("PCKETLM_Q4_MOE_LOW_RAM_TRIM_MB", "2500")

    result = run_warm_agent_prompt(
        "qwen3-q4-moe-trim-test",
        "hello",
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )

    assert result.ready is True
    assert trims == [True]
    trim = result.prefix_reuse["q4_moe_low_ram_trim"]
    assert trim["applied"] is True
    assert trim["threshold_mb"] == 2500
    assert trim["free_before_mb"] == 1500
    assert result.memory_after["free_ram_mb"] == 3200


def test_warm_runner_trims_q4_moe_cache_before_memory_guard_blocks(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage
    from pcketlm.core.runtime import layer_bridge

    snapshots = [
        MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
        MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=1400 * 1024**2),
        MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=5200 * 1024**2),
        MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=5000 * 1024**2),
    ]
    trims = []
    calls = []

    def fake_memory_snapshot():
        if snapshots:
            return snapshots.pop(0)
        return MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=5000 * 1024**2)

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls.append((model_id, kwargs))
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="hello OK",
            generated_token_ids=[1],
            steps_completed=1,
            max_new_tokens=1,
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={},
            reusable_token_ids=[1, 2],
            final_decode_state=SimpleNamespace(ready=True),
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(runtime.warm_runner, "_memory_snapshot", fake_memory_snapshot)
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)
    monkeypatch.setattr(
        layer_bridge,
        "load_layer_bridge_config",
        lambda model_id: SimpleNamespace(
            ready=True,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=48,
        ),
    )
    monkeypatch.setattr(runtime.warm_runner, "clear_dequantized_tensor_residency_cache", lambda: trims.append(True))
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.setenv("PCKETLM_DISABLE_Q4_MOE_AUTO_PRIME_REQUEST", "1")
    monkeypatch.setenv("PCKETLM_Q4_MOE_LOW_RAM_TRIM_MB", "2500")

    result = run_warm_agent_prompt(
        "qwen3-q4-moe-guard-trim-test",
        "hello",
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )

    assert result.ready is True
    assert trims == [True]
    assert len(calls) == 1
    assert result.memory_before["free_ram_mb"] == 5200


def test_warm_runner_preserves_explicit_q4_moe_cache_settings(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage
    from pcketlm.core.runtime import layer_bridge

    captured_env = {}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        del model_id, kwargs
        captured_env.update(
            {
                "PCKETLM_Q4_PACKED_CACHE_MB": __import__("os").environ.get("PCKETLM_Q4_PACKED_CACHE_MB"),
                "PCKETLM_TENSOR_CACHE_MB": __import__("os").environ.get("PCKETLM_TENSOR_CACHE_MB"),
                "PCKETLM_TENSOR_CACHE_FRONT_LAYERS": __import__("os").environ.get("PCKETLM_TENSOR_CACHE_FRONT_LAYERS"),
            }
        )
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="hello OK",
            generated_token_ids=[1],
            steps_completed=1,
            max_new_tokens=1,
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={},
            reusable_token_ids=[1, 2],
            final_decode_state=SimpleNamespace(ready=True),
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)
    monkeypatch.setattr(
        layer_bridge,
        "load_layer_bridge_config",
        lambda model_id: SimpleNamespace(
            ready=True,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=48,
        ),
    )
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.setenv("PCKETLM_DISABLE_Q4_MOE_AUTO_PRIME_REQUEST", "1")
    monkeypatch.setenv("PCKETLM_Q4_PACKED_CACHE_MB", "1024")
    monkeypatch.setenv("PCKETLM_TENSOR_CACHE_MB", "768")
    monkeypatch.setenv("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", "12")

    result = run_warm_agent_prompt(
        "qwen3-q4-moe-test",
        "hello",
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )

    assert result.ready is True
    assert captured_env["PCKETLM_Q4_PACKED_CACHE_MB"] == "1024"
    assert captured_env["PCKETLM_TENSOR_CACHE_MB"] == "768"
    assert captured_env["PCKETLM_TENSOR_CACHE_FRONT_LAYERS"] == "12"


def test_warm_runner_keeps_decode_state_scoped_to_session(tmp_path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage

    calls = []
    states = {
        "a": SimpleNamespace(ready=True, label="state-a"),
        "b": SimpleNamespace(ready=True, label="state-b"),
    }

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls.append(kwargs)
        session_state = states["a"] if len(calls) in {1, 3} else states["b"]
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text=f"{kwargs['prompt']} OK",
            generated_token_ids=[1],
            steps_completed=1,
            max_new_tokens=kwargs["max_new_tokens"],
            blockers=[],
            timings={"total": 1.0},
            prefix_reuse={"used": bool(kwargs.get("initial_decode_state"))},
            reusable_token_ids=[1, len(calls)],
            final_decode_state=session_state,
        )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.warm_runner,
        "_memory_snapshot",
        lambda: MemorySnapshot(total_bytes=16 * 1024**3, free_bytes=8 * 1024**3),
    )
    monkeypatch.setattr(runtime.warm_runner, "_process_working_set_bytes", lambda: 512 * 1024**2)

    first_a = run_warm_agent_prompt(
        "qwen-test-session",
        "hello",
        session_id="chat-a",
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )
    first_b = run_warm_agent_prompt(
        "qwen-test-session",
        "hello",
        session_id="chat-b",
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )
    second_a = run_warm_agent_prompt(
        "qwen-test-session",
        "hello again",
        session_id="chat-a",
        run_prompt_decode_loop_fn=fake_run_prompt_decode_loop,
    )

    assert first_a.runner_status["session_id"] == "chat-a"
    assert first_b.runner_status["session_id"] == "chat-b"
    assert second_a.ready is True
    assert calls[0]["initial_decode_state"] is None
    assert calls[1]["initial_decode_state"] is None
    assert calls[2]["initial_decode_state"] is states["a"]
    assert calls[2]["initial_decode_state"] is not states["b"]
    assert warm_runner_status("qwen-test-session", session_id="chat-a")["reusable_token_count"] == 2
    assert warm_runner_status("qwen-test-session", session_id="chat-b")["reusable_token_count"] == 2


def test_warm_runner_cli_sequence_chains_second_prompt(monkeypatch, capsys) -> None:
    from pcketlm.app.chat_shell import warm_runner_cli

    calls = []

    def fake_start(
        model_id: str,
        session_id: str = "default",
        prime_prompt=None,
        prime_apply_chat_format: bool = False,
    ):
        del prime_prompt, prime_apply_chat_format
        return {"model_id": model_id, "session_id": session_id, "state": "ready"}

    def fake_status(model_id: str, session_id: str = "default"):
        return {"model_id": model_id, "session_id": session_id, "state": "ready", "request_count": 2}

    def fake_run(
        model_id: str,
        prompt: str,
        *,
        session_id: str,
        max_new_tokens: int,
        min_free_memory_mb: int,
        apply_chat_format: bool,
    ):
        calls.append(
            {
                "model_id": model_id,
                "session_id": session_id,
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
            "--session-id",
            "probe",
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
    assert calls[0]["model_id"] == "qwen-test"
    assert calls[0]["session_id"] == "probe"
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
