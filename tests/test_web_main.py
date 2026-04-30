import socket
from types import SimpleNamespace

import pytest

from pcketlm.app.web import main as web_main
from pcketlm.app.web.main import (
    ChatJob,
    _acquire_single_instance_lock,
    _existing_server_is_pocket_llm,
    _release_single_instance_lock,
    _cancel_chat_job,
    _chat_layer_count,
    _chat_layer_count_for_request,
    _chat_job_payload,
    _clear_chat_response_cache,
    _clear_session_prefix_cache,
    _CHAT_JOBS,
    _CHAT_JOBS_LOCK,
    _conversation_prompt,
    _direct_model_guardrails,
    _formatted_chat_prompt,
    _local_runtime_context_answer,
    _memory_guard_response,
    _normalize_chat_messages,
    _prompt_result_payload,
    _run_chat_payload,
    _runtime_identity_system_prompt,
    _start_chat_job,
    _update_runtime_settings,
    _warm_runner_control_payload,
)


@pytest.fixture(autouse=True)
def clear_chat_response_cache():
    _clear_chat_response_cache()
    _clear_session_prefix_cache()
    yield
    _clear_chat_response_cache()
    _clear_session_prefix_cache()


def test_chat_layer_count_maps_web_modes() -> None:
    assert _chat_layer_count("Quick") is None
    assert _chat_layer_count("Agent") is None
    assert _chat_layer_count("Fast") == 8
    assert _chat_layer_count("Balanced") == 32
    assert _chat_layer_count("Quality") is None
    assert _chat_layer_count("unknown") is None


def test_chat_layer_count_for_experimental_direct_uses_full_stack(monkeypatch) -> None:
    monkeypatch.setattr(
        web_main,
        "load_layer_bridge_config",
        lambda model_id: SimpleNamespace(num_hidden_layers=48, blockers=[]),
    )

    assert _chat_layer_count_for_request("qwen2.5-14b-instruct", "Quality", 12) == 48
    assert _chat_layer_count_for_request("qwen2.5-14b-instruct", "Quality", 8) is None


def test_existing_server_check_returns_false_for_closed_port() -> None:
    assert _existing_server_is_pocket_llm(9) is False


def test_single_instance_lock_blocks_second_acquire(monkeypatch) -> None:
    probe_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe_socket.bind(("127.0.0.1", 0))
        lock_port = int(probe_socket.getsockname()[1])
    finally:
        probe_socket.close()

    monkeypatch.setattr(web_main, "SINGLE_INSTANCE_LOCK_PORT", lock_port)

    assert _acquire_single_instance_lock() is True
    try:
        competing_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            try:
                competing_socket.bind(("127.0.0.1", lock_port))
                bound = True
            except OSError:
                bound = False
            assert bound is False
        finally:
            competing_socket.close()
    finally:
        _release_single_instance_lock()


def test_normalize_chat_messages_accepts_ui_roles_and_skips_placeholders() -> None:
    messages = _normalize_chat_messages(
        [
            {"role": "user", "text": "hello"},
            {"role": "ai", "text": "Working locally..."},
            {"role": "ai", "text": "hi"},
            {"role": "debug", "text": "hidden"},
            "not a message",
        ]
    )

    assert messages == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "hi"},
    ]


def test_conversation_prompt_wraps_recent_history() -> None:
    prompt, turn_count = _conversation_prompt(
        [
            {"role": "user", "text": "my name is Sam"},
            {"role": "assistant", "text": "Nice to meet you, Sam."},
        ],
        "what is my name?",
    )

    assert turn_count == 2
    assert "Previous conversation:" in prompt
    assert "User: my name is Sam" in prompt
    assert "Assistant: Nice to meet you, Sam." in prompt
    assert prompt.endswith("Current user message:\nwhat is my name?")


def test_conversation_prompt_returns_plain_prompt_without_history() -> None:
    assert _conversation_prompt([], "hello") == ("hello", 0)


def test_runtime_identity_system_prompt_names_current_model_and_profile() -> None:
    profile = SimpleNamespace(profile_id="low-memory", label="Low Memory")

    prompt = _runtime_identity_system_prompt("qwen2.5-14b-instruct", "Quality", profile)

    assert "Pocket LLM" in prompt
    assert "Qwen2.5-14B-Instruct" in prompt
    assert "qwen2.5-14b-instruct" in prompt
    assert "Mode: Quality" in prompt
    assert "Profile: Low Memory" in prompt


def test_local_runtime_context_answer_handles_model_identity_without_generation() -> None:
    profile = SimpleNamespace(profile_id="low-memory", label="Low Memory")

    payload = _local_runtime_context_answer("Which model do you run on?", "qwen2.5-14b-instruct", "Quality", profile)

    assert payload is not None
    assert payload["ready"] is True
    assert payload["generated_text"] == "Qwen2.5-14B-Instruct (qwen2.5-14b-instruct)"
    assert payload["strategy"] == "local-runtime-context-answer"
    assert payload["runtime_context"]["profile_label"] == "Low Memory"


def test_memory_guard_response_blocks_generation_when_free_ram_is_too_low(monkeypatch) -> None:
    profile = SimpleNamespace(profile_id="low-memory", label="Low Memory")
    monkeypatch.setattr(
        web_main,
        "tensor_residency_policy_snapshot",
        lambda: {
            "free_memory_bytes": 1 * 1024**3,
            "free_memory_gb": 1.0,
            "memory_guard_active": True,
        },
    )
    monkeypatch.setattr(web_main, "_runtime_settings_payload", lambda: {"tensor_residency_policy": {}})

    payload = _memory_guard_response("hello", "qwen2.5-14b-instruct", "Quality", profile)

    assert payload is not None
    assert payload["ready"] is False
    assert payload["stop_reason"] == "memory-guard"
    assert payload["strategy"] == "local-memory-guard"
    assert "too low" in payload["blockers"][0]
    assert payload["profile_label"] == "Low Memory"


def test_direct_model_guardrails_marks_qwen_32b_as_stable_slow(monkeypatch) -> None:
    monkeypatch.setattr(
        web_main,
        "tensor_residency_policy_snapshot",
        lambda: {
            "free_memory_bytes": 6 * 1024**3,
            "free_memory_gb": 6.0,
            "memory_guard_active": False,
        },
    )

    payload = _direct_model_guardrails("qwen2.5-32b-instruct", requested_max_new_tokens=8)

    assert payload["ready"] is True
    assert payload["status"] == "stable-slow"
    assert payload["proven_max_new_tokens"] == 8
    assert payload["recommended_mode"] == "Agent"
    assert payload["baseline_estimate"]["bottleneck"] == "tensor loading"
    assert payload["scoped_safetensor_handle_cache"]["default_enabled"] is False
    assert payload["warnings"] == []


def test_direct_model_guardrails_marks_qwen_14b_as_proven_short_path(monkeypatch) -> None:
    monkeypatch.setattr(
        web_main,
        "tensor_residency_policy_snapshot",
        lambda: {
            "free_memory_bytes": 6 * 1024**3,
            "free_memory_gb": 6.0,
            "memory_guard_active": False,
        },
    )

    payload = _direct_model_guardrails("qwen2.5-14b-instruct", requested_max_new_tokens=8)

    assert payload["ready"] is True
    assert payload["status"] == "stable-slow"
    assert payload["model_size_class"] == "direct-14b"
    assert payload["proven_max_new_tokens"] == 8
    assert payload["recommended_mode"] == "Quality"
    assert payload["baseline_estimate"]["measured_tokens"] == 8
    assert payload["baseline_estimate"]["tensor_load_share"] >= 0.8
    assert payload["warnings"] == []


def test_direct_model_guardrails_warns_on_unproven_qwen_32b_length(monkeypatch) -> None:
    monkeypatch.setattr(
        web_main,
        "tensor_residency_policy_snapshot",
        lambda: {
            "free_memory_bytes": 6 * 1024**3,
            "free_memory_gb": 6.0,
            "memory_guard_active": False,
        },
    )

    payload = _direct_model_guardrails("qwen2.5-32b-instruct", requested_max_new_tokens=12)

    assert payload["ready"] is True
    assert payload["status"] == "stable-slow"
    assert "proven to 8 new tokens" in payload["warnings"][0]
    assert payload["recommended_mode"] == "Agent"


def test_direct_model_guardrails_blocks_qwen_32b_below_ram_floor(monkeypatch) -> None:
    monkeypatch.setattr(
        web_main,
        "tensor_residency_policy_snapshot",
        lambda: {
            "free_memory_bytes": 3 * 1024**3,
            "free_memory_gb": 3.0,
            "memory_guard_active": True,
        },
    )

    payload = _direct_model_guardrails("qwen2.5-32b-instruct", requested_max_new_tokens=1)

    assert payload["ready"] is False
    assert payload["status"] == "blocked-low-ram"
    assert "below the direct-runtime guard" in payload["blockers"][0]


def test_formatted_chat_prompt_uses_qwen_chat_turns(tmp_path, monkeypatch) -> None:
    from pcketlm.app import web

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text(
        """
        {
          "added_tokens_decoder": {
            "1": {"content": "<|im_start|>"},
            "2": {"content": "<|im_end|>"}
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(web.main, "original_model_root", lambda model_id: model_dir)

    prompt, turn_count, preformatted = _formatted_chat_prompt(
        "qwen-test",
        [
            {"role": "user", "text": "my name is Sam"},
            {"role": "assistant", "text": "Nice to meet you, Sam."},
        ],
        "what is my name?",
    )

    assert turn_count == 2
    assert preformatted is True
    assert prompt.startswith("<|im_start|>system\nYou are Pocket LLM.")
    assert "<|im_start|>user\nmy name is Sam<|im_end|>" in prompt
    assert "<|im_start|>assistant\nNice to meet you, Sam.<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_formatted_chat_prompt_falls_back_without_qwen_tokens(tmp_path, monkeypatch) -> None:
    from pcketlm.app import web

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr(web.main, "original_model_root", lambda model_id: model_dir)

    prompt, turn_count, preformatted = _formatted_chat_prompt(
        "plain-test",
        [{"role": "user", "text": "hello"}],
        "continue",
    )

    assert turn_count == 1
    assert preformatted is False
    assert "Previous conversation:" in prompt


def test_prompt_result_payload_serializes_runtime_result() -> None:
    result = SimpleNamespace(
        ready=True,
        generated_text="Hello!",
        full_text="User: hello\nAssistant: Hello!",
        generated_token_ids=[9707, 0],
        prompt_token_ids=[1, 2],
        steps_completed=2,
        max_new_tokens=2,
        stop_reason="step-limit",
        strategy="greedy-prompt-kv-cache-rope",
        cache_sequence_lengths={"0": 32},
        blockers=[],
        timings={
            "total": 12.0,
            "prefill_stack_op_load_tensors": 7.0,
            "continuation_stack_op_load_tensors": 2.0,
            "prefill_decode_tail": 1.0,
        },
    )

    payload = _prompt_result_payload(result, elapsed_seconds=12.5)

    assert payload["ready"] is True
    assert payload["generated_text"] == "Hello!"
    assert payload["generated_token_ids"] == [9707, 0]
    assert payload["prompt_token_count"] == 2
    assert payload["elapsed_seconds"] == 12.5
    assert payload["performance_summary"]["bottleneck"] == "tensor loading"
    assert payload["performance_summary"]["tensor_load_seconds"] == 9.0
    assert payload["performance_summary"]["tensor_load_share"] == 0.75
    assert "runtime_settings" in payload


def test_run_chat_payload_uses_runtime_and_returns_web_metadata(monkeypatch) -> None:
    from pcketlm.app import web

    captured = {}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        captured["model_id"] = model_id
        captured.update(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="Hello!",
            full_text="Hello!",
            generated_token_ids=[1],
            prompt_token_ids=[1, 2, 3],
            steps_completed=1,
            max_new_tokens=4,
            stop_reason="step-limit",
            strategy="fake",
            cache_sequence_lengths={"0": 3},
            blockers=[],
        )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(web.main, "_supports_im_chat_tokens", lambda model_id: True)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 8 * 1024**3, "free_memory_gb": 8.0, "memory_guard_active": False},
    )

    payload = _run_chat_payload(
        {
            "model_id": "qwen-test",
            "prompt": "What is my name?",
            "messages": [{"role": "user", "text": "My name is Sam."}],
            "max_new_tokens": 4,
        },
        should_cancel=lambda: False,
    )

    assert payload["ready"] is True
    assert payload["conversation_turn_count"] == 1
    assert payload["preformatted_chat"] is True
    assert "runtime_settings" in payload
    assert payload["runtime_context"]["model_id"] == "qwen-test"
    assert payload["conversation_state"]["state_kind"] == "bounded-recent-chat"
    assert payload["conversation_state"]["reuse_ready"] is False
    assert payload["response_reuse"]["hit"] is False
    assert captured["apply_chat_format"] is False
    assert captured["layer_count"] is None
    assert "Model: qwen-test (qwen-test)" in captured["system_prompt"]
    assert callable(captured["should_cancel"])


def test_run_chat_payload_quick_mode_caps_tokens_without_reduced_layers(monkeypatch) -> None:
    from pcketlm.app import web

    captured = {}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="OK",
            generated_token_ids=[1],
            prompt_token_ids=[1, 2],
            steps_completed=1,
            max_new_tokens=1,
            stop_reason="step-limit",
            strategy="fake",
            cache_sequence_lengths={},
            blockers=[],
        )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 8 * 1024**3, "free_memory_gb": 8.0, "memory_guard_active": False},
    )

    payload = _run_chat_payload(
        {
            "model_id": "qwen-test",
            "prompt": "reply OK",
            "mode": "Quick",
            "max_new_tokens": 8,
        }
    )

    assert payload["ready"] is True
    assert captured["layer_count"] is None
    assert captured["max_new_tokens"] == 1
    assert captured["min_new_tokens"] == 1


def test_run_chat_payload_includes_qwen_32b_guardrails(monkeypatch) -> None:
    from pcketlm.app import web

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        return SimpleNamespace(
            ready=True,
            generated_text="Hello World! It",
            full_text="Hello World! It",
            generated_token_ids=[9707, 4337, 0, 1084],
            prompt_token_ids=[1, 2, 3],
            steps_completed=4,
            max_new_tokens=4,
            stop_reason="step-limit",
            strategy="fake",
            cache_sequence_lengths={"0": 34},
            blockers=[],
        )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(
        web.main,
        "_load_saved_runtime_settings",
        lambda: {"tensor_cache_preset": "standard", "agent_warm_runner": "off"},
    )
    monkeypatch.setattr(web.main, "_apply_runtime_settings", lambda settings: None)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 6 * 1024**3, "free_memory_gb": 6.0, "memory_guard_active": False},
    )

    payload = _run_chat_payload(
        {
            "model_id": "qwen2.5-32b-instruct",
            "prompt": "hello world",
            "mode": "Quality",
            "max_new_tokens": 4,
        }
    )

    assert payload["ready"] is True
    assert payload["model_guardrails"]["status"] == "stable-slow"
    assert payload["model_guardrails"]["proven_max_new_tokens"] == 8
    assert payload["model_guardrails"]["scoped_safetensor_handle_cache"]["default_enabled"] is False


def test_run_chat_payload_caps_qwen_32b_to_proven_token_range(monkeypatch) -> None:
    from pcketlm.app import web

    captured = {}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="Hello World! It's great to see",
            full_text="Hello World! It's great to see",
            generated_token_ids=[9707, 4337, 0, 1084, 594, 2244, 311, 1490],
            prompt_token_ids=[1, 2, 3],
            steps_completed=8,
            max_new_tokens=kwargs["max_new_tokens"],
            stop_reason="step-limit",
            strategy="fake",
            cache_sequence_lengths={"0": 34},
            blockers=[],
        )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 6 * 1024**3, "free_memory_gb": 6.0, "memory_guard_active": False},
    )

    payload = _run_chat_payload(
        {
            "model_id": "qwen2.5-32b-instruct",
            "prompt": "hello world",
            "mode": "Quality",
            "max_new_tokens": 12,
        }
    )

    assert captured["max_new_tokens"] == 8
    assert payload["max_new_tokens"] == 8
    assert payload["model_guardrails"]["requested_max_new_tokens"] == 8
    assert payload["model_guardrails"]["warnings"] == []


def test_run_chat_payload_caps_qwen_14b_to_proven_token_range(monkeypatch) -> None:
    from pcketlm.app import web

    captured = {}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="Hello! How can I assist you today",
            full_text="Hello! How can I assist you today",
            generated_token_ids=[9707, 0, 2585, 646, 358, 7789, 498, 3351],
            prompt_token_ids=[1, 2, 3],
            steps_completed=8,
            max_new_tokens=kwargs["max_new_tokens"],
            stop_reason="step-limit",
            strategy="fake",
            cache_sequence_lengths={"0": 38},
            blockers=[],
        )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 6 * 1024**3, "free_memory_gb": 6.0, "memory_guard_active": False},
    )

    payload = _run_chat_payload(
        {
            "model_id": "qwen2.5-14b-instruct",
            "prompt": "hello world",
            "mode": "Quality",
            "max_new_tokens": 16,
        }
    )

    assert captured["max_new_tokens"] == 8
    assert payload["max_new_tokens"] == 8
    assert payload["model_guardrails"]["proven_max_new_tokens"] == 8
    assert payload["model_guardrails"]["warnings"] == []


def test_run_chat_payload_experimental_qwen_14b_uses_full_stack(monkeypatch) -> None:
    from pcketlm.app import web

    captured = {}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="longer",
            full_text="longer",
            generated_token_ids=[1] * kwargs["max_new_tokens"],
            prompt_token_ids=[1, 2, 3],
            steps_completed=kwargs["max_new_tokens"],
            max_new_tokens=kwargs["max_new_tokens"],
            stop_reason="step-limit",
            strategy="fake",
            cache_sequence_lengths={"0": 42},
            blockers=[],
        )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(web.main, "load_layer_bridge_config", lambda model_id: SimpleNamespace(num_hidden_layers=48, blockers=[]))
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 6 * 1024**3, "free_memory_gb": 6.0, "memory_guard_active": False},
    )

    payload = _run_chat_payload(
        {
            "model_id": "qwen2.5-14b-instruct",
            "prompt": "hello world",
            "mode": "Quality",
            "max_new_tokens": 12,
            "allow_experimental_direct_tokens": True,
        }
    )

    assert captured["max_new_tokens"] == 12
    assert captured["layer_count"] == 48
    assert payload["model_guardrails"]["warnings"]


def test_run_chat_payload_agent_mode_caps_direct_runtime_to_short_work(monkeypatch) -> None:
    from pcketlm.app import web

    captured = {}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="OK",
            generated_token_ids=[9707, 0],
            prompt_token_ids=[1, 2, 3],
            steps_completed=2,
            max_new_tokens=kwargs["max_new_tokens"],
            stop_reason="step-limit",
            strategy="fake",
            cache_sequence_lengths={"0": 32},
            blockers=[],
        )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(
        web.main,
        "_load_saved_runtime_settings",
        lambda: {"tensor_cache_preset": "standard", "agent_warm_runner": "off"},
    )
    monkeypatch.setattr(web.main, "_apply_runtime_settings", lambda settings: None)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 6 * 1024**3, "free_memory_gb": 6.0, "memory_guard_active": False},
    )

    payload = _run_chat_payload(
        {
            "model_id": "qwen2.5-14b-instruct",
            "prompt": "next action",
            "mode": "Agent",
            "max_new_tokens": 8,
        }
    )

    assert captured["max_new_tokens"] == 2
    assert payload["max_new_tokens"] == 2


def test_run_chat_payload_allows_explicit_experimental_qwen_32b_length(monkeypatch) -> None:
    from pcketlm.app import web

    captured = {}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="longer",
            full_text="longer",
            generated_token_ids=[1] * kwargs["max_new_tokens"],
            prompt_token_ids=[1, 2, 3],
            steps_completed=kwargs["max_new_tokens"],
            max_new_tokens=kwargs["max_new_tokens"],
            stop_reason="step-limit",
            strategy="fake",
            cache_sequence_lengths={"0": 40},
            blockers=[],
        )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 6 * 1024**3, "free_memory_gb": 6.0, "memory_guard_active": False},
    )

    payload = _run_chat_payload(
        {
            "model_id": "qwen2.5-32b-instruct",
            "prompt": "hello world",
            "mode": "Quality",
            "max_new_tokens": 12,
            "allow_experimental_32b_tokens": True,
        }
    )

    assert captured["max_new_tokens"] == 12
    assert payload["max_new_tokens"] == 12
    assert "proven to 8 new tokens" in payload["model_guardrails"]["warnings"][0]


def test_run_chat_payload_reuses_session_prefix_when_followup_matches(monkeypatch) -> None:
    from pcketlm.app import web

    calls = []
    reusable_state = SimpleNamespace(ready=True)

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="OK",
            generated_token_ids=[1],
            prompt_token_ids=[1, 2, 3],
            steps_completed=1,
            max_new_tokens=1,
            stop_reason="step-limit",
            strategy="fake",
            cache_sequence_lengths={},
            blockers=[],
            prefix_reuse={"used": bool(kwargs.get("initial_decode_state")), "summary": "test"},
            reusable_token_ids=[1, 2, 3, len(calls)],
            final_decode_state=reusable_state,
        )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 8 * 1024**3, "free_memory_gb": 8.0, "memory_guard_active": False},
    )

    request = {
        "model_id": "qwen-test",
        "prompt": "reply OK",
        "mode": "Quick",
        "session_id": "session-test",
    }

    first = _run_chat_payload(request)
    second = _run_chat_payload({**request, "prompt": "reply OK again"})

    assert first["conversation_state"]["reuse_ready"] is True
    assert first["conversation_state"]["reuse_used"] is False
    assert second["conversation_state"]["reuse_used"] is True
    assert calls[0]["initial_decode_state"] is None
    assert calls[1]["initial_decode_state"] is reusable_state
    assert calls[1]["initial_token_ids"] == [1, 2, 3, 1]


def test_run_chat_payload_reuses_exact_cached_response(monkeypatch) -> None:
    from pcketlm.app import web

    calls = {"count": 0}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="OK",
            generated_token_ids=[1],
            prompt_token_ids=[1, 2],
            steps_completed=1,
            max_new_tokens=2,
            stop_reason="step-limit",
            strategy="fake-runtime",
            cache_sequence_lengths={"0": 2},
            blockers=[],
        )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 8 * 1024**3, "free_memory_gb": 8.0, "memory_guard_active": False},
    )

    request = {
        "model_id": "qwen-test",
        "prompt": "Reply with exactly one word: OK",
        "mode": "Quality",
        "max_new_tokens": 2,
    }
    first = _run_chat_payload(request)
    second = _run_chat_payload(request)

    assert calls["count"] == 1
    assert first["generated_text"] == "OK"
    assert first["response_reuse"]["hit"] is False
    assert second["generated_text"] == "OK"
    assert second["response_reuse"]["hit"] is True
    assert second["elapsed_seconds"] == 0.0
    assert second["strategy"] == "fake-runtime+response-cache"


def test_start_chat_job_completes_immediately_for_cached_response(monkeypatch) -> None:
    from pcketlm.app import web

    calls = {"count": 0}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            full_text="OK",
            generated_token_ids=[1],
            prompt_token_ids=[1],
            steps_completed=1,
            max_new_tokens=2,
            stop_reason="step-limit",
            strategy="fake-runtime",
            cache_sequence_lengths={},
            blockers=[],
        )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 8 * 1024**3, "free_memory_gb": 8.0, "memory_guard_active": False},
    )

    request = {
        "model_id": "qwen-test",
        "prompt": "Reply with exactly one word: OK",
        "mode": "Quality",
        "max_new_tokens": 2,
    }
    _run_chat_payload(request)
    job = _start_chat_job(request)

    assert calls["count"] == 1
    assert job.status == "completed"
    assert job.result is not None
    assert job.result["response_reuse"]["hit"] is True
    assert job.result["job_fast_path"] == "response-cache"


def test_run_chat_payload_applies_saved_profile_defaults(monkeypatch) -> None:
    from pcketlm.app import web

    captured = {}

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            ready=True,
            generated_text="ok",
            full_text="ok",
            generated_token_ids=[1],
            prompt_token_ids=[1],
            steps_completed=1,
            max_new_tokens=3,
            stop_reason="step-limit",
            strategy="fake",
            cache_sequence_lengths={},
            blockers=[],
        )

    profile = SimpleNamespace(
        profile_id="low-memory",
        label="Low Memory",
        runtime_mode="Quality",
        settings={"runtime_mode": "Quality", "default_max_new_tokens": 3},
    )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(web.main, "get_saved_profile", lambda _model_id, _profile_id: profile)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 8 * 1024**3, "free_memory_gb": 8.0, "memory_guard_active": False},
    )

    payload = _run_chat_payload({"model_id": "qwen-test", "prompt": "hello", "profile_id": "low-memory"})

    assert payload["profile_id"] == "low-memory"
    assert payload["profile_label"] == "Low Memory"
    assert payload["runtime_context"]["runtime_mode"] == "Quality"
    assert captured["layer_count"] is None
    assert captured["max_new_tokens"] == 3
    assert captured["stop_strings"] == ["<|im_end|>", "<|im_start|>"]
    assert "Profile: Low Memory" in captured["system_prompt"]


def test_run_chat_payload_short_circuits_model_identity_questions(monkeypatch) -> None:
    from pcketlm.app import web

    def fake_run_prompt_decode_loop(*args, **kwargs):
        raise AssertionError("identity questions should not spend a model run")

    profile = SimpleNamespace(
        profile_id="low-memory",
        label="Low Memory",
        runtime_mode="Quality",
        settings={"runtime_mode": "Quality", "default_max_new_tokens": 3},
    )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(web.main, "get_saved_profile", lambda _model_id, _profile_id: profile)

    payload = _run_chat_payload(
        {"model_id": "qwen2.5-14b-instruct", "prompt": "Which model do you run on?", "profile_id": "low-memory"}
    )

    assert payload["generated_text"] == "Qwen2.5-14B-Instruct (qwen2.5-14b-instruct)"
    assert payload["stop_reason"] == "local-runtime-context"
    assert payload["profile_label"] == "Low Memory"


def test_run_chat_payload_can_use_gguf_backend(monkeypatch) -> None:
    from pcketlm.app import web

    def fake_run_gguf_prompt(model_id: str, prompt: str, **kwargs):
        assert model_id == "qwen-test"
        assert "User: reply OK" in prompt
        assert prompt.endswith("Assistant:")
        assert kwargs["max_tokens"] == 2
        assert kwargs["stop_strings"] == ["<|im_end|>", "<|im_start|>"]
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            backend="llama-cpp-gguf",
            blockers=[],
            timings={"total": 1.25},
        )

    monkeypatch.setattr(web.main, "run_gguf_prompt", fake_run_gguf_prompt)
    monkeypatch.setattr(web.main, "build_gguf_backend_status", lambda model_id: SimpleNamespace(to_dict=lambda: {"ready": True}))
    monkeypatch.setattr(web.main, "_supports_im_chat_tokens", lambda model_id: False)
    monkeypatch.setattr(web.main, "_supports_im_chat_tokens", lambda model_id: False)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {"free_memory_bytes": 8 * 1024**3, "free_memory_gb": 8.0},
    )

    payload = _run_chat_payload({"model_id": "qwen-test", "prompt": "reply OK", "mode": "GGUF", "max_new_tokens": 2})

    assert payload["ready"] is True
    assert payload["generated_text"] == "OK"
    assert payload["strategy"] == "llama-cpp-gguf"
    assert payload["stop_reason"] == "gguf-complete"
    assert payload["gguf_backend"]["ready"] is True


def test_run_chat_payload_gguf_bypasses_direct_memory_guard(monkeypatch) -> None:
    from pcketlm.app import web

    def fake_run_gguf_prompt(model_id: str, prompt: str, **kwargs):
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            backend="llama-cpp-gguf-server",
            blockers=[],
            timings={"total": 0.4},
        )

    monkeypatch.setattr(web.main, "run_gguf_prompt", fake_run_gguf_prompt)
    monkeypatch.setattr(web.main, "build_gguf_backend_status", lambda model_id: SimpleNamespace(to_dict=lambda: {"ready": True}))
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {
            "free_memory_bytes": 1 * 1024**3,
            "free_memory_gb": 1.0,
            "memory_guard_active": True,
        },
    )

    payload = _run_chat_payload({"model_id": "qwen-test", "prompt": "Say OK only.", "mode": "GGUF", "max_new_tokens": 1})

    assert payload["ready"] is True
    assert payload["generated_text"] == "OK"
    assert payload["strategy"] == "llama-cpp-gguf-server"
    assert payload["stop_reason"] == "gguf-complete"


def test_run_chat_payload_gguf_uses_qwen_chat_template(monkeypatch) -> None:
    from pcketlm.app import web

    captured = {}

    def fake_run_gguf_prompt(model_id: str, prompt: str, **kwargs):
        captured["prompt"] = prompt
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            backend="llama-cpp-gguf-server",
            blockers=[],
            timings={"total": 0.4},
        )

    monkeypatch.setattr(web.main, "run_gguf_prompt", fake_run_gguf_prompt)
    monkeypatch.setattr(web.main, "_supports_im_chat_tokens", lambda model_id: True)

    payload = _run_chat_payload({"model_id": "qwen-test", "prompt": "Reply with OK only.", "mode": "GGUF", "max_new_tokens": 1})

    assert payload["ready"] is True
    assert captured["prompt"].startswith("<|im_start|>system\n")
    assert "<|im_start|>user\nReply with OK only.<|im_end|>" in captured["prompt"]
    assert captured["prompt"].endswith("<|im_start|>assistant\n")
    assert payload["preformatted_chat"] is True


def test_run_chat_payload_gguf_allows_longer_replies_than_direct_modes(monkeypatch) -> None:
    from pcketlm.app import web

    captured = {}

    def fake_run_gguf_prompt(model_id: str, prompt: str, **kwargs):
        captured["max_tokens"] = kwargs["max_tokens"]
        return SimpleNamespace(
            ready=True,
            generated_text="1. Load the model.\n2. Run a short check.",
            backend="llama-cpp-gguf-server",
            blockers=[],
            timings={"total": 2.0},
        )

    monkeypatch.setattr(web.main, "run_gguf_prompt", fake_run_gguf_prompt)
    monkeypatch.setattr(web.main, "_supports_im_chat_tokens", lambda model_id: True)

    payload = _run_chat_payload(
        {
            "model_id": "qwen-test",
            "prompt": "List two steps.",
            "mode": "GGUF",
            "max_new_tokens": 48,
        }
    )

    assert payload["ready"] is True
    assert captured["max_tokens"] == 48
    assert payload["max_new_tokens"] == 48


def test_run_chat_payload_blocks_before_generation_when_free_ram_is_too_low(monkeypatch) -> None:
    from pcketlm.app import web

    def fake_run_prompt_decode_loop(*args, **kwargs):
        raise AssertionError("memory guard should block before model generation")

    profile = SimpleNamespace(
        profile_id="low-memory",
        label="Low Memory",
        runtime_mode="Quality",
        settings={"runtime_mode": "Quality", "default_max_new_tokens": 3},
    )

    monkeypatch.setattr(web.main, "run_prompt_decode_loop", fake_run_prompt_decode_loop)
    monkeypatch.setattr(web.main, "get_saved_profile", lambda _model_id, _profile_id: profile)
    monkeypatch.setattr(
        web.main,
        "tensor_residency_policy_snapshot",
        lambda: {
            "free_memory_bytes": 1 * 1024**3,
            "free_memory_gb": 1.0,
            "memory_guard_active": True,
        },
    )

    payload = _run_chat_payload(
        {"model_id": "qwen2.5-14b-instruct", "prompt": "If 2 plus 3 equals 5, answer YES only.", "profile_id": "low-memory"}
    )

    assert payload["ready"] is False
    assert payload["stop_reason"] == "memory-guard"
    assert payload["profile_label"] == "Low Memory"


def test_status_payload_includes_engine_decision(monkeypatch) -> None:
    from pcketlm.app import web

    monkeypatch.setattr(web.main, "list_status_screen_options", lambda: [])
    monkeypatch.setattr(
        web.main,
        "build_status_screen_model",
        lambda model_id, model_dir: SimpleNamespace(
            model_id=model_id,
            model_label="Qwen",
            family_label="Qwen",
            runtime_status="Ready",
            runtime_summary="Ready",
            streaming_status="Advancing",
            model_dir=model_dir,
            blockers=[],
            warnings=[],
        ),
    )
    monkeypatch.setattr(web.main, "_latest_json", lambda path: None)
    monkeypatch.setattr(web.main, "latest_optimized_artifact_manifest", lambda model_id: None)
    monkeypatch.setattr(web.main, "list_saved_profiles", lambda model_id: [])
    monkeypatch.setattr(web.main, "build_profile_compare_summary", lambda model_id, latest_benchmark: {})
    monkeypatch.setattr(web.main, "build_measured_benchmark_history", lambda model_id: {})
    monkeypatch.setattr(
        web.main,
        "select_runtime_engine",
        lambda model_id: SimpleNamespace(to_dict=lambda: {"selected_engine": "direct-cpu"}),
    )
    monkeypatch.setattr(
        web.main,
        "build_runtime_backend_report",
        lambda model_id: SimpleNamespace(to_dict=lambda: {"recommended_backend_id": "llama-cpp-gguf"}),
    )
    monkeypatch.setattr(
        web.main,
        "build_gguf_backend_status",
        lambda model_id: SimpleNamespace(
            to_dict=lambda: {
                "ready": True,
                "model_files": [
                    {
                        "name": "queen-q4.gguf",
                        "path": "C:/models/queen-q4.gguf",
                        "size_gb": 8.37,
                        "state": "ready",
                    }
                ],
                "load_estimate": {
                    "state": "ready",
                    "model_file": "queen-q4.gguf",
                    "expected_ram_mb": 9000,
                    "estimated_cold_load_seconds": 52.0,
                },
            }
        ),
    )
    monkeypatch.setattr(
        web.main,
        "warm_runner_status",
        lambda model_id: {"model_id": model_id, "state": "ready", "request_count": 2},
    )

    payload = web.main._status_payload()

    assert payload["engine_decision"]["selected_engine"] == "direct-cpu"
    assert payload["backend_report"]["recommended_backend_id"] == "llama-cpp-gguf"
    assert payload["backend_comparison"]["recommended_backend_id"] == "llama-cpp-gguf"
    assert payload["gguf_backend"]["load_estimate"]["expected_ram_mb"] == 9000
    assert payload["gguf_backend"]["model_files"][0]["name"] == "queen-q4.gguf"
    assert payload["warm_runner"]["state"] == "ready"
    assert payload["warm_runner"]["request_count"] == 2


def test_status_payload_includes_live_download_meter(monkeypatch, tmp_path) -> None:
    from pcketlm.app import web

    download_root = tmp_path / "downloads"
    download_root.mkdir()
    (download_root / "qwen2.5-32b-instruct.json").write_text(
        """
        {
          "model_id": "qwen2.5-32b-instruct",
          "repo_id": "Qwen/Qwen2.5-32B-Instruct",
          "status": "downloading",
          "bytes_on_disk_gb": 1.25,
          "expected_bytes_gb": 61.04,
          "progress_pct": 2.05,
          "expected_file_count": 24,
          "present_expected_file_count": 3
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(web.main, "state_root", lambda: tmp_path)

    payload = web.main._download_status_payload()

    assert payload["active_count"] == 1
    assert payload["records"][0]["model_id"] == "qwen2.5-32b-instruct"
    assert payload["records"][0]["progress_pct"] == 2.05
    assert payload["active"][0]["status"] == "downloading"


def test_update_runtime_settings_selects_boosted_cache(monkeypatch, tmp_path) -> None:
    from pcketlm.app import web

    calls = {"resident_cleared": 0, "response_cleared": 0, "prefix_cleared": 0}
    monkeypatch.setattr(web.main, "state_root", lambda: tmp_path)
    monkeypatch.setattr(
        web.main,
        "clear_tensor_residency_cache",
        lambda: calls.__setitem__("resident_cleared", calls["resident_cleared"] + 1),
    )
    monkeypatch.setattr(
        web.main,
        "_clear_session_prefix_cache",
        lambda: calls.__setitem__("prefix_cleared", calls["prefix_cleared"] + 1),
    )
    monkeypatch.setattr(
        web.main,
        "_clear_chat_response_cache",
        lambda: calls.__setitem__("response_cleared", calls["response_cleared"] + 1),
    )
    monkeypatch.setattr(web.main, "tensor_residency_policy_snapshot", lambda: {"tensor_cache_preset": "boosted"})
    monkeypatch.setattr(web.main, "tensor_residency_stats", lambda: SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(web.main, "tensor_load_stats_snapshot", lambda: SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setenv("PCKETLM_TENSOR_CACHE_PRESET", "standard")

    payload = _update_runtime_settings({"tensor_cache_preset": "boosted", "agent_warm_runner": "safe"})

    assert calls == {"resident_cleared": 1, "response_cleared": 1, "prefix_cleared": 1}
    assert payload["runtime_settings"]["saved_runtime_settings"]["tensor_cache_preset"] == "boosted"
    assert payload["runtime_settings"]["saved_runtime_settings"]["agent_warm_runner"] == "safe"
    assert payload["runtime_settings"]["agent_warm_runner"] == "safe"
    assert payload["runtime_settings"]["tensor_residency_policy"]["tensor_cache_preset"] == "boosted"
    assert (tmp_path / "runtime-settings.json").exists()


def test_warm_runner_control_start_and_stop(monkeypatch) -> None:
    from pcketlm.app import web

    calls = {"prefix_cleared": 0, "response_cleared": 0}
    monkeypatch.setattr(web.main, "start_warm_runner", lambda model_id, mode: {"model_id": model_id, "state": "ready"})
    monkeypatch.setattr(web.main, "stop_warm_runner", lambda model_id: {"model_id": model_id, "state": "stopped"})
    monkeypatch.setattr(web.main, "_runtime_settings_payload", lambda: {"agent_warm_runner": "safe"})
    monkeypatch.setattr(
        web.main,
        "_clear_session_prefix_cache",
        lambda: calls.__setitem__("prefix_cleared", calls["prefix_cleared"] + 1),
    )
    monkeypatch.setattr(
        web.main,
        "_clear_chat_response_cache",
        lambda: calls.__setitem__("response_cleared", calls["response_cleared"] + 1),
    )

    started = _warm_runner_control_payload({"model_id": "qwen-test", "action": "start"})
    stopped = _warm_runner_control_payload({"model_id": "qwen-test", "action": "stop"})

    assert started["warm_runner"]["state"] == "ready"
    assert stopped["warm_runner"]["state"] == "stopped"
    assert calls == {"prefix_cleared": 1, "response_cleared": 1}


def test_agent_mode_can_use_opt_in_warm_runner(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(web_main, "_local_runtime_context_answer", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_main, "_memory_guard_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        web_main,
        "_load_saved_runtime_settings",
        lambda: {"tensor_cache_preset": "standard", "agent_warm_runner": "safe"},
    )
    monkeypatch.setattr(web_main, "_apply_runtime_settings", lambda settings: None)
    monkeypatch.setattr(web_main, "_runtime_settings_payload", lambda: {"agent_warm_runner": "safe"})
    monkeypatch.setattr(web_main, "_speed_status_payload", lambda model_id: {"status": "Stable"})
    monkeypatch.setattr(web_main, "_runtime_context_payload", lambda model_id, mode, profile: {"mode": mode})
    monkeypatch.setattr(web_main, "_direct_model_guardrails", lambda model_id, max_new_tokens: {"blockers": []})
    monkeypatch.setattr(web_main, "_supports_im_chat_tokens", lambda model_id: True)

    def fake_warm_runner(model_id, prompt, *, mode, session_id, max_new_tokens, apply_chat_format):
        captured.update(
            {
                "model_id": model_id,
                "prompt": prompt,
                "mode": mode,
                "session_id": session_id,
                "max_new_tokens": max_new_tokens,
                "apply_chat_format": apply_chat_format,
            }
        )
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            generated_token_ids=[111, 222],
            steps_completed=2,
            max_new_tokens=max_new_tokens,
            blockers=[],
            performance_summary={"bottleneck": "tensor loading"},
            prefix_reuse={"enabled": True, "used": True, "reason": "warm-runner-state"},
            runner_status={"state": "ready", "reusable_token_count": 6},
        )

    monkeypatch.setattr(web_main, "run_warm_agent_prompt", fake_warm_runner)

    payload = _run_chat_payload(
        {
            "model_id": "qwen2.5-14b-instruct",
            "mode": "Agent",
            "prompt": "reply ok only",
            "max_new_tokens": 16,
            "session_id": "warm-agent-test",
            "messages": [{"role": "user", "text": "hello"}],
        }
    )

    assert captured["model_id"] == "qwen2.5-14b-instruct"
    assert captured["mode"] == "Agent"
    assert captured["session_id"] == "warm-agent-test"
    assert captured["max_new_tokens"] == 2
    assert captured["apply_chat_format"] is False
    assert "reply ok only" in captured["prompt"]
    assert payload["strategy"] == "warm-agent-runner"
    assert payload["generated_text"] == "OK"
    assert payload["warm_runner"]["state"] == "ready"
    assert payload["prefix_reuse"]["used"] is True


def test_agent_warm_runner_formats_first_qwen_turn(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(web_main, "_local_runtime_context_answer", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_main, "_memory_guard_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        web_main,
        "_load_saved_runtime_settings",
        lambda: {"tensor_cache_preset": "standard", "agent_warm_runner": "safe"},
    )
    monkeypatch.setattr(web_main, "_apply_runtime_settings", lambda settings: None)
    monkeypatch.setattr(web_main, "_runtime_settings_payload", lambda: {"agent_warm_runner": "safe"})
    monkeypatch.setattr(web_main, "_speed_status_payload", lambda model_id: {"status": "Stable"})
    monkeypatch.setattr(web_main, "_runtime_context_payload", lambda model_id, mode, profile: {"mode": mode})
    monkeypatch.setattr(web_main, "_direct_model_guardrails", lambda model_id, max_new_tokens: {"blockers": []})
    monkeypatch.setattr(web_main, "_supports_im_chat_tokens", lambda model_id: True)

    def fake_warm_runner(model_id, prompt, *, mode, session_id, max_new_tokens, apply_chat_format):
        captured["prompt"] = prompt
        captured["apply_chat_format"] = apply_chat_format
        return SimpleNamespace(
            ready=True,
            generated_text="OK",
            generated_token_ids=[111],
            steps_completed=1,
            max_new_tokens=max_new_tokens,
            blockers=[],
            performance_summary={},
            prefix_reuse={"enabled": False, "used": False},
            runner_status={"state": "ready", "reusable_token_count": 1},
        )

    monkeypatch.setattr(web_main, "run_warm_agent_prompt", fake_warm_runner)

    payload = _run_chat_payload(
        {
            "model_id": "qwen2.5-14b-instruct",
            "mode": "Agent",
            "prompt": "hello",
            "max_new_tokens": 2,
            "session_id": "warm-agent-format-test",
            "messages": [],
        }
    )

    assert payload["strategy"] == "warm-agent-runner"
    assert "<|im_start|>system" in captured["prompt"]
    assert "<|im_start|>user\nhello<|im_end|>" in captured["prompt"]
    assert captured["prompt"].endswith("<|im_start|>assistant\n")
    assert captured["apply_chat_format"] is False


def test_chat_job_cancel_marks_queued_job_canceled() -> None:
    job = ChatJob(job_id="cancel-test", payload={"prompt": "hello"})
    with _CHAT_JOBS_LOCK:
        _CHAT_JOBS[job.job_id] = job
    try:
        canceled = _cancel_chat_job("cancel-test")
        assert canceled is not None
        payload = _chat_job_payload(canceled)
        assert payload["status"] == "canceled"
        assert payload["cancel_requested"] is True
    finally:
        with _CHAT_JOBS_LOCK:
            _CHAT_JOBS.pop(job.job_id, None)
