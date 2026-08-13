from types import SimpleNamespace

from pcketlm.app.desktop.main import (
    _chat_layer_budget,
    _chat_mode_hint,
    _chat_runtime_hint,
    _parse_stop_strings,
    _parse_stop_token_ids,
    _planned_action_message,
    _run_desktop_chat_generation,
    _summarize_prompt_result,
)


def test_parse_stop_token_ids_accepts_comma_separated_integers() -> None:
    values, blockers = _parse_stop_token_ids("1, 2,3")

    assert values == [1, 2, 3]
    assert blockers == []


def test_parse_stop_token_ids_reports_invalid_values() -> None:
    values, blockers = _parse_stop_token_ids("7, nope, 9")

    assert values == [7, 9]
    assert blockers == ["Stop token id 'nope' is not a valid integer."]


def test_parse_stop_strings_accepts_pipe_separated_values() -> None:
    values = _parse_stop_strings("hello|world| assistant ")

    assert values == ["hello", "world", "assistant"]


def test_summarize_prompt_result_includes_key_session_fields() -> None:
    result = SimpleNamespace(
        strategy="greedy-prompt-kv-cache-rope",
        ready=True,
        min_new_tokens=2,
        steps_completed=3,
        max_new_tokens=8,
        stop_reason="step-limit",
        stop_token_ids=[151645],
        stop_strings=["assistant"],
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[4, 5, 6],
        cache_sequence_lengths={"0": 3, "1": 3},
        blockers=["Still rough output quality."],
    )

    summary = _summarize_prompt_result(result)

    assert "Strategy: greedy-prompt-kv-cache-rope" in summary
    assert "Min new tokens: 2" in summary
    assert "Steps: 3 / 8" in summary
    assert "Stop reason: step-limit" in summary
    assert "Stop tokens: 151645" in summary
    assert "Stop strings: assistant" in summary
    assert "Prompt token count: 3" in summary
    assert "Generated token count: 3" in summary
    assert "Still rough output quality." in summary


def test_planned_action_message_explains_unfinished_home_actions() -> None:
    assert "original model untouched" in _planned_action_message("Personalize")
    assert "proven instead of guessed" in _planned_action_message("Compare")
    assert "repeatable checks" in _planned_action_message("Benchmark")
    assert _planned_action_message("Unknown") == "Unknown is planned but not wired yet."


def test_chat_runtime_hint_sets_alpha_speed_expectations() -> None:
    assert "about 1 minute" in _chat_runtime_hint(2)
    assert "about 2 minutes" in _chat_runtime_hint(4)
    assert "several minutes" in _chat_runtime_hint(8)
    assert "about 40 seconds" in _chat_runtime_hint(2, "Balanced (32 layers)")
    assert "under 30 seconds" in _chat_runtime_hint(2, "Fast (8 layers)")


def test_chat_layer_budget_maps_desktop_modes() -> None:
    assert _chat_layer_budget("Fast (8 layers)") == 8
    assert _chat_layer_budget("Balanced (32 layers)") == 32
    assert _chat_layer_budget("Quality (full stack)") is None
    assert _chat_layer_budget("GPU (CUDA paged)") is None
    assert "quick smoke tests" in _chat_mode_hint("Fast (8 layers)")
    assert "full current stack" in _chat_mode_hint("Quality (full stack)")
    assert "CUDA resident pager" in _chat_mode_hint("GPU (CUDA paged)")
    assert "CUDA GPU runtime" in _chat_runtime_hint(1, "GPU (CUDA paged)")


def test_desktop_chat_generation_routes_deepseek_to_fp8(monkeypatch) -> None:
    from pcketlm.app import desktop

    captured = {}

    def fake_run_fp8_decode_loop(model_id, token_ids, *, layer_count, max_new_tokens):
        captured.update(
            {
                "model_id": model_id,
                "token_ids": list(token_ids),
                "layer_count": layer_count,
                "max_new_tokens": max_new_tokens,
            }
        )
        return SimpleNamespace(
            ready=True,
            blockers=[],
            generated_token_ids=[9],
            positions_completed=2,
            next_kv_caches={},
        )

    monkeypatch.setattr(desktop.main, "_encode_with_catalog_tokenizer", lambda model_id, prompt: ([1, 2, 3], []))
    monkeypatch.setattr(desktop.main, "_decode_with_catalog_tokenizer", lambda model_id, ids: ("ok", []))
    monkeypatch.setattr(desktop.main, "run_fp8_decode_loop", fake_run_fp8_decode_loop)

    result = _run_desktop_chat_generation(
        "deepseek-v3",
        prompt="hello",
        max_new_tokens=1,
        min_new_tokens=1,
        top_p=0.9,
        layer_count=8,
        repetition_penalty=1.1,
        system_prompt=None,
        apply_chat_format=False,
        stop_token_ids=[],
        stop_strings=[],
    )

    assert result.ready is True
    assert result.strategy == "deepseek-fp8-pack"
    assert result.generated_text == "ok"
    assert captured == {"model_id": "deepseek-v3", "token_ids": [1, 2, 3], "layer_count": 8, "max_new_tokens": 1}


def test_desktop_chat_generation_uses_exact_mtp_for_full_deepseek(monkeypatch) -> None:
    from pcketlm.app import desktop

    captured = {}

    def fake_run_fp8_mtp_batched_generate_from_tokens(
        model_id,
        token_ids,
        *,
        prompt_text,
        max_visible_tokens,
        k,
        layer_count,
        max_passes,
        prefer_eos_after_punctuation,
    ):
        captured.update(
            {
                "model_id": model_id,
                "token_ids": list(token_ids),
                "prompt_text": prompt_text,
                "max_visible_tokens": max_visible_tokens,
                "k": k,
                "layer_count": layer_count,
                "max_passes": max_passes,
                "prefer_eos_after_punctuation": prefer_eos_after_punctuation,
            }
        )
        return {
            "ready": True,
            "blockers": [],
            "generated_token_ids": [9],
            "visible_token_ids": [9],
            "generated_text": "ok",
            "accepted_token_count": 1,
            "seconds_per_visible_token": 12.345,
        }

    def fake_warm_fp8_mtp_prompt_prefill_cache_from_tokens(model_id, token_ids, *, prompt_text, layer_count):
        captured["warmup"] = {
            "model_id": model_id,
            "token_ids": list(token_ids),
            "prompt_text": prompt_text,
            "layer_count": layer_count,
        }
        return {
            "ready": True,
            "blockers": [],
            "layers_executed": 61,
            "expected_layers_executed": 61,
            "anti_cheat_passed": True,
        }

    monkeypatch.setattr(desktop.main, "_encode_with_catalog_tokenizer", lambda model_id, prompt: ([1, 2, 3], []))
    monkeypatch.setattr(
        desktop.main,
        "warm_fp8_mtp_prompt_prefill_cache_from_tokens",
        fake_warm_fp8_mtp_prompt_prefill_cache_from_tokens,
    )
    monkeypatch.setattr(desktop.main, "run_fp8_mtp_batched_generate_from_tokens", fake_run_fp8_mtp_batched_generate_from_tokens)

    result = _run_desktop_chat_generation(
        "deepseek-v3",
        prompt="hello",
        max_new_tokens=1,
        min_new_tokens=1,
        top_p=0.9,
        layer_count=None,
        repetition_penalty=1.1,
        system_prompt=None,
        apply_chat_format=False,
        stop_token_ids=[],
        stop_strings=[],
    )

    assert result.ready is True
    assert result.strategy == "deepseek-fp8-mtp-batched-exact"
    assert result.generated_text == "ok"
    assert result.generation_seconds_per_token == 12.345
    assert result.mtp_batched_exact["warmup_proof"]["anti_cheat_passed"] is True
    assert "Token speed: 12.345s/token" in _summarize_prompt_result(result)
    assert captured["warmup"] == {
        "model_id": "deepseek-v3",
        "token_ids": [1, 2, 3],
        "prompt_text": "hello",
        "layer_count": None,
    }
    captured.pop("warmup")
    assert captured == {
        "model_id": "deepseek-v3",
        "token_ids": [1, 2, 3],
        "prompt_text": "hello",
        "max_visible_tokens": 1,
        "k": 8,
        "layer_count": None,
        "max_passes": 20,
        "prefer_eos_after_punctuation": True,
    }


def test_desktop_chat_generation_can_route_deepseek_to_gpu_paged(monkeypatch) -> None:
    from pcketlm.app import desktop

    captured = {}

    def fake_run_local_deepseek_paged_decode_loop(model_id, token_ids, **kwargs):
        captured.update({"model_id": model_id, "token_ids": list(token_ids), **kwargs})
        return SimpleNamespace(
            passed=True,
            cuda_available=True,
            device="cuda",
            device_name="Fake RTX",
            torch_version="test",
            prompt_token_ids=list(token_ids),
            generated_token_ids=[9],
            start_layer=0,
            layer_count=61,
            config_hidden_layers=61,
            positions_completed=4,
            final_top_token_ids=[9, 8],
            final_top_logits=[2.0, 1.0],
            cache_sequence_lengths={0: 4},
            step_summaries=[],
            resident_budget_bytes=12 * 1024**3,
            peak_resident_bytes=2 * 1024**3,
            final_resident_bytes=1 * 1024**3,
            pager_loads=61,
            pager_evictions=1,
            pager_cache_hits=2,
            pager_cache_misses=61,
            pager_prefetch_submitted=0,
            pager_prefetch_completed=0,
            tail_elapsed_seconds=0.25,
            elapsed_seconds=2.0,
            blockers=[],
            note="fake gpu pass",
        )

    def fail_mtp(*args, **kwargs):
        raise AssertionError("GPU mode should not call the CPU MTP route")

    monkeypatch.setattr(desktop.main, "_encode_with_catalog_tokenizer", lambda model_id, prompt: ([1, 2, 3], []))
    monkeypatch.setattr(desktop.main, "_decode_with_catalog_tokenizer", lambda model_id, ids: ("gpu ok", []))
    monkeypatch.setattr(desktop.main, "run_local_deepseek_paged_decode_loop", fake_run_local_deepseek_paged_decode_loop)
    monkeypatch.setattr(desktop.main, "run_fp8_mtp_batched_generate_from_tokens", fail_mtp)

    result = _run_desktop_chat_generation(
        "deepseek-v3",
        prompt="hello",
        max_new_tokens=1,
        min_new_tokens=1,
        top_p=0.9,
        layer_count=None,
        repetition_penalty=1.1,
        system_prompt=None,
        apply_chat_format=False,
        stop_token_ids=[],
        stop_strings=[],
        runtime_mode="GPU (CUDA paged)",
    )

    assert result.ready is True
    assert result.strategy == "deepseek-fp8-gpu-paged"
    assert result.generated_text == "gpu ok"
    assert result.generation_seconds_per_token == 2.0
    assert result.gpu_paged["device_name"] == "Fake RTX"
    assert "Token speed: 2.0s/token" in _summarize_prompt_result(result)
    assert captured["model_id"] == "deepseek-v3"
    assert captured["token_ids"] == [1, 2, 3]
    assert captured["device"] == "cuda"
    assert captured["require_cuda"] is True
    assert captured["layer_count"] == 61
