from types import SimpleNamespace

from pcketlm.app.desktop.main import (
    _chat_layer_budget,
    _chat_mode_hint,
    _chat_runtime_hint,
    _parse_stop_strings,
    _parse_stop_token_ids,
    _planned_action_message,
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
    assert "quick smoke tests" in _chat_mode_hint("Fast (8 layers)")
    assert "full current stack" in _chat_mode_hint("Quality (full stack)")
