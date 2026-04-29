import json
from types import SimpleNamespace

from pcketlm.app.chat_shell import runtime_layer_bridge_cli


class _FakePromptResult:
    def to_dict(self) -> dict:
        return {
            "ready": True,
            "generated_text": "OK",
            "generated_token_ids": [123],
            "timings": {"total": 0.01},
        }


def test_runtime_layer_bridge_cli_measure_memory(monkeypatch, capsys) -> None:
    gb = 1024**3
    calls = {"count": 0}

    def fake_memory_snapshot():
        calls["count"] += 1
        free_bytes = 8 * gb if calls["count"] == 1 else 6 * gb
        return SimpleNamespace(total_bytes=16 * gb, free_bytes=free_bytes)

    def fake_prompt_decode_loop(model_id: str, **kwargs):
        assert model_id == "qwen-test"
        assert kwargs["prompt"] == "hello"
        assert kwargs["selection_policy"] == "greedy"
        assert kwargs["max_new_tokens"] == 1
        return _FakePromptResult()

    monkeypatch.setattr(runtime_layer_bridge_cli, "_memory_snapshot", fake_memory_snapshot)
    monkeypatch.setattr(runtime_layer_bridge_cli, "run_prompt_decode_loop", fake_prompt_decode_loop)

    exit_code = runtime_layer_bridge_cli.main(
        ["qwen-test", "--prompt", "hello", "--policy", "greedy", "--max-new-tokens", "1", "--measure-memory"]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["generated_text"] == "OK"
    assert payload["measurement"]["free_ram_start_mb"] == 8192
    assert payload["measurement"]["free_ram_end_mb"] == 6144
    assert payload["measurement"]["peak_ram_delta_mb"] == 2048
