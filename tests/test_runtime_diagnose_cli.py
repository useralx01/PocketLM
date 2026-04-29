import json
from types import SimpleNamespace

import torch

from pcketlm.app.chat_shell import runtime_diagnose_cli


def test_runtime_diagnose_cli_load_config_outputs_checkpoints(monkeypatch, capsys, tmp_path) -> None:
    gb = 1024**3
    monkeypatch.setattr(
        runtime_diagnose_cli,
        "_memory_snapshot",
        lambda: SimpleNamespace(total_bytes=16 * gb, free_bytes=8 * gb),
    )
    monkeypatch.setattr(runtime_diagnose_cli, "_working_set_mb", lambda: 123)
    monkeypatch.setattr(runtime_diagnose_cli, "original_model_root", lambda model_id: tmp_path)
    monkeypatch.setattr(
        runtime_diagnose_cli,
        "load_layer_bridge_config",
        lambda model_id: SimpleNamespace(
            ready=True,
            blockers=[],
            hidden_size=5120,
            num_hidden_layers=64,
            num_attention_heads=40,
            num_key_value_heads=8,
            intermediate_size=27648,
            vocab_size=152064,
            source_dtype="bfloat16",
        ),
    )

    exit_code = runtime_diagnose_cli.main(["--model", "qwen-test", "--slice", "load-config"])

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["event"] for line in lines] == ["start", "before", "after", "complete"]
    assert lines[0]["free_ram_mb"] == 8192
    assert lines[0]["process_working_set_mb"] == 123
    assert lines[2]["result"]["num_hidden_layers"] == 64
    assert lines[-1]["ready"] is True


def test_runtime_diagnose_cli_full_honors_max_new_tokens(monkeypatch, capsys, tmp_path) -> None:
    gb = 1024**3
    captured = {}

    class _FakeResult:
        def to_dict(self) -> dict:
            return {
                "ready": True,
                "generated_text": "Hello! How can",
                "blockers": [],
            }

    monkeypatch.setattr(
        runtime_diagnose_cli,
        "_memory_snapshot",
        lambda: SimpleNamespace(total_bytes=16 * gb, free_bytes=8 * gb),
    )
    monkeypatch.setattr(runtime_diagnose_cli, "_working_set_mb", lambda: 123)
    monkeypatch.setattr(runtime_diagnose_cli, "original_model_root", lambda model_id: tmp_path)

    def fake_run_prompt_decode_loop(model_id: str, **kwargs):
        captured.update(kwargs)
        return _FakeResult()

    monkeypatch.setattr(runtime_diagnose_cli, "run_prompt_decode_loop", fake_run_prompt_decode_loop)

    exit_code = runtime_diagnose_cli.main(
        ["--model", "qwen-test", "--slice", "full", "--max-new-tokens", "4"]
    )

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert captured["max_new_tokens"] == 4
    assert lines[2]["result"]["generated_text"] == "Hello! How can"


def test_runtime_diagnose_cli_summarizes_kv_cache_bytes() -> None:
    key = torch.zeros((1, 8, 3, 128), dtype=torch.float16)
    value = torch.zeros((1, 8, 3, 128), dtype=torch.float16)

    summary = runtime_diagnose_cli._kv_cache_summary({0: (key, value), 1: (key, value)})

    assert summary["kv_cache_layers"] == 2
    assert summary["kv_cache_total_bytes"] == key.nelement() * key.element_size() * 4
    assert summary["kv_cache_by_layer"]["0"]["key_shape"] == [1, 8, 3, 128]
