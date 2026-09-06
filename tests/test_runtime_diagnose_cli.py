import json
from types import SimpleNamespace

import torch
import pytest

from pcketlm.app.chat_shell import runtime_diagnose_cli


def test_runtime_diagnose_cli_model_path_override_is_restored(monkeypatch, capsys, tmp_path) -> None:
    gb = 1024**3
    original_cli_root = lambda _model_id: tmp_path / "cli-root"
    original_bridge_root = lambda _model_id: tmp_path / "bridge-root"
    original_tokenizer_root = lambda _model_id: tmp_path / "tokenizer-root"
    monkeypatch.setattr(runtime_diagnose_cli, "original_model_root", original_cli_root)
    monkeypatch.setattr(runtime_diagnose_cli.layer_bridge_module, "original_model_root", original_bridge_root)
    monkeypatch.setattr(runtime_diagnose_cli.tokenizer_runtime_module, "original_model_root", original_tokenizer_root)
    monkeypatch.setattr(
        runtime_diagnose_cli,
        "_memory_snapshot",
        lambda: SimpleNamespace(total_bytes=16 * gb, free_bytes=8 * gb),
    )
    monkeypatch.setattr(runtime_diagnose_cli, "_working_set_mb", lambda: 123)
    monkeypatch.setattr(runtime_diagnose_cli, "build_tensor_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime_diagnose_cli,
        "load_layer_bridge_config",
        lambda _model_id: SimpleNamespace(
            ready=True,
            blockers=[],
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=16,
            vocab_size=32,
            source_dtype="bfloat16",
        ),
    )

    exit_code = runtime_diagnose_cli.main(
        ["--model", "override-test", "--model-path", str(tmp_path), "--slice", "load-config"]
    )

    assert exit_code == 0
    assert runtime_diagnose_cli.original_model_root is original_cli_root
    assert runtime_diagnose_cli.layer_bridge_module.original_model_root is original_bridge_root
    assert runtime_diagnose_cli.tokenizer_runtime_module.original_model_root is original_tokenizer_root
    assert json.loads(capsys.readouterr().out.splitlines()[0])["model_dir"] == str(tmp_path)


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
        [
            "--model",
            "qwen-test",
            "--slice",
            "full",
            "--max-new-tokens",
            "4",
            "--prompt",
            "Write a short paragraph about local AI.",
        ]
    )

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert captured["max_new_tokens"] == 4
    assert captured["prompt"] == "Write a short paragraph about local AI."
    assert lines[2]["result"]["generated_text"] == "Hello! How can"
    assert "tensor_load_stats" in lines[2]["result"]


def test_runtime_diagnose_cli_sets_tensor_source(monkeypatch, capsys, tmp_path) -> None:
    gb = 1024**3
    captured = {}

    class _FakeResult:
        def to_dict(self) -> dict:
            captured["source_seen"] = runtime_diagnose_cli.os.environ.get("PCKETLM_TENSOR_SOURCE")
            return {
                "ready": True,
                "generated_text": "The capital",
                "blockers": [],
            }

    monkeypatch.setattr(
        runtime_diagnose_cli,
        "_memory_snapshot",
        lambda: SimpleNamespace(total_bytes=16 * gb, free_bytes=8 * gb),
    )
    monkeypatch.setattr(runtime_diagnose_cli, "_working_set_mb", lambda: 123)
    monkeypatch.setattr(runtime_diagnose_cli, "original_model_root", lambda model_id: tmp_path)
    monkeypatch.setattr(runtime_diagnose_cli, "run_prompt_decode_loop", lambda *_args, **_kwargs: _FakeResult())

    exit_code = runtime_diagnose_cli.main(["--model", "qwen-test", "--slice", "full", "--source", "q4"])

    assert exit_code == 0
    assert captured["source_seen"] == "q4"
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["tensor_source"] == "q4"


def test_runtime_diagnose_cli_speculative_outputs_metrics(monkeypatch, capsys, tmp_path) -> None:
    gb = 1024**3
    captured = {}

    class _FakeSpeculativeResult:
        def to_dict(self) -> dict:
            return {
                "ready": True,
                "generated_text": "Paris is the capital.",
                "elapsed_seconds": 4.0,
                "effective_tokens_per_second": 2.5,
                "effective_seconds_per_token": 0.4,
                "verifier_passes": 3,
                "average_accepted_per_pass": 2.0,
                "layers_executed": 192,
                "expected_layers_executed": 192,
                "anti_cheat_passed": True,
                "blockers": [],
            }

    monkeypatch.setattr(
        runtime_diagnose_cli,
        "_memory_snapshot",
        lambda: SimpleNamespace(total_bytes=16 * gb, free_bytes=8 * gb),
    )
    monkeypatch.setattr(runtime_diagnose_cli, "_working_set_mb", lambda: 123)
    monkeypatch.setattr(runtime_diagnose_cli, "original_model_root", lambda model_id: tmp_path)

    def fake_speculative_generate(verifier_model_id, speculator_model_id, prompt, **kwargs):
        captured.update(
            {
                "verifier_model_id": verifier_model_id,
                "speculator_model_id": speculator_model_id,
                "prompt": prompt,
                **kwargs,
            }
        )
        return _FakeSpeculativeResult()

    monkeypatch.setattr(runtime_diagnose_cli, "speculative_generate", fake_speculative_generate)

    exit_code = runtime_diagnose_cli.main(
        [
            "--model",
            "ignored",
            "--slice",
            "speculative",
            "--verifier-model",
            "qwen3-30b-a3b",
            "--speculator-model",
            "qwen2.5-14b-instruct",
            "--prompt",
            "The capital of France is",
            "--max-new-tokens",
            "10",
            "--k",
            "4",
        ]
    )

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[1]["operation"] == "speculative-prompt-decode"
    assert lines[2]["result"]["generated_text"] == "Paris is the capital."
    assert lines[2]["result"]["effective_tokens_per_second"] == 2.5
    assert lines[2]["result"]["anti_cheat_passed"] is True
    assert captured["verifier_model_id"] == "qwen3-30b-a3b"
    assert captured["speculator_model_id"] == "qwen2.5-14b-instruct"
    assert captured["max_new_tokens"] == 10
    assert captured["k"] == 4


def test_runtime_diagnose_cli_full_repeat_runs_in_one_process(monkeypatch, capsys, tmp_path) -> None:
    gb = 1024**3
    calls = {"count": 0}

    class _FakeResult:
        def to_dict(self) -> dict:
            calls["count"] += 1
            return {
                "ready": True,
                "generated_text": f"Hello {calls['count']}",
                "blockers": [],
                "timings": {"total": float(calls["count"])},
            }

    monkeypatch.setattr(
        runtime_diagnose_cli,
        "_memory_snapshot",
        lambda: SimpleNamespace(total_bytes=16 * gb, free_bytes=8 * gb),
    )
    monkeypatch.setattr(runtime_diagnose_cli, "_working_set_mb", lambda: 123)
    monkeypatch.setattr(runtime_diagnose_cli, "original_model_root", lambda model_id: tmp_path)
    monkeypatch.setattr(runtime_diagnose_cli, "run_prompt_decode_loop", lambda *_args, **_kwargs: _FakeResult())

    exit_code = runtime_diagnose_cli.main(
        ["--model", "qwen-test", "--slice", "full", "--max-new-tokens", "1", "--repeat", "3"]
    )

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    after_lines = [line for line in lines if line["event"] == "after"]
    assert len(after_lines) == 3
    assert lines[-1]["ready"] is True
    assert calls["count"] == 3


def test_runtime_diagnose_cli_top_k_experts_repeat_runs_special_callback(monkeypatch, capsys, tmp_path) -> None:
    gb = 1024**3
    calls = {"count": 0}

    monkeypatch.setattr(
        runtime_diagnose_cli,
        "_memory_snapshot",
        lambda: SimpleNamespace(total_bytes=16 * gb, free_bytes=8 * gb),
    )
    monkeypatch.setattr(runtime_diagnose_cli, "_working_set_mb", lambda: 123)
    monkeypatch.setattr(runtime_diagnose_cli, "original_model_root", lambda model_id: tmp_path)
    monkeypatch.setattr(
        runtime_diagnose_cli,
        "expert_residency_snapshot",
        lambda: {"expert_residency": {"expert_hits": calls["count"]}},
    )

    def fake_top_k(model_id: str, slice_name: str, started_at: float) -> dict:
        calls["count"] += 1
        return {
            "ready": True,
            "blockers": [],
            "selected_experts": [1, 5],
            "fp16_packed_cache_stats": {"hits": calls["count"] - 1, "misses": 1},
        }

    monkeypatch.setattr(runtime_diagnose_cli, "_moe_top_k_experts", fake_top_k)

    exit_code = runtime_diagnose_cli.main(
        ["--model", "mixtral-test", "--slice", "top-k-experts", "--repeat", "2"]
    )

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    after_lines = [line for line in lines if line["event"] == "after"]
    assert len(after_lines) == 2
    assert calls["count"] == 2
    assert after_lines[0]["result"]["selected_experts"] == [1, 5]
    assert after_lines[1]["result"]["fp16_packed_cache_stats"]["hits"] == 1
    assert lines[-1]["ready"] is True


def test_runtime_diagnose_cli_compare_with_reference_reports_token_overlap(monkeypatch, capsys, tmp_path) -> None:
    gb = 1024**3
    fixture_dir = tmp_path / "fixtures" / "qwen_moe_test_moe_reference"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "reference.json").write_text(
        json.dumps(
            {
                "generated_token_ids": [10, 11, 12],
                "decoded_generated_text": " Paris.",
            }
        ),
        encoding="utf-8",
    )

    class _FakeResult:
        ready = True
        generated_token_ids = [10, 11, 99]
        generated_text = " Paris?"
        blockers = []
        timings = {}

        def to_dict(self) -> dict:
            return {
                "ready": True,
                "generated_token_ids": list(self.generated_token_ids),
                "generated_text": self.generated_text,
                "blockers": [],
            }

    monkeypatch.setattr(
        runtime_diagnose_cli,
        "_memory_snapshot",
        lambda: SimpleNamespace(total_bytes=16 * gb, free_bytes=8 * gb),
    )
    monkeypatch.setattr(runtime_diagnose_cli, "_working_set_mb", lambda: 123)
    monkeypatch.setattr(runtime_diagnose_cli, "original_model_root", lambda model_id: tmp_path)
    monkeypatch.setattr(runtime_diagnose_cli, "run_prompt_decode_loop", lambda *_args, **_kwargs: _FakeResult())

    exit_code = runtime_diagnose_cli.main(
        [
            "--model",
            "qwen-moe-test",
            "--slice",
            "compare-with-reference",
            "--max-new-tokens",
            "3",
            "--prompt",
            "The capital of France is",
            "--reference-root",
            str(tmp_path / "fixtures"),
        ]
    )

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    result = lines[2]["result"]
    assert result["shared_prefix_positions"] == 2
    assert result["actual_token_ids"] == [10, 11, 99]
    assert result["expected_token_ids"] == [10, 11, 12]


@pytest.mark.skip(reason="tensor fixture intentionally omitted from public source snapshot")
def test_runtime_diagnose_cli_compare_with_tiny_qwen3_oracle(monkeypatch, capsys) -> None:
    fixture_root = runtime_diagnose_cli.Path("tests/fixtures")
    model_path = fixture_root / "tiny_moe_qwen3"
    monkeypatch.setenv("PCKETLM_RUNTIME_MATH_DTYPE", "float32")

    exit_code = runtime_diagnose_cli.main(
        [
            "--model",
            "tiny_moe_qwen3",
            "--model-path",
            str(model_path),
            "--slice",
            "compare-with-reference",
            "--reference-root",
            str(fixture_root),
            "--max-new-tokens",
            "10",
        ]
    )

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    result = lines[2]["result"]
    assert result["ready"] is True
    assert result["shared_prefix_positions"] == 10
    assert result["actual_token_ids"] == result["expected_token_ids"]
    assert all(checkpoint["ready"] for checkpoint in result["checkpoint_comparisons"])


@pytest.mark.skip(reason="tensor fixture intentionally omitted from public source snapshot")
def test_runtime_diagnose_cli_compare_accepts_token_exact_mixtral_float_drift(monkeypatch, capsys) -> None:
    fixture_root = runtime_diagnose_cli.Path("tests/fixtures")
    model_path = fixture_root / "tiny_moe_mixtral"
    monkeypatch.setenv("PCKETLM_RUNTIME_MATH_DTYPE", "float32")

    exit_code = runtime_diagnose_cli.main(
        [
            "--model",
            "tiny_moe_mixtral",
            "--model-path",
            str(model_path),
            "--slice",
            "compare-with-reference",
            "--reference-root",
            str(fixture_root),
            "--max-new-tokens",
            "10",
        ]
    )

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    result = lines[2]["result"]
    assert result["ready"] is True
    assert result["shared_prefix_positions"] == 10
    assert result["actual_token_ids"] == result["expected_token_ids"]
    layer0 = next(item for item in result["checkpoint_comparisons"] if item["name"] == "layer0_combined_hidden")
    assert layer0["ready"] is True
    assert layer0["cosine_similarity"] >= 0.9999
    assert layer0["max_abs_diff"] <= 1e-3


def test_runtime_diagnose_cli_summarizes_kv_cache_bytes() -> None:
    key = torch.zeros((1, 8, 3, 128), dtype=torch.float16)
    value = torch.zeros((1, 8, 3, 128), dtype=torch.float16)

    summary = runtime_diagnose_cli._kv_cache_summary({0: (key, value), 1: (key, value)})

    assert summary["kv_cache_layers"] == 2
    assert summary["kv_cache_total_bytes"] == key.nelement() * key.element_size() * 4
    assert summary["kv_cache_by_layer"]["0"]["key_shape"] == [1, 8, 3, 128]


def test_runtime_diagnose_cli_moe_router_slice_reports_selected_experts(monkeypatch, capsys, tmp_path) -> None:
    gb = 1024**3
    hidden = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
    router = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
    )

    monkeypatch.setattr(
        runtime_diagnose_cli,
        "_memory_snapshot",
        lambda: SimpleNamespace(total_bytes=16 * gb, free_bytes=8 * gb),
    )
    monkeypatch.setattr(runtime_diagnose_cli, "_working_set_mb", lambda: 123)
    monkeypatch.setattr(runtime_diagnose_cli, "original_model_root", lambda model_id: tmp_path)
    monkeypatch.setattr(
        runtime_diagnose_cli,
        "_load_hello_hidden_checkpoint",
        lambda *_args, **_kwargs: (1, hidden),
    )
    monkeypatch.setattr(
        runtime_diagnose_cli,
        "load_layer_bridge_config",
        lambda _model_id: SimpleNamespace(
            ready=True,
            blockers=[],
            num_experts=4,
            num_experts_per_tok=2,
        ),
    )
    monkeypatch.setattr(
        runtime_diagnose_cli,
        "load_tensor_by_name",
        lambda *_args, **_kwargs: SimpleNamespace(tensor=router, blockers=[]),
    )

    exit_code = runtime_diagnose_cli.main(["--model", "qwen-moe-test", "--slice", "router-only"])

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    result = lines[2]["result"]
    assert result["selected_experts"] == [1, 0]
    assert result["router_logits_shape"] == [1, 1, 4]
