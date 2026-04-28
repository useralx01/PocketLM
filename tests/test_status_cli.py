from pathlib import Path

from pcketlm.app.chat_shell.status_cli import build_status_payload


def test_build_status_payload_partial(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2"}',
        encoding="utf-8",
    )

    payload = build_status_payload(tmp_path)
    assert payload["status"] == "partial"
    assert "tokenizer.json" in payload["missing_files"]
    assert payload["inspection"]["config"]["model_type"] == "qwen2"
