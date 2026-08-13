from types import SimpleNamespace

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from pcketlm.core.runtime import tokenizer_runtime
from tools.analyze_fp8_generation_quality import _quality


def test_load_local_tokenizer_falls_back_to_tensor_catalog_model_dir(tmp_path, monkeypatch) -> None:
    missing_original = tmp_path / "missing" / "original"
    catalog_dir = tmp_path / "external-source"
    catalog_dir.mkdir(parents=True)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(catalog_dir / "tokenizer.json"))

    monkeypatch.setattr(tokenizer_runtime, "original_model_root", lambda _model_id: missing_original)

    from pcketlm.core.runtime import tensor_catalog

    monkeypatch.setattr(tensor_catalog, "load_tensor_catalog", lambda _model_id: SimpleNamespace(model_dir=catalog_dir))

    result = tokenizer_runtime.load_local_tokenizer("deepseek-v3")

    assert result.ready is True
    assert result.tokenizer_path == catalog_dir / "tokenizer.json"


def test_prepare_prompt_text_supports_deepseek_chat_template(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "models" / "deepseek-test" / "original"
    model_dir.mkdir(parents=True)
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "<｜begin▁of▁sentence｜>": 1,
                "<｜User｜>": 2,
                "<｜Assistant｜>": 3,
                "Be": 4,
                "brief.": 5,
                "hello": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(
        """
{
  "bos_token": {"content": "<｜begin▁of▁sentence｜>"},
  "chat_template": "{{bos_token}}{{'<｜User｜>' + message['content']}}{{'<｜Assistant｜>'}}"
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(tokenizer_runtime, "original_model_root", lambda _model_id: model_dir)

    wrapped = tokenizer_runtime.prepare_prompt_text(
        "deepseek-test",
        "hello",
        system_prompt="Be brief.",
        apply_chat_format=True,
    )
    raw = tokenizer_runtime.prepare_prompt_text("deepseek-test", "hello", apply_chat_format=False)

    assert wrapped.ready is True
    assert raw.ready is True
    assert wrapped.prepared_prompt == "<｜begin▁of▁sentence｜>Be brief.<｜User｜>hello<｜Assistant｜>"
    assert raw.prepared_prompt == "hello"


def test_generation_quality_rejects_whitespace_runs() -> None:
    blank = _quality("        ", [223] * 8, min_non_whitespace=3, min_unique_tokens=2)
    useful = _quality("Hello.", [100, 101, 102], min_non_whitespace=3, min_unique_tokens=2)

    assert blank["is_useful_text"] is False
    assert useful["is_useful_text"] is True
