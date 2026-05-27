from types import SimpleNamespace

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from pcketlm.core.runtime import tokenizer_runtime


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
