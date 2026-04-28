from pathlib import Path

from pcketlm.core.model_import.service import ImportRequest, import_model
from pcketlm.core.registry.catalog import build_model_catalog
from pcketlm.core import storage


def test_build_model_catalog(tmp_path: Path) -> None:
    storage.paths.project_root = lambda: tmp_path

    source = tmp_path / "source-model"
    source.mkdir()
    for name in ("config.json", "tokenizer.json", "vocab.json", "merges.txt"):
        (source / name).write_text("{}", encoding="utf-8")
    (source / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":1},"weight_map":{"a":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    (source / "model-00001-of-00001.safetensors").write_text("x", encoding="utf-8")

    import_model(
        ImportRequest(
            model_id="qwen-test",
            label="Qwen Test",
            family="qwen",
            source_path=source,
            source_origin="test",
            format_name="safetensors-sharded",
        )
    )

    catalog = build_model_catalog()
    assert len(catalog) == 1
    assert catalog[0].model_id == "qwen-test"
    assert catalog[0].source_status == "ready"
