from pathlib import Path

from pcketlm.core.model_compatibility import build_compatibility_matrix, profile_for_family
from pcketlm.core.model_import.inspect import inspect_model_source
from pcketlm.core.registry.catalog import build_model_catalog
from pcketlm.core.registry.models import ModelRecord
from pcketlm.core.validation.files import validate_model_source


def test_gemma_single_file_source_is_recognized_without_loading_weights(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"gemma3_text","architectures":["Gemma3ForCausalLM"]}', encoding="utf-8")
    (tmp_path / "tokenizer.model").write_bytes(b"fixture")
    (tmp_path / "model.safetensors").write_bytes(b"fixture")

    inspection = inspect_model_source(tmp_path)
    validation = validate_model_source(tmp_path, family="gemma")

    assert inspection.format_name == "safetensors"
    assert inspection.config.model_type == "gemma3_text"
    assert validation.result == "ok"


def test_kimi_gguf_source_is_recognized_without_full_download(tmp_path: Path) -> None:
    (tmp_path / "kimi-k2-fixture.gguf").write_bytes(b"GGUF")

    inspection = inspect_model_source(tmp_path)
    validation = validate_model_source(tmp_path, family="kimi")

    assert inspection.format_name == "gguf"
    assert validation.result == "ok"


def test_compatibility_matrix_is_truthful_when_weights_are_missing(monkeypatch, tmp_path: Path) -> None:
    from pcketlm.core import model_compatibility

    monkeypatch.setattr(model_compatibility, "llama_server_path", lambda: tmp_path / "llama-server.exe")
    rows = {row.profile.key: row for row in build_compatibility_matrix({})}

    assert rows["kimi-k2"].status == "Missing"
    assert rows["gemma-3"].status == "Missing"
    assert rows["kronos"].status == "Missing"
    assert rows["kronos"].profile.chat_status == "Unsupported"
    assert rows["kronos"].profile.capability == "forecast"


def test_profile_detection_uses_model_id_when_registry_family_is_absent() -> None:
    assert profile_for_family(None, "moonshot-kimi-k2").key == "kimi-k2"
    assert profile_for_family(None, "gemma-3-270m").key == "gemma-3"
    assert profile_for_family(None, "kronos-small").key == "kronos"


def test_catalog_never_repeats_stale_runnable_state(monkeypatch, tmp_path: Path) -> None:
    from pcketlm.core.registry import catalog

    record = ModelRecord(
        model_id="gemma-missing",
        label="Gemma Missing",
        family="gemma",
        model_type="dense",
        source_path=tmp_path / "gone",
        original_path=tmp_path / "gone",
        runnable=True,
    )
    monkeypatch.setattr(catalog, "load_model_registry", lambda: {record.model_id: record})

    entry = build_model_catalog()[0]

    assert entry.runnable is False
    assert entry.source_status == "missing"
    assert any("stale" in warning.lower() for warning in entry.warnings)
