import json
from pathlib import Path

from pcketlm.core.registry.repository import load_model_registry


def test_load_model_registry_relocates_stale_project_paths(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    new_original = tmp_path / "models" / "qwen-test" / "original"
    new_original.mkdir(parents=True)

    registry_dir = tmp_path / "state" / "registry"
    registry_dir.mkdir(parents=True)
    registry_path = registry_dir / "models.json"
    registry_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "qwen-test",
                        "label": "Qwen Test",
                        "family": "qwen",
                        "model_type": "dense",
                        "source_path": "C:/missing-old-root/models/qwen-test/original",
                        "original_path": "C:/missing-old-root/models/qwen-test/original",
                        "source_kind": "local-folder",
                        "source_origin": "huggingface",
                        "repo_id": "Qwen/Test",
                        "format_name": "safetensors-sharded",
                        "config": None,
                        "imported": True,
                        "validated": True,
                        "runnable": True,
                        "validation": {"result": "ok", "missing_files": [], "warnings": []},
                        "artifact_paths": [],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    records = load_model_registry()

    assert records["qwen-test"].source_path == new_original
    assert records["qwen-test"].original_path == new_original

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["models"][0]["source_path"] == str(new_original)
    assert payload["models"][0]["original_path"] == str(new_original)
