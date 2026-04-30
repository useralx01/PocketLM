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


def test_registry_lists_qwen_14b_32b_and_moe_entries(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    registry_dir = tmp_path / "state" / "registry"
    registry_dir.mkdir(parents=True)
    registry_path = registry_dir / "models.json"
    records = []
    for size in ("14b", "32b"):
        model_id = f"qwen2.5-{size}-instruct"
        original = tmp_path / "models" / model_id / "original"
        original.mkdir(parents=True)
        records.append(
            {
                "model_id": model_id,
                "label": f"Qwen2.5-{size.upper()}-Instruct",
                "family": "qwen",
                "model_type": "dense",
                "source_path": str(original),
                "original_path": str(original),
                "source_kind": "local-folder",
                "source_origin": "huggingface",
                "repo_id": f"Qwen/Qwen2.5-{size.upper()}-Instruct",
                "format_name": "safetensors-sharded",
                "config": None,
                "imported": size == "14b",
                "validated": size == "14b",
                "runnable": size == "14b",
                "validation": {"result": "ok" if size == "14b" else "pending-download", "missing_files": [], "warnings": []},
                "artifact_paths": [],
            }
        )
    moe_original = tmp_path / "models" / "qwen3-30b-a3b" / "original"
    moe_original.mkdir(parents=True)
    records.append(
        {
            "model_id": "qwen3-30b-a3b",
            "label": "Qwen3-30B-A3B",
            "family": "qwen-moe",
            "model_type": "moe",
            "source_path": str(moe_original),
            "original_path": str(moe_original),
            "source_kind": "local-folder",
            "source_origin": "huggingface",
            "repo_id": "Qwen/Qwen3-30B-A3B",
            "format_name": "safetensors-sharded",
            "config": None,
            "imported": False,
            "validated": False,
            "runnable": False,
            "validation": {"result": "pending-download", "missing_files": [], "warnings": []},
            "artifact_paths": [],
        }
    )
    registry_path.write_text(json.dumps({"models": records}, indent=2), encoding="utf-8")

    loaded = load_model_registry()

    assert set(loaded) == {"qwen2.5-14b-instruct", "qwen2.5-32b-instruct", "qwen3-30b-a3b"}
    assert loaded["qwen2.5-14b-instruct"].repo_id == "Qwen/Qwen2.5-14B-Instruct"
    assert loaded["qwen2.5-32b-instruct"].repo_id == "Qwen/Qwen2.5-32B-Instruct"
    assert loaded["qwen3-30b-a3b"].family == "qwen-moe"
    assert loaded["qwen3-30b-a3b"].repo_id == "Qwen/Qwen3-30B-A3B"
