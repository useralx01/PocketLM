import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from pcketlm.core.optimize import (
    build_optimized_artifact_manifest,
    latest_optimized_artifact_manifest,
    select_runtime_artifact_manifest,
)
from pcketlm.core.runtime.tensor_catalog import TensorCatalog, TensorCatalogEntry, tensor_catalog_path


def test_build_optimized_artifact_manifest_records_source_catalog(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "artifact-test"
    shard_path = tmp_path / "models" / model_id / "original" / "model.safetensors"
    shard_path.parent.mkdir(parents=True)
    save_file(
        {
            "model.layers.0.input_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
            "model.embed_tokens.weight": torch.ones((16, 8), dtype=torch.bfloat16),
        },
        str(shard_path),
    )
    catalog_path = tensor_catalog_path(model_id)
    catalog_path.parent.mkdir(parents=True)
    catalog = TensorCatalog(
        model_id=model_id,
        model_dir=shard_path.parent,
        catalog_path=catalog_path,
        tensor_count=1,
        shard_count=1,
        layer_count=1,
        tensors=[
            TensorCatalogEntry(
                tensor_name="model.layers.0.input_layernorm.weight",
                shard_name=shard_path.name,
                shard_path=shard_path,
                dtype="BF16",
                shape=[8],
                data_offset_start=0,
                data_offset_end=16,
                data_nbytes=16,
                layer_index=0,
                component_group="layer_norm",
            )
        ],
        ready=True,
    )
    catalog_path.write_text(json.dumps(catalog.to_dict()), encoding="utf-8")

    manifest = build_optimized_artifact_manifest(model_id, profile_id="low-memory")
    latest = latest_optimized_artifact_manifest(model_id)

    assert manifest.ready is True
    assert manifest.source_tensor_count == 1
    assert manifest.source_layer_count == 1
    assert manifest.profile_id == "low-memory"
    assert manifest.artifact_path.exists()
    assert latest is not None
    assert latest.artifact_id == manifest.artifact_id


def test_build_optimized_artifact_manifest_can_materialize_small_tensor_pack(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "artifact-pack-test"
    shard_path = tmp_path / "models" / model_id / "original" / "model.safetensors"
    shard_path.parent.mkdir(parents=True)
    save_file(
        {
            "model.layers.0.input_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
            "model.layers.0.self_attn.q_proj.weight": torch.full((8, 8), 3, dtype=torch.bfloat16),
            "model.layers.0.self_attn.k_proj.weight": torch.full((4, 8), 4, dtype=torch.bfloat16),
            "model.layers.0.self_attn.v_proj.weight": torch.full((4, 8), 5, dtype=torch.bfloat16),
            "model.layers.1.self_attn.q_proj.weight": torch.full((8, 8), 6, dtype=torch.bfloat16),
            "model.embed_tokens.weight": torch.ones((16, 8), dtype=torch.bfloat16),
        },
        str(shard_path),
    )
    catalog_path = tensor_catalog_path(model_id)
    catalog_path.parent.mkdir(parents=True)
    catalog = TensorCatalog(
        model_id=model_id,
        model_dir=shard_path.parent,
        catalog_path=catalog_path,
        tensor_count=2,
        shard_count=1,
        layer_count=1,
        tensors=[
            TensorCatalogEntry(
                tensor_name="model.layers.0.input_layernorm.weight",
                shard_name=shard_path.name,
                shard_path=shard_path,
                dtype="BF16",
                shape=[8],
                data_offset_start=0,
                data_offset_end=16,
                data_nbytes=16,
                layer_index=0,
                component_group="layer_norm",
            ),
            TensorCatalogEntry(
                tensor_name="model.layers.0.self_attn.q_proj.weight",
                shard_name=shard_path.name,
                shard_path=shard_path,
                dtype="BF16",
                shape=[8, 8],
                data_offset_start=16,
                data_offset_end=144,
                data_nbytes=128,
                layer_index=0,
                component_group="attention",
            ),
            TensorCatalogEntry(
                tensor_name="model.layers.0.self_attn.k_proj.weight",
                shard_name=shard_path.name,
                shard_path=shard_path,
                dtype="BF16",
                shape=[4, 8],
                data_offset_start=144,
                data_offset_end=208,
                data_nbytes=64,
                layer_index=0,
                component_group="attention",
            ),
            TensorCatalogEntry(
                tensor_name="model.layers.0.self_attn.v_proj.weight",
                shard_name=shard_path.name,
                shard_path=shard_path,
                dtype="BF16",
                shape=[4, 8],
                data_offset_start=208,
                data_offset_end=272,
                data_nbytes=64,
                layer_index=0,
                component_group="attention",
            ),
            TensorCatalogEntry(
                tensor_name="model.layers.1.self_attn.q_proj.weight",
                shard_name=shard_path.name,
                shard_path=shard_path,
                dtype="BF16",
                shape=[8, 8],
                data_offset_start=272,
                data_offset_end=400,
                data_nbytes=128,
                layer_index=1,
                component_group="attention",
            ),
            TensorCatalogEntry(
                tensor_name="model.embed_tokens.weight",
                shard_name=shard_path.name,
                shard_path=shard_path,
                dtype="BF16",
                shape=[16, 8],
                data_offset_start=16,
                data_offset_end=272,
                data_nbytes=256,
                layer_index=None,
                component_group="embeddings",
            ),
        ],
        ready=True,
    )
    catalog_path.write_text(json.dumps(catalog.to_dict()), encoding="utf-8")

    manifest = build_optimized_artifact_manifest(model_id, profile_id="low-memory", materialize_small_pack=True)

    assert manifest.ready is True
    assert manifest.tensor_pack_path is not None
    assert manifest.tensor_pack_path.exists()
    assert manifest.packed_tensor_names == ["model.layers.0.input_layernorm.weight"]
    with safe_open(manifest.tensor_pack_path, framework="pt", device="cpu") as handle:
        assert torch.allclose(handle.get_tensor("model.layers.0.input_layernorm.weight").float(), torch.ones(8))


def test_build_optimized_artifact_manifest_can_materialize_front_attention_pack(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "artifact-attention-pack-test"
    shard_path = tmp_path / "models" / model_id / "original" / "model.safetensors"
    shard_path.parent.mkdir(parents=True)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.full((8, 8), 3, dtype=torch.bfloat16),
            "model.layers.0.self_attn.k_proj.weight": torch.full((4, 8), 4, dtype=torch.bfloat16),
            "model.layers.0.self_attn.v_proj.weight": torch.full((4, 8), 5, dtype=torch.bfloat16),
            "model.layers.1.self_attn.q_proj.weight": torch.full((8, 8), 6, dtype=torch.bfloat16),
        },
        str(shard_path),
    )
    catalog_path = tensor_catalog_path(model_id)
    catalog_path.parent.mkdir(parents=True)

    def attention_entry(name: str, shape: list[int], nbytes: int, layer: int) -> TensorCatalogEntry:
        return TensorCatalogEntry(
            tensor_name=name,
            shard_name=shard_path.name,
            shard_path=shard_path,
            dtype="BF16",
            shape=shape,
            data_offset_start=0,
            data_offset_end=nbytes,
            data_nbytes=nbytes,
            layer_index=layer,
            component_group="attention",
        )

    catalog = TensorCatalog(
        model_id=model_id,
        model_dir=shard_path.parent,
        catalog_path=catalog_path,
        tensor_count=4,
        shard_count=1,
        layer_count=2,
        tensors=[
            attention_entry("model.layers.0.self_attn.q_proj.weight", [8, 8], 128, 0),
            attention_entry("model.layers.0.self_attn.k_proj.weight", [4, 8], 64, 0),
            attention_entry("model.layers.0.self_attn.v_proj.weight", [4, 8], 64, 0),
            attention_entry("model.layers.1.self_attn.q_proj.weight", [8, 8], 128, 1),
        ],
        ready=True,
    )
    catalog_path.write_text(json.dumps(catalog.to_dict()), encoding="utf-8")

    manifest = build_optimized_artifact_manifest(
        model_id,
        profile_id="low-memory",
        materialize_front_attention_pack=True,
        front_attention_layers=1,
    )

    assert manifest.ready is True
    assert manifest.tensor_pack_path is not None
    assert manifest.packed_tensor_names == [
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    with safe_open(manifest.tensor_pack_path, framework="pt", device="cpu") as handle:
        assert "model.layers.1.self_attn.q_proj.weight" not in handle.keys()
        assert torch.allclose(handle.get_tensor("model.layers.0.self_attn.q_proj.weight").float(), torch.full((8, 8), 3.0))


def test_build_optimized_artifact_manifest_can_include_attention_output_projection(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "artifact-output-pack-test"
    shard_path = tmp_path / "models" / model_id / "original" / "model.safetensors"
    shard_path.parent.mkdir(parents=True)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.full((8, 8), 3, dtype=torch.bfloat16),
            "model.layers.0.self_attn.o_proj.weight": torch.full((8, 8), 7, dtype=torch.bfloat16),
        },
        str(shard_path),
    )
    catalog_path = tensor_catalog_path(model_id)
    catalog_path.parent.mkdir(parents=True)

    catalog = TensorCatalog(
        model_id=model_id,
        model_dir=shard_path.parent,
        catalog_path=catalog_path,
        tensor_count=2,
        shard_count=1,
        layer_count=1,
        tensors=[
            TensorCatalogEntry(
                tensor_name="model.layers.0.self_attn.q_proj.weight",
                shard_name=shard_path.name,
                shard_path=shard_path,
                dtype="BF16",
                shape=[8, 8],
                data_offset_start=0,
                data_offset_end=128,
                data_nbytes=128,
                layer_index=0,
                component_group="attention",
            ),
            TensorCatalogEntry(
                tensor_name="model.layers.0.self_attn.o_proj.weight",
                shard_name=shard_path.name,
                shard_path=shard_path,
                dtype="BF16",
                shape=[8, 8],
                data_offset_start=128,
                data_offset_end=256,
                data_nbytes=128,
                layer_index=0,
                component_group="attention",
            ),
        ],
        ready=True,
    )
    catalog_path.write_text(json.dumps(catalog.to_dict()), encoding="utf-8")

    manifest = build_optimized_artifact_manifest(
        model_id,
        profile_id="speed",
        materialize_front_attention_pack=True,
        include_attention_output_projection=True,
    )
    selected = select_runtime_artifact_manifest(model_id, free_memory_bytes=8 * 1024**3)

    assert manifest.ready is True
    assert "model.layers.0.self_attn.o_proj.weight" in manifest.packed_tensor_names
    assert selected is not None
    assert selected.artifact_id == manifest.artifact_id


def test_build_optimized_artifact_manifest_can_materialize_front_kv_pack(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "artifact-kv-pack-test"
    shard_path = tmp_path / "models" / model_id / "original" / "model.safetensors"
    shard_path.parent.mkdir(parents=True)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.full((8, 8), 3, dtype=torch.bfloat16),
            "model.layers.0.self_attn.k_proj.weight": torch.full((4, 8), 4, dtype=torch.bfloat16),
            "model.layers.0.self_attn.v_proj.weight": torch.full((4, 8), 5, dtype=torch.bfloat16),
            "model.layers.1.self_attn.k_proj.weight": torch.full((4, 8), 6, dtype=torch.bfloat16),
        },
        str(shard_path),
    )
    catalog_path = tensor_catalog_path(model_id)
    catalog_path.parent.mkdir(parents=True)

    def attention_entry(name: str, shape: list[int], nbytes: int, layer: int) -> TensorCatalogEntry:
        return TensorCatalogEntry(
            tensor_name=name,
            shard_name=shard_path.name,
            shard_path=shard_path,
            dtype="BF16",
            shape=shape,
            data_offset_start=0,
            data_offset_end=nbytes,
            data_nbytes=nbytes,
            layer_index=layer,
            component_group="attention",
        )

    catalog = TensorCatalog(
        model_id=model_id,
        model_dir=shard_path.parent,
        catalog_path=catalog_path,
        tensor_count=4,
        shard_count=1,
        layer_count=2,
        tensors=[
            attention_entry("model.layers.0.self_attn.q_proj.weight", [8, 8], 128, 0),
            attention_entry("model.layers.0.self_attn.k_proj.weight", [4, 8], 64, 0),
            attention_entry("model.layers.0.self_attn.v_proj.weight", [4, 8], 64, 0),
            attention_entry("model.layers.1.self_attn.k_proj.weight", [4, 8], 64, 1),
        ],
        ready=True,
    )
    catalog_path.write_text(json.dumps(catalog.to_dict()), encoding="utf-8")

    manifest = build_optimized_artifact_manifest(
        model_id,
        profile_id="boosted",
        materialize_front_kv_pack=True,
        front_kv_layers=2,
    )

    assert manifest.ready is True
    assert manifest.packed_tensor_names == [
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
    ]
    assert "model.layers.0.self_attn.q_proj.weight" not in manifest.packed_tensor_names


def test_select_runtime_artifact_manifest_uses_boosted_budget(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage
    from pcketlm.core.optimize import OptimizedArtifactManifest
    from pcketlm.core.storage.paths import artifacts_root

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "artifact-budget-test"
    root = artifacts_root(model_id)
    root.mkdir(parents=True)

    def write_manifest(profile_id: str, packed_nbytes: int) -> None:
        pack_path = root / f"runtime-pack-plan.{profile_id}.small.safetensors"
        pack_path.write_bytes(b"pack")
        manifest_path = root / f"runtime-pack-plan.{profile_id}.artifact.json"
        manifest = OptimizedArtifactManifest(
            model_id=model_id,
            artifact_id=manifest_path.stem,
            strategy="runtime-pack-plan",
            profile_id=profile_id,
            created_at=f"2026-04-28T00:00:0{len(profile_id)}+00:00",
            artifact_path=manifest_path,
            source_tensor_count=1,
            source_shard_count=1,
            source_layer_count=1,
            tensor_pack_path=pack_path,
            packed_tensor_count=1,
            packed_nbytes=packed_nbytes,
            packed_tensor_names=["tensor"],
            ready=True,
        )
        manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    write_manifest("low-memory", 140 * 1024 * 1024)
    write_manifest("boosted", 360 * 1024 * 1024)
    write_manifest("speed-core", 481 * 1024 * 1024)

    standard = select_runtime_artifact_manifest(model_id, runtime_preset="standard")
    boosted = select_runtime_artifact_manifest(model_id, runtime_preset="boosted")

    assert standard is not None
    assert standard.profile_id == "low-memory"
    assert boosted is not None
    assert boosted.profile_id == "boosted"
