import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.pack_weights_row8 import build_row8_packed_artifact, pack_rows8_tensor


def test_row8_runtime_loader_reads_only_requested_tensor(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage
    from pcketlm.core.runtime.packed_artifact_loader import (
        clear_row8_manifest_cache,
        clear_row8_tensor_cache,
        load_row8_packed_tensor,
        row8_artifact_status,
        row8_tensor_cache_stats,
        row8_tensor_available,
    )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setenv("PCKETLM_ROW8_TENSOR_CACHE_MB", "1")
    model_id = "row8-loader-test"
    model_dir = _write_model_fixture(tmp_path, model_id)
    out_dir = tmp_path / "models" / model_id / "artifacts" / "row8"
    build_row8_packed_artifact(model_dir, out_dir, include_patterns=["down_proj"])
    clear_row8_manifest_cache()
    clear_row8_tensor_cache()

    tensor_name = "model.layers.0.mlp.down_proj.weight"
    loaded, entry = load_row8_packed_tensor(model_id, tensor_name)
    expected = pack_rows8_tensor(torch.arange(130, dtype=torch.float32).reshape(10, 13).to(torch.float16))

    assert row8_artifact_status(model_id)["ready"] is True
    assert row8_tensor_available(model_id, tensor_name) is True
    assert row8_tensor_available(model_id, "missing.weight") is False
    assert entry["shape"] == [10, 13]
    assert torch.equal(loaded, expected)
    assert row8_tensor_cache_stats()["resident_count"] == 1


def test_row8_runtime_loader_caches_repeated_tensor_reads(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage
    from pcketlm.core.runtime.packed_artifact_loader import (
        clear_row8_manifest_cache,
        clear_row8_tensor_cache,
        load_row8_packed_tensor,
        row8_tensor_cache_stats,
    )

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setenv("PCKETLM_ROW8_TENSOR_CACHE_MB", "1")
    model_id = "row8-cache-test"
    model_dir = _write_model_fixture(tmp_path, model_id)
    out_dir = tmp_path / "models" / model_id / "artifacts" / "row8"
    build_row8_packed_artifact(model_dir, out_dir, include_patterns=["down_proj"])
    clear_row8_manifest_cache()
    clear_row8_tensor_cache()

    tensor_name = "model.layers.0.mlp.down_proj.weight"
    first, _ = load_row8_packed_tensor(model_id, tensor_name)
    data_file = out_dir / "row8_packed.bin"
    renamed = out_dir / "row8_packed.bin.hidden"
    data_file.rename(renamed)
    try:
        second, _ = load_row8_packed_tensor(model_id, tensor_name)
    finally:
        renamed.rename(data_file)

    assert torch.equal(first, second)
    assert row8_tensor_cache_stats()["resident_count"] == 1


def _write_model_fixture(tmp_path: Path, model_id: str) -> Path:
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.layers.0.mlp.down_proj.weight": torch.arange(130, dtype=torch.float32).reshape(10, 13).to(torch.float16),
        "model.layers.0.mlp.up_proj.weight": torch.ones((10, 13), dtype=torch.float16),
    }
    save_file(tensors, str(shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": shard.stat().st_size}, "weight_map": {name: shard.name for name in tensors}}),
        encoding="utf-8",
    )
    return model_dir
