import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.pack_weights_row8 import (
    build_row8_packed_artifact,
    load_row8_packed_tensor,
    load_row8_unpacked_tensor,
    pack_rows8_tensor,
    unpack_rows8_tensor,
)


def test_row8_pack_unpack_round_trips_fp16_and_bf16() -> None:
    for dtype in (torch.float16, torch.bfloat16):
        tensor = torch.arange(130, dtype=torch.float32).reshape(10, 13).to(dtype)
        packed = pack_rows8_tensor(tensor)
        restored = unpack_rows8_tensor(packed, 10, 13, dtype=dtype)
        assert torch.equal(restored, tensor)


def test_row8_artifact_writes_manifest_and_loads_packed_tensor(tmp_path: Path) -> None:
    model_dir = _write_fixture(tmp_path)
    out_dir = tmp_path / "row8"

    manifest = build_row8_packed_artifact(model_dir, out_dir, include_patterns=["down_proj"])
    packed, entry = load_row8_packed_tensor(out_dir, "model.layers.0.mlp.down_proj.weight")
    restored = load_row8_unpacked_tensor(out_dir, "model.layers.0.mlp.down_proj.weight")

    assert manifest["tensor_count"] == 1
    assert entry["shape"] == [10, 13]
    assert packed.dtype == torch.uint16
    assert torch.equal(restored, torch.arange(130, dtype=torch.float32).reshape(10, 13).to(torch.float16))
    assert (out_dir / "row8_packed.bin").stat().st_size == entry["nbytes"]


def test_row8_artifact_feeds_native_packed_gemv(tmp_path: Path) -> None:
    from pcketlm.native import packed_gemv_rows8

    model_dir = _write_fixture(tmp_path)
    out_dir = tmp_path / "row8"
    build_row8_packed_artifact(model_dir, out_dir, include_patterns=["down_proj"])
    packed, entry = load_row8_packed_tensor(out_dir, "model.layers.0.mlp.down_proj.weight")
    hidden = torch.arange(13, dtype=torch.float16) / 10

    native = packed_gemv_rows8(hidden, packed, rows=entry["shape"][0], cols=entry["shape"][1])
    expected = torch.mv(
        torch.arange(130, dtype=torch.float32).reshape(10, 13),
        hidden.float(),
    )

    assert torch.allclose(native, expected, atol=1e-4, rtol=1e-4)


def _write_fixture(tmp_path: Path) -> Path:
    model_dir = tmp_path / "source"
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
