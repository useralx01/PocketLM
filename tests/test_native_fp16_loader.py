import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

from pcketlm.core.runtime.tensor_execution_plan import build_tensor_execution_plan
from pcketlm.core.runtime.tensor_loader import (
    load_tensor_by_name,
    reset_tensor_load_stats,
    tensor_load_stats_snapshot,
)


def _write_native_loader_fixture(tmp_path: Path, monkeypatch) -> tuple[str, Path, torch.Tensor]:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "native-fp16-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "vocab_size": 8,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "vocab.json"):
        (model_dir / name).write_text("{}", encoding="utf-8")
    (model_dir / "merges.txt").write_text("", encoding="utf-8")
    tensor = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    shard = model_dir / "model-00001-of-00001.safetensors"
    save_file({"model.layers.0.input_layernorm.weight": tensor}, str(shard))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {"model.layers.0.input_layernorm.weight": shard.name},
            }
        ),
        encoding="utf-8",
    )
    build_tensor_execution_plan(model_id, model_dir)
    return model_id, shard, tensor


def _tensor_absolute_offset(shard: Path, tensor_name: str) -> tuple[int, int]:
    with shard.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length).decode("utf-8"))
    start, end = header[tensor_name]["data_offsets"]
    return 8 + int(header_length) + int(start), int(end) - int(start)


def test_native_read_tensor_bytes_matches_safetensors_payload(tmp_path: Path, monkeypatch) -> None:
    _model_id, shard, expected = _write_native_loader_fixture(tmp_path, monkeypatch)
    offset, nbytes = _tensor_absolute_offset(shard, "model.layers.0.input_layernorm.weight")

    from pcketlm.native import native_fp16_loader_available, native_read_tensor_bytes

    assert native_fp16_loader_available() is True
    out = torch.empty_like(expected)
    native_read_tensor_bytes(shard, offset, nbytes, out)

    assert torch.equal(out, expected)


def test_tensor_loader_uses_native_fp16_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_FP16_LOAD", raising=False)
    monkeypatch.setenv("PCKETLM_RUNTIME_PACK", "0")
    monkeypatch.setenv("PCKETLM_SAFETENSOR_HANDLE_CACHE", "0")
    reset_tensor_load_stats()
    model_id, _shard, expected = _write_native_loader_fixture(tmp_path, monkeypatch)

    loaded = load_tensor_by_name(model_id, "model.layers.0.input_layernorm.weight")
    stats = tensor_load_stats_snapshot()

    assert loaded.ready is True
    assert loaded.tensor is not None
    assert torch.equal(loaded.tensor, expected)
    assert stats.native_fp16_loads == 1
    assert stats.native_fp16_loaded_nbytes == expected.nelement() * expected.element_size()
    assert stats.shard_opens == 0


def test_tensor_loader_native_fp16_kill_switch_falls_back(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_FP16_LOAD", "1")
    monkeypatch.setenv("PCKETLM_RUNTIME_PACK", "0")
    monkeypatch.setenv("PCKETLM_SAFETENSOR_HANDLE_CACHE", "0")
    reset_tensor_load_stats()
    model_id, _shard, expected = _write_native_loader_fixture(tmp_path, monkeypatch)

    loaded = load_tensor_by_name(model_id, "model.layers.0.input_layernorm.weight")
    stats = tensor_load_stats_snapshot()

    assert loaded.ready is True
    assert loaded.tensor is not None
    assert torch.equal(loaded.tensor, expected)
    assert stats.native_fp16_loads == 0
    assert stats.shard_opens == 1
