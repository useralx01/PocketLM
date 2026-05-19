import ctypes
import json
import struct
from pathlib import Path

import pytest

from pcketlm.core.runtime.fp8_pack import FP8PackReader
from tools.pack_fp8 import pack_model_dir_to_fp8


def test_ds_session_registers_fp8_pack_callbacks(tmp_path: Path, monkeypatch) -> None:
    import tools.pack_fp8 as pack_tool
    from pcketlm.native import DeepSeekNativeSession, native_ds_forward_available

    monkeypatch.setattr(pack_tool, "state_root", lambda: tmp_path / "state")
    model_dir = _write_ds_pack_fixture(tmp_path)
    pack_model_dir_to_fp8(
        model_dir,
        model_dir / "artifacts" / "fp8_pack",
        model_id="ds-session-test",
        pack_bytes=128,
    )
    reader = FP8PackReader(model_dir)
    callback_counts = {"fp8": 0, "scale": 0}
    keepalive: list[ctypes.Array] = []

    def _buffer_from_view(view: memoryview) -> ctypes.Array:
        raw = ctypes.create_string_buffer(bytes(view))
        keepalive.append(raw)
        return raw

    def fp8_callback(layer_idx, layer_kind, weight_role, out_ptr, out_nbytes):  # noqa: ANN001
        callback_counts["fp8"] += 1
        name = f"model.layers.{int(layer_idx)}.mlp.experts.0.gate_proj.weight"
        tensor = reader.get_tensor(name)
        raw = _buffer_from_view(tensor.fp8_bytes)
        out_ptr[0] = ctypes.cast(raw, ctypes.c_void_p)
        out_nbytes[0] = len(raw.raw) - 1
        return 0

    def scale_callback(layer_idx, layer_kind, weight_role, out_ptr, out_nbytes):  # noqa: ANN001
        callback_counts["scale"] += 1
        name = f"model.layers.{int(layer_idx)}.mlp.experts.0.gate_proj.weight"
        tensor = reader.get_tensor(name)
        raw = _buffer_from_view(tensor.scale_bytes)
        out_ptr[0] = ctypes.cast(raw, ctypes.c_void_p)
        out_nbytes[0] = len(raw.raw) - 1
        return 0

    assert native_ds_forward_available() is True
    with DeepSeekNativeSession(num_layers=2, hidden_dim=4, num_experts=1, top_k=1) as session:
        session.register_layer(
            layer_idx=0,
            layer_kind=1,
            weight_role=7,
            fp8_callback=fp8_callback,
            scale_callback=scale_callback,
        )
        session.register_layer(
            layer_idx=1,
            layer_kind=1,
            weight_role=7,
            fp8_callback=fp8_callback,
            scale_callback=scale_callback,
        )

        assert session.registered_layer_count() == 2
        assert session.callback_invocation_count() == 4
        assert session.monolithic_call_count() == 0
        assert session.registered_fp8_nbytes(0) == 8
        assert session.registered_scale_nbytes(0) == 4

    assert callback_counts == {"fp8": 2, "scale": 2}


def test_ds_session_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_DS_MONOLITHIC", "1")
    from pcketlm.native import DeepSeekNativeSession, native_ds_forward_available

    assert native_ds_forward_available() is False
    with pytest.raises(RuntimeError, match="unavailable"):
        DeepSeekNativeSession(num_layers=2, hidden_dim=4, num_experts=1, top_k=1)


def _write_ds_pack_fixture(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": ("F8_E4M3", [2, 4], b"abcdefgh"),
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": ("F32", [1, 1], b"SCAL"),
        "model.layers.1.mlp.experts.0.gate_proj.weight": ("F8_E4M3", [2, 4], b"ABCDEFGH"),
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv": ("F32", [1, 1], b"scal"),
    }
    offset = 0
    header = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    header_bytes = json.dumps(header).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in tensors}}),
        encoding="utf-8",
    )
    return model_dir
