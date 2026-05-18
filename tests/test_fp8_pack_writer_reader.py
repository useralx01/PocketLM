import json
import struct
from pathlib import Path

from pcketlm.core.runtime.fp8_pack import FP8PackReader
from tools.pack_fp8 import pack_model_dir_to_fp8


def test_pack_writer_preserves_bytes_and_reader_returns_views(tmp_path: Path, monkeypatch) -> None:
    import tools.pack_fp8 as pack_tool

    monkeypatch.setattr(pack_tool, "state_root", lambda: tmp_path / "state")
    model_dir = _write_pack_fixture(tmp_path)
    output_dir = model_dir / "artifacts" / "fp8_pack"

    state = pack_model_dir_to_fp8(model_dir, output_dir, model_id="pack-test", pack_bytes=128)
    reader = FP8PackReader(model_dir)

    tensor = reader.get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")

    assert state["status"] == "complete"
    assert bytes(tensor.fp8_bytes) == b"abcdefgh"
    assert bytes(tensor.scale_bytes) == b"SCAL"
    assert reader.telemetry()["sequential_reads"] == 1


def test_pack_reader_reports_native_slice_and_span_locations(tmp_path: Path, monkeypatch) -> None:
    import tools.pack_fp8 as pack_tool

    monkeypatch.setattr(pack_tool, "state_root", lambda: tmp_path / "state")
    model_dir = _write_pack_fixture(tmp_path)
    output_dir = model_dir / "artifacts" / "fp8_pack"

    pack_model_dir_to_fp8(model_dir, output_dir, model_id="span-pack-test", pack_bytes=128)
    reader = FP8PackReader(model_dir)

    slice_location = reader.get_tensor_slice_location(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        2,
        3,
    )
    span_location = reader.get_tensor_span_location(
        [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
        ]
    )
    with slice_location.path.open("rb") as handle:
        handle.seek(slice_location.byte_offset)
        slice_bytes = handle.read(slice_location.byte_length)
    with span_location.path.open("rb") as handle:
        handle.seek(span_location.byte_offset)
        span_bytes = handle.read(span_location.byte_length)

    assert slice_bytes == b"cde"
    assert span_bytes == b"abcdefghSCAL"
    assert span_location.tensor_slices["model.layers.0.mlp.experts.0.gate_proj.weight"] == (0, 8)
    assert span_location.tensor_slices["model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"] == (8, 4)
    assert reader.telemetry()["sequential_reads"] == 2


def test_pack_writer_resumes_without_rewriting_finalized_tensors(tmp_path: Path, monkeypatch) -> None:
    import tools.pack_fp8 as pack_tool

    monkeypatch.setattr(pack_tool, "state_root", lambda: tmp_path / "state")
    model_dir = _write_pack_fixture(tmp_path)
    output_dir = model_dir / "artifacts" / "fp8_pack"

    first = pack_model_dir_to_fp8(
        model_dir,
        output_dir,
        model_id="resume-pack-test",
        pack_bytes=128,
        max_new_tensors=1,
    )
    first_pack = output_dir / "pack_0000.bin"
    first_bytes = first_pack.read_bytes()
    second = pack_model_dir_to_fp8(model_dir, output_dir, model_id="resume-pack-test", pack_bytes=128)
    manifest = json.loads((output_dir / "pack_manifest.json").read_text(encoding="utf-8"))

    assert first["status"] == "paused"
    assert first["completed_tensor_count"] == 1
    assert second["status"] == "complete"
    assert first_pack.read_bytes() == first_bytes
    assert len(manifest["tensors"]) == 5
    assert bytes(FP8PackReader(model_dir).get_tensor("lm_head.weight").fp8_bytes) == b"LMHD"


def _write_pack_fixture(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": ("F8_E4M3", [2, 4], b"abcdefgh"),
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": ("F32", [1, 1], b"SCAL"),
        "model.layers.0.self_attn.q_a_proj.weight": ("F8_E4M3", [2, 4], b"ABCDEFGH"),
        "model.layers.0.self_attn.q_a_proj.weight_scale_inv": ("F32", [1, 1], b"scal"),
        "lm_head.weight": ("BF16", [2, 2], b"LMHD"),
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
