import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

from pcketlm.app.chat_shell.acquisition_cli import format_plain_snapshot
from pcketlm.core.acquisition.state import build_acquisition_snapshot
from pcketlm.core.runtime.tensor_catalog import build_tensor_catalog


def test_build_acquisition_snapshot_partial(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    snapshot = build_acquisition_snapshot(tmp_path)

    assert snapshot.status == "partial"
    assert snapshot.bytes_on_disk_gb >= 0
    assert snapshot.progress_bar.startswith("[")
    assert "not usable yet" in snapshot.plain_english_summary
    assert "recheck acquisition state" in snapshot.recommended_next_step


def test_build_acquisition_snapshot_missing(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    snapshot = build_acquisition_snapshot(missing_dir)

    assert snapshot.status == "missing"
    assert snapshot.progress_bar == "[" + ("?" * 24) + "]"
    assert "does not exist yet" in snapshot.plain_english_summary


def test_build_acquisition_snapshot_ready_includes_compact_q4_plan(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file(
        {"model.layers.0.block_sparse_moe.experts.0.gate_proj.weight": torch.ones((8, 8), dtype=torch.bfloat16)},
        str(shard),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {
                    "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight": shard.name,
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = build_acquisition_snapshot(tmp_path)

    assert snapshot.status == "ready"
    assert snapshot.compact_q4_plan is not None
    assert snapshot.compact_q4_plan["ready_for_conversion"] is True
    assert snapshot.compact_q4_plan["estimated_expert_q4_bytes"] > 0
    assert "compact Q4 artifact first" in snapshot.recommended_next_step


def test_build_acquisition_snapshot_fp8_recommends_paged_runtime_not_q4(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"num_experts_per_tok": 1}), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    shard = tmp_path / "model-00001-of-00001.safetensors"
    tensor_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    scale_name = f"{tensor_name}_scale_inv"
    _write_header_only_safetensors(
        shard,
        {
            tensor_name: ("F8_E4M3", [128, 128]),
            scale_name: ("F32", [1, 1]),
        },
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {tensor_name: shard.name, scale_name: shard.name},
            }
        ),
        encoding="utf-8",
    )

    snapshot = build_acquisition_snapshot(tmp_path)

    assert snapshot.status == "ready"
    assert snapshot.compact_q4_plan is not None
    assert snapshot.compact_q4_plan["fp8_native"] is True
    assert snapshot.compact_q4_plan["ready_for_conversion"] is False
    assert "FP8 paged runtime planning" in snapshot.recommended_next_step


def test_build_acquisition_snapshot_fp8_includes_runtime_status(
    tmp_path: Path, monkeypatch
) -> None:
    from pcketlm.core import storage

    project_root = tmp_path / "project"
    monkeypatch.setattr(storage.paths, "project_root", lambda: project_root)
    model_dir = tmp_path / "models" / "deepseek-v3-test" / "original"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "first_k_dense_replace": 1,
                "num_experts_per_tok": 1,
                "quantization_config": {"fmt": "e4m3", "quant_method": "fp8"},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensor_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    scale_name = f"{tensor_name}_scale_inv"
    _write_header_only_safetensors(
        shard,
        {
            tensor_name: ("F8_E4M3", [128, 128]),
            scale_name: ("F32", [1, 1]),
        },
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {tensor_name: shard.name, scale_name: shard.name},
            }
        ),
        encoding="utf-8",
    )
    build_tensor_catalog("deepseek-v3-test", model_dir)

    snapshot = build_acquisition_snapshot(model_dir)

    assert snapshot.fp8_runtime_status is not None
    assert snapshot.fp8_runtime_status["ready"] is True
    assert snapshot.fp8_runtime_status["runtime_policy"]["weight_residency"] == "paged_fp8_source"
    assert snapshot.fp8_runtime_status["runtime_policy"]["layer_count"] == 1
    assert "Use the FP8 paged runtime path" in snapshot.recommended_next_step


def test_acquisition_plain_summary_includes_fp8_runtime_status(
    tmp_path: Path, monkeypatch
) -> None:
    from pcketlm.core import storage

    project_root = tmp_path / "project"
    monkeypatch.setattr(storage.paths, "project_root", lambda: project_root)
    model_dir = tmp_path / "models" / "deepseek-v3-test" / "original"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "num_experts_per_tok": 1,
                "quantization_config": {"fmt": "e4m3", "quant_method": "fp8"},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensor_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    scale_name = f"{tensor_name}_scale_inv"
    _write_header_only_safetensors(
        shard,
        {
            tensor_name: ("F8_E4M3", [128, 128]),
            scale_name: ("F32", [1, 1]),
        },
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {tensor_name: shard.name, scale_name: shard.name},
            }
        ),
        encoding="utf-8",
    )
    build_tensor_catalog("deepseek-v3-test", model_dir)
    snapshot = build_acquisition_snapshot(model_dir)

    summary = format_plain_snapshot(snapshot)

    assert "Status: ready" in summary
    assert "on disk" in summary
    assert "FP8 runtime: ready" in summary
    assert "path paged_fp8_source" in summary
    assert "blockers none" in summary


def _write_header_only_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    offset = 0
    header = {}
    for name, (dtype, shape) in tensors.items():
        size = {"F8_E4M3": 1, "F32": 4}[dtype]
        for value in shape:
            size *= int(value)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)
