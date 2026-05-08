import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from pcketlm.core.acquisition.state import build_acquisition_snapshot


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
