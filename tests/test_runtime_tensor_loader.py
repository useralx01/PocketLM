import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from pcketlm.core.optimize import build_optimized_artifact_manifest
from pcketlm.core.runtime.tensor_execution_plan import build_tensor_execution_plan
from pcketlm.core.runtime.tensor_loader import (
    clear_runtime_pack_cache,
    load_execution_unit,
    load_tensor_by_name,
    load_tensors_by_name,
    reset_tensor_load_stats,
    runtime_pack_selection_snapshot,
    scoped_tensor_handle_cache,
    tensor_load_stats_snapshot,
    verify_execution_unit,
    verify_loaded_tensor,
)


def _bootstrap_tensor_fixture(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    model_id = "qwen-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)

    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 128,
                "vocab_size": 16,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (model_dir / "merges.txt").write_text("", encoding="utf-8")

    shard_1 = model_dir / "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": torch.arange(128, dtype=torch.bfloat16).reshape(16, 8),
            "model.layers.0.input_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
            "model.layers.0.post_attention_layernorm.weight": torch.full((8,), 2, dtype=torch.bfloat16),
            "model.layers.0.self_attn.q_proj.weight": torch.full((8, 8), 3, dtype=torch.bfloat16),
            "model.layers.0.mlp.gate_proj.weight": torch.full((8, 8), 4, dtype=torch.bfloat16),
            "model.norm.weight": torch.full((8,), 5, dtype=torch.bfloat16),
            "lm_head.weight": torch.full((16, 8), 6, dtype=torch.bfloat16),
        },
        str(shard_1),
    )

    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard_1.stat().st_size},
                "weight_map": {
                    "model.embed_tokens.weight": shard_1.name,
                    "model.layers.0.input_layernorm.weight": shard_1.name,
                    "model.layers.0.post_attention_layernorm.weight": shard_1.name,
                    "model.layers.0.self_attn.q_proj.weight": shard_1.name,
                    "model.layers.0.mlp.gate_proj.weight": shard_1.name,
                    "model.norm.weight": shard_1.name,
                    "lm_head.weight": shard_1.name,
                },
            }
        ),
        encoding="utf-8",
    )

    build_tensor_execution_plan(model_id, model_dir)
    return model_id, model_dir


def test_load_tensor_by_name_reads_real_tensor(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_tensor_fixture(tmp_path, monkeypatch)

    loaded = load_tensor_by_name(model_id, "model.layers.0.input_layernorm.weight")

    assert loaded.ready is True
    assert loaded.tensor is not None
    assert loaded.shape == [8]
    assert loaded.component_group == "layer_norm"
    assert torch.allclose(loaded.tensor.float(), torch.ones(8))


def test_load_tensor_by_name_prefers_runtime_pack_when_available(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_tensor_fixture(tmp_path, monkeypatch)
    build_optimized_artifact_manifest(model_id, profile_id="low-memory", materialize_small_pack=True)
    clear_runtime_pack_cache()
    reset_tensor_load_stats()

    loaded = load_tensor_by_name(model_id, "model.layers.0.input_layernorm.weight")
    stats = tensor_load_stats_snapshot()

    assert loaded.ready is True
    assert torch.allclose(loaded.tensor.float(), torch.ones(8))
    assert stats.artifact_tensor_hits == 1
    assert stats.artifact_pack_opens == 1
    assert stats.shard_opens == 0


def test_runtime_pack_selection_snapshot_reports_selected_pack(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_tensor_fixture(tmp_path, monkeypatch)
    build_optimized_artifact_manifest(model_id, profile_id="low-memory", materialize_small_pack=True)
    clear_runtime_pack_cache()

    snapshot = runtime_pack_selection_snapshot(model_id)

    assert snapshot["enabled"] is True
    assert snapshot["selected"] is True
    assert snapshot["packed_tensor_count"] > 0
    assert "runtime pack" in snapshot["summary"]


def test_load_tensors_by_name_reads_real_tensors_in_one_call(tmp_path: Path, monkeypatch) -> None:
    reset_tensor_load_stats()
    model_id, _model_dir = _bootstrap_tensor_fixture(tmp_path, monkeypatch)

    loaded = load_tensors_by_name(
        model_id,
        [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ],
    )

    assert set(loaded) == {
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    }
    assert all(result.ready for result in loaded.values())
    assert torch.allclose(loaded["model.layers.0.input_layernorm.weight"].tensor.float(), torch.ones(8))
    assert torch.allclose(
        loaded["model.layers.0.post_attention_layernorm.weight"].tensor.float(),
        torch.full((8,), 2.0),
    )
    stats = tensor_load_stats_snapshot()
    assert stats.batch_load_calls == 1
    assert stats.tensors_loaded == 2
    assert (stats.native_fp16_loads, stats.shard_opens) in {(2, 0), (0, 1)}


def test_scoped_tensor_handle_cache_reuses_handle_across_load_calls(tmp_path: Path, monkeypatch) -> None:
    reset_tensor_load_stats()
    model_id, _model_dir = _bootstrap_tensor_fixture(tmp_path, monkeypatch)
    calls = {"opens": 0}
    tensors = {
        "model.layers.0.input_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.full((8,), 2, dtype=torch.bfloat16),
    }

    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_tensor(self, tensor_name: str) -> torch.Tensor:
            return tensors[tensor_name]

    def fake_safe_open(*_args, **_kwargs):
        calls["opens"] += 1
        return FakeHandle()

    monkeypatch.setattr("pcketlm.core.runtime.tensor_loader.safe_open", fake_safe_open)

    with scoped_tensor_handle_cache():
        first = load_tensor_by_name(model_id, "model.layers.0.input_layernorm.weight")
        second = load_tensor_by_name(model_id, "model.layers.0.post_attention_layernorm.weight")

    assert first.ready is True
    assert second.ready is True
    assert calls["opens"] == 1
    stats = tensor_load_stats_snapshot()
    assert stats.single_load_calls == 2
    assert stats.shard_opens == 1
    assert stats.scoped_handle_reuses == 1
    assert first.borrowed_from_live_handle is True
    assert second.borrowed_from_live_handle is True


def test_persistent_tensor_handle_cache_reuses_handle_across_calls(tmp_path: Path, monkeypatch) -> None:
    reset_tensor_load_stats()
    model_id, _model_dir = _bootstrap_tensor_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("PCKETLM_SAFETENSOR_HANDLE_CACHE", "1")
    monkeypatch.delenv("PCKETLM_DISABLE_PERSISTENT_HANDLES", raising=False)
    calls = {"opens": 0}
    tensors = {
        "model.layers.0.input_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.full((8,), 2, dtype=torch.bfloat16),
    }

    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_tensor(self, tensor_name: str) -> torch.Tensor:
            return tensors[tensor_name]

    def fake_safe_open(*_args, **_kwargs):
        calls["opens"] += 1
        return FakeHandle()

    monkeypatch.setattr("pcketlm.core.runtime.tensor_loader.safe_open", fake_safe_open)

    first = load_tensor_by_name(model_id, "model.layers.0.input_layernorm.weight")
    second = load_tensor_by_name(model_id, "model.layers.0.post_attention_layernorm.weight")

    assert first.ready is True
    assert second.ready is True
    assert calls["opens"] == 1
    assert first.borrowed_from_live_handle is True
    assert second.borrowed_from_live_handle is True
    stats = tensor_load_stats_snapshot()
    assert stats.shard_opens == 1
    assert stats.persistent_handle_reuses == 1


def test_load_execution_unit_reads_grouped_runtime_unit(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_tensor_fixture(tmp_path, monkeypatch)

    loaded = load_execution_unit(model_id, "layer-00-layer_norm")

    assert loaded.ready is True
    assert loaded.tensor_count == 2
    assert loaded.phase == "layer-entry"
    assert {tensor.tensor_name for tensor in loaded.tensors} == {
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    }
    assert loaded.total_nbytes > 0


def test_verify_loaded_tensor_matches_catalog_metadata(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_tensor_fixture(tmp_path, monkeypatch)

    verified = verify_loaded_tensor(model_id, "model.layers.0.self_attn.q_proj.weight")

    assert verified.ready is True
    assert verified.expected_dtype == "torch.bfloat16"
    assert verified.actual_dtype == "torch.bfloat16"
    assert verified.expected_shape == [8, 8]
    assert verified.actual_shape == [8, 8]
    assert verified.expected_nbytes == verified.actual_nbytes


def test_verify_execution_unit_matches_plan_metadata(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_tensor_fixture(tmp_path, monkeypatch)

    verified = verify_execution_unit(model_id, "layer-00-layer_norm")

    assert verified.ready is True
    assert verified.tensor_count == 2
    assert verified.total_nbytes > 0
    assert all(result.ready for result in verified.tensor_results)
