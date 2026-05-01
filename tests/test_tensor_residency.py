from pathlib import Path
from types import SimpleNamespace

import torch

from pcketlm.core.runtime.tensor_catalog import TensorCatalogEntry
from pcketlm.core.runtime.tensor_loader import LoadedTensorSlice
from pcketlm.core.runtime.tensor_residency import (
    TensorResidencyPolicy,
    advance_tensor_residency_step,
    clear_tensor_residency_cache,
    current_tensor_residency_step,
    expert_residency_snapshot,
    load_resident_tensor,
    load_resident_tensors,
    record_expert_activation,
    tensor_residency_stats,
)
from pcketlm.core.runtime.tensor_residency import _dequantize_q4_tensor, _quantize_q4_tensor


def _entry(tmp_path: Path, tensor_name: str = "model.layers.0.self_attn.k_proj.weight") -> TensorCatalogEntry:
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"test")
    return TensorCatalogEntry(
        tensor_name=tensor_name,
        shard_name=shard_path.name,
        shard_path=shard_path,
        dtype="BF16",
        shape=[2, 2],
        data_offset_start=0,
        data_offset_end=8,
        data_nbytes=8,
        layer_index=0,
        component_group="attention",
    )


def _loaded_tensor(model_id: str, entry: TensorCatalogEntry, value: float = 1.0) -> LoadedTensorSlice:
    tensor = torch.full((2, 2), value, dtype=torch.bfloat16)
    return LoadedTensorSlice(
        model_id=model_id,
        tensor_name=entry.tensor_name,
        shard_name=entry.shard_name,
        dtype=str(tensor.dtype),
        shape=[2, 2],
        tensor=tensor,
        layer_index=entry.layer_index,
        component_group=entry.component_group,
        loaded_nbytes=tensor.element_size() * tensor.nelement(),
        blockers=[],
        ready=True,
    )


def test_load_resident_tensor_reuses_converted_tensor_when_within_policy(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "resident-cache-test"
    entry = _entry(tmp_path)
    calls = {"count": 0}

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        calls["count"] += 1
        assert model_id_arg == model_id
        assert tensor_name == entry.tensor_name
        return _loaded_tensor(model_id_arg, entry)

    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._find_tensor_entry", lambda *_args: entry)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(max_resident_bytes=1024, max_tensor_bytes=1024)
    first = load_resident_tensor(model_id, entry.tensor_name, policy=policy)
    second = load_resident_tensor(model_id, entry.tensor_name, policy=policy)
    stats = tensor_residency_stats()

    assert first.ready is True
    assert second.ready is True
    assert first.tensor is second.tensor
    assert second.dtype == "torch.float32"
    assert calls["count"] == 1
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.stores == 1
    assert stats.resident_count == 1


def test_large_attention_tensor_is_evictable_under_memory_pressure(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "resident-large-attention-test"
    entries = {
        "a": _entry(tmp_path, tensor_name="model.layers.0.self_attn.q_proj.weight"),
        "b": _entry(tmp_path, tensor_name="model.layers.1.self_attn.q_proj.weight"),
    }
    for entry in entries.values():
        entry.component_group = "attention"
        entry.data_nbytes = 1024
    calls: list[str] = []

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        calls.append(tensor_name)
        entry = entries[tensor_name]
        tensor = torch.ones((512,), dtype=torch.bfloat16)
        return LoadedTensorSlice(
            model_id=model_id_arg,
            tensor_name=entry.tensor_name,
            shard_name=entry.shard_name,
            dtype=str(tensor.dtype),
            shape=[512],
            tensor=tensor,
            layer_index=entry.layer_index,
            component_group=entry.component_group,
            loaded_nbytes=tensor.element_size() * tensor.nelement(),
            blockers=[],
            ready=True,
        )

    monkeypatch.setenv("PCKETLM_ALWAYS_RESIDENT_TENSOR_MB", "0")
    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_residency._find_tensor_entry",
        lambda _model_id, tensor_name: entries.get(tensor_name),
    )
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(max_resident_bytes=2048, max_tensor_bytes=2048, sticky_residency_steps=0)
    first = load_resident_tensor(model_id, "a", policy=policy)
    second = load_resident_tensor(model_id, "b", policy=policy)
    stats = tensor_residency_stats()

    assert first.ready is True
    assert second.ready is True
    assert calls == ["a", "b"]
    assert stats.evictions == 1
    assert stats.resident_bytes <= 2048
    assert stats.resident_count == 1


def test_sticky_residency_prefers_evicting_stale_tensor_over_recent_tensor(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "resident-sticky-test"
    entries = {
        "a": _entry(tmp_path, tensor_name="a"),
        "b": _entry(tmp_path, tensor_name="b"),
        "c": _entry(tmp_path, tensor_name="c"),
    }
    for entry in entries.values():
        entry.component_group = "mlp"
    calls: list[str] = []

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        calls.append(tensor_name)
        return _loaded_tensor(model_id_arg, entries[tensor_name], value=float(len(calls)))

    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_residency._find_tensor_entry",
        lambda _model_id, tensor_name: entries.get(tensor_name),
    )
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(max_resident_bytes=32, max_tensor_bytes=1024, sticky_residency_steps=1)
    load_resident_tensor(model_id, "a", policy=policy)
    advance_tensor_residency_step()
    load_resident_tensor(model_id, "b", policy=policy)
    load_resident_tensor(model_id, "a", policy=policy)
    advance_tensor_residency_step()
    load_resident_tensor(model_id, "c", policy=policy)

    stats = tensor_residency_stats()
    load_resident_tensor(model_id, "b", policy=policy)
    load_resident_tensor(model_id, "a", policy=policy)

    assert current_tensor_residency_step() == 2
    assert stats.evictions == 1
    assert calls == ["a", "b", "c", "a"]


def test_live_handle_tensor_can_skip_hot_path_clone(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "resident-zero-copy-test"
    entry = _entry(tmp_path, tensor_name="borrowed")
    source_tensor = torch.ones((2, 2), dtype=torch.bfloat16)
    loaded = LoadedTensorSlice(
        model_id=model_id,
        tensor_name="borrowed",
        shard_name=entry.shard_name,
        dtype=str(source_tensor.dtype),
        shape=[2, 2],
        tensor=source_tensor,
        layer_index=entry.layer_index,
        component_group=entry.component_group,
        loaded_nbytes=source_tensor.nelement() * source_tensor.element_size(),
        ready=True,
        borrowed_from_live_handle=True,
    )

    monkeypatch.delenv("PCKETLM_DISABLE_ZERO_COPY_TENSORS", raising=False)
    monkeypatch.setenv("PCKETLM_ENABLE_ZERO_COPY_TENSORS", "1")
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._find_tensor_entry", lambda *_args: entry)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", lambda *_args: loaded)

    result = load_resident_tensor(
        model_id,
        "borrowed",
        dtype=torch.bfloat16,
        policy=TensorResidencyPolicy(enabled=False),
    )

    assert result.tensor is not None
    assert result.tensor.data_ptr() == source_tensor.data_ptr()


def test_zero_copy_kill_switch_restores_clone_for_live_handle_tensor(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "resident-zero-copy-off-test"
    entry = _entry(tmp_path, tensor_name="borrowed")
    source_tensor = torch.ones((2, 2), dtype=torch.bfloat16)
    loaded = LoadedTensorSlice(
        model_id=model_id,
        tensor_name="borrowed",
        shard_name=entry.shard_name,
        dtype=str(source_tensor.dtype),
        shape=[2, 2],
        tensor=source_tensor,
        layer_index=entry.layer_index,
        component_group=entry.component_group,
        loaded_nbytes=source_tensor.nelement() * source_tensor.element_size(),
        ready=True,
        borrowed_from_live_handle=True,
    )

    monkeypatch.setenv("PCKETLM_DISABLE_ZERO_COPY_TENSORS", "1")
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._find_tensor_entry", lambda *_args: entry)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", lambda *_args: loaded)

    result = load_resident_tensor(
        model_id,
        "borrowed",
        dtype=torch.bfloat16,
        policy=TensorResidencyPolicy(enabled=False),
    )

    assert result.tensor is not None
    assert result.tensor.data_ptr() != source_tensor.data_ptr()


def test_expert_residency_evicts_cold_expert_before_hot_expert(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "expert-residency-test"
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"test")
    entries = {
        f"expert-{index}": TensorCatalogEntry(
            tensor_name=f"model.layers.0.mlp.experts.{index}.gate_proj.weight",
            shard_name=shard_path.name,
            shard_path=shard_path,
            dtype="BF16",
            shape=[2, 2],
            data_offset_start=0,
            data_offset_end=8,
            data_nbytes=8,
            layer_index=0,
            component_group="expert_mlp",
            expert_index=index,
        )
        for index in range(3)
    }

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        entry = entries[tensor_name]
        return _loaded_tensor(model_id_arg, entry, value=float(entry.expert_index + 1))

    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_residency._find_tensor_entry",
        lambda _model_id, tensor_name: entries.get(tensor_name),
    )
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)
    policy = TensorResidencyPolicy(
        max_resident_bytes=64,
        max_tensor_bytes=1024,
        expert_max_resident_bytes=32,
        sticky_residency_steps=0,
    )

    record_expert_activation(0, 0)
    record_expert_activation(0, 0)
    load_resident_tensor(model_id, "expert-0", policy=policy)
    advance_tensor_residency_step()
    record_expert_activation(0, 1)
    load_resident_tensor(model_id, "expert-1", policy=policy)
    advance_tensor_residency_step()
    record_expert_activation(0, 2)
    load_resident_tensor(model_id, "expert-2", policy=policy)

    stats = tensor_residency_stats()
    snapshot = expert_residency_snapshot()
    hot = load_resident_tensor(model_id, "expert-0", policy=policy)
    cold = load_resident_tensor(model_id, "expert-1", policy=policy)

    assert stats.expert_evictions == 1
    assert snapshot["expert_resident_count"] == 2
    assert hot.tensor is not None
    assert cold.tensor is not None
    assert tensor_residency_stats().expert_hits >= 1
    assert tensor_residency_stats().expert_misses >= 4


def test_expert_residency_snapshot_reports_hit_miss_and_touch_rankings(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "expert-telemetry-test"
    hot_name = "model.layers.0.mlp.experts.5.gate_proj.weight"
    cold_name = "model.layers.0.mlp.experts.12.gate_proj.weight"
    hot_entry = _entry(tmp_path, hot_name)
    hot_entry.component_group = "expert_mlp"
    hot_entry.expert_index = 5
    cold_entry = _entry(tmp_path, cold_name)
    cold_entry.component_group = "expert_mlp"
    cold_entry.expert_index = 12
    entries = {hot_name: hot_entry, cold_name: cold_entry}
    calls = {"count": 0}

    def fake_load_tensors_by_name(model_id_arg: str, tensor_names: list[str]) -> dict[str, LoadedTensorSlice]:
        calls["count"] += 1
        return {tensor_name: _loaded_tensor(model_id_arg, entries[tensor_name]) for tensor_name in tensor_names}

    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_residency._find_tensor_entry",
        lambda _model_id, tensor_name: entries.get(tensor_name),
    )
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensors_by_name", fake_load_tensors_by_name)

    policy = TensorResidencyPolicy(
        max_resident_bytes=1024,
        max_tensor_bytes=1024,
        all_layer_small_tensor_bytes=0,
        front_layer_count=1,
        expert_max_resident_bytes=1024,
    )
    record_expert_activation(0, 5)
    record_expert_activation(0, 5)
    record_expert_activation(0, 12)
    load_resident_tensors(model_id, [hot_name, cold_name], policy=policy)
    load_resident_tensors(model_id, [hot_name], policy=policy)

    snapshot = expert_residency_snapshot(top_k=1)

    assert snapshot["total_expert_requests"] == 3
    assert snapshot["expert_hits"] == 1
    assert snapshot["expert_misses"] == 2
    assert snapshot["expert_hit_rate"] == 0.3333
    assert snapshot["expert_activation_total"] == 3
    assert snapshot["top_touched_experts"][0]["layer"] == 0
    assert snapshot["top_touched_experts"][0]["expert"] == 5
    assert snapshot["top_touched_experts"][0]["touches"] == 2
    assert snapshot["top_touched_experts"][0]["score"] > snapshot["least_touched_experts"][0]["score"]
    assert snapshot["least_touched_experts"][0]["layer"] == 0
    assert snapshot["least_touched_experts"][0]["expert"] == 12
    assert snapshot["least_touched_experts"][0]["touches"] == 1
    assert snapshot["expert_resident_count"] == 2
    assert calls["count"] == 1


def test_expert_residency_respects_zero_expert_budget(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "expert-zero-budget-test"
    expert_name = "model.layers.0.mlp.experts.5.gate_proj.weight"
    entry = _entry(tmp_path, expert_name)
    entry.component_group = "expert_mlp"
    entry.expert_index = 5

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        assert tensor_name == expert_name
        return _loaded_tensor(model_id_arg, entry)

    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._find_tensor_entry", lambda *_args: entry)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(
        max_resident_bytes=1024,
        max_tensor_bytes=1024,
        all_layer_small_tensor_bytes=0,
        front_layer_count=1,
        expert_max_resident_bytes=0,
    )
    record_expert_activation(0, 5)
    load_resident_tensor(model_id, expert_name, policy=policy)
    load_resident_tensor(model_id, expert_name, policy=policy)
    snapshot = expert_residency_snapshot()

    assert snapshot["expert_hits"] == 0
    assert snapshot["expert_misses"] == 2
    assert snapshot["expert_hit_rate"] == 0.0
    assert snapshot["expert_resident_count"] == 0


def test_q4_expert_residency_round_trips_known_tensor() -> None:
    tensor = torch.linspace(-1.0, 1.0, steps=128, dtype=torch.float32).reshape(8, 16)

    q4 = _quantize_q4_tensor(tensor)
    restored = _dequantize_q4_tensor(q4)

    assert restored is not None
    assert restored.shape == tensor.shape
    assert torch.nn.functional.cosine_similarity(tensor.flatten(), restored.flatten(), dim=0).item() > 0.995
    assert q4.nbytes <= tensor.element_size() * tensor.nelement()


def test_q4_expert_residency_holds_more_expert_tensors_under_same_budget(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "expert-q4-budget-test"
    entries = {}
    for index in range(4):
        entry = _entry(tmp_path, f"model.layers.0.mlp.experts.{index}.gate_proj.weight")
        entry.component_group = "expert_mlp"
        entry.expert_index = index
        entry.shape = [64, 64]
        entry.data_nbytes = 64 * 64 * 4
        entries[entry.tensor_name] = entry

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        entry = entries[tensor_name]
        tensor = torch.linspace(-1.0, 1.0, steps=64 * 64, dtype=torch.float32).reshape(64, 64)
        return LoadedTensorSlice(
            model_id=model_id_arg,
            tensor_name=entry.tensor_name,
            shard_name=entry.shard_name,
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
            tensor=tensor + float(entry.expert_index),
            layer_index=entry.layer_index,
            component_group=entry.component_group,
            loaded_nbytes=tensor.element_size() * tensor.nelement(),
            blockers=[],
            ready=True,
        )

    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_residency._find_tensor_entry",
        lambda _model_id, tensor_name: entries.get(tensor_name),
    )
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    fp_policy = TensorResidencyPolicy(
        max_resident_bytes=128 * 1024,
        max_tensor_bytes=128 * 1024,
        expert_max_resident_bytes=32 * 1024,
        max_resident_experts_per_layer=8,
    )
    for tensor_name in entries:
        record_expert_activation(0, entries[tensor_name].expert_index)
        load_resident_tensor(model_id, tensor_name, dtype=torch.float32, policy=fp_policy)
    fp_count = expert_residency_snapshot()["expert_resident_count"]

    clear_tensor_residency_cache()
    q4_policy = TensorResidencyPolicy(
        max_resident_bytes=128 * 1024,
        max_tensor_bytes=128 * 1024,
        expert_max_resident_bytes=32 * 1024,
        max_resident_experts_per_layer=8,
        expert_q4_residency=True,
    )
    for tensor_name in entries:
        record_expert_activation(0, entries[tensor_name].expert_index)
        load_resident_tensor(model_id, tensor_name, dtype=torch.float32, policy=q4_policy)
    q4_count = expert_residency_snapshot()["expert_resident_count"]

    assert fp_count < 4
    assert q4_count == 4


def test_expert_residency_uses_separate_budget_from_dense_cache(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "expert-separate-budget-test"
    expert_name = "model.layers.0.mlp.experts.5.gate_proj.weight"
    entry = _entry(tmp_path, expert_name)
    entry.component_group = "expert_mlp"
    entry.expert_index = 5

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        return _loaded_tensor(model_id_arg, entry)

    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._find_tensor_entry", lambda *_args: entry)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(
        max_resident_bytes=1,
        max_tensor_bytes=1024,
        all_layer_small_tensor_bytes=0,
        front_layer_count=1,
        expert_max_resident_bytes=1024,
    )
    record_expert_activation(0, 5)
    first = load_resident_tensor(model_id, expert_name, policy=policy)
    second = load_resident_tensor(model_id, expert_name, policy=policy)
    snapshot = expert_residency_snapshot()

    assert first.ready is True
    assert second.ready is True
    assert snapshot["expert_hits"] == 1
    assert snapshot["expert_misses"] == 1
    assert snapshot["expert_resident_count"] == 1


def test_expert_residency_allows_large_expert_tensors_with_expert_budget(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "expert-large-budget-test"
    expert_name = "model.layers.0.block_sparse_moe.experts.5.w1.weight"
    entry = _entry(tmp_path, expert_name)
    entry.component_group = "expert_mlp"
    entry.expert_index = 5

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        tensor = torch.ones((8, 8), dtype=torch.bfloat16)
        return LoadedTensorSlice(
            model_id=model_id_arg,
            tensor_name=tensor_name,
            shard_name=entry.shard_name,
            dtype=str(tensor.dtype),
            shape=[8, 8],
            tensor=tensor,
            layer_index=entry.layer_index,
            component_group=entry.component_group,
            loaded_nbytes=tensor.element_size() * tensor.nelement(),
            blockers=[],
            ready=True,
        )

    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._find_tensor_entry", lambda *_args: entry)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(
        max_resident_bytes=0,
        max_tensor_bytes=4,
        all_layer_small_tensor_bytes=0,
        front_layer_count=0,
        expert_max_resident_bytes=1024,
    )
    record_expert_activation(0, 5)
    first = load_resident_tensor(model_id, expert_name, policy=policy)
    second = load_resident_tensor(model_id, expert_name, policy=policy)
    snapshot = expert_residency_snapshot()

    assert first.ready is True
    assert second.ready is True
    assert snapshot["expert_hits"] == 1
    assert snapshot["expert_misses"] == 1
    assert snapshot["expert_resident_count"] == 1


def test_expert_residency_decay_lets_new_hot_expert_replace_old_one(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    monkeypatch.setenv("PCKETLM_EXPERT_CACHE_DECAY", "0.5")
    model_id = "expert-decay-test"
    names = {
        5: "model.layers.0.mlp.experts.5.gate_proj.weight",
        12: "model.layers.0.mlp.experts.12.gate_proj.weight",
    }
    entries = {}
    for expert_index, tensor_name in names.items():
        entry = _entry(tmp_path, tensor_name)
        entry.component_group = "expert_mlp"
        entry.expert_index = expert_index
        entries[tensor_name] = entry

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        return _loaded_tensor(model_id_arg, entries[tensor_name])

    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_residency._find_tensor_entry",
        lambda _model_id, tensor_name: entries.get(tensor_name),
    )
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(
        max_resident_bytes=1024,
        max_tensor_bytes=1024,
        all_layer_small_tensor_bytes=0,
        front_layer_count=1,
        expert_max_resident_bytes=16,
        expert_decay_rate=0.5,
    )
    record_expert_activation(0, 5)
    load_resident_tensor(model_id, names[5], policy=policy)
    advance_tensor_residency_step()
    for _ in range(50):
        record_expert_activation(0, 12)
    load_resident_tensor(model_id, names[12], policy=policy)

    new = load_resident_tensor(model_id, names[12], policy=policy)
    old = load_resident_tensor(model_id, names[5], policy=policy)

    assert old.tensor is not None
    assert new.tensor is not None
    assert tensor_residency_stats().expert_misses >= 3
    assert tensor_residency_stats().expert_hits >= 1


def test_expert_residency_per_layer_cap_evicts_lower_score_expert(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "expert-layer-cap-test"
    names = {
        5: "model.layers.0.mlp.experts.5.gate_proj.weight",
        12: "model.layers.0.mlp.experts.12.gate_proj.weight",
    }
    entries = {}
    for expert_index, tensor_name in names.items():
        entry = _entry(tmp_path, tensor_name)
        entry.component_group = "expert_mlp"
        entry.expert_index = expert_index
        entries[tensor_name] = entry

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        return _loaded_tensor(model_id_arg, entries[tensor_name])

    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_residency._find_tensor_entry",
        lambda _model_id, tensor_name: entries.get(tensor_name),
    )
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(
        max_resident_bytes=1024,
        max_tensor_bytes=1024,
        all_layer_small_tensor_bytes=0,
        front_layer_count=1,
        expert_max_resident_bytes=1024,
        max_resident_experts_per_layer=1,
    )
    record_expert_activation(0, 5)
    load_resident_tensor(model_id, names[5], policy=policy)
    advance_tensor_residency_step()
    record_expert_activation(0, 12)
    record_expert_activation(0, 12)
    load_resident_tensor(model_id, names[12], policy=policy)
    load_resident_tensor(model_id, names[5], policy=policy)
    load_resident_tensor(model_id, names[12], policy=policy)

    assert tensor_residency_stats().expert_evictions >= 1
    assert tensor_residency_stats().expert_hits >= 1


def test_expert_residency_hard_budget_can_evict_current_step_experts(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "expert-hard-budget-test"
    names = {
        5: "model.layers.0.mlp.experts.5.gate_proj.weight",
        12: "model.layers.0.mlp.experts.12.gate_proj.weight",
    }
    entries = {}
    for expert_index, tensor_name in names.items():
        entry = _entry(tmp_path, tensor_name)
        entry.component_group = "expert_mlp"
        entry.expert_index = expert_index
        entries[tensor_name] = entry

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        return _loaded_tensor(model_id_arg, entries[tensor_name])

    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_residency._find_tensor_entry",
        lambda _model_id, tensor_name: entries.get(tensor_name),
    )
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(
        max_resident_bytes=1024,
        max_tensor_bytes=1024,
        all_layer_small_tensor_bytes=0,
        front_layer_count=1,
        expert_max_resident_bytes=16,
        max_resident_experts_per_layer=8,
    )
    record_expert_activation(0, 5)
    record_expert_activation(0, 12)
    load_resident_tensor(model_id, names[5], policy=policy)
    load_resident_tensor(model_id, names[12], policy=policy)

    snapshot = expert_residency_snapshot()

    assert snapshot["expert_resident_bytes"] <= 16
    assert snapshot["expert_resident_count"] == 1
    assert tensor_residency_stats().expert_evictions == 1


def test_default_tensor_residency_policy_stays_standard_when_memory_has_headroom(monkeypatch) -> None:
    clear_tensor_residency_cache()
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_MB", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_PRESET", raising=False)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._free_memory_bytes", lambda: 8 * 1024**3)

    policy = TensorResidencyPolicy.from_environment()

    assert policy.tensor_cache_preset == "standard"
    assert policy.adaptive_boost_active is False
    assert policy.front_layer_count == 12
    assert policy.max_resident_bytes == 256 * 1024**2
    assert policy.max_tensor_bytes == 32 * 1024**2


def test_tensor_residency_policy_boosts_when_preset_is_selected(monkeypatch) -> None:
    clear_tensor_residency_cache()
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_MB", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", raising=False)
    monkeypatch.setenv("PCKETLM_TENSOR_CACHE_PRESET", "boosted")
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._free_memory_bytes", lambda: 8 * 1024**3)

    policy = TensorResidencyPolicy.from_environment()

    assert policy.tensor_cache_preset == "boosted"
    assert policy.adaptive_boost_active is True
    assert policy.front_layer_count == 13
    assert policy.max_resident_bytes == 288 * 1024**2
    assert policy.max_tensor_bytes == 32 * 1024**2


def test_default_tensor_residency_policy_stays_conservative_without_headroom(monkeypatch) -> None:
    clear_tensor_residency_cache()
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_MB", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_PRESET", raising=False)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._free_memory_bytes", lambda: 6 * 1024**3)

    policy = TensorResidencyPolicy.from_environment()

    assert policy.tensor_cache_preset == "standard"
    assert policy.adaptive_boost_active is False
    assert policy.front_layer_count == 12
    assert policy.max_resident_bytes == 256 * 1024**2


def test_tensor_residency_policy_reduces_cache_when_memory_is_low(monkeypatch) -> None:
    clear_tensor_residency_cache()
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_MB", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_PRESET", raising=False)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._free_memory_bytes", lambda: 1 * 1024**3)

    policy = TensorResidencyPolicy.from_environment()

    assert policy.tensor_cache_preset == "standard"
    assert policy.memory_guard_active is True
    assert policy.adaptive_boost_active is False
    assert policy.max_resident_bytes == 128 * 1024**2
    assert policy.front_layer_count == 6
    assert policy.free_memory_bytes == 1 * 1024**3


def test_tensor_residency_policy_guards_below_three_gb_by_default(monkeypatch) -> None:
    clear_tensor_residency_cache()
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_MB", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_LOW_MEMORY_GUARD_MB", raising=False)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._free_memory_bytes", lambda: int(2.5 * 1024**3))

    policy = TensorResidencyPolicy.from_environment()

    assert policy.memory_guard_active is True
    assert policy.max_resident_bytes == 128 * 1024**2
    assert policy.front_layer_count == 6


def test_tensor_residency_policy_honors_guard_threshold_override(monkeypatch) -> None:
    clear_tensor_residency_cache()
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_MB", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", raising=False)
    monkeypatch.setenv("PCKETLM_TENSOR_CACHE_LOW_MEMORY_GUARD_MB", "2048")
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._free_memory_bytes", lambda: int(2.5 * 1024**3))

    policy = TensorResidencyPolicy.from_environment()

    assert policy.memory_guard_active is False
    assert policy.max_resident_bytes == 256 * 1024**2
    assert policy.front_layer_count == 12


def test_tensor_residency_policy_honors_explicit_cache_overrides_when_memory_is_low(monkeypatch) -> None:
    clear_tensor_residency_cache()
    monkeypatch.setenv("PCKETLM_TENSOR_CACHE_MB", "64")
    monkeypatch.setenv("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", "3")
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._free_memory_bytes", lambda: 1 * 1024**3)

    policy = TensorResidencyPolicy.from_environment()

    assert policy.memory_guard_active is True
    assert policy.max_resident_bytes == 64 * 1024**2
    assert policy.front_layer_count == 3


def test_tensor_residency_policy_uses_model_aware_budget_for_deep_models(monkeypatch) -> None:
    clear_tensor_residency_cache()
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_MB", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_FRONT_LAYERS", raising=False)
    monkeypatch.delenv("PCKETLM_TENSOR_CACHE_PRESET", raising=False)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._free_memory_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_catalog.load_tensor_catalog",
        lambda _model_id: SimpleNamespace(num_hidden_layers=64),
    )

    policy = TensorResidencyPolicy.from_environment("qwen32b-test")

    assert policy.model_aware_budget_active is True
    assert policy.max_resident_bytes == 4 * 1024**3
    assert policy.free_memory_bytes == 10 * 1024**3


def test_moe_residency_policy_keeps_per_layer_cap_at_least_top_k(monkeypatch) -> None:
    clear_tensor_residency_cache()
    monkeypatch.delenv("PCKETLM_MAX_RESIDENT_EXPERTS_PER_LAYER", raising=False)
    monkeypatch.delenv("PCKETLM_EXPERT_TENSOR_CACHE_MB", raising=False)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._free_memory_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_catalog.load_tensor_catalog",
        lambda _model_id: SimpleNamespace(num_hidden_layers=48, num_experts=128, num_experts_per_tok=8),
    )

    policy = TensorResidencyPolicy.from_environment("qwen3-topk-test")

    assert policy.max_resident_experts_per_layer == 8


def test_moe_residency_policy_honors_explicit_per_layer_cap(monkeypatch) -> None:
    clear_tensor_residency_cache()
    monkeypatch.setenv("PCKETLM_MAX_RESIDENT_EXPERTS_PER_LAYER", "2")
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._free_memory_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_catalog.load_tensor_catalog",
        lambda _model_id: SimpleNamespace(num_hidden_layers=48, num_experts=128, num_experts_per_tok=8),
    )

    policy = TensorResidencyPolicy.from_environment("qwen3-topk-test")

    assert policy.max_resident_experts_per_layer == 2


def test_load_resident_tensors_batches_misses_and_reuses_cached_results(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "resident-batch-cache-test"
    first_entry = _entry(tmp_path, tensor_name="model.layers.0.self_attn.q_proj.weight")
    second_entry = _entry(tmp_path, tensor_name="model.layers.0.self_attn.k_proj.weight")
    entries = {
        first_entry.tensor_name: first_entry,
        second_entry.tensor_name: second_entry,
    }
    calls = {"count": 0}

    def fake_load_tensors_by_name(model_id_arg: str, tensor_names: list[str]) -> dict[str, LoadedTensorSlice]:
        calls["count"] += 1
        return {
            tensor_name: _loaded_tensor(model_id_arg, entries[tensor_name], value=float(index + 1))
            for index, tensor_name in enumerate(tensor_names)
        }

    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_residency._find_tensor_entry",
        lambda _model_id, tensor_name: entries.get(tensor_name),
    )
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensors_by_name", fake_load_tensors_by_name)

    policy = TensorResidencyPolicy(max_resident_bytes=1024, max_tensor_bytes=1024)
    first = load_resident_tensors(model_id, list(entries), policy=policy)
    second = load_resident_tensors(model_id, list(entries), policy=policy)
    stats = tensor_residency_stats()

    assert all(result.ready for result in first.values())
    assert all(result.ready for result in second.values())
    assert first[first_entry.tensor_name].tensor is second[first_entry.tensor_name].tensor
    assert calls["count"] == 1
    assert stats.hits == 2
    assert stats.misses == 2
    assert stats.stores == 2


def test_load_resident_tensors_evicts_under_small_budget_for_32b_shaped_catalog(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "qwen32b-resident-test"
    entries = {
        f"model.layers.{index}.input_layernorm.weight": _entry(
            tmp_path,
            tensor_name=f"model.layers.{index}.input_layernorm.weight",
        )
        for index in range(64)
    }
    for index, entry in enumerate(entries.values()):
        entry.layer_index = index
        entry.component_group = "mlp"

    def fake_load_tensors_by_name(model_id_arg: str, tensor_names: list[str]) -> dict[str, LoadedTensorSlice]:
        return {tensor_name: _loaded_tensor(model_id_arg, entries[tensor_name], value=1.0) for tensor_name in tensor_names}

    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_residency._find_tensor_entry",
        lambda _model_id, tensor_name: entries.get(tensor_name),
    )
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensors_by_name", fake_load_tensors_by_name)

    policy = TensorResidencyPolicy(
        max_resident_bytes=64,
        max_tensor_bytes=1024,
        all_layer_small_tensor_bytes=0,
        front_layer_count=64,
    )
    loaded = load_resident_tensors(model_id, list(entries), policy=policy)
    stats = tensor_residency_stats()

    assert all(result.ready for result in loaded.values())
    assert stats.evictions > 0
    assert stats.resident_bytes <= 64
    assert stats.resident_count < 64


def test_load_resident_tensor_skips_tensors_larger_than_policy(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "resident-skip-test"
    entry = _entry(tmp_path)
    calls = {"count": 0}

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        calls["count"] += 1
        return _loaded_tensor(model_id_arg, entry, value=float(calls["count"]))

    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._find_tensor_entry", lambda *_args: entry)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(max_resident_bytes=1024, max_tensor_bytes=4, all_layer_small_tensor_bytes=0)
    first = load_resident_tensor(model_id, entry.tensor_name, policy=policy)
    second = load_resident_tensor(model_id, entry.tensor_name, policy=policy)
    stats = tensor_residency_stats()

    assert first.ready is True
    assert second.ready is True
    assert first.tensor is not second.tensor
    assert calls["count"] == 2
    assert stats.hits == 0
    assert stats.misses == 2
    assert stats.stores == 0
    assert stats.skips == 2
    assert stats.resident_count == 0


def test_load_resident_tensor_skips_layers_outside_front_cache_window(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "resident-layer-window-test"
    entry = _entry(tmp_path, tensor_name="model.layers.9.self_attn.k_proj.weight")
    entry.layer_index = 9
    calls = {"count": 0}

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        calls["count"] += 1
        return _loaded_tensor(model_id_arg, entry)

    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._find_tensor_entry", lambda *_args: entry)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(
        max_resident_bytes=1024,
        max_tensor_bytes=1024,
        all_layer_small_tensor_bytes=0,
        front_layer_count=2,
    )
    first = load_resident_tensor(model_id, entry.tensor_name, policy=policy)
    second = load_resident_tensor(model_id, entry.tensor_name, policy=policy)
    stats = tensor_residency_stats()

    assert first.ready is True
    assert second.ready is True
    assert calls["count"] == 2
    assert stats.hits == 0
    assert stats.misses == 2
    assert stats.stores == 0
    assert stats.skips == 2


def test_load_resident_tensor_keeps_small_tensors_across_all_layers(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "resident-small-all-layer-test"
    entry = _entry(tmp_path, tensor_name="model.layers.9.self_attn.q_proj.bias")
    entry.layer_index = 9
    calls = {"count": 0}

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        calls["count"] += 1
        return _loaded_tensor(model_id_arg, entry)

    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._find_tensor_entry", lambda *_args: entry)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    policy = TensorResidencyPolicy(
        max_resident_bytes=1024,
        max_tensor_bytes=0,
        all_layer_small_tensor_bytes=1024,
        front_layer_count=0,
    )
    first = load_resident_tensor(model_id, entry.tensor_name, policy=policy)
    second = load_resident_tensor(model_id, entry.tensor_name, policy=policy)
    stats = tensor_residency_stats()

    assert first.ready is True
    assert second.ready is True
    assert calls["count"] == 1
    assert stats.hits == 1
    assert stats.stores == 1


def test_load_resident_tensor_clones_same_dtype_safetensors_view(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "resident-same-dtype-test"
    entry = _entry(tmp_path)
    source_tensor = torch.ones((2, 2), dtype=torch.bfloat16)

    def fake_load_tensor_by_name(model_id_arg: str, tensor_name: str) -> LoadedTensorSlice:
        return LoadedTensorSlice(
            model_id=model_id_arg,
            tensor_name=tensor_name,
            shard_name=entry.shard_name,
            dtype=str(source_tensor.dtype),
            shape=[2, 2],
            tensor=source_tensor,
            layer_index=entry.layer_index,
            component_group=entry.component_group,
            loaded_nbytes=source_tensor.element_size() * source_tensor.nelement(),
            blockers=[],
            ready=True,
        )

    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._find_tensor_entry", lambda *_args: entry)
    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency.load_tensor_by_name", fake_load_tensor_by_name)

    loaded = load_resident_tensor(
        model_id,
        entry.tensor_name,
        dtype=torch.bfloat16,
        policy=TensorResidencyPolicy(max_resident_bytes=1024, max_tensor_bytes=1024),
    )

    assert loaded.ready is True
    assert loaded.tensor is not None
    assert loaded.tensor.dtype == torch.bfloat16
    assert loaded.tensor.data_ptr() != source_tensor.data_ptr()


def test_clear_tensor_residency_cache_resets_counters(tmp_path: Path, monkeypatch) -> None:
    clear_tensor_residency_cache()
    model_id = "resident-clear-test"
    entry = _entry(tmp_path)

    monkeypatch.setattr("pcketlm.core.runtime.tensor_residency._find_tensor_entry", lambda *_args: entry)
    monkeypatch.setattr(
        "pcketlm.core.runtime.tensor_residency.load_tensor_by_name",
        lambda model_id_arg, _tensor_name: _loaded_tensor(model_id_arg, entry),
    )

    load_resident_tensor(model_id, entry.tensor_name, policy=TensorResidencyPolicy(max_resident_bytes=1024))
    assert tensor_residency_stats().resident_count == 1

    clear_tensor_residency_cache()
    stats = tensor_residency_stats()

    assert stats.to_dict() == {
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "evictions": 0,
        "skips": 0,
        "resident_bytes": 0,
        "resident_count": 0,
        "expert_hits": 0,
        "expert_misses": 0,
        "expert_evictions": 0,
        "expert_resident_bytes": 0,
        "expert_resident_count": 0,
    }
