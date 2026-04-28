from pathlib import Path

import torch

from pcketlm.core.runtime.tensor_catalog import TensorCatalogEntry
from pcketlm.core.runtime.tensor_loader import LoadedTensorSlice
from pcketlm.core.runtime.tensor_residency import (
    TensorResidencyPolicy,
    clear_tensor_residency_cache,
    load_resident_tensor,
    load_resident_tensors,
    tensor_residency_stats,
)


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
    }
