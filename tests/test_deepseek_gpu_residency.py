import importlib.util
from pathlib import Path

import torch

from pcketlm.core.runtime.deepseek_gpu_residency import (
    LocalDeepSeekResidentLayer,
    LocalDeepSeekResidentLayerPager,
    estimate_deepseek_gpu_residency,
    run_local_deepseek_paged_decode_loop,
    run_local_deepseek_paged_decode_probe,
)
from pcketlm.core.runtime.tensor_catalog import build_tensor_catalog

_RUNTIME_TEST_PATH = Path(__file__).with_name("test_runtime_fp8_source.py")
_RUNTIME_SPEC = importlib.util.spec_from_file_location("_runtime_fp8_source_helpers", _RUNTIME_TEST_PATH)
assert _RUNTIME_SPEC is not None and _RUNTIME_SPEC.loader is not None
_RUNTIME_HELPERS = importlib.util.module_from_spec(_RUNTIME_SPEC)
_RUNTIME_SPEC.loader.exec_module(_RUNTIME_HELPERS)
_write_fp8_runtime_fixture = _RUNTIME_HELPERS._write_fp8_runtime_fixture


class _DummyLayer:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = int(nbytes)
        self.released = False

    def resident_nbytes(self) -> int:
        return 0 if self.released else self.nbytes

    def release(self) -> None:
        self.released = True


def test_estimate_deepseek_gpu_residency_reports_window_size(tmp_path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    estimate = estimate_deepseek_gpu_residency(
        model_id,
        layer_count=1,
        selected_experts=[1],
        resident_budget_bytes=1024,
        measured_seconds_per_layer=0.02,
    )

    assert estimate.ready is True
    assert estimate.config_hidden_layers == 1
    assert estimate.layer_count == 1
    assert estimate.max_source_layer_bytes > 0
    assert estimate.max_dequantized_layer_bytes > estimate.max_source_layer_bytes
    assert estimate.layers_fit_by_source_bytes >= 1
    assert estimate.speed_target_met is True
    assert estimate.layer_plans[0]["selected_experts"] == [1]


def test_local_paged_decode_probe_runs_fixture_on_cpu(tmp_path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    result = run_local_deepseek_paged_decode_probe(
        model_id,
        token_id=1,
        layer_count=1,
        resident_budget_bytes=1024 * 1024,
        device="cpu",
        require_cuda=False,
        dtype=torch.float32,
    )

    assert result.passed is True
    assert result.device == "cpu"
    assert result.executed_layers == [0]
    assert result.output_shape == [1, 1, 4]
    assert result.cache_sequence_lengths == {0: 1}
    assert result.pager_loads == 1
    assert result.pager_cache_misses == 1
    assert result.peak_resident_bytes > 0
    assert result.tail_top_token_ids == []
    assert result.blockers == []


def test_local_resident_layer_carries_kv_cache(tmp_path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)
    config = {
        "hidden_size": 4,
        "num_attention_heads": 1,
        "qk_nope_head_dim": 1,
        "qk_rope_head_dim": 2,
        "v_head_dim": 1,
        "kv_lora_rank": 1,
        "rms_norm_eps": 1e-6,
        "first_k_dense_replace": 1,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 2.5,
        "n_shared_experts": 0,
        "rope_theta": 10000.0,
    }
    layer = LocalDeepSeekResidentLayer(
        model_id,
        0,
        config=config,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    hidden = torch.ones((1, 1, 4), dtype=torch.float32)

    first, first_cache = layer.forward_with_cache(hidden, start_pos=0)
    second, second_cache = layer.forward_with_cache(hidden, start_pos=1, previous_kv_cache=first_cache)

    assert first.shape == second.shape == (1, 1, 4)
    assert first_cache is not None
    assert second_cache is not None
    assert first_cache[0].shape[1] == 1
    assert second_cache[0].shape[1] == 2


def test_local_paged_decode_probe_can_stream_tail_topk(tmp_path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    result = run_local_deepseek_paged_decode_probe(
        model_id,
        token_id=1,
        layer_count=1,
        resident_budget_bytes=1024 * 1024,
        device="cpu",
        require_cuda=False,
        dtype=torch.float32,
        include_tail=True,
        tail_top_k=2,
        tail_chunk_rows=2,
    )

    assert result.passed is True
    assert len(result.tail_top_token_ids) == 2
    assert len(result.tail_top_logits) == 2
    assert result.tail_elapsed_seconds > 0
    assert result.blockers == []


def test_local_paged_decode_loop_generates_with_cache(tmp_path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    result = run_local_deepseek_paged_decode_loop(
        model_id,
        [1, 2],
        layer_count=1,
        max_new_tokens=1,
        resident_budget_bytes=1024 * 1024,
        device="cpu",
        require_cuda=False,
        dtype=torch.float32,
        tail_top_k=2,
        tail_chunk_rows=2,
    )

    assert result.passed is True
    assert result.prompt_token_ids == [1, 2]
    assert len(result.generated_token_ids) == 1
    assert result.positions_completed == 3
    assert result.cache_sequence_lengths == {0: 3}
    assert len(result.step_summaries) == 3
    assert result.pager_cache_hits >= 2
    assert result.final_top_token_ids
    assert result.blockers == []


def test_local_layer_pager_evicts_lru_without_evicting_protected_layer() -> None:
    layers = {3: _DummyLayer(70), 4: _DummyLayer(70)}
    pager = LocalDeepSeekResidentLayerPager(
        "dummy",
        config={},
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_resident_bytes=100,
        layer_factory=lambda index: layers[index],  # type: ignore[arg-type]
    )

    pager.get(3)
    pager.get(4)
    pager.enforce_budget(protected_layer=4)

    assert sorted(pager.layers) == [4]
    assert pager.evictions == 1
    assert pager.resident_nbytes() == 70


def test_local_layer_pager_prefetches_next_layer() -> None:
    layers = {3: _DummyLayer(70), 4: _DummyLayer(70)}
    pager = LocalDeepSeekResidentLayerPager(
        "dummy",
        config={},
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_resident_bytes=200,
        layer_factory=lambda index: layers[index],  # type: ignore[arg-type]
        prefetch_workers=1,
    )

    assert pager.prefetch(4) is True
    layer = pager.get(4)
    pager.close()

    assert layer is layers[4]
    assert pager.loads == 1
    assert pager.prefetch_submitted == 1
    assert pager.prefetch_completed == 1
    assert pager.cache_misses == 1


def test_deepseek_gpu_validator_can_run_local_only(tmp_path, monkeypatch, capsys) -> None:
    from tools.deepseek_gpu_validate import main

    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    exit_code = main(
        [
            "--skip-synthetic",
            "--skip-remote",
            "--local-model-id",
            model_id,
            "--layer",
            "0",
            "--local-paged-decode-layers",
            "1",
            "--local-include-tail",
            "--local-max-new-tokens",
            "1",
            "--json",
        ]
    )
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert '"local_deepseek_gpu_residency_estimate"' in captured
    assert '"local_deepseek_paged_decode_loop"' in captured
    assert '"tail_top_token_ids"' in captured
