import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from pcketlm.core.runtime.layer_bridge import (
    CANCEL_BLOCKER,
    LayerBridgeResult,
    _recommended_prompt_layer_count,
    _moe_expert_tensor_name_map,
    _moe_router_tensor_name,
    _native_q4_moe_prefill_enabled,
    _native_lm_head_topk_enabled,
    _q4_prefix_scoped_handles_enabled,
    _run_q4_moe_mlp_token_loop,
    _run_moe_mlp,
    _trim_generated_text_at_stop_string,
    _auto_torch_thread_count,
    _use_scoped_safetensor_handles,
    build_history_summary_hidden_state,
    initialize_kv_decode_state,
    select_next_token,
    run_decode_benchmark,
    run_kv_decode_step,
    run_kv_decode_loop,
    run_decode_tail,
    load_token_entry_hidden_state,
    load_layer_bridge_config,
    run_layer_bridge_stack,
    run_minimal_layer_forward_bridge,
    run_prompt_decode_loop,
    run_prompt_prefill_session,
    run_repeated_decode_loop,
    runtime_torch_thread_count,
    runtime_math_dtype_name,
    run_token_decode_step,
    run_token_entry_layer_bridge,
)
from pcketlm.core.runtime.tokenizer_runtime import prepare_prompt_text
from pcketlm.core.runtime.tensor_execution_plan import build_tensor_execution_plan
from pcketlm.core.runtime import layer_bridge as layer_bridge_module


class _PromptBudgetConfig:
    ready = True
    num_hidden_layers = 48
    num_experts = 0
    num_experts_per_tok = 0


class _MoePromptBudgetConfig(_PromptBudgetConfig):
    num_experts = 128
    num_experts_per_tok = 8


def test_recommended_prompt_layer_count_keeps_short_chat_at_full_stack() -> None:
    assert _recommended_prompt_layer_count(_PromptBudgetConfig(), prompt_token_count=80, max_new_tokens=4) == 48
    assert _recommended_prompt_layer_count(_PromptBudgetConfig(), prompt_token_count=260, max_new_tokens=4) == 48


def test_recommended_prompt_layer_count_keeps_moe_at_full_stack_for_correctness() -> None:
    assert _recommended_prompt_layer_count(_MoePromptBudgetConfig(), prompt_token_count=80, max_new_tokens=20) == 48
    assert _recommended_prompt_layer_count(_PromptBudgetConfig(), prompt_token_count=80, max_new_tokens=20) == 48


def test_moe_config_defaults_topk_normalization_when_field_is_absent(tmp_path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_dir = tmp_path / "models" / "mixtral-style-test" / "original"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mixtral",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "num_local_experts": 8,
                "num_experts_per_tok": 2,
                "vocab_size": 128,
                "rms_norm_eps": 1e-5,
                "hidden_act": "silu",
            }
        ),
        encoding="utf-8",
    )
    layer_bridge_module._load_layer_bridge_config_cached.cache_clear()

    config = load_layer_bridge_config("mixtral-style-test")

    assert config.ready is True
    assert config.norm_topk_prob is True


def test_moe_tensor_name_helpers_select_mixtral_layout(monkeypatch) -> None:
    available = {
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.3.w1.weight",
        "model.layers.0.block_sparse_moe.experts.3.w2.weight",
        "model.layers.0.block_sparse_moe.experts.3.w3.weight",
    }

    monkeypatch.setattr(
        layer_bridge_module,
        "find_tensor_catalog_entry",
        lambda _model_id, tensor_name: object() if tensor_name in available else None,
    )

    assert _moe_router_tensor_name("mixtral-test", 0) == "model.layers.0.block_sparse_moe.gate.weight"
    assert _moe_expert_tensor_name_map("mixtral-test", 0, 3) == {
        "gate_proj": "model.layers.0.block_sparse_moe.experts.3.w1.weight",
        "up_proj": "model.layers.0.block_sparse_moe.experts.3.w3.weight",
        "down_proj": "model.layers.0.block_sparse_moe.experts.3.w2.weight",
    }


def test_runtime_torch_thread_count_uses_safe_auto_and_env_override(monkeypatch) -> None:
    assert _auto_torch_thread_count(16) == 14
    assert _auto_torch_thread_count(8) == 6
    assert _auto_torch_thread_count(4) == 4

    monkeypatch.setenv("PCKETLM_TORCH_THREADS", "3")
    assert runtime_torch_thread_count() == 3
    monkeypatch.setenv("PCKETLM_TORCH_THREADS", "bad")
    assert runtime_torch_thread_count() >= 1


def test_scoped_safetensor_handles_default_off_for_qwen_32b(monkeypatch) -> None:
    monkeypatch.delenv("PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE", raising=False)

    assert _use_scoped_safetensor_handles("qwen2.5-14b-instruct", 1) is True
    assert _use_scoped_safetensor_handles("qwen2.5-14b-instruct", 4) is True
    assert _use_scoped_safetensor_handles("qwen2.5-14b-instruct", 5) is False
    assert _use_scoped_safetensor_handles("qwen2.5-32b-instruct", 1) is False
    assert _use_scoped_safetensor_handles("qwen2.5-32b-instruct", 2) is False
    assert _use_scoped_safetensor_handles("qwen3-30b-a3b", 1) is False
    assert _use_scoped_safetensor_handles("mixtral-8x7b-instruct-v01", 1) is False

    monkeypatch.setenv("PCKETLM_SCOPED_SAFETENSOR_HANDLE_CACHE", "1")
    assert _use_scoped_safetensor_handles("qwen2.5-32b-instruct", 1) is True
    assert _use_scoped_safetensor_handles("qwen3-30b-a3b", 1) is True
    assert _use_scoped_safetensor_handles("mixtral-8x7b-instruct-v01", 1) is True


def test_layer_prefetch_starts_next_load_before_current_compute_finishes(monkeypatch) -> None:
    events: list[tuple[str, int, float]] = []

    monkeypatch.delenv("PCKETLM_DISABLE_LAYER_PREFETCH", raising=False)
    monkeypatch.setenv("PCKETLM_ENABLE_LAYER_PREFETCH", "1")
    monkeypatch.setattr(
        layer_bridge_module,
        "load_layer_bridge_config",
        lambda _model_id: SimpleNamespace(ready=True, blockers=[], num_hidden_layers=2),
    )
    monkeypatch.setattr(
        layer_bridge_module.TensorResidencyPolicy,
        "from_environment",
        classmethod(lambda cls, _model_id=None: cls(enabled=False)),
    )

    def fake_load_resident_tensors(model_id, tensor_names, **_kwargs):
        del model_id
        layer_index = 1 if any(".1." in name for name in tensor_names) else 0
        events.append(("load-start", layer_index, time.perf_counter()))
        time.sleep(0.02)
        events.append(("load-end", layer_index, time.perf_counter()))
        return {
            name: SimpleNamespace(ready=True, tensor=torch.zeros((1,), dtype=torch.bfloat16), blockers=[])
            for name in tensor_names
        }

    def fake_run_minimal_layer_forward_bridge(*_args, layer_index: int, prefetched_tensors=None, **_kwargs):
        events.append(("compute-start", layer_index, time.perf_counter()))
        if layer_index == 1:
            assert prefetched_tensors
        time.sleep(0.05)
        events.append(("compute-end", layer_index, time.perf_counter()))
        return LayerBridgeResult(
            model_id="qwen-test",
            layer_index=layer_index,
            input_mode="provided",
            input_shape=[1, 1, 1],
            output_shape=[1, 1, 1],
            output_dtype="torch.bfloat16",
            cache_sequence_length=1,
            ready=True,
            output_tensor=torch.zeros((1, 1, 1), dtype=torch.bfloat16),
        )

    monkeypatch.setattr(layer_bridge_module, "load_resident_tensors", fake_load_resident_tensors)
    monkeypatch.setattr(
        layer_bridge_module,
        "run_minimal_layer_forward_bridge",
        fake_run_minimal_layer_forward_bridge,
    )

    result = layer_bridge_module.run_layer_bridge_stack(
        "qwen-test",
        start_layer=0,
        layer_count=2,
        input_hidden=torch.zeros((1, 1, 1), dtype=torch.bfloat16),
        collect_step_summaries=False,
        collect_metrics=False,
    )

    assert result.ready is True
    load_start = next(ts for event, layer, ts in events if event == "load-start" and layer == 1)
    compute_end = next(ts for event, layer, ts in events if event == "compute-end" and layer == 0)
    assert load_start < compute_end


def test_native_dense_decode_dispatch_runs_one_token_dense_layer(monkeypatch) -> None:
    hidden_size = 8
    intermediate_size = 16
    num_heads = 2
    num_kv_heads = 1
    head_dim = hidden_size // num_heads
    kv_width = num_kv_heads * head_dim
    config = SimpleNamespace(
        model_id="native-dense-test",
        ready=True,
        blockers=[],
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=0,
        num_experts_per_tok=0,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    torch.manual_seed(321)
    tensors = {
        "input_layernorm.weight": torch.ones((hidden_size,), dtype=torch.bfloat16),
        "post_attention_layernorm.weight": torch.ones((hidden_size,), dtype=torch.bfloat16),
        "self_attn.q_proj.weight": (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.k_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.v_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.o_proj.weight": (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "mlp.gate_proj.weight": (torch.randn((intermediate_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "mlp.up_proj.weight": (torch.randn((intermediate_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "mlp.down_proj.weight": (torch.randn((hidden_size, intermediate_size)) * 0.1).to(torch.bfloat16),
        "self_attn.q_proj.bias": (torch.randn((hidden_size,)) * 0.01).to(torch.bfloat16),
        "self_attn.k_proj.bias": (torch.randn((kv_width,)) * 0.01).to(torch.bfloat16),
        "self_attn.v_proj.bias": (torch.randn((kv_width,)) * 0.01).to(torch.bfloat16),
    }

    def fake_exists(_model_id, tensor_name):
        return any(tensor_name.endswith(suffix) for suffix in tensors)

    def fake_load_resident_tensors(_model_id, tensor_names, **_kwargs):
        loaded = {}
        for name in tensor_names:
            suffix = next(suffix for suffix in tensors if name.endswith(suffix))
            loaded[name] = SimpleNamespace(ready=True, tensor=tensors[suffix], blockers=[])
        return loaded

    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_LAYER", raising=False)
    monkeypatch.setattr(layer_bridge_module, "_tensor_entry_exists", fake_exists)
    monkeypatch.setattr(layer_bridge_module, "load_resident_tensors", fake_load_resident_tensors)

    timings: dict[str, float] = {}
    result = layer_bridge_module._try_native_dense_decode_bridge(
        "native-dense-test",
        0,
        torch.randn((1, 1, hidden_size), dtype=torch.bfloat16),
        (
            torch.zeros((1, num_kv_heads, 2, head_dim), dtype=torch.bfloat16),
            torch.zeros((1, num_kv_heads, 2, head_dim), dtype=torch.bfloat16),
        ),
        None,
        config,
        tensor_policy=None,
        collect_metrics=True,
        timings=timings,
    )

    assert result is not None
    assert result.ready is True
    assert result.output_tensor is not None
    assert result.output_tensor.shape == (1, 1, hidden_size)
    assert result.next_kv_cache is None
    assert result.native_kv_session is not None
    assert result.native_kv_session.committed_length(0) == 3
    assert "native_layer" in result.timings
    second = layer_bridge_module._try_native_dense_decode_bridge(
        "native-dense-test",
        0,
        torch.randn((1, 1, hidden_size), dtype=torch.bfloat16),
        None,
        result.native_kv_session,
        config,
        tensor_policy=None,
        collect_metrics=True,
        timings=timings,
    )
    try:
        assert second is not None
        assert second.ready is True
        assert second.native_kv_session is result.native_kv_session
        assert second.native_kv_session.committed_length(0) == 4
        assert second.next_kv_cache is None
    finally:
        result.native_kv_session.close()


def test_native_dense_decode_uses_prefetched_tensor_bundle(monkeypatch) -> None:
    hidden_size = 8
    intermediate_size = 16
    num_heads = 2
    num_kv_heads = 1
    head_dim = hidden_size // num_heads
    kv_width = num_kv_heads * head_dim
    config = SimpleNamespace(
        model_id="native-dense-prefetch-test",
        ready=True,
        blockers=[],
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=0,
        num_experts_per_tok=0,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    torch.manual_seed(1122)
    tensors = {
        f"model.layers.0.input_layernorm.weight": torch.ones((hidden_size,), dtype=torch.bfloat16),
        f"model.layers.0.post_attention_layernorm.weight": torch.ones((hidden_size,), dtype=torch.bfloat16),
        f"model.layers.0.self_attn.q_proj.weight": (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16),
        f"model.layers.0.self_attn.k_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        f"model.layers.0.self_attn.v_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        f"model.layers.0.self_attn.o_proj.weight": (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16),
        f"model.layers.0.mlp.gate_proj.weight": (torch.randn((intermediate_size, hidden_size)) * 0.1).to(torch.bfloat16),
        f"model.layers.0.mlp.up_proj.weight": (torch.randn((intermediate_size, hidden_size)) * 0.1).to(torch.bfloat16),
        f"model.layers.0.mlp.down_proj.weight": (torch.randn((hidden_size, intermediate_size)) * 0.1).to(torch.bfloat16),
    }

    def fake_exists(_model_id, tensor_name):
        return tensor_name in tensors

    def fail_load_resident_tensors(*_args, **_kwargs):
        raise AssertionError("native dense decode should consume the prefetched tensor bundle")

    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_LAYER", raising=False)
    monkeypatch.setattr(layer_bridge_module, "_tensor_entry_exists", fake_exists)
    monkeypatch.setattr(layer_bridge_module, "load_resident_tensors", fail_load_resident_tensors)

    timings: dict[str, float] = {}
    result = layer_bridge_module._try_native_dense_decode_bridge(
        "native-dense-prefetch-test",
        0,
        torch.randn((1, 1, hidden_size), dtype=torch.bfloat16),
        None,
        None,
        config,
        tensor_policy=None,
        collect_metrics=True,
        timings=timings,
        prefetched_tensors=tensors,
    )

    assert result is not None
    try:
        assert result.ready is True
        assert result.output_tensor is not None
        assert result.native_kv_session is not None
        assert result.timings.get("load_tensors", 0.0) == 0.0
        assert result.timings.get("native_layer", 0.0) > 0.0
    finally:
        if result.native_kv_session is not None:
            result.native_kv_session.close()


def test_native_dense_decode_uses_row8_artifact_without_original_projection_load(monkeypatch) -> None:
    from tools.pack_weights_row8 import pack_rows8_tensor
    from pcketlm.core.runtime import packed_artifact_loader

    hidden_size = 8
    intermediate_size = 16
    num_heads = 2
    num_kv_heads = 1
    head_dim = hidden_size // num_heads
    kv_width = num_kv_heads * head_dim
    config = SimpleNamespace(
        model_id="native-dense-row8-artifact-test",
        ready=True,
        blockers=[],
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=0,
        num_experts_per_tok=0,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    torch.manual_seed(4422)
    full_names = {
        "model.layers.0.input_layernorm.weight": torch.ones((hidden_size,), dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones((hidden_size,), dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        "model.layers.0.self_attn.v_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        "model.layers.0.self_attn.o_proj.weight": (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "model.layers.0.mlp.gate_proj.weight": (torch.randn((intermediate_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight": (torch.randn((intermediate_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": (torch.randn((hidden_size, intermediate_size)) * 0.1).to(torch.bfloat16),
    }
    projection_names = {
        name
        for name in full_names
        if any(
            suffix in name
            for suffix in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
            )
        )
    }
    packed_by_name = {
        name: pack_rows8_tensor(tensor)
        for name, tensor in full_names.items()
        if name in projection_names
    }
    loaded_names: list[str] = []

    def fake_exists(_model_id, tensor_name):
        return tensor_name in full_names

    def fake_load_resident_tensors(_model_id, tensor_names, **_kwargs):
        loaded = {}
        for name in tensor_names:
            assert name not in projection_names
            loaded_names.append(name)
            loaded[name] = SimpleNamespace(ready=True, tensor=full_names[name], blockers=[])
        return loaded

    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_LAYER", raising=False)
    monkeypatch.setenv("PCKETLM_ENABLE_NATIVE_PACKED_ARTIFACT_LAYER", "1")
    monkeypatch.setenv("PCKETLM_ROW8_ARTIFACT_NAME", "row8-test")
    monkeypatch.setattr(layer_bridge_module, "_tensor_entry_exists", fake_exists)
    monkeypatch.setattr(layer_bridge_module, "load_resident_tensors", fake_load_resident_tensors)
    monkeypatch.setattr(
        packed_artifact_loader,
        "row8_tensor_available",
        lambda _model_id, tensor_name, artifact_name="row8": artifact_name == "row8-test" and tensor_name in packed_by_name,
    )
    monkeypatch.setattr(
        packed_artifact_loader,
        "load_row8_packed_tensor",
        lambda _model_id, tensor_name, artifact_name="row8": (packed_by_name[tensor_name], {"shape": list(full_names[tensor_name].shape)}),
    )

    timings: dict[str, float] = {}
    result = layer_bridge_module._try_native_dense_decode_bridge(
        "native-dense-row8-artifact-test",
        0,
        torch.randn((1, 1, hidden_size), dtype=torch.bfloat16),
        None,
        None,
        config,
        tensor_policy=None,
        collect_metrics=True,
        timings=timings,
    )

    assert result is not None
    try:
        assert result.ready is True
        assert result.output_tensor is not None
        assert set(loaded_names) == {
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        }
        assert timings.get("load_packed_artifact", 0.0) >= 0.0
        assert "pack_weights" not in timings
    finally:
        if result.native_kv_session is not None:
            result.native_kv_session.close()


def test_native_attention_decode_dispatch_runs_one_token_moe_attention(monkeypatch) -> None:
    hidden_size = 8
    num_heads = 2
    num_kv_heads = 1
    head_dim = 6
    attention_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim
    config = SimpleNamespace(
        model_id="native-moe-attention-test",
        ready=True,
        blockers=[],
        hidden_size=hidden_size,
        intermediate_size=16,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    torch.manual_seed(654)
    tensors = {
        "self_attn.q_proj.weight": (torch.randn((attention_width, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.k_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.v_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.o_proj.weight": (torch.randn((hidden_size, attention_width)) * 0.1).to(torch.bfloat16),
        "self_attn.q_proj.bias": (torch.randn((attention_width,)) * 0.01).to(torch.bfloat16),
        "self_attn.k_proj.bias": (torch.randn((kv_width,)) * 0.01).to(torch.bfloat16),
        "self_attn.v_proj.bias": (torch.randn((kv_width,)) * 0.01).to(torch.bfloat16),
    }

    def fake_exists(_model_id, tensor_name):
        return any(tensor_name.endswith(suffix) for suffix in tensors)

    def fake_load_resident_tensors(_model_id, tensor_names, **_kwargs):
        loaded = {}
        for name in tensor_names:
            suffix = next(suffix for suffix in tensors if name.endswith(suffix))
            loaded[name] = SimpleNamespace(ready=True, tensor=tensors[suffix], blockers=[])
        return loaded

    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_LAYER", raising=False)
    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_ATTENTION", raising=False)
    monkeypatch.setattr(layer_bridge_module, "_tensor_entry_exists", fake_exists)
    monkeypatch.setattr(layer_bridge_module, "load_resident_tensors", fake_load_resident_tensors)

    timings: dict[str, float] = {}
    payload = layer_bridge_module._try_native_attention_decode_bridge(
        "native-moe-attention-test",
        0,
        torch.randn((1, 1, hidden_size), dtype=torch.bfloat16),
        (
            torch.zeros((1, num_kv_heads, 2, head_dim), dtype=torch.bfloat16),
            torch.zeros((1, num_kv_heads, 2, head_dim), dtype=torch.bfloat16),
        ),
        None,
        config,
        tensor_policy=None,
        timings=timings,
    )

    assert payload is not None
    output, cache_sequence_length, session = payload
    try:
        assert output.shape == (1, 1, hidden_size)
        assert output.dtype == torch.bfloat16
        assert cache_sequence_length == 3
        assert session.committed_length(0) == 3
        assert "native_attention" in timings
    finally:
        session.close()


def test_native_attention_decode_can_leave_kv_tentative_for_speculation(monkeypatch) -> None:
    hidden_size = 8
    num_heads = 2
    num_kv_heads = 1
    head_dim = 4
    kv_width = num_kv_heads * head_dim
    config = SimpleNamespace(
        model_id="native-attention-tentative-test",
        ready=True,
        blockers=[],
        hidden_size=hidden_size,
        intermediate_size=16,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    torch.manual_seed(987)
    tensors = {
        "self_attn.q_proj.weight": (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.k_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.v_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.o_proj.weight": (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16),
    }

    def fake_exists(_model_id, tensor_name):
        return any(tensor_name.endswith(suffix) for suffix in tensors)

    def fake_load_resident_tensors(_model_id, tensor_names, **_kwargs):
        loaded = {}
        for name in tensor_names:
            suffix = next(suffix for suffix in tensors if name.endswith(suffix))
            loaded[name] = SimpleNamespace(ready=True, tensor=tensors[suffix], blockers=[])
        return loaded

    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_LAYER", raising=False)
    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_ATTENTION", raising=False)
    monkeypatch.setattr(layer_bridge_module, "_tensor_entry_exists", fake_exists)
    monkeypatch.setattr(layer_bridge_module, "load_resident_tensors", fake_load_resident_tensors)

    payload = layer_bridge_module._try_native_attention_decode_bridge(
        "native-attention-tentative-test",
        0,
        torch.randn((1, 1, hidden_size), dtype=torch.bfloat16),
        (
            torch.zeros((1, num_kv_heads, 2, head_dim), dtype=torch.bfloat16),
            torch.zeros((1, num_kv_heads, 2, head_dim), dtype=torch.bfloat16),
        ),
        None,
        config,
        tensor_policy=None,
        timings={},
        native_kv_commit=False,
    )

    assert payload is not None
    _output, cache_sequence_length, session = payload
    try:
        assert cache_sequence_length == 3
        assert session.committed_length(0) == 2
        assert session.tentative_length(0) == 1
        session.rollback()
        assert session.committed_length(0) == 2
        assert session.tentative_length(0) == 0
    finally:
        session.close()


def test_native_dense_prefill_dispatch_commits_prompt_kv(monkeypatch) -> None:
    hidden_size = 8
    intermediate_size = 16
    num_heads = 2
    num_kv_heads = 1
    head_dim = 4
    kv_width = num_kv_heads * head_dim
    config = SimpleNamespace(
        model_id="native-dense-prefill-test",
        ready=True,
        blockers=[],
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=0,
        num_experts_per_tok=0,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    torch.manual_seed(24601)
    tensors = {
        "input_layernorm.weight": (torch.rand((hidden_size,)) + 0.5).to(torch.bfloat16),
        "post_attention_layernorm.weight": (torch.rand((hidden_size,)) + 0.5).to(torch.bfloat16),
        "self_attn.q_proj.weight": (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.k_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.v_proj.weight": (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16),
        "self_attn.o_proj.weight": (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "mlp.gate_proj.weight": (torch.randn((intermediate_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "mlp.up_proj.weight": (torch.randn((intermediate_size, hidden_size)) * 0.1).to(torch.bfloat16),
        "mlp.down_proj.weight": (torch.randn((hidden_size, intermediate_size)) * 0.1).to(torch.bfloat16),
    }

    def fake_config(_model_id):
        return config

    def fake_exists(_model_id, tensor_name):
        return any(tensor_name.endswith(suffix) for suffix in tensors)

    def fake_load_resident_tensors(_model_id, tensor_names, **_kwargs):
        loaded = {}
        for name in tensor_names:
            suffix = next(suffix for suffix in tensors if name.endswith(suffix))
            loaded[name] = SimpleNamespace(ready=True, tensor=tensors[suffix], blockers=[])
        return loaded

    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_LAYER", raising=False)
    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_ATTENTION", raising=False)
    monkeypatch.setenv("PCKETLM_ENABLE_NATIVE_DENSE_PREFILL", "1")
    monkeypatch.setattr(layer_bridge_module, "load_layer_bridge_config", fake_config)
    monkeypatch.setattr(layer_bridge_module, "_tensor_entry_exists", fake_exists)
    monkeypatch.setattr(layer_bridge_module, "load_resident_tensors", fake_load_resident_tensors)

    result = layer_bridge_module.run_minimal_layer_forward_bridge(
        "native-dense-prefill-test",
        layer_index=0,
        input_hidden=torch.randn((1, 3, hidden_size), dtype=torch.bfloat16),
        return_kv_cache=True,
    )

    assert result.ready is True
    assert result.output_tensor is not None
    assert tuple(result.output_tensor.shape) == (1, 3, hidden_size)
    assert result.cache_sequence_length == 3
    assert result.native_kv_session is not None
    try:
        assert result.native_kv_session.committed_length(0) == 3
        assert result.native_kv_session.tentative_length(0) == 0
    finally:
        result.native_kv_session.close()
    assert result.timings.get("native_layer", 0.0) > 0.0


def test_trim_generated_text_at_stop_string_removes_visible_marker() -> None:
    text, marker = _trim_generated_text_at_stop_string("Hello<|im_end|>ignored", ["<|im_end|>"])

    assert text == "Hello"
    assert marker == "<|im_end|>"


def test_run_moe_mlp_routes_top_k_experts_with_real_math() -> None:
    hidden = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
    router_weight = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    expert_tensors = {
        0: {
            "gate_proj": torch.eye(2),
            "up_proj": torch.eye(2),
            "down_proj": torch.eye(2),
        },
        1: {
            "gate_proj": torch.eye(2) * 2,
            "up_proj": torch.eye(2),
            "down_proj": torch.eye(2),
        },
    }

    output, touched, selected = _run_moe_mlp(
        hidden_states=hidden,
        router_weight=router_weight,
        expert_tensors=expert_tensors,
        top_k=2,
        norm_topk_prob=True,
    )

    probs = torch.softmax(torch.tensor([1.0, 2.0, -1.0, -2.0]), dim=-1)
    top_values, _top_indices = torch.topk(probs, 2)
    weights = top_values / top_values.sum()
    expert0 = torch.nn.functional.silu(hidden) * hidden
    expert1 = torch.nn.functional.silu(hidden * 2) * hidden
    expected = expert1 * weights[0] + expert0 * weights[1]

    assert touched == [0, 1]
    assert selected.tolist() == [[[1, 0]]]
    assert torch.allclose(output, expected, atol=1e-6)


def test_run_moe_mlp_native_selected_path_matches_python_bfloat16() -> None:
    torch.manual_seed(222)
    hidden = (torch.randn((1, 1, 4), dtype=torch.float32) * 0.2).to(torch.bfloat16)
    router_weight = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    expert_tensors = {}
    for expert in (0, 1, 2):
        expert_tensors[expert] = {
            "gate_proj": (torch.randn((6, 4), dtype=torch.float32) * 0.15).to(torch.bfloat16),
            "up_proj": (torch.randn((6, 4), dtype=torch.float32) * 0.15).to(torch.bfloat16),
            "down_proj": (torch.randn((4, 6), dtype=torch.float32) * 0.15).to(torch.bfloat16),
        }

    native_output, touched, selected = _run_moe_mlp(
        hidden_states=hidden,
        router_weight=router_weight,
        expert_tensors=expert_tensors,
        top_k=2,
        norm_topk_prob=True,
    )

    router_probs = torch.softmax(F.linear(hidden.float(), router_weight.float()), dim=-1)
    weights, expected_selected = torch.topk(router_probs, 2, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    expected = torch.zeros_like(hidden.float())
    for rank, expert in enumerate(expected_selected.reshape(-1).tolist()):
        tensors = expert_tensors[int(expert)]
        expert_hidden = F.silu(F.linear(hidden.float(), tensors["gate_proj"].float())) * F.linear(
            hidden.float(),
            tensors["up_proj"].float(),
        )
        expected += F.linear(expert_hidden, tensors["down_proj"].float()) * weights[..., rank : rank + 1]

    assert touched == sorted({int(value) for value in expected_selected.reshape(-1).tolist()})
    assert torch.equal(selected, expected_selected)
    assert torch.allclose(native_output.float(), expected.to(torch.bfloat16).float(), atol=3e-2, rtol=3e-2)


def _bootstrap_layer_bridge_fixture(
    tmp_path: Path, monkeypatch, *, explicit_head_dim: int | None = None
) -> tuple[str, Path]:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    model_id = "qwen-bridge-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)
    attention_head_dim = explicit_head_dim or 4
    q_projection_size = 2 * attention_head_dim
    kv_projection_size = attention_head_dim
    config_payload = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "bos_token_id": 0,
        "eos_token_id": 6,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 12,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "vocab_size": 8,
        "torch_dtype": "bfloat16",
    }
    if explicit_head_dim is not None:
        config_payload["head_dim"] = explicit_head_dim

    (model_dir / "config.json").write_text(
        json.dumps(config_payload),
        encoding="utf-8",
    )
    (model_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": 0,
                "pad_token_id": 0,
                "eos_token_id": [6, 0],
                "do_sample": True,
                "top_k": 3,
                "temperature": 0.8,
                "repetition_penalty": 1.1,
            }
        ),
        encoding="utf-8",
    )
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[PAD]": 0,
                "hello": 1,
                "world": 2,
                "there": 3,
                "friend": 4,
                "again": 5,
                "[EOS]": 6,
                "[UNK]": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (model_dir / "merges.txt").write_text("", encoding="utf-8")

    shard_path = model_dir / "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": torch.arange(64, dtype=torch.bfloat16).reshape(8, 8),
            "model.layers.0.input_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
            "model.layers.0.post_attention_layernorm.weight": torch.full((8,), 1.5, dtype=torch.bfloat16),
            "model.layers.0.self_attn.q_proj.weight": torch.eye(
                q_projection_size, 8, dtype=torch.bfloat16
            ),
            "model.layers.0.self_attn.q_proj.bias": torch.zeros((q_projection_size,), dtype=torch.bfloat16),
            "model.layers.0.self_attn.k_proj.weight": torch.eye(
                kv_projection_size, 8, dtype=torch.bfloat16
            ),
            "model.layers.0.self_attn.k_proj.bias": torch.zeros((kv_projection_size,), dtype=torch.bfloat16),
            "model.layers.0.self_attn.v_proj.weight": torch.eye(
                kv_projection_size, 8, dtype=torch.bfloat16
            ),
            "model.layers.0.self_attn.v_proj.bias": torch.zeros((kv_projection_size,), dtype=torch.bfloat16),
            "model.layers.0.self_attn.o_proj.weight": torch.eye(8, q_projection_size, dtype=torch.bfloat16),
            "model.layers.0.mlp.gate_proj.weight": torch.ones((12, 8), dtype=torch.bfloat16),
            "model.layers.0.mlp.up_proj.weight": torch.full((12, 8), 0.5, dtype=torch.bfloat16),
            "model.layers.0.mlp.down_proj.weight": torch.full((8, 12), 0.25, dtype=torch.bfloat16),
            "model.layers.1.input_layernorm.weight": torch.full((8,), 0.75, dtype=torch.bfloat16),
            "model.layers.1.post_attention_layernorm.weight": torch.full((8,), 1.25, dtype=torch.bfloat16),
            "model.layers.1.self_attn.q_proj.weight": torch.eye(
                q_projection_size, 8, dtype=torch.bfloat16
            ),
            "model.layers.1.self_attn.q_proj.bias": torch.full(
                (q_projection_size,), 0.1, dtype=torch.bfloat16
            ),
            "model.layers.1.self_attn.k_proj.weight": torch.eye(
                kv_projection_size, 8, dtype=torch.bfloat16
            ),
            "model.layers.1.self_attn.k_proj.bias": torch.full(
                (kv_projection_size,), 0.05, dtype=torch.bfloat16
            ),
            "model.layers.1.self_attn.v_proj.weight": torch.eye(
                kv_projection_size, 8, dtype=torch.bfloat16
            ),
            "model.layers.1.self_attn.v_proj.bias": torch.full(
                (kv_projection_size,), -0.05, dtype=torch.bfloat16
            ),
            "model.layers.1.self_attn.o_proj.weight": torch.eye(8, q_projection_size, dtype=torch.bfloat16),
            "model.layers.1.mlp.gate_proj.weight": torch.full((12, 8), 0.8, dtype=torch.bfloat16),
            "model.layers.1.mlp.up_proj.weight": torch.full((12, 8), 0.3, dtype=torch.bfloat16),
            "model.layers.1.mlp.down_proj.weight": torch.full((8, 12), 0.2, dtype=torch.bfloat16),
            "model.norm.weight": torch.ones((8,), dtype=torch.bfloat16),
            "lm_head.weight": torch.arange(64, dtype=torch.bfloat16).reshape(8, 8),
        },
        str(shard_path),
    )

    weight_names = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.bias",
        "model.layers.1.self_attn.k_proj.weight",
        "model.layers.1.self_attn.k_proj.bias",
        "model.layers.1.self_attn.v_proj.weight",
        "model.layers.1.self_attn.v_proj.bias",
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard_path.stat().st_size},
                "weight_map": {name: shard_path.name for name in weight_names},
            }
        ),
        encoding="utf-8",
    )

    build_tensor_execution_plan(model_id, model_dir)
    return model_id, model_dir


def test_load_layer_bridge_config_reads_required_qwen_values(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    config = load_layer_bridge_config(model_id)

    assert config.ready is True
    assert config.hidden_size == 8
    assert config.num_hidden_layers == 2
    assert config.num_attention_heads == 2
    assert config.num_key_value_heads == 1
    assert config.intermediate_size == 12
    assert config.eos_token_ids == [6, 0]
    assert config.bos_token_id == 0
    assert config.pad_token_id == 0


def test_run_minimal_layer_forward_bridge_uses_configured_attention_head_dim(
    tmp_path: Path, monkeypatch
) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(
        tmp_path, monkeypatch, explicit_head_dim=6
    )

    config = load_layer_bridge_config(model_id)
    result = run_minimal_layer_forward_bridge(model_id)

    assert config.head_dim == 6
    assert result.ready is True
    assert result.output_shape == [1, 1, 8]
    assert result.attention_head_dim == 6
    assert result.output_tensor is not None
    assert torch.isfinite(result.output_tensor).all()


def test_run_minimal_layer_forward_bridge_executes_real_layer_slice(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_minimal_layer_forward_bridge(model_id)

    assert result.ready is True
    assert result.loaded_unit_ids == ["layer-00-layer_norm", "layer-00-attention", "layer-00-mlp"]
    assert result.input_shape == [1, 1, 8]
    assert result.output_shape == [1, 1, 8]
    assert result.output_dtype == "torch.bfloat16"
    assert result.output_tensor is not None
    assert torch.isfinite(result.output_tensor).all()
    assert result.output_mean_abs > 0
    assert result.output_l2_norm > 0


def test_run_minimal_layer_forward_bridge_supports_multi_token_input_with_causal_mask(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    input_hidden = torch.zeros((1, 2, 8), dtype=torch.float32)
    result = run_minimal_layer_forward_bridge(model_id, input_hidden=input_hidden)

    assert result.ready is True
    assert result.input_shape == [1, 2, 8]
    assert result.output_shape == [1, 2, 8]
    assert result.output_tensor is not None
    assert torch.isfinite(result.output_tensor).all()


def test_run_minimal_layer_forward_bridge_supports_bfloat16_math_mode(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("PCKETLM_RUNTIME_MATH_DTYPE", "bf16")

    result = run_minimal_layer_forward_bridge(model_id)

    assert result.ready is True
    assert result.output_dtype == "torch.bfloat16"
    assert result.output_tensor is not None
    assert torch.isfinite(result.output_tensor.float()).all()


def test_q4_tensor_source_defaults_to_float16_math(monkeypatch) -> None:
    monkeypatch.delenv("PCKETLM_RUNTIME_MATH_DTYPE", raising=False)
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")

    assert runtime_math_dtype_name() == "float16"


def test_explicit_runtime_dtype_overrides_q4_default(monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.setenv("PCKETLM_RUNTIME_MATH_DTYPE", "bf16")

    assert runtime_math_dtype_name() == "bfloat16"


def test_q4_tensor_source_skips_stack_heavy_native_moe_by_default(monkeypatch) -> None:
    import pcketlm.native as native_module

    class NativeMoeCalled(BaseException):
        pass

    def fail_native(*_args, **_kwargs):
        raise NativeMoeCalled()

    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.delenv("PCKETLM_ENABLE_NATIVE_MOE_FOR_Q4", raising=False)
    monkeypatch.setattr(native_module, "moe_selected_forward_u16", fail_native)
    hidden = torch.tensor([[[0.1, -0.2, 0.3, -0.4]]], dtype=torch.float16)
    router = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float16)
    expert_tensors = {
        expert: {
            "gate_proj": torch.ones((3, 4), dtype=torch.float16) * (expert + 1),
            "up_proj": torch.ones((3, 4), dtype=torch.float16) * 0.5,
            "down_proj": torch.ones((4, 3), dtype=torch.float16) * 0.25,
        }
        for expert in (0, 1)
    }

    output, touched, selected = _run_moe_mlp(
        hidden_states=hidden,
        router_weight=router,
        expert_tensors=expert_tensors,
        top_k=1,
        norm_topk_prob=True,
    )

    assert output.shape == hidden.shape
    assert touched
    assert selected.shape == (1, 1, 1)


def test_q4_tensor_source_can_opt_into_native_moe(monkeypatch) -> None:
    import pcketlm.native as native_module

    calls = {"count": 0}

    def fake_native(hidden, _gate, _up, _down, _route):
        calls["count"] += 1
        return torch.zeros_like(hidden)

    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.setenv("PCKETLM_ENABLE_NATIVE_MOE_FOR_Q4", "1")
    monkeypatch.setattr(native_module, "moe_selected_forward_u16", fake_native)
    hidden = torch.tensor([[[0.1, -0.2, 0.3, -0.4]]], dtype=torch.float16)
    router = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float16)
    expert_tensors = {
        expert: {
            "gate_proj": torch.ones((3, 4), dtype=torch.float16),
            "up_proj": torch.ones((3, 4), dtype=torch.float16),
            "down_proj": torch.ones((4, 3), dtype=torch.float16),
        }
        for expert in (0, 1)
    }

    output, _touched, _selected = _run_moe_mlp(
        hidden_states=hidden,
        router_weight=router,
        expert_tensors=expert_tensors,
        top_k=1,
        norm_topk_prob=True,
    )

    assert calls["count"] == 1
    assert torch.equal(output, torch.zeros_like(hidden))


def test_native_q4_moe_prefill_defaults_on_with_kill_switch(monkeypatch) -> None:
    monkeypatch.delenv("PCKETLM_ENABLE_NATIVE_Q4_MOE_PREFILL", raising=False)
    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_Q4_MOE_PREFILL", raising=False)
    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_Q4_MOE", raising=False)

    assert _native_q4_moe_prefill_enabled() is True

    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_Q4_MOE_PREFILL", "1")
    assert _native_q4_moe_prefill_enabled() is False

    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_Q4_MOE_PREFILL", raising=False)
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_Q4_MOE", "1")
    assert _native_q4_moe_prefill_enabled() is False


def test_q4_prefix_scoped_handles_enabled_only_for_q4_moe(monkeypatch) -> None:
    moe_config = SimpleNamespace(num_experts=8, num_experts_per_tok=2)
    dense_config = SimpleNamespace(num_experts=0, num_experts_per_tok=0)

    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.delenv("PCKETLM_DISABLE_Q4_PREFIX_SCOPED_HANDLES", raising=False)
    assert _q4_prefix_scoped_handles_enabled(moe_config) is True
    assert _q4_prefix_scoped_handles_enabled(dense_config) is False

    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "fp16")
    assert _q4_prefix_scoped_handles_enabled(moe_config) is False

    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.setenv("PCKETLM_DISABLE_Q4_PREFIX_SCOPED_HANDLES", "1")
    assert _q4_prefix_scoped_handles_enabled(moe_config) is False


def test_native_lm_head_topk_stays_opt_in_for_q4_moe(monkeypatch) -> None:
    moe_config = SimpleNamespace(num_experts=8, num_experts_per_tok=2)
    dense_config = SimpleNamespace(num_experts=0, num_experts_per_tok=0)

    monkeypatch.setenv("PCKETLM_TENSOR_SOURCE", "q4")
    monkeypatch.delenv("PCKETLM_ENABLE_NATIVE_LM_HEAD_TOPK", raising=False)
    monkeypatch.delenv("PCKETLM_DISABLE_NATIVE_LM_HEAD_TOPK", raising=False)
    assert _native_lm_head_topk_enabled(moe_config) is False
    assert _native_lm_head_topk_enabled(dense_config) is False

    monkeypatch.setenv("PCKETLM_ENABLE_NATIVE_LM_HEAD_TOPK", "1")
    assert _native_lm_head_topk_enabled(dense_config) is True

    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_LM_HEAD_TOPK", "1")
    assert _native_lm_head_topk_enabled(moe_config) is False


def test_q4_moe_token_loop_matches_dequantized_selected_experts() -> None:
    from tools.quantize_to_q4 import dequantize_q4_tensor, quantize_tensor_to_q4

    class CountingPackedMap(dict):
        def __init__(self) -> None:
            super().__init__()
            self.lookup_count = 0

        def __getitem__(self, key):
            self.lookup_count += 1
            return super().__getitem__(key)

    torch.manual_seed(777)
    hidden_size = 8
    intermediate_size = 6
    hidden = torch.randn((1, 2, hidden_size), dtype=torch.float16)
    selected = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    routing = torch.tensor([[[0.75, 0.25], [0.6, 0.4]]], dtype=torch.float32)
    expert_name_maps = {
        0: {
            "gate_proj": "expert.0.gate",
            "up_proj": "expert.0.up",
            "down_proj": "expert.0.down",
        },
        1: {
            "gate_proj": "expert.1.gate",
            "up_proj": "expert.1.up",
            "down_proj": "expert.1.down",
        },
    }
    packed_by_name = CountingPackedMap()
    dequantized = {}
    for expert_index in (0, 1):
        dequantized[expert_index] = {}
        for role, shape in {
            "gate_proj": (intermediate_size, hidden_size),
            "up_proj": (intermediate_size, hidden_size),
            "down_proj": (hidden_size, intermediate_size),
        }.items():
            weight = torch.randn(shape, dtype=torch.float16)
            packed, scales, metadata = quantize_tensor_to_q4(weight)
            name = expert_name_maps[expert_index][role]
            packed_by_name[name] = (packed, scales, metadata["shape"])
            dequantized[expert_index][role] = dequantize_q4_tensor(
                packed,
                scales,
                metadata["shape"],
                dtype=torch.float16,
            )

    expected = torch.zeros_like(hidden.float())
    for token_index in range(hidden.shape[1]):
        token_hidden = hidden[:, token_index : token_index + 1, :].float()
        for slot_index, expert_id in enumerate(selected[0, token_index].tolist()):
            tensors = dequantized[int(expert_id)]
            expert_hidden = F.silu(F.linear(token_hidden, tensors["gate_proj"].float())) * F.linear(
                token_hidden, tensors["up_proj"].float()
            )
            expected[:, token_index : token_index + 1, :] += (
                F.linear(expert_hidden, tensors["down_proj"].float()) * routing[0, token_index, slot_index]
            )

    actual = _run_q4_moe_mlp_token_loop(
        hidden_states=hidden,
        selected_experts=selected,
        routing_weights=routing,
        expert_name_maps=expert_name_maps,
        packed_by_name=packed_by_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    assert torch.allclose(actual.float(), expected.to(torch.float16).float(), atol=1e-2, rtol=1e-2)
    assert packed_by_name.lookup_count == 6


def test_run_layer_bridge_stack_executes_two_real_layers_in_sequence(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_layer_bridge_stack(model_id, start_layer=0, layer_count=2)

    assert result.ready is True
    assert result.executed_layers == [0, 1]
    assert result.input_shape == [1, 1, 8]
    assert result.output_shape == [1, 1, 8]
    assert result.output_dtype == "torch.bfloat16"
    assert len(result.step_summaries) == 2
    assert all(summary.ready for summary in result.step_summaries)
    assert result.output_tensor is not None
    assert torch.isfinite(result.output_tensor).all()


def test_run_layer_bridge_stack_iterates_qwen32b_layer_count(monkeypatch) -> None:
    import pcketlm.core.runtime.layer_bridge as bridge_module

    monkeypatch.setattr(
        bridge_module,
        "load_layer_bridge_config",
        lambda _model_id: SimpleNamespace(ready=True, blockers=[], num_hidden_layers=64),
    )

    def fake_layer_forward(model_id: str, layer_index: int, input_hidden: torch.Tensor | None = None, **_kwargs) -> LayerBridgeResult:
        output = torch.zeros((1, 1, 1), dtype=torch.float32) if input_hidden is None else input_hidden + 1
        return LayerBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_mode="mock",
            input_shape=[1, 1, 1],
            output_shape=[1, 1, 1],
            output_dtype=str(output.dtype),
            loaded_unit_ids=[f"layer-{layer_index:02d}-mock"],
            attention_head_dim=1,
            cache_sequence_length=1,
            output_mean_abs=float(output.abs().mean().item()),
            output_l2_norm=float(output.norm().item()),
            blockers=[],
            ready=True,
            output_tensor=output,
        )

    monkeypatch.setattr(bridge_module, "run_minimal_layer_forward_bridge", fake_layer_forward)

    result = run_layer_bridge_stack(
        "qwen32b-loop-test",
        start_layer=0,
        layer_count=64,
        input_hidden=torch.zeros((1, 1, 1), dtype=torch.float32),
        collect_step_summaries=False,
    )

    assert result.ready is True
    assert result.executed_layers == list(range(64))
    assert result.output_shape == [1, 1, 1]
    assert result.output_tensor is not None
    assert result.output_tensor.item() == 64


def test_load_token_entry_hidden_state_reads_one_embedding_row_without_full_table(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    hidden_state, blockers = load_token_entry_hidden_state(model_id, [3])

    assert not blockers
    assert hidden_state is not None
    assert list(hidden_state.shape) == [1, 1, 8]
    expected = torch.arange(24, 32, dtype=hidden_state.dtype).view(1, 1, 8)
    assert torch.equal(hidden_state, expected)


def test_run_token_entry_layer_bridge_executes_from_token_id(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_token_entry_layer_bridge(model_id, token_ids=[3], start_layer=0, layer_count=2)

    assert result.ready is True
    assert result.executed_layers == [0, 1]
    assert result.embedding_shape == [1, 1, 8]
    assert result.output_shape == [1, 1, 8]
    assert result.output_tensor is not None
    assert torch.isfinite(result.output_tensor).all()


def test_run_decode_tail_streams_lm_head_and_returns_logits(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)
    stack_result = run_layer_bridge_stack(model_id, start_layer=0, layer_count=2)

    result = run_decode_tail(model_id, stack_result.output_tensor, lm_head_chunk_rows=3, top_k=3)

    assert result.ready is True
    assert result.logits_shape == [1, 1, 8]
    assert result.logits_dtype == "torch.bfloat16"
    assert result.chunk_count == 3
    assert len(result.top_token_ids) == 3
    assert result.logits is not None
    assert torch.isfinite(result.logits).all()


def test_run_decode_tail_can_stream_topk_without_full_logits(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)
    stack_result = run_layer_bridge_stack(model_id, start_layer=0, layer_count=2)

    result = run_decode_tail(
        model_id,
        stack_result.output_tensor,
        lm_head_chunk_rows=3,
        top_k=3,
        return_logits=False,
        recent_token_ids=[1, 2],
        repetition_penalty=1.1,
    )

    assert result.ready is True
    assert result.logits_shape == [1, 1, 8]
    assert result.chunk_count == 3
    assert len(result.top_token_ids) == 3
    assert len(result.top_logits) == 3
    assert result.logits is None


def test_run_token_decode_step_produces_logits_from_real_token_entry(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_token_decode_step(model_id, token_ids=[3], start_layer=0, layer_count=2, lm_head_chunk_rows=3, top_k=3)

    assert result.ready is True
    assert result.context_mode == "single-token-entry"
    assert result.chosen_token_id is not None
    assert result.executed_layers == [0, 1]
    assert result.embedding_shape == [1, 1, 8]
    assert result.hidden_shape == [1, 1, 8]
    assert result.logits_shape == [1, 1, 8]
    assert len(result.top_token_ids) == 3
    assert result.logits is not None
    assert torch.isfinite(result.logits).all()


def test_run_repeated_decode_loop_greedily_selects_next_tokens(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_repeated_decode_loop(model_id, seed_token_id=3, steps=2, start_layer=0, layer_count=2, lm_head_chunk_rows=3, top_k=3)

    assert result.ready is True
    assert result.steps_completed == 2
    assert result.generated_token_ids[0] == 3
    assert len(result.generated_token_ids) == 3
    assert len(result.step_summaries) == 2
    assert all(summary.ready for summary in result.step_summaries)
    assert any("history summary" in blocker for blocker in result.blockers)


def test_build_history_summary_hidden_state_carries_recent_tokens(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    hidden_state, context_tokens, context_mode, blockers = build_history_summary_hidden_state(
        model_id,
        token_ids=[1, 2, 3],
        history_window=2,
    )

    assert not blockers
    assert context_tokens == [2, 3]
    assert context_mode == "history-summary-no-kv-cache"
    assert hidden_state is not None
    assert list(hidden_state.shape) == [1, 1, 8]


def test_run_token_decode_step_can_use_history_summary_context(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_token_decode_step(
        model_id,
        token_ids=[1, 2, 3],
        start_layer=0,
        layer_count=2,
        lm_head_chunk_rows=3,
        top_k=3,
        history_window=3,
    )

    assert result.ready is True
    assert result.context_mode == "history-summary-no-kv-cache"
    assert result.context_token_ids == [1, 2, 3]
    assert result.history_window == 3
    assert result.logits_shape == [1, 1, 8]


def test_select_next_token_supports_greedy_and_top_k_sample() -> None:
    logits = torch.tensor([[[1.0, 3.0, 2.0, 0.5]]], dtype=torch.float32)

    greedy = select_next_token(logits, policy="greedy", top_k=3)
    sampled = select_next_token(
        logits,
        policy="top-k-sample",
        top_k=3,
        temperature=1.0,
        sample_seed=7,
    )

    assert greedy.ready is True
    assert greedy.chosen_token_id == 1
    assert sampled.ready is True
    assert sampled.chosen_token_id in sampled.top_token_ids


def test_select_next_token_applies_top_p_filtering() -> None:
    logits = torch.tensor([[[5.0, 4.0, 1.0, 0.1]]], dtype=torch.float32)

    sampled = select_next_token(
        logits,
        policy="top-k-sample",
        top_k=4,
        top_p=0.55,
        temperature=1.0,
        sample_seed=7,
    )

    assert sampled.ready is True
    assert sampled.chosen_token_id in {0, 1}


def test_run_repeated_decode_loop_can_use_top_k_sampling_policy(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_repeated_decode_loop(
        model_id,
        seed_token_id=3,
        steps=2,
        start_layer=0,
        layer_count=2,
        lm_head_chunk_rows=3,
        top_k=3,
        history_window=2,
        selection_policy="top-k-sample",
        sample_seed=11,
    )

    assert result.ready is True
    assert result.selection_policy == "top-k-sample"
    assert result.strategy == "top-k-sample-single-token-no-kv-cache"
    assert len(result.generated_token_ids) == 3


def test_run_kv_decode_loop_carries_cache_lengths_across_steps(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_kv_decode_loop(
        model_id,
        seed_token_id=3,
        steps=2,
        start_layer=0,
        layer_count=2,
        lm_head_chunk_rows=3,
        top_k=3,
    )

    assert result.ready is True
    assert result.strategy == "greedy-kv-cache-rope"
    assert len(result.generated_token_ids) == 3
    assert result.cache_sequence_lengths["0"] == 2
    assert result.cache_sequence_lengths["1"] == 2
    assert result.stop_reason == "step-limit"
    assert any("RoPE" in blocker for blocker in result.blockers)


def test_run_decode_benchmark_compares_history_and_kv_paths(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_decode_benchmark(
        model_id,
        seed_token_id=3,
        steps=2,
        start_layer=0,
        layer_count=2,
        lm_head_chunk_rows=3,
        top_k=3,
        history_window=2,
        sample_seed=11,
    )

    assert result.ready is True
    assert result.sample_seed == 11
    assert result.to_dict()["case_count"] == 3
    assert len(result.cases) == 3
    assert [case.label for case in result.cases] == [
        "history-greedy",
        "history-top-k-sample",
        "kv-greedy",
    ]
    assert result.cases[0].steps_completed == 2
    assert result.cases[0].final_token_id == result.cases[0].generated_token_ids[-1]
    assert result.cases[0].unique_token_count >= 2
    assert result.cases[0].stop_reason == "step-limit"
    assert result.cases[2].strategy == "greedy-kv-cache-rope"
    assert result.cases[2].cache_sequence_lengths["0"] == 2
    assert result.cases[2].stop_reason == "step-limit"


def test_run_kv_decode_step_can_advance_explicit_decode_state(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)
    decode_state = initialize_kv_decode_state(model_id, seed_token_id=3)

    first_step = run_kv_decode_step(
        model_id,
        input_token_id=decode_state.next_token_id,
        decode_state=decode_state,
        start_layer=0,
        layer_count=2,
        lm_head_chunk_rows=3,
        top_k=3,
    )

    assert first_step.ready is True
    assert first_step.next_decode_state is not None
    assert first_step.next_decode_state.ready is True
    assert first_step.next_decode_state.next_position == 1
    assert first_step.next_decode_state.generated_token_ids[0] == 3
    assert len(first_step.next_decode_state.generated_token_ids) == 2
    assert first_step.next_decode_state.cache_sequence_lengths["0"] == 1
    assert first_step.next_decode_state.finished is False
    assert first_step.next_decode_state.stop_reason is None


def test_run_prompt_decode_loop_uses_real_prompt_tokenization(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=2,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
    )

    assert result.ready is True
    assert result.prompt_token_ids == [1, 2]
    assert result.steps_completed == 2
    assert result.stop_reason == "step-limit"
    assert result.strategy == "greedy-prompt-kv-cache-rope"
    assert len(result.generated_token_ids) == 2
    assert result.cache_sequence_lengths["0"] >= 3
    assert isinstance(result.generated_text, str)
    assert isinstance(result.full_text, str)
    assert result.timings["total"] >= 0
    assert "prefill_decode_tail" in result.timings
    assert result.configured_layer_count == 2
    assert result.prompt_layer_count == 2
    assert result.layers_executed == 4
    assert result.expected_layers_executed == 4
    assert result.anti_cheat_passed is True


def test_run_prompt_decode_loop_prefix_reuse_returns_native_session_and_counts_layers(
    tmp_path: Path, monkeypatch
) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    first = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=1,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
    )
    assert first.ready is True
    assert first.final_decode_state is not None
    assert first.reusable_token_ids == [1, 2]

    second = run_prompt_decode_loop(
        model_id,
        prompt="hello world again",
        steps=1,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
        initial_decode_state=first.final_decode_state,
        initial_token_ids=first.reusable_token_ids,
    )

    assert second.ready is True
    assert second.prefix_reuse["used"] is True
    assert second.prefix_reuse["matched_token_count"] == 2
    assert second.prefix_reuse["appended_token_count"] == 1
    assert second.layers_executed == 2
    assert second.expected_layers_executed == 2
    assert second.anti_cheat_passed is True
    assert second.final_decode_state is not None
    assert isinstance(second.final_decode_state.native_kv_sessions, dict)


def test_run_prompt_decode_loop_can_commit_single_generated_token_for_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    first = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=1,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
        commit_generated_prefix=True,
    )
    assert first.ready is True
    assert first.final_decode_state is not None
    assert first.reusable_token_ids == first.prompt_token_ids
    assert first.final_decode_state.next_position == len(first.prompt_token_ids)
    assert first.prefix_reuse["generated_token_commit_skipped"]
    assert first.layers_executed == 2
    assert first.anti_cheat_passed is True

    chained_prompt = first.full_text + " again"
    second = run_prompt_decode_loop(
        model_id,
        prompt=chained_prompt,
        steps=1,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
        initial_decode_state=first.final_decode_state,
        initial_token_ids=first.reusable_token_ids,
    )

    assert second.ready is True
    assert second.prefix_reuse["used"] is True
    assert second.prefix_reuse["matched_token_count"] == len(first.reusable_token_ids)
    assert second.prefix_reuse["appended_token_count"] == len(second.prompt_token_ids) - first.final_decode_state.next_position


def test_run_prompt_decode_loop_reuses_pending_generated_token_without_full_prefill(
    tmp_path: Path, monkeypatch
) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    first = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=1,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
    )
    assert first.ready is True
    assert first.final_decode_state is not None

    second = run_prompt_decode_loop(
        model_id,
        prompt=first.full_text,
        steps=1,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
        initial_decode_state=first.final_decode_state,
        initial_token_ids=first.reusable_token_ids,
    )

    assert second.ready is True
    assert second.prefix_reuse["used"] is True
    assert second.prefix_reuse["appended_token_count"] == (
        len(second.prompt_token_ids) - first.final_decode_state.next_position
    )
    assert "prefill_stack" not in second.timings
    assert "prefix_append" in second.timings
    assert second.layers_executed == 2
    assert second.expected_layers_executed == 2
    assert second.anti_cheat_passed is True

    direct = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=2,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
    )
    assert direct.ready is True
    assert second.generated_token_ids == direct.generated_token_ids[1:2]


def test_run_prompt_decode_loop_reuses_exact_pending_prefix_token(
    tmp_path: Path, monkeypatch
) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    first = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=1,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
    )
    assert first.ready is True
    assert first.final_decode_state is not None

    second = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=1,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
        initial_decode_state=first.final_decode_state,
        initial_token_ids=first.reusable_token_ids,
    )

    assert second.ready is True
    assert second.generated_token_ids == first.generated_token_ids
    assert second.prefix_reuse["used"] is True
    assert second.prefix_reuse["pending_token_reused"] is True
    assert second.prefix_reuse["appended_token_count"] == 0
    assert "prefill_stack" not in second.timings
    assert "pending_prefix_token" in second.timings
    assert second.layers_executed == 0
    assert second.expected_layers_executed == 0
    assert second.anti_cheat_passed is True


def test_run_prompt_decode_loop_reuses_pending_prefix_then_continues(
    tmp_path: Path, monkeypatch
) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    first = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=1,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
    )
    assert first.ready is True
    assert first.final_decode_state is not None
    direct = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=2,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
    )
    assert direct.ready is True

    reused = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=2,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
        initial_decode_state=first.final_decode_state,
        initial_token_ids=first.reusable_token_ids,
    )

    assert reused.ready is True
    assert reused.generated_token_ids == direct.generated_token_ids
    assert reused.prefix_reuse["pending_token_reused"] is True
    assert reused.steps_completed == 2
    assert reused.layers_executed == 2
    assert reused.expected_layers_executed == 2
    assert reused.anti_cheat_passed is True
    assert "pending_prefix_continuation_total" in reused.timings


def test_run_prompt_prefill_session_returns_reusable_cached_tail_state(
    tmp_path: Path, monkeypatch
) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_prompt_prefill_session(
        model_id,
        prompt="hello world",
        start_layer=0,
        layer_count=2,
        apply_chat_format=False,
    )

    assert result.ready is True
    assert result.generated_token_ids == []
    assert result.steps_completed == 0
    assert result.reusable_token_ids == [1, 2]
    assert result.layers_executed == 2
    assert result.expected_layers_executed == 2
    assert result.anti_cheat_passed is True
    assert result.final_decode_state is not None
    assert result.final_decode_state.next_token_id == -1
    assert result.final_decode_state.next_position == 2
    assert result.final_decode_state.generated_token_ids == [1, 2]
    assert result.final_decode_state.last_hidden_state is not None


def test_run_prompt_decode_loop_reuses_cached_prefill_tail_for_first_token(
    tmp_path: Path, monkeypatch
) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    prefill = run_prompt_prefill_session(
        model_id,
        prompt="hello world",
        start_layer=0,
        layer_count=2,
        apply_chat_format=False,
    )
    assert prefill.ready is True
    direct = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=1,
        start_layer=0,
        layer_count=2,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
    )
    assert direct.ready is True

    reused = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=1,
        start_layer=0,
        layer_count=2,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
        initial_decode_state=prefill.final_decode_state,
        initial_token_ids=prefill.reusable_token_ids,
    )

    assert reused.ready is True
    assert reused.generated_token_ids == direct.generated_token_ids
    assert reused.prefix_reuse["used"] is True
    assert reused.prefix_reuse["cached_prefill_tail_used"] is True
    assert "prefill_stack" not in reused.timings
    assert "cached_prefill_tail" in reused.timings
    assert reused.layers_executed == 0
    assert reused.expected_layers_executed == 0
    assert reused.anti_cheat_passed is True


def test_run_prompt_decode_loop_reports_per_token_expert_telemetry(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=2,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="greedy",
        apply_chat_format=False,
    )

    payload = result.to_dict()
    assert result.ready is True
    assert [summary["token_index"] for summary in payload["token_summaries"]] == [1, 2]
    assert all("elapsed_seconds" in summary for summary in payload["token_summaries"])
    assert all("expert_hit_rate" in summary for summary in payload["token_summaries"])


def test_run_prompt_decode_loop_can_cancel_before_heavy_generation(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=2,
        start_layer=0,
        lm_head_chunk_rows=3,
        top_k=3,
        apply_chat_format=False,
        should_cancel=lambda: True,
    )

    assert result.ready is False
    assert result.stop_reason == "canceled"
    assert CANCEL_BLOCKER in result.blockers
    assert result.steps_completed == 0


def test_run_prompt_decode_loop_supports_raw_prompt_and_custom_system_controls(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    raw_result = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=1,
        start_layer=0,
        layer_count=2,
        lm_head_chunk_rows=3,
        top_k=3,
        selection_policy="top-k-sample",
        repetition_penalty=1.2,
        apply_chat_format=False,
        sample_seed=5,
    )

    assert raw_result.ready is True
    assert raw_result.prompt == "hello world"


def test_run_prompt_decode_loop_supports_min_new_tokens_and_stop_strings(tmp_path: Path, monkeypatch) -> None:
    model_id, _model_dir = _bootstrap_layer_bridge_fixture(tmp_path, monkeypatch)

    result = run_prompt_decode_loop(
        model_id,
        prompt="hello world",
        steps=2,
        start_layer=0,
        top_p=0.9,
        min_new_tokens=2,
        stop_strings=["friend"],
        apply_chat_format=False,
        sample_seed=5,
    )

    assert result.ready is True
    assert result.min_new_tokens == 2
    assert result.stop_strings == ["friend"]


def test_prepare_prompt_text_supports_chat_wrapping_when_metadata_is_present(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "prompt-wrap-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)

    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "<|im_start|>": 1,
                "<|im_end|>": 2,
                "system": 3,
                "user": 4,
                "assistant": 5,
                "Be": 6,
                "brief.": 7,
                "hello": 8,
                "world": 9,
                "You": 10,
                "are": 11,
                "Qwen,": 12,
                "created": 13,
                "by": 14,
                "Alibaba": 15,
                "Cloud.": 16,
                "a": 17,
                "helpful": 18,
                "assistant.": 19,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    "1": {"content": "<|im_start|>", "special": True},
                    "2": {"content": "<|im_end|>", "special": True},
                }
            }
        ),
        encoding="utf-8",
    )

    wrapped = prepare_prompt_text(model_id, "hello world", system_prompt="Be brief.", apply_chat_format=True)
    raw = prepare_prompt_text(model_id, "hello world", apply_chat_format=False)

    assert wrapped.ready is True
    assert raw.ready is True
    assert wrapped.prepared_prompt.startswith("<|im_start|>system\nBe brief.")
    assert raw.prepared_prompt == "hello world"
    assert len(wrapped.token_ids) > len(raw.token_ids)
