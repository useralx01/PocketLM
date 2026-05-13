import json
import struct
from pathlib import Path

import torch

import pcketlm.core.runtime.fp8_source as fp8_source
from pcketlm.core.runtime.fp8_source import (
    dequantize_fp8_block_scaled,
    load_fp8_token_embedding,
    run_fp8_decode_tail_topk,
    run_fp8_decode_loop,
    run_fp8_dense_mlp,
    run_fp8_prompt_prefill,
    load_dequantized_fp8_weight,
    load_fp8_weight_pair,
    plan_fp8_layer_working_set,
    run_fp8_expert_mlp,
    run_fp8_moe,
    run_fp8_router,
    run_fp8_single_token_forward,
)
from pcketlm.core.runtime.tensor_catalog import build_tensor_catalog, find_tensor_catalog_entry


def test_tensor_catalog_records_fp8_weight_scale_pairs(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)

    catalog = build_tensor_catalog(model_id, model_dir)

    assert catalog.ready is True
    assert catalog.fp8_weight_count == 16
    assert catalog.fp8_scale_count == 16
    assert catalog.fp8_pair_count == 16
    weight = find_tensor_catalog_entry(model_id, "model.layers.0.mlp.experts.1.gate_proj.weight")
    scale = find_tensor_catalog_entry(model_id, "model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv")
    assert weight is not None
    assert scale is not None
    assert weight.dtype == "F8_E4M3"
    assert weight.tensor_role == "weight"
    assert weight.scale_tensor_name == scale.tensor_name
    assert weight.physical_format == "fp8_block_scaled"
    assert scale.tensor_role == "scale_companion"
    assert scale.weight_tensor_name == weight.tensor_name


def test_plan_fp8_layer_working_set_keeps_selected_expert_subset(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    plan = plan_fp8_layer_working_set(model_id, 0, [1])

    assert plan.ready is True
    assert "model.layers.0.self_attn.q_proj.weight" in plan.tensor_names
    assert "model.layers.0.mlp.experts.1.gate_proj.weight" in plan.tensor_names
    assert "model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv" in plan.tensor_names
    assert "model.layers.0.mlp.experts.1.up_proj.weight" in plan.tensor_names
    assert "model.layers.0.mlp.experts.1.down_proj.weight" in plan.tensor_names
    assert "model.layers.0.mlp.shared_experts.gate_proj.weight" in plan.tensor_names
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in plan.tensor_names
    assert plan.fp8_weight_count == 15
    assert plan.scale_count == 15
    assert plan.total_nbytes == plan.fp8_weight_bytes + plan.scale_bytes + plan.non_fp8_bytes


def test_load_fp8_weight_pair_reads_raw_weight_bytes_and_scale(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    loaded = load_fp8_weight_pair(model_id, "model.layers.0.self_attn.q_proj.weight")

    assert loaded.ready is True
    assert loaded.fp8_bytes is not None
    assert loaded.scale_tensor is not None
    assert loaded.fp8_bytes.dtype == torch.uint8
    assert loaded.fp8_bytes.shape == (2, 4)
    assert loaded.fp8_bytes.flatten().tolist() == _fp8_bytes(
        torch.tensor([[0.5, 1.0, 2.0, -1.0], [3.0, 4.0, -2.0, -0.5]], dtype=torch.float32)
    )
    assert loaded.scale_tensor.dtype == torch.float32
    assert torch.allclose(loaded.scale_tensor, torch.tensor([[0.5]], dtype=torch.float32))


def test_dequantize_fp8_block_scaled_matches_torch_float8_reference() -> None:
    values = torch.tensor([[0.5, 1.0, 2.0, -1.0], [3.0, 4.0, -2.0, -0.5]], dtype=torch.float32)
    fp8_bytes = values.to(torch.float8_e4m3fn).view(torch.uint8)
    scale = torch.tensor([[2.0]], dtype=torch.float32)

    dequantized = dequantize_fp8_block_scaled(fp8_bytes, scale, dtype=torch.float32)

    assert torch.allclose(dequantized, values.to(torch.float8_e4m3fn).float() * 2.0)


def test_load_dequantized_fp8_weight_pair_applies_scale(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    loaded = load_dequantized_fp8_weight(model_id, "model.layers.0.self_attn.q_proj.weight", dtype=torch.float32)

    expected = torch.tensor([[0.5, 1.0, 2.0, -1.0], [3.0, 4.0, -2.0, -0.5]], dtype=torch.float32)
    assert loaded.ready is True
    assert loaded.tensor is not None
    assert torch.allclose(loaded.tensor, expected.to(torch.float8_e4m3fn).float() * 0.5)


def test_run_fp8_expert_mlp_materializes_one_selected_expert(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)
    hidden = torch.ones((1, 4), dtype=torch.bfloat16)

    result = run_fp8_expert_mlp(model_id, 0, 1, hidden, dtype=torch.float32)

    assert result.ready is True
    assert result.output_tensor is not None
    assert result.output_shape == [1, 4]
    assert result.loaded_weight_bytes > 0
    assert result.dequantized_weight_bytes > 0


def test_run_fp8_router_uses_sigmoid_bias_for_selection(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)
    hidden = torch.ones((1, 4), dtype=torch.bfloat16)

    result = run_fp8_router(model_id, 0, hidden)

    assert result.ready is True
    assert result.selected_experts == [1]
    assert result.bias_loaded is True
    assert result.scoring_func == "sigmoid"
    assert result.weights_tensor is not None
    assert torch.allclose(result.weights_tensor.float(), torch.tensor([[2.5]], dtype=torch.float32))


def test_run_fp8_moe_combines_routed_and_shared_experts(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)
    hidden = torch.ones((1, 4), dtype=torch.bfloat16)

    result = run_fp8_moe(model_id, 0, hidden, dtype=torch.float32)

    assert result.ready is True
    assert result.output_tensor is not None
    assert result.output_shape == [1, 4]
    assert result.selected_experts == [1]
    assert result.routed_weight_bytes > 0
    assert result.shared_weight_bytes > 0
    assert result.dequantized_weight_bytes > 0


def test_run_fp8_dense_mlp_materializes_dense_layer(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)
    hidden = torch.ones((1, 1, 4), dtype=torch.bfloat16)

    result = run_fp8_dense_mlp(model_id, 0, hidden, dtype=torch.float32)

    assert result.ready is True
    assert result.output_tensor is not None
    assert result.output_shape == [1, 1, 4]


def test_run_fp8_dense_mlp_uses_native_streamed_linear_when_available(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)
    calls = {"count": 0}

    def fake_native_linear(fp8_rows: torch.Tensor, scale_rows: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        weight = dequantize_fp8_block_scaled(fp8_rows, scale_rows, dtype=torch.float32)
        return torch.nn.functional.linear(hidden.reshape(-1, hidden.shape[-1]).float(), weight.float())

    monkeypatch.setattr(fp8_source, "native_fp8_linear_available", lambda: True)
    monkeypatch.setattr(fp8_source, "native_fp8_dual_linear_available", lambda: False)
    monkeypatch.setattr(fp8_source, "fp8_e4m3_block_linear_f32", fake_native_linear)
    monkeypatch.setenv("PCKETLM_ENABLE_NATIVE_FP8_LINEAR", "1")
    hidden = torch.ones((1, 1, 4), dtype=torch.bfloat16)

    result = run_fp8_dense_mlp(model_id, 0, hidden, dtype=torch.float32)

    assert result.ready is True
    assert result.output_tensor is not None
    assert calls["count"] == 3


def test_run_fp8_dense_mlp_uses_native_dual_gate_up_when_available(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)
    calls = {"dual": 0, "single": 0}

    def fake_dual_linear(
        fp8_a: torch.Tensor,
        scale_a: torch.Tensor,
        fp8_b: torch.Tensor,
        scale_b: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls["dual"] += 1
        weight_a = dequantize_fp8_block_scaled(fp8_a, scale_a, dtype=torch.float32)
        weight_b = dequantize_fp8_block_scaled(fp8_b, scale_b, dtype=torch.float32)
        flat = hidden.reshape(-1, hidden.shape[-1]).float()
        return torch.nn.functional.linear(flat, weight_a.float()), torch.nn.functional.linear(flat, weight_b.float())

    def fake_native_linear(fp8_rows: torch.Tensor, scale_rows: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        calls["single"] += 1
        weight = dequantize_fp8_block_scaled(fp8_rows, scale_rows, dtype=torch.float32)
        return torch.nn.functional.linear(hidden.reshape(-1, hidden.shape[-1]).float(), weight.float())

    monkeypatch.setattr(fp8_source, "native_fp8_dual_linear_available", lambda: True)
    monkeypatch.setattr(fp8_source, "fp8_e4m3_block_dual_linear_f32", fake_dual_linear)
    monkeypatch.setattr(fp8_source, "native_fp8_linear_available", lambda: True)
    monkeypatch.setattr(fp8_source, "fp8_e4m3_block_linear_f32", fake_native_linear)
    hidden = torch.ones((1, 1, 4), dtype=torch.bfloat16)

    result = run_fp8_dense_mlp(model_id, 0, hidden, dtype=torch.float32)

    assert result.ready is True
    assert result.output_tensor is not None
    assert calls == {"dual": 1, "single": 1}


def test_run_fp8_decode_tail_streams_lm_head_chunks(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.bfloat16)

    result = run_fp8_decode_tail_topk(model_id, hidden, top_k=2, chunk_rows=2)

    assert result.ready is True
    assert result.chunk_count == 2
    assert result.top_token_ids[0] == 1
    assert result.loaded_lm_head_bytes == 3 * 4 * 2


def test_load_fp8_token_embedding_reads_one_row(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    result = load_fp8_token_embedding(model_id, 1)

    assert result.ready is True
    assert result.output_tensor is not None
    assert result.output_shape == [1, 1, 4]
    assert torch.allclose(result.output_tensor.float(), torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))


def test_run_fp8_single_token_forward_carries_kv_cache(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    result = run_fp8_single_token_forward(model_id, 1, layer_count=1, include_tail=False, dtype=torch.float32)

    assert result.ready is True
    assert result.executed_layers == [0]
    assert result.output_tensor is not None
    assert result.hidden_shape == [1, 1, 4]
    assert result.step_summaries[0]["ffn_type"] == "dense"
    assert result.step_summaries[0]["cache_sequence_length"] == 1

    next_result = run_fp8_single_token_forward(
        model_id,
        2,
        layer_count=1,
        include_tail=False,
        dtype=torch.float32,
        position=1,
        previous_kv_caches=result.next_kv_caches,
    )

    assert next_result.ready is True
    assert next_result.step_summaries[0]["cache_sequence_length"] == 2


def test_run_fp8_decode_loop_generates_from_prompt_tail(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    result = run_fp8_decode_loop(model_id, [1, 2], layer_count=0, max_new_tokens=1, dtype=torch.float32)

    assert result.ready is True
    assert result.prompt_token_ids == [1, 2]
    assert result.generated_token_ids == [result.final_top_token_ids[0]]
    assert result.positions_completed == 3
    assert result.step_summaries[-1]["cache_sequence_lengths"] == {}


def test_run_fp8_prompt_prefill_processes_prompt_layer_wise(tmp_path: Path, monkeypatch) -> None:
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)

    result = run_fp8_prompt_prefill(model_id, [1, 2], layer_count=1, include_tail=False, dtype=torch.float32)

    assert result.ready is True
    assert result.executed_layers == [0]
    assert result.output_tensor is not None
    assert result.hidden_shape == [1, 2, 4]
    assert result.next_kv_caches[0][0].shape[1] == 2


def _write_fp8_runtime_fixture(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "deepseek-fp8-test"
    model_dir = tmp_path / "models" / model_id / "original"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "intermediate_size": 4,
                "qk_nope_head_dim": 1,
                "qk_rope_head_dim": 2,
                "v_head_dim": 1,
                "kv_lora_rank": 1,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "rms_norm_eps": 1e-6,
                "first_k_dense_replace": 1,
                "n_group": 1,
                "topk_group": 1,
                "scoring_func": "sigmoid",
                "routed_scaling_factor": 2.5,
                "vocab_size": 8,
                "quantization_config": {"fmt": "e4m3", "quant_method": "fp8"},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    attention = torch.tensor([[0.5, 1.0, 2.0, -1.0], [3.0, 4.0, -2.0, -0.5]], dtype=torch.float32)
    router_gate = torch.tensor([[4.0, 4.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    dense_gate = torch.tensor([[0.5, 1.0, -0.5, 0.25], [1.0, 0.0, 0.5, -0.25]], dtype=torch.float32)
    dense_up = torch.tensor([[0.25, -0.25, 1.0, 0.5], [0.75, 0.5, -0.5, 0.25]], dtype=torch.float32)
    dense_down = torch.tensor([[1.0, -0.5], [0.25, 0.75], [-0.25, 0.5], [0.5, -1.0]], dtype=torch.float32)
    expert_gate = torch.tensor([[1.0, 0.5, -1.0, 2.0], [0.25, -0.5, 1.5, -2.0]], dtype=torch.float32)
    expert_up = torch.tensor([[0.5, -1.0, 1.0, 0.25], [1.5, 0.5, -0.25, 1.0]], dtype=torch.float32)
    expert_down = torch.tensor([[1.0, -0.5], [0.25, 1.5], [-1.0, 0.5], [2.0, -1.5]], dtype=torch.float32)
    shared_gate = torch.tensor([[0.25, 0.5, 0.75, 1.0], [1.0, -0.5, 0.25, -0.75]], dtype=torch.float32)
    shared_up = torch.tensor([[0.5, 0.5, -0.5, -0.5], [1.0, 0.25, 0.5, -0.25]], dtype=torch.float32)
    shared_down = torch.tensor([[0.25, -0.25], [0.5, 1.0], [-0.75, 0.5], [1.25, -1.0]], dtype=torch.float32)
    lm_head = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]], dtype=torch.float32)
    embedding = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": ("F8_E4M3", [2, 4], bytes(_fp8_bytes(attention))),
        "model.layers.0.self_attn.q_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 0.5)),
        "model.layers.0.mlp.gate.weight": ("F32", [2, 4], _f32_payload(router_gate)),
        "model.layers.0.mlp.gate.e_score_correction_bias": ("F32", [2], struct.pack("<ff", -20.0, 0.0)),
        "model.layers.0.mlp.gate_proj.weight": ("F8_E4M3", [2, 4], bytes(_fp8_bytes(dense_gate))),
        "model.layers.0.mlp.gate_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.mlp.up_proj.weight": ("F8_E4M3", [2, 4], bytes(_fp8_bytes(dense_up))),
        "model.layers.0.mlp.up_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.mlp.down_proj.weight": ("F8_E4M3", [4, 2], bytes(_fp8_bytes(dense_down))),
        "model.layers.0.mlp.down_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.mlp.experts.0.gate_proj.weight": ("F8_E4M3", [2, 4], bytes([9, 10, 11, 12, 13, 14, 15, 16])),
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.mlp.experts.1.gate_proj.weight": ("F8_E4M3", [2, 4], bytes(_fp8_bytes(expert_gate))),
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 2.0)),
        "model.layers.0.mlp.experts.1.up_proj.weight": ("F8_E4M3", [2, 4], bytes(_fp8_bytes(expert_up))),
        "model.layers.0.mlp.experts.1.up_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.5)),
        "model.layers.0.mlp.experts.1.down_proj.weight": ("F8_E4M3", [4, 2], bytes(_fp8_bytes(expert_down))),
        "model.layers.0.mlp.experts.1.down_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 0.75)),
        "model.layers.0.mlp.shared_experts.gate_proj.weight": ("F8_E4M3", [2, 4], bytes(_fp8_bytes(shared_gate))),
        "model.layers.0.mlp.shared_experts.gate_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.mlp.shared_experts.up_proj.weight": ("F8_E4M3", [2, 4], bytes(_fp8_bytes(shared_up))),
        "model.layers.0.mlp.shared_experts.up_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.mlp.shared_experts.down_proj.weight": ("F8_E4M3", [4, 2], bytes(_fp8_bytes(shared_down))),
        "model.layers.0.mlp.shared_experts.down_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.input_layernorm.weight": ("BF16", [4], b"\x00\x00" * 4),
        "model.layers.0.post_attention_layernorm.weight": ("BF16", [4], _bf16_payload(torch.ones(4))),
        "model.layers.0.self_attn.q_a_proj.weight": ("F8_E4M3", [2, 4], bytes(_fp8_bytes(attention))),
        "model.layers.0.self_attn.q_a_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.self_attn.q_b_proj.weight": ("F8_E4M3", [3, 2], bytes(_fp8_bytes(torch.ones((3, 2))))),
        "model.layers.0.self_attn.q_b_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": ("F8_E4M3", [3, 4], bytes(_fp8_bytes(torch.ones((3, 4))))),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.self_attn.kv_b_proj.weight": ("F8_E4M3", [2, 1], bytes(_fp8_bytes(torch.ones((2, 1))))),
        "model.layers.0.self_attn.kv_b_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.self_attn.o_proj.weight": ("F8_E4M3", [4, 1], bytes(_fp8_bytes(torch.ones((4, 1))))),
        "model.layers.0.self_attn.o_proj.weight_scale_inv": ("F32", [1, 1], struct.pack("<f", 1.0)),
        "model.layers.0.self_attn.q_a_layernorm.weight": ("BF16", [2], _bf16_payload(torch.ones(2))),
        "model.layers.0.self_attn.kv_a_layernorm.weight": ("BF16", [1], _bf16_payload(torch.ones(1))),
        "model.embed_tokens.weight": ("BF16", [3, 4], _bf16_payload(embedding)),
        "model.norm.weight": ("BF16", [4], _bf16_payload(torch.ones(4))),
        "lm_head.weight": ("BF16", [3, 4], _bf16_payload(lm_head)),
    }
    shard = model_dir / "model-00001-of-00001.safetensors"
    _write_safetensors_bytes(shard, tensors)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": shard.stat().st_size}, "weight_map": {name: shard.name for name in tensors}}),
        encoding="utf-8",
    )
    return model_id, model_dir


def _write_safetensors_bytes(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    offset = 0
    data = bytearray()
    header = {}
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(payload)]}
        data.extend(payload)
        offset += len(payload)
    header_payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_payload)) + header_payload + bytes(data))


def _fp8_bytes(values: torch.Tensor) -> list[int]:
    return values.to(torch.float8_e4m3fn).view(torch.uint8).flatten().tolist()


def _f32_payload(values: torch.Tensor) -> bytes:
    return bytes(values.contiguous().to(dtype=torch.float32).view(torch.uint8).flatten().tolist())


def _bf16_payload(values: torch.Tensor) -> bytes:
    return bytes(values.contiguous().to(dtype=torch.bfloat16).view(torch.uint8).flatten().tolist())
