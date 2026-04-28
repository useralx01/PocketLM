"""Minimal layer-level forward bridge using real loaded Qwen tensors."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from pcketlm.core.runtime.tensor_loader import LoadedExecutionUnit, load_execution_unit


@dataclass(slots=True)
class LayerForwardBridgeResult:
    """One minimal real layer-forward pass using loaded execution units."""

    model_id: str
    layer_index: int
    input_shape: list[int]
    output_shape: list[int]
    output_dtype: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    loaded_unit_ids: list[str] = field(default_factory=list)
    output_mean: float = 0.0
    output_std: float = 0.0
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "layer_index": self.layer_index,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "loaded_unit_ids": list(self.loaded_unit_ids),
            "output_mean": self.output_mean,
            "output_std": self.output_std,
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


def _read_config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))


def _tensor_lookup(unit: LoadedExecutionUnit) -> dict[str, torch.Tensor]:
    return {
        tensor.tensor_name: tensor.tensor
        for tensor in unit.tensors
        if tensor.tensor is not None
    }


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden * torch.rsqrt(variance + eps)
    return normalized * weight.view(1, 1, -1)


def _linear(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    projected = F.linear(hidden, weight, bias)
    return projected


def _repeat_kv(hidden: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden
    return hidden.repeat_interleave(repeats, dim=1)


def run_minimal_layer_forward(model_id: str, model_dir: Path, layer_index: int = 0) -> LayerForwardBridgeResult:
    """Run a minimal real forward bridge for one Qwen layer on a synthetic single-token input."""
    config = _read_config(model_dir)
    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    intermediate_size = int(config["intermediate_size"])
    rms_norm_eps = float(config["rms_norm_eps"])
    head_dim = hidden_size // num_attention_heads
    kv_repeat = num_attention_heads // num_key_value_heads

    layer_norm_unit = load_execution_unit(model_id, f"layer-{layer_index:02d}-layer_norm")
    attention_unit = load_execution_unit(model_id, f"layer-{layer_index:02d}-attention")
    mlp_unit = load_execution_unit(model_id, f"layer-{layer_index:02d}-mlp")

    blockers = list(layer_norm_unit.blockers) + list(attention_unit.blockers) + list(mlp_unit.blockers)
    if not layer_norm_unit.ready or not attention_unit.ready or not mlp_unit.ready:
        if not blockers:
            blockers.append("One or more required execution units are not ready for the layer forward bridge.")
        return LayerForwardBridgeResult(
            model_id=model_id,
            layer_index=layer_index,
            input_shape=[],
            output_shape=[],
            output_dtype="unknown",
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            loaded_unit_ids=[layer_norm_unit.unit_id, attention_unit.unit_id, mlp_unit.unit_id],
            blockers=blockers,
            ready=False,
        )

    norm_tensors = _tensor_lookup(layer_norm_unit)
    attn_tensors = _tensor_lookup(attention_unit)
    mlp_tensors = _tensor_lookup(mlp_unit)

    input_norm = norm_tensors[f"model.layers.{layer_index}.input_layernorm.weight"].float()
    post_attn_norm = norm_tensors[f"model.layers.{layer_index}.post_attention_layernorm.weight"].float()

    q_weight = attn_tensors[f"model.layers.{layer_index}.self_attn.q_proj.weight"].float()
    q_bias = attn_tensors[f"model.layers.{layer_index}.self_attn.q_proj.bias"].float()
    k_weight = attn_tensors[f"model.layers.{layer_index}.self_attn.k_proj.weight"].float()
    k_bias = attn_tensors[f"model.layers.{layer_index}.self_attn.k_proj.bias"].float()
    v_weight = attn_tensors[f"model.layers.{layer_index}.self_attn.v_proj.weight"].float()
    v_bias = attn_tensors[f"model.layers.{layer_index}.self_attn.v_proj.bias"].float()
    o_weight = attn_tensors[f"model.layers.{layer_index}.self_attn.o_proj.weight"].float()

    gate_weight = mlp_tensors[f"model.layers.{layer_index}.mlp.gate_proj.weight"].float()
    up_weight = mlp_tensors[f"model.layers.{layer_index}.mlp.up_proj.weight"].float()
    down_weight = mlp_tensors[f"model.layers.{layer_index}.mlp.down_proj.weight"].float()

    torch.manual_seed(0)
    hidden = torch.randn((1, 1, hidden_size), dtype=torch.float32)
    residual = hidden

    hidden_norm = _rms_norm(hidden, input_norm, rms_norm_eps)
    q = _linear(hidden_norm, q_weight, q_bias)
    k = _linear(hidden_norm, k_weight, k_bias)
    v = _linear(hidden_norm, v_weight, v_bias)

    q = q.view(1, 1, num_attention_heads, head_dim).transpose(1, 2)
    k = k.view(1, 1, num_key_value_heads, head_dim).transpose(1, 2)
    v = v.view(1, 1, num_key_value_heads, head_dim).transpose(1, 2)
    k = _repeat_kv(k, kv_repeat)
    v = _repeat_kv(v, kv_repeat)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    context = torch.matmul(attn_weights, v)
    context = context.transpose(1, 2).contiguous().view(1, 1, hidden_size)
    attn_output = _linear(context, o_weight, None)
    hidden_after_attn = residual + attn_output

    post_norm_hidden = _rms_norm(hidden_after_attn, post_attn_norm, rms_norm_eps)
    gate = _linear(post_norm_hidden, gate_weight, None)
    up = _linear(post_norm_hidden, up_weight, None)
    mlp_hidden = F.silu(gate) * up
    mlp_output = _linear(mlp_hidden, down_weight, None)
    final_hidden = hidden_after_attn + mlp_output

    return LayerForwardBridgeResult(
        model_id=model_id,
        layer_index=layer_index,
        input_shape=[1, 1, hidden_size],
        output_shape=[int(value) for value in final_hidden.shape],
        output_dtype=str(final_hidden.dtype),
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        loaded_unit_ids=[layer_norm_unit.unit_id, attention_unit.unit_id, mlp_unit.unit_id],
        output_mean=float(final_hidden.mean().item()),
        output_std=float(final_hidden.std().item()),
        blockers=[],
        ready=True,
    )
