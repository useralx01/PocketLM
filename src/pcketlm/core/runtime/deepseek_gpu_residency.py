"""Local DeepSeek FP8 GPU residency and paging helpers."""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from pcketlm.core.runtime.deepseek_remote_gpu import _rms_norm, _softmax_scale
from pcketlm.core.runtime.fp8_source import (
    dequantize_fp8_weight_pair,
    load_fp8_weight_pair,
    plan_fp8_layer_working_set,
)
from pcketlm.core.runtime.tensor_catalog import find_tensor_catalog_entry, load_tensor_catalog


DEFAULT_T4_RESIDENT_SECONDS_PER_LAYER = 0.02892160244443984


@dataclass(frozen=True, slots=True)
class DeepSeekGPUResidencyEstimate:
    """Header-only estimate for the local FP8 GPU residency path."""

    model_id: str
    ready: bool
    model_dir: str
    config_hidden_layers: int
    start_layer: int
    layer_count: int
    selected_experts: list[int]
    resident_budget_bytes: int
    max_source_layer_bytes: int
    max_dequantized_layer_bytes: int
    layers_fit_by_source_bytes: int
    layers_fit_by_dequantized_bytes: int
    projected_hot_seconds_per_token: float
    measured_seconds_per_resident_layer: float
    speed_target_met: bool
    layer_plans: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocalDeepSeekPagedDecodeResult:
    """Bounded local decode probe through the resident layer pager."""

    passed: bool
    cuda_available: bool
    device: str
    device_name: str
    torch_version: str
    model_id: str
    token_id: int
    start_layer: int
    layer_count: int
    config_hidden_layers: int
    executed_layers: list[int]
    selected_experts_by_layer: dict[int, list[int]]
    tail_top_token_ids: list[int]
    tail_top_logits: list[float]
    output_shape: list[int]
    cache_sequence_lengths: dict[int, int]
    checksum: float
    max_abs_value: float
    resident_budget_bytes: int
    peak_resident_bytes: int
    final_resident_bytes: int
    pager_loads: int
    pager_evictions: int
    pager_cache_hits: int
    pager_cache_misses: int
    pager_prefetch_submitted: int
    pager_prefetch_completed: int
    tail_elapsed_seconds: float
    elapsed_seconds: float
    seconds_per_layer: float
    projected_config_layers_seconds_per_token: float
    blockers: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalDeepSeekResidentLayer:
    """One DeepSeek layer with dequantized weights resident on a target device."""

    def __init__(self, model_id: str, layer_index: int, *, config: dict[str, Any], dtype: torch.dtype, device: torch.device) -> None:
        self.model_id = str(model_id)
        self.layer_index = int(layer_index)
        self.config = config
        self.dtype = dtype
        self.device = device
        self.weights: dict[str, torch.Tensor] = {}
        self.regular: dict[str, torch.Tensor] = {}
        self.selected_experts: list[int] = []
        self._expert_gate: torch.Tensor | None = None
        self._expert_up: torch.Tensor | None = None
        self._expert_down: torch.Tensor | None = None
        self._load_attention_and_router()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output, _cache = self.forward_with_cache(hidden)
        return output

    def forward_with_cache(
        self,
        hidden: torch.Tensor,
        *,
        start_pos: int = 0,
        previous_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        hidden = hidden.to(device=self.device, dtype=self.dtype)
        input_norm = self.regular[f"model.layers.{self.layer_index}.input_layernorm.weight"]
        post_norm = self.regular[f"model.layers.{self.layer_index}.post_attention_layernorm.weight"]
        normed = _rms_norm(hidden, input_norm, float(self.config["rms_norm_eps"]))
        attn_out, next_cache = self._attention(normed, start_pos=int(start_pos), previous_kv_cache=previous_kv_cache)
        hidden_after_attn = hidden + attn_out.to(dtype=self.dtype)
        ffn_input = _rms_norm(hidden_after_attn, post_norm, float(self.config["rms_norm_eps"]))
        if int(self.layer_index) < int(self.config.get("first_k_dense_replace", 0) or 0):
            ffn_out = self._dense_mlp(ffn_input)
        else:
            route_weights, route_indices = self._route(ffn_input)
            selected = sorted({int(value) for value in route_indices.reshape(-1).tolist()})
            if selected != self.selected_experts:
                self._load_moe(selected)
            ffn_out = self._moe(ffn_input, route_weights, route_indices)
        return (hidden_after_attn + ffn_out.to(dtype=self.dtype)).contiguous(), next_cache

    def resident_nbytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for tensor in list(self.weights.values()) + list(self.regular.values()):
            ident = id(tensor)
            if ident in seen:
                continue
            seen.add(ident)
            total += int(tensor.nelement() * tensor.element_size())
        for tensor in (self._expert_gate, self._expert_up, self._expert_down):
            if tensor is not None:
                total += int(tensor.nelement() * tensor.element_size())
        return int(total)

    def release(self) -> None:
        self.weights.clear()
        self.regular.clear()
        self.selected_experts = []
        self._expert_gate = None
        self._expert_up = None
        self._expert_down = None

    def _load_attention_and_router(self) -> None:
        layer = self.layer_index
        for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            name = f"model.layers.{layer}.{suffix}"
            self.regular[name] = _load_regular_tensor(self.model_id, name, device=self.device)
        prefix = f"model.layers.{layer}.self_attn"
        for role in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
            name = f"{prefix}.{role}.weight"
            self.weights[name] = _load_dequantized_weight(self.model_id, name, dtype=self.dtype, device=self.device)
        for role in ("q_a_layernorm", "kv_a_layernorm"):
            name = f"{prefix}.{role}.weight"
            self.regular[name] = _load_regular_tensor(self.model_id, name, device=self.device)
        router_name = f"model.layers.{layer}.mlp.gate.weight"
        self.regular[router_name] = _load_regular_tensor(self.model_id, router_name, device=self.device).float()
        bias_name = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
        try:
            self.regular[bias_name] = _load_regular_tensor(self.model_id, bias_name, device=self.device).float()
        except KeyError:
            pass

    def _load_dense_mlp(self) -> None:
        prefix = f"model.layers.{self.layer_index}.mlp"
        for role in ("gate_proj", "up_proj", "down_proj"):
            name = f"{prefix}.{role}.weight"
            self.weights.setdefault(name, _load_dequantized_weight(self.model_id, name, dtype=self.dtype, device=self.device))

    def _load_moe(self, selected: list[int]) -> None:
        layer = self.layer_index
        for expert_index in selected:
            prefix = f"model.layers.{layer}.mlp.experts.{expert_index}"
            for role in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}.{role}.weight"
                self.weights.setdefault(name, _load_dequantized_weight(self.model_id, name, dtype=self.dtype, device=self.device))
        if int(self.config.get("n_shared_experts", 0) or 0) > 0:
            prefix = f"model.layers.{layer}.mlp.shared_experts"
            for role in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}.{role}.weight"
                self.weights.setdefault(name, _load_dequantized_weight(self.model_id, name, dtype=self.dtype, device=self.device))
        self._expert_gate = torch.stack(
            [self.weights[f"model.layers.{layer}.mlp.experts.{idx}.gate_proj.weight"] for idx in selected]
        ).contiguous()
        self._expert_up = torch.stack(
            [self.weights[f"model.layers.{layer}.mlp.experts.{idx}.up_proj.weight"] for idx in selected]
        ).contiguous()
        self._expert_down = torch.stack(
            [self.weights[f"model.layers.{layer}.mlp.experts.{idx}.down_proj.weight"] for idx in selected]
        ).contiguous()
        self.selected_experts = list(selected)

    def _attention(
        self,
        hidden: torch.Tensor,
        *,
        start_pos: int,
        previous_kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from pcketlm.core.runtime.fp8_source import _apply_rope_real

        prefix = f"model.layers.{self.layer_index}.self_attn"
        q_a = self._linear(f"{prefix}.q_a_proj.weight", hidden)
        q_a_norm = _rms_norm(q_a, self.regular[f"{prefix}.q_a_layernorm.weight"], float(self.config["rms_norm_eps"]))
        q = self._linear(f"{prefix}.q_b_proj.weight", q_a_norm).float()
        kv = self._linear(f"{prefix}.kv_a_proj_with_mqa.weight", hidden).float()

        heads = int(self.config["num_attention_heads"])
        qk_nope = int(self.config["qk_nope_head_dim"])
        qk_rope = int(self.config["qk_rope_head_dim"])
        v_head_dim = int(self.config["v_head_dim"])
        kv_lora_rank = int(self.config["kv_lora_rank"])
        q = q.view(q.shape[0], q.shape[1], heads, qk_nope + qk_rope)
        q_nope, q_pe = torch.split(q, [qk_nope, qk_rope], dim=-1)
        kv_latent, k_pe = torch.split(kv, [kv_lora_rank, qk_rope], dim=-1)
        kv_latent = _rms_norm(kv_latent, self.regular[f"{prefix}.kv_a_layernorm.weight"], float(self.config["rms_norm_eps"]))
        kv_b = self.weights[f"{prefix}.kv_b_proj.weight"].float()
        wkv_b = kv_b.view(heads, qk_nope + v_head_dim, kv_lora_rank)
        q_nope_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope])
        q_pe = _apply_rope_real(q_pe, start_pos=int(start_pos), config=self.config)
        k_pe = _apply_rope_real(k_pe.unsqueeze(2), start_pos=int(start_pos), config=self.config).squeeze(2)
        if previous_kv_cache is not None:
            previous_kv, previous_pe = previous_kv_cache
            kv_cache = torch.cat([previous_kv.to(device=hidden.device, dtype=kv_latent.dtype), kv_latent], dim=1)
            pe_cache = torch.cat([previous_pe.to(device=hidden.device, dtype=k_pe.dtype), k_pe], dim=1)
        else:
            kv_cache = kv_latent
            pe_cache = k_pe
        scores = (
            torch.einsum("bshc,btc->bsht", q_nope_absorbed, kv_cache)
            + torch.einsum("bshr,btr->bsht", q_pe, pe_cache)
        ) * _softmax_scale(self.config)
        if previous_kv_cache is None and int(hidden.shape[1]) > 1:
            seq_len = int(hidden.shape[1])
            causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden.device).triu(1)
            scores = scores.masked_fill(causal_mask.view(1, seq_len, 1, seq_len), float("-inf"))
        probs = scores.softmax(dim=-1, dtype=torch.float32).to(dtype=hidden.dtype)
        latent = torch.einsum("bsht,btc->bshc", probs, kv_cache)
        attention_heads = torch.einsum("bshc,hdc->bshd", latent, wkv_b[:, -v_head_dim:])
        return self._linear(f"{prefix}.o_proj.weight", attention_heads.flatten(2)), (
            kv_cache.detach().contiguous(),
            pe_cache.detach().contiguous(),
        )

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layer_index
        router = self.regular[f"model.layers.{layer}.mlp.gate.weight"]
        bias = self.regular.get(f"model.layers.{layer}.mlp.gate.e_score_correction_bias")
        scores = F.linear(hidden.reshape(-1, hidden.shape[-1]).float(), router)
        scoring = str(self.config.get("scoring_func", "sigmoid"))
        scores = scores.sigmoid() if scoring == "sigmoid" else scores.softmax(dim=-1, dtype=torch.float32)
        choice_scores = scores + bias if bias is not None else scores
        n_groups = int(self.config.get("n_group", 1) or 1)
        topk_groups = int(self.config.get("topk_group", 1) or 1)
        if n_groups > 1:
            grouped = choice_scores.view(choice_scores.shape[0], n_groups, -1)
            group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1) if bias is not None else grouped.amax(dim=-1)
            keep_groups = group_scores.topk(max(1, min(topk_groups, n_groups)), dim=-1)[1]
            mask = grouped.new_ones(choice_scores.shape[0], n_groups, dtype=torch.bool).scatter_(1, keep_groups, False)
            choice_scores = grouped.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
        top_k = max(1, min(int(self.config.get("num_experts_per_tok", 1) or 1), choice_scores.shape[-1]))
        indices = torch.topk(choice_scores, top_k, dim=-1)[1]
        weights = scores.gather(1, indices)
        if scoring == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return weights * float(self.config.get("routed_scaling_factor", 1.0) or 1.0), indices

    def _dense_mlp(self, hidden: torch.Tensor) -> torch.Tensor:
        self._load_dense_mlp()
        return self._mlp(f"model.layers.{self.layer_index}.mlp", hidden)

    def _moe(self, hidden: torch.Tensor, route_weights: torch.Tensor, route_indices: torch.Tensor) -> torch.Tensor:
        h = hidden.reshape(-1, hidden.shape[-1]).float()
        selected = list(self.selected_experts)
        if self._expert_gate is None or self._expert_up is None or self._expert_down is None:
            self._load_moe(selected)
        if self._expert_gate is None or self._expert_up is None or self._expert_down is None:
            raise RuntimeError("Resident expert stacks did not materialize.")
        gate = self._expert_gate.float()
        up = self._expert_up.float()
        down = self._expert_down.float()
        expert_hidden = h[0]
        gate_out = torch.einsum("eih,h->ei", gate, expert_hidden)
        up_out = torch.einsum("eih,h->ei", up, expert_hidden)
        act = F.silu(gate_out) * up_out
        down_out = torch.einsum("ehi,ei->eh", down, act)
        by_expert = {int(route_indices[0, pos].item()): float(route_weights[0, pos].item()) for pos in range(route_indices.shape[1])}
        weights = torch.tensor([by_expert[idx] for idx in selected], device=hidden.device, dtype=down_out.dtype).view(-1, 1)
        out = (down_out * weights).sum(dim=0)
        if int(self.config.get("n_shared_experts", 0) or 0) > 0:
            out = out + self._mlp(f"model.layers.{self.layer_index}.mlp.shared_experts", hidden).reshape(-1).float()
        return out.reshape_as(h).to(dtype=self.dtype).reshape_as(hidden).contiguous()

    def _mlp(self, prefix: str, hidden: torch.Tensor) -> torch.Tensor:
        gate = self._linear(f"{prefix}.gate_proj.weight", hidden)
        up = self._linear(f"{prefix}.up_proj.weight", hidden)
        act = F.silu(gate.float()) * up.float()
        return self._linear(f"{prefix}.down_proj.weight", act)

    def _linear(self, name: str, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.weights[name]
        out = F.linear(hidden.reshape(-1, hidden.shape[-1]).float(), weight.float())
        return out.reshape(*hidden.shape[:-1], weight.shape[0]).to(dtype=self.dtype).contiguous()


class LocalDeepSeekResidentLayerPager:
    """LRU resident-layer pager for local DeepSeek FP8 sources."""

    def __init__(
        self,
        model_id: str,
        *,
        config: dict[str, Any],
        dtype: torch.dtype,
        device: torch.device,
        max_resident_bytes: int,
        layer_factory: Callable[[int], LocalDeepSeekResidentLayer] | None = None,
        prefetch_workers: int = 0,
    ) -> None:
        self.model_id = str(model_id)
        self.config = config
        self.dtype = dtype
        self.device = device
        self.max_resident_bytes = int(max_resident_bytes)
        self.layer_factory = layer_factory
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(prefetch_workers))) if int(prefetch_workers) > 0 else None
        self._prefetch_futures: dict[int, Future[LocalDeepSeekResidentLayer]] = {}
        self.layers: dict[int, LocalDeepSeekResidentLayer] = {}
        self._lru: list[int] = []
        self.loads = 0
        self.evictions = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.prefetch_submitted = 0
        self.prefetch_completed = 0
        self.peak_resident_bytes = 0

    def get(self, layer_index: int) -> LocalDeepSeekResidentLayer:
        layer_index = int(layer_index)
        resident = self.layers.get(layer_index)
        if resident is not None:
            self.cache_hits += 1
            self._touch(layer_index)
            self._record_peak()
            return resident
        self.cache_misses += 1
        future = self._prefetch_futures.pop(layer_index, None)
        if future is not None:
            resident = future.result()
            self.prefetch_completed += 1
        else:
            self.loads += 1
            resident = self._load_layer(layer_index)
        self.layers[layer_index] = resident
        self._touch(layer_index)
        self._record_peak()
        return resident

    def prefetch(self, layer_index: int) -> bool:
        layer_index = int(layer_index)
        if self._executor is None:
            return False
        if layer_index in self.layers or layer_index in self._prefetch_futures:
            return False
        self.loads += 1
        self.prefetch_submitted += 1
        self._prefetch_futures[layer_index] = self._executor.submit(self._load_layer, layer_index)
        return True

    def enforce_budget(self, *, protected_layer: int | None = None) -> None:
        protected = None if protected_layer is None else int(protected_layer)
        self._record_peak()
        while self.resident_nbytes() > self.max_resident_bytes and self._evict_one(protected_layer=protected):
            pass
        self._record_peak()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def resident_nbytes(self) -> int:
        return int(sum(layer.resident_nbytes() for layer in self.layers.values()))

    def _load_layer(self, layer_index: int) -> LocalDeepSeekResidentLayer:
        if self.layer_factory is None:
            return LocalDeepSeekResidentLayer(
                self.model_id,
                int(layer_index),
                config=self.config,
                dtype=self.dtype,
                device=self.device,
            )
        return self.layer_factory(int(layer_index))

    def _touch(self, layer_index: int) -> None:
        if layer_index in self._lru:
            self._lru.remove(layer_index)
        self._lru.append(layer_index)

    def _evict_one(self, *, protected_layer: int | None) -> bool:
        for layer_index in list(self._lru):
            if protected_layer is not None and layer_index == protected_layer:
                continue
            resident = self.layers.pop(layer_index, None)
            if resident is None:
                self._lru.remove(layer_index)
                continue
            resident.release()
            self._lru.remove(layer_index)
            self.evictions += 1
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return True
        return False

    def _record_peak(self) -> None:
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_nbytes())


def estimate_deepseek_gpu_residency(
    model_id: str,
    *,
    start_layer: int = 0,
    layer_count: int | None = None,
    selected_experts: list[int] | None = None,
    resident_budget_bytes: int = 12 * 1024**3,
    measured_seconds_per_layer: float = DEFAULT_T4_RESIDENT_SECONDS_PER_LAYER,
) -> DeepSeekGPUResidencyEstimate:
    """Estimate local GPU resident-window size from tensor catalog headers only."""
    blockers: list[str] = []
    try:
        catalog = load_tensor_catalog(model_id)
        config = _load_config(catalog.model_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        return DeepSeekGPUResidencyEstimate(
            model_id=str(model_id),
            ready=False,
            model_dir="",
            config_hidden_layers=0,
            start_layer=int(start_layer),
            layer_count=0,
            selected_experts=[] if selected_experts is None else [int(value) for value in selected_experts],
            resident_budget_bytes=int(resident_budget_bytes),
            max_source_layer_bytes=0,
            max_dequantized_layer_bytes=0,
            layers_fit_by_source_bytes=0,
            layers_fit_by_dequantized_bytes=0,
            projected_hot_seconds_per_token=0.0,
            measured_seconds_per_resident_layer=float(measured_seconds_per_layer),
            speed_target_met=False,
            blockers=[str(exc)],
        )

    config_layers = int(config.get("num_hidden_layers", catalog.num_hidden_layers or catalog.layer_count or 0) or 0)
    layer_count = config_layers - int(start_layer) if layer_count is None else int(layer_count)
    layer_count = max(0, min(int(layer_count), max(0, config_layers - int(start_layer))))
    selected = (
        [int(value) for value in selected_experts]
        if selected_experts is not None
        else list(range(int(config.get("num_experts_per_tok", catalog.num_experts_per_tok or 1) or 1)))
    )
    layer_plans: list[dict[str, Any]] = []
    max_source = 0
    max_dequant = 0
    for layer_index in range(int(start_layer), int(start_layer) + layer_count):
        plan = plan_fp8_layer_working_set(model_id, layer_index, selected)
        layer_dequant = _dequantized_layer_bytes(plan)
        layer_plans.append({**plan.to_dict(), "estimated_dequantized_bytes": int(layer_dequant)})
        max_source = max(max_source, int(plan.total_nbytes))
        max_dequant = max(max_dequant, int(layer_dequant))
        blockers.extend(plan.blockers)

    budget = max(1, int(resident_budget_bytes))
    projected = float(max(1, config_layers) * float(measured_seconds_per_layer))
    return DeepSeekGPUResidencyEstimate(
        model_id=str(model_id),
        ready=bool(layer_plans) and not blockers,
        model_dir=str(catalog.model_dir),
        config_hidden_layers=int(config_layers),
        start_layer=int(start_layer),
        layer_count=int(layer_count),
        selected_experts=selected,
        resident_budget_bytes=budget,
        max_source_layer_bytes=int(max_source),
        max_dequantized_layer_bytes=int(max_dequant),
        layers_fit_by_source_bytes=max(0, budget // max(1, int(max_source))),
        layers_fit_by_dequantized_bytes=max(0, budget // max(1, int(max_dequant))),
        projected_hot_seconds_per_token=float(projected),
        measured_seconds_per_resident_layer=float(measured_seconds_per_layer),
        speed_target_met=bool(projected <= 2.0),
        layer_plans=layer_plans,
        blockers=blockers,
    )


def run_local_deepseek_paged_decode_probe(
    model_id: str,
    *,
    token_id: int = 0,
    start_layer: int = 0,
    layer_count: int = 1,
    resident_budget_bytes: int = 12 * 1024**3,
    prefetch_window: int = 0,
    prefetch_workers: int = 1,
    include_tail: bool = False,
    tail_top_k: int = 5,
    tail_chunk_rows: int = 8192,
    device: str | None = None,
    require_cuda: bool = False,
    dtype: torch.dtype = torch.float16,
) -> LocalDeepSeekPagedDecodeResult:
    """Run a bounded local DeepSeek FP8 decode slice through the GPU resident pager."""
    cuda_available = torch.cuda.is_available()
    if device is None:
        if cuda_available:
            device = "cuda"
        elif require_cuda:
            return _failed_decode(model_id, token_id, start_layer, layer_count, "CUDA is required but is not available.")
        else:
            device = "cpu"
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not cuda_available:
        if require_cuda:
            return _failed_decode(model_id, token_id, start_layer, layer_count, "CUDA device was requested but is not available.")
        torch_device = torch.device("cpu")
    device_name = torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else "cpu"
    catalog = load_tensor_catalog(model_id)
    config = _load_config(catalog.model_dir)
    config_layers = int(config.get("num_hidden_layers", catalog.num_hidden_layers or catalog.layer_count or 0) or 0)
    start_layer = max(0, int(start_layer))
    layer_count = max(1, min(int(layer_count), max(1, config_layers - start_layer)))
    pager = LocalDeepSeekResidentLayerPager(
        model_id,
        config=config,
        dtype=dtype,
        device=torch_device,
        max_resident_bytes=max(1, int(resident_budget_bytes)),
        prefetch_workers=max(0, int(prefetch_workers)) if int(prefetch_window) > 0 else 0,
    )
    executed = list(range(start_layer, start_layer + layer_count))
    selected: dict[int, list[int]] = {}
    caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    blockers: list[str] = []
    started = time.perf_counter()
    try:
        embedding = _load_regular_tensor(model_id, "model.embed_tokens.weight", device=torch_device)
        hidden = embedding[int(token_id) : int(token_id) + 1].reshape(1, 1, int(config["hidden_size"])).to(dtype=dtype)
        out = hidden
        for offset, layer_index in enumerate(executed):
            for ahead in range(1, max(0, int(prefetch_window)) + 1):
                prefetch_index = offset + ahead
                if prefetch_index >= len(executed):
                    break
                pager.prefetch(executed[prefetch_index])
            resident = pager.get(layer_index)
            out, next_cache = resident.forward_with_cache(out, start_pos=0, previous_kv_cache=caches.get(layer_index))
            caches[layer_index] = next_cache
            selected[layer_index] = list(resident.selected_experts)
            pager.enforce_budget(protected_layer=layer_index)
        tail_ids: list[int] = []
        tail_logits: list[float] = []
        tail_elapsed = 0.0
        if include_tail:
            tail_started = time.perf_counter()
            tail_ids, tail_logits = _run_tail_topk(
                model_id,
                out,
                device=torch_device,
                top_k=tail_top_k,
                chunk_rows=tail_chunk_rows,
            )
            tail_elapsed = time.perf_counter() - tail_started
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        elapsed = time.perf_counter() - started
        seconds_per_layer = elapsed / float(max(1, layer_count))
        projected = seconds_per_layer * max(1, config_layers)
        passed = bool(torch.isfinite(out).all().item() and out.shape == (1, 1, int(config["hidden_size"])))
        pager.close()
        return LocalDeepSeekPagedDecodeResult(
            passed=passed,
            cuda_available=cuda_available,
            device=str(torch_device),
            device_name=device_name,
            torch_version=torch.__version__,
            model_id=str(model_id),
            token_id=int(token_id),
            start_layer=int(start_layer),
            layer_count=int(layer_count),
            config_hidden_layers=int(config_layers),
            executed_layers=executed,
            selected_experts_by_layer=selected,
            tail_top_token_ids=tail_ids,
            tail_top_logits=tail_logits,
            output_shape=[int(value) for value in out.shape],
            cache_sequence_lengths={int(key): int(value[0].shape[1]) for key, value in caches.items()},
            checksum=float(out.float().sum().detach().cpu().item()),
            max_abs_value=float(out.float().abs().max().detach().cpu().item()),
            resident_budget_bytes=int(resident_budget_bytes),
            peak_resident_bytes=int(pager.peak_resident_bytes),
            final_resident_bytes=int(pager.resident_nbytes()),
            pager_loads=int(pager.loads),
            pager_evictions=int(pager.evictions),
            pager_cache_hits=int(pager.cache_hits),
            pager_cache_misses=int(pager.cache_misses),
            pager_prefetch_submitted=int(pager.prefetch_submitted),
            pager_prefetch_completed=int(pager.prefetch_completed),
            tail_elapsed_seconds=float(tail_elapsed),
            elapsed_seconds=float(elapsed),
            seconds_per_layer=float(seconds_per_layer),
            projected_config_layers_seconds_per_token=float(projected),
            blockers=blockers,
            note="Bounded local DeepSeek FP8 decode slice through the resident-layer pager.",
        )
    except Exception as exc:  # noqa: BLE001 - caller needs a machine-readable failure.
        pager.close()
        return LocalDeepSeekPagedDecodeResult(
            passed=False,
            cuda_available=cuda_available,
            device=str(torch_device),
            device_name=device_name,
            torch_version=torch.__version__,
            model_id=str(model_id),
            token_id=int(token_id),
            start_layer=int(start_layer),
            layer_count=int(layer_count),
            config_hidden_layers=int(config_layers),
            executed_layers=[],
            selected_experts_by_layer=selected,
            tail_top_token_ids=[],
            tail_top_logits=[],
            output_shape=[],
            cache_sequence_lengths={int(key): int(value[0].shape[1]) for key, value in caches.items()},
            checksum=0.0,
            max_abs_value=0.0,
            resident_budget_bytes=int(resident_budget_bytes),
            peak_resident_bytes=int(pager.peak_resident_bytes),
            final_resident_bytes=int(pager.resident_nbytes()),
            pager_loads=int(pager.loads),
            pager_evictions=int(pager.evictions),
            pager_cache_hits=int(pager.cache_hits),
            pager_cache_misses=int(pager.cache_misses),
            pager_prefetch_submitted=int(pager.prefetch_submitted),
            pager_prefetch_completed=int(pager.prefetch_completed),
            tail_elapsed_seconds=0.0,
            elapsed_seconds=float(time.perf_counter() - started),
            seconds_per_layer=0.0,
            projected_config_layers_seconds_per_token=0.0,
            blockers=[f"{type(exc).__name__}: {exc}"],
            note="Local DeepSeek resident pager failed.",
        )


def _load_dequantized_weight(model_id: str, name: str, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    pair = load_fp8_weight_pair(model_id, name)
    if pair.fp8_bytes is not None:
        pair.fp8_bytes = pair.fp8_bytes.to(device=device)
    if pair.scale_tensor is not None:
        pair.scale_tensor = pair.scale_tensor.to(device=device)
    loaded = dequantize_fp8_weight_pair(pair, dtype=dtype)
    if not loaded.ready or loaded.tensor is None:
        raise KeyError("; ".join(loaded.blockers) or f"Could not load {name}.")
    return loaded.tensor.to(device=device, dtype=dtype).contiguous()


def _run_tail_topk(
    model_id: str,
    hidden: torch.Tensor,
    *,
    device: torch.device,
    top_k: int,
    chunk_rows: int,
) -> tuple[list[int], list[float]]:
    from pcketlm.core.runtime.fp8_source import _read_tensor_rows

    lm_head = find_tensor_catalog_entry(model_id, "lm_head.weight")
    if lm_head is None:
        raise KeyError("lm_head.weight is not present in the tensor catalog.")
    norm = _load_regular_tensor(model_id, "model.norm.weight", device=device)
    config = _load_config(load_tensor_catalog(model_id).model_dir)
    hidden_2d = hidden.reshape(-1, hidden.shape[-1])[-1:].to(device=device, dtype=torch.float16)
    normalized = _rms_norm(hidden_2d, norm, float(config["rms_norm_eps"])).float()
    vocab_size = int(lm_head.shape[0])
    hidden_size = int(lm_head.shape[1])
    if int(normalized.shape[-1]) != hidden_size:
        raise ValueError(f"Hidden size {int(normalized.shape[-1])} does not match lm_head hidden size {hidden_size}.")
    keep_k = max(1, min(int(top_k), vocab_size))
    rows = max(1, int(chunk_rows))
    best_values: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None
    for start in range(0, vocab_size, rows):
        end = min(vocab_size, start + rows)
        weight = _read_tensor_rows(lm_head, start, end).to(device=device)
        logits = F.linear(normalized, weight.float()).reshape(-1)
        values, indices = torch.topk(logits, min(keep_k, int(logits.numel())))
        indices = indices + int(start)
        if best_values is None or best_indices is None:
            best_values = values
            best_indices = indices
            continue
        merged_values = torch.cat([best_values, values])
        merged_indices = torch.cat([best_indices, indices])
        next_values, next_positions = torch.topk(merged_values, min(keep_k, int(merged_values.numel())))
        best_values = next_values
        best_indices = merged_indices[next_positions]
    if best_values is None or best_indices is None:
        return [], []
    return [int(value) for value in best_indices.detach().cpu().tolist()], [
        float(value) for value in best_values.detach().cpu().tolist()
    ]


def _load_regular_tensor(model_id: str, name: str, *, device: torch.device) -> torch.Tensor:
    from pcketlm.core.runtime.fp8_source import _load_regular_tensor as load_regular

    tensor, blockers = load_regular(model_id, name)
    if tensor is None or blockers:
        raise KeyError("; ".join(blockers) or f"Could not load {name}.")
    return tensor.to(device=device).contiguous()


def _load_config(model_dir: Path) -> dict[str, Any]:
    payload = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    if "num_hidden_layers" not in payload:
        payload["num_hidden_layers"] = 0
    if "hidden_size" not in payload:
        raise ValueError("DeepSeek config is missing hidden_size.")
    payload.setdefault("num_attention_heads", 1)
    payload.setdefault("qk_nope_head_dim", 128)
    payload.setdefault("qk_rope_head_dim", 64)
    payload.setdefault("v_head_dim", 128)
    payload.setdefault("kv_lora_rank", 512)
    payload.setdefault("rms_norm_eps", 1e-6)
    payload.setdefault("first_k_dense_replace", payload.get("n_dense_layers", 0))
    payload.setdefault("num_experts_per_tok", payload.get("n_activated_experts", 1))
    payload.setdefault("n_group", payload.get("n_expert_groups", 1))
    payload.setdefault("topk_group", payload.get("n_limited_groups", 1))
    payload.setdefault("scoring_func", payload.get("score_func", "sigmoid"))
    payload.setdefault("routed_scaling_factor", payload.get("route_scale", 1.0))
    payload.setdefault("n_shared_experts", payload.get("num_shared_experts", 0))
    payload.setdefault("rope_theta", 10000.0)
    payload.setdefault("original_max_position_embeddings", payload.get("max_position_embeddings", 4096))
    payload.setdefault("max_position_embeddings", payload.get("original_max_position_embeddings", 4096))
    return payload


def _dequantized_layer_bytes(plan: Any) -> int:
    return int(plan.fp8_weight_bytes * torch.empty((), dtype=torch.float16).element_size() + plan.non_fp8_bytes)


def _failed_decode(
    model_id: str,
    token_id: int,
    start_layer: int,
    layer_count: int,
    note: str,
) -> LocalDeepSeekPagedDecodeResult:
    return LocalDeepSeekPagedDecodeResult(
        passed=False,
        cuda_available=False,
        device="cpu",
        device_name="cpu",
        torch_version=torch.__version__,
        model_id=str(model_id),
        token_id=int(token_id),
        start_layer=int(start_layer),
        layer_count=int(layer_count),
        config_hidden_layers=0,
        executed_layers=[],
        selected_experts_by_layer={},
        tail_top_token_ids=[],
        tail_top_logits=[],
        output_shape=[],
        cache_sequence_lengths={},
        checksum=0.0,
        max_abs_value=0.0,
        resident_budget_bytes=0,
        peak_resident_bytes=0,
        final_resident_bytes=0,
        pager_loads=0,
        pager_evictions=0,
        pager_cache_hits=0,
        pager_cache_misses=0,
        pager_prefetch_submitted=0,
        pager_prefetch_completed=0,
        tail_elapsed_seconds=0.0,
        elapsed_seconds=0.0,
        seconds_per_layer=0.0,
        projected_config_layers_seconds_per_token=0.0,
        blockers=[note],
        note=note,
    )
