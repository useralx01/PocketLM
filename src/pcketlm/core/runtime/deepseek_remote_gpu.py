"""Range-streamed real DeepSeek V3 CUDA validation helpers."""

from __future__ import annotations

import json
import math
import os
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_REPO_ID = "deepseek-ai/DeepSeek-V3"
DEFAULT_REVISION = "main"


@dataclass(frozen=True)
class RemoteDeepSeekLayerProbeResult:
    passed: bool
    cuda_available: bool
    device: str
    device_name: str
    torch_version: str
    repo_id: str
    revision: str
    layer_index: int
    token_id: int
    config_hidden_layers: int
    layers_executed: int
    selected_experts: list[int]
    output_shape: list[int]
    checksum: float
    max_abs_value: float
    bytes_downloaded: int
    tensors_downloaded: int
    elapsed_seconds: float
    attention_elapsed_seconds: float
    router_elapsed_seconds: float
    moe_elapsed_seconds: float
    projected_config_layers_seconds_per_token: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RemoteSafeTensorShard:
    def __init__(self, *, repo_id: str, revision: str, filename: str) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.filename = filename
        self.url = _hf_url(repo_id, revision, filename)
        header_prefix = _http_range(self.url, 0, 7)
        if len(header_prefix) != 8:
            raise RuntimeError(f"Could not read safetensors header prefix for {filename}.")
        self.header_len = struct.unpack("<Q", header_prefix)[0]
        header_bytes = _http_range(self.url, 8, 8 + self.header_len - 1)
        self.header = json.loads(header_bytes.decode("utf-8"))
        self.data_base = 8 + self.header_len
        self.bytes_downloaded = len(header_prefix) + len(header_bytes)
        self.tensors_downloaded = 0

    def tensor_info(self, name: str) -> dict[str, Any]:
        info = self.header.get(name)
        if not isinstance(info, dict):
            raise KeyError(name)
        return info

    def read_tensor(self, name: str, *, device: torch.device) -> torch.Tensor:
        info = self.tensor_info(name)
        start, end = [int(value) for value in info["data_offsets"]]
        raw = _http_range(self.url, self.data_base + start, self.data_base + end - 1)
        self.bytes_downloaded += len(raw)
        self.tensors_downloaded += 1
        return _tensor_from_safetensors_bytes(raw, str(info["dtype"]), [int(v) for v in info["shape"]], device=device)

    def read_rows(self, name: str, start_row: int, end_row: int, *, device: torch.device) -> torch.Tensor:
        info = self.tensor_info(name)
        shape = [int(v) for v in info["shape"]]
        if len(shape) != 2:
            raise ValueError(f"{name} is not a row-readable 2D tensor.")
        dtype = str(info["dtype"])
        elem_size = _dtype_size(dtype)
        rows = max(0, int(end_row) - int(start_row))
        cols = shape[1]
        tensor_start, _tensor_end = [int(value) for value in info["data_offsets"]]
        byte_start = self.data_base + tensor_start + int(start_row) * cols * elem_size
        byte_end = byte_start + rows * cols * elem_size - 1
        raw = _http_range(self.url, byte_start, byte_end)
        self.bytes_downloaded += len(raw)
        self.tensors_downloaded += 1
        return _tensor_from_safetensors_bytes(raw, dtype, [rows, cols], device=device)


class RemoteDeepSeekStore:
    def __init__(self, repo_id: str = DEFAULT_REPO_ID, revision: str = DEFAULT_REVISION, *, device: torch.device) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.device = device
        self.config = json.loads(_http_get(_hf_url(repo_id, revision, "config.json")).decode("utf-8"))
        self.weight_map = json.loads(_http_get(_hf_url(repo_id, revision, "model.safetensors.index.json")).decode("utf-8"))[
            "weight_map"
        ]
        self._shards: dict[str, RemoteSafeTensorShard] = {}
        self._cache: dict[str, torch.Tensor] = {}
        self._base_downloaded = 0

    @property
    def bytes_downloaded(self) -> int:
        return self._base_downloaded + sum(shard.bytes_downloaded for shard in self._shards.values())

    @property
    def tensors_downloaded(self) -> int:
        return sum(shard.tensors_downloaded for shard in self._shards.values())

    def tensor(self, name: str, *, cache: bool = True) -> torch.Tensor:
        if cache and name in self._cache:
            return self._cache[name]
        shard = self._shard_for(name)
        tensor = shard.read_tensor(name, device=self.device)
        if cache:
            self._cache[name] = tensor
        return tensor

    def rows(self, name: str, start: int, end: int) -> torch.Tensor:
        return self._shard_for(name).read_rows(name, start, end, device=self.device)

    def fp8_pair(self, weight_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        scale_name = self._scale_name(weight_name)
        return self.tensor(weight_name, cache=False), self.tensor(scale_name, cache=False).float()

    def _scale_name(self, weight_name: str) -> str:
        for suffix in ("_scale_inv", ".scale_inv", ".scale"):
            candidate = f"{weight_name}{suffix}"
            if candidate in self.weight_map:
                return candidate
        raise KeyError(f"No FP8 scale companion found for {weight_name}.")

    def _shard_for(self, name: str) -> RemoteSafeTensorShard:
        filename = self.weight_map[name]
        shard = self._shards.get(filename)
        if shard is None:
            shard = RemoteSafeTensorShard(repo_id=self.repo_id, revision=self.revision, filename=filename)
            self._shards[filename] = shard
        return shard


def run_remote_deepseek_layer_probe(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    layer_index: int = 3,
    token_id: int = 0,
    require_cuda: bool = True,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> RemoteDeepSeekLayerProbeResult:
    """Run one real DeepSeek V3 layer on CUDA using remote FP8 safetensors ranges."""
    cuda_available = torch.cuda.is_available()
    if device is None:
        if cuda_available:
            device = "cuda"
        elif require_cuda:
            return _failed_remote_probe(
                repo_id=repo_id,
                revision=revision,
                layer_index=layer_index,
                token_id=token_id,
                torch_version=torch.__version__,
                note="CUDA is required but is not available.",
            )
        else:
            device = "cpu"
    torch_device = torch.device(device)
    device_name = torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else "cpu"
    store = RemoteDeepSeekStore(repo_id, revision, device=torch_device)
    config = store.config
    started = time.perf_counter()
    attention_elapsed = 0.0
    router_elapsed = 0.0
    moe_elapsed = 0.0
    selected: list[int] = []

    try:
        hidden = store.rows("model.embed_tokens.weight", int(token_id), int(token_id) + 1).reshape(
            1, 1, int(config["hidden_size"])
        )
        input_norm = store.tensor(f"model.layers.{int(layer_index)}.input_layernorm.weight")
        post_norm = store.tensor(f"model.layers.{int(layer_index)}.post_attention_layernorm.weight")

        attn_start = time.perf_counter()
        normed = _rms_norm(hidden.to(dtype=dtype), input_norm, float(config["rms_norm_eps"]))
        attn_out = _attention_layer(store, int(layer_index), normed, config=config, dtype=dtype)
        attention_elapsed = time.perf_counter() - attn_start

        hidden_after_attn = hidden.to(dtype=dtype) + attn_out.to(dtype=dtype)
        ffn_input = _rms_norm(hidden_after_attn, post_norm, float(config["rms_norm_eps"]))

        router_start = time.perf_counter()
        route_weights, route_indices = _route_layer(store, int(layer_index), ffn_input, config=config)
        router_elapsed = time.perf_counter() - router_start
        selected = sorted({int(value) for value in route_indices.reshape(-1).tolist()})

        moe_start = time.perf_counter()
        if int(layer_index) < int(config.get("first_k_dense_replace", 0)):
            ffn_out = _mlp_prefix(store, f"model.layers.{int(layer_index)}.mlp", ffn_input, dtype=dtype)
        else:
            ffn_out = _moe_layer(store, int(layer_index), ffn_input, route_weights, route_indices, config=config, dtype=dtype)
        moe_elapsed = time.perf_counter() - moe_start

        out = (hidden_after_attn + ffn_out.to(dtype=dtype)).contiguous()
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        elapsed = time.perf_counter() - started
        checksum = float(out.float().sum().detach().cpu().item())
        max_abs = float(out.float().abs().max().detach().cpu().item())
        layer_count = int(config.get("num_hidden_layers", 0) or 0)
        passed = bool(torch.isfinite(out).all().item() and out.shape == (1, 1, int(config["hidden_size"])))
        return RemoteDeepSeekLayerProbeResult(
            passed=passed,
            cuda_available=cuda_available,
            device=str(torch_device),
            device_name=device_name,
            torch_version=torch.__version__,
            repo_id=repo_id,
            revision=revision,
            layer_index=int(layer_index),
            token_id=int(token_id),
            config_hidden_layers=layer_count,
            layers_executed=1,
            selected_experts=selected,
            output_shape=[int(v) for v in out.shape],
            checksum=checksum,
            max_abs_value=max_abs,
            bytes_downloaded=int(store.bytes_downloaded),
            tensors_downloaded=int(store.tensors_downloaded),
            elapsed_seconds=float(elapsed),
            attention_elapsed_seconds=float(attention_elapsed),
            router_elapsed_seconds=float(router_elapsed),
            moe_elapsed_seconds=float(moe_elapsed),
            projected_config_layers_seconds_per_token=float(elapsed * max(1, layer_count)),
            note=(
                "Real DeepSeek V3 FP8 weights streamed by HTTP range and executed on CUDA for one bounded layer. "
                "This is a real-weight layer gate, not the full decode done gate."
            ),
        )
    except Exception as exc:  # noqa: BLE001 - validation JSON should carry the exact remote failure.
        elapsed = time.perf_counter() - started
        return RemoteDeepSeekLayerProbeResult(
            passed=False,
            cuda_available=cuda_available,
            device=str(torch_device),
            device_name=device_name,
            torch_version=torch.__version__,
            repo_id=repo_id,
            revision=revision,
            layer_index=int(layer_index),
            token_id=int(token_id),
            config_hidden_layers=int(store.config.get("num_hidden_layers", 0) or 0),
            layers_executed=0,
            selected_experts=[],
            output_shape=[],
            checksum=0.0,
            max_abs_value=0.0,
            bytes_downloaded=int(store.bytes_downloaded),
            tensors_downloaded=int(store.tensors_downloaded),
            elapsed_seconds=float(elapsed),
            attention_elapsed_seconds=float(attention_elapsed),
            router_elapsed_seconds=float(router_elapsed),
            moe_elapsed_seconds=float(moe_elapsed),
            projected_config_layers_seconds_per_token=0.0,
            note=f"Remote DeepSeek layer probe failed: {type(exc).__name__}: {exc}",
        )


def _attention_layer(
    store: RemoteDeepSeekStore,
    layer_index: int,
    hidden: torch.Tensor,
    *,
    config: dict[str, Any],
    dtype: torch.dtype,
) -> torch.Tensor:
    prefix = f"model.layers.{layer_index}.self_attn"
    q_a = _fp8_linear(store, f"{prefix}.q_a_proj.weight", hidden, dtype=dtype)
    q_a_norm = _rms_norm(q_a, store.tensor(f"{prefix}.q_a_layernorm.weight"), float(config["rms_norm_eps"]))
    q = _fp8_linear(store, f"{prefix}.q_b_proj.weight", q_a_norm, dtype=dtype).float()
    kv = _fp8_linear(store, f"{prefix}.kv_a_proj_with_mqa.weight", hidden, dtype=dtype).float()

    heads = int(config["num_attention_heads"])
    qk_nope = int(config["qk_nope_head_dim"])
    qk_rope = int(config["qk_rope_head_dim"])
    v_head_dim = int(config["v_head_dim"])
    kv_lora_rank = int(config["kv_lora_rank"])
    q = q.view(1, 1, heads, qk_nope + qk_rope)
    q_nope, q_pe = torch.split(q, [qk_nope, qk_rope], dim=-1)
    kv_latent, k_pe = torch.split(kv, [kv_lora_rank, qk_rope], dim=-1)
    kv_latent = _rms_norm(kv_latent, store.tensor(f"{prefix}.kv_a_layernorm.weight"), float(config["rms_norm_eps"]))
    kv_b = _dequant_pair(store, f"{prefix}.kv_b_proj.weight", dtype=dtype).float()
    wkv_b = kv_b.view(heads, qk_nope + v_head_dim, kv_lora_rank)
    q_nope_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope])
    q_pe = _apply_rope(q_pe, config=config)
    k_pe = _apply_rope(k_pe.unsqueeze(2), config=config).squeeze(2)
    scores = (
        torch.einsum("bshc,btc->bsht", q_nope_absorbed, kv_latent)
        + torch.einsum("bshr,btr->bsht", q_pe, k_pe)
    ) * _softmax_scale(config)
    probs = scores.softmax(dim=-1, dtype=torch.float32).to(dtype=hidden.dtype)
    latent = torch.einsum("bsht,btc->bshc", probs, kv_latent)
    heads_out = torch.einsum("bshc,hdc->bshd", latent, wkv_b[:, -v_head_dim:])
    return _fp8_linear(store, f"{prefix}.o_proj.weight", heads_out.flatten(2), dtype=dtype)


def _route_layer(
    store: RemoteDeepSeekStore,
    layer_index: int,
    hidden: torch.Tensor,
    *,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    router = store.tensor(f"model.layers.{layer_index}.mlp.gate.weight").float()
    bias_name = f"model.layers.{layer_index}.mlp.gate.e_score_correction_bias"
    bias = store.tensor(bias_name).float() if bias_name in store.weight_map else None
    scores = F.linear(hidden.reshape(-1, hidden.shape[-1]).float(), router)
    scoring = str(config.get("scoring_func", "sigmoid"))
    scores = scores.sigmoid() if scoring == "sigmoid" else scores.softmax(dim=-1, dtype=torch.float32)
    choice_scores = scores + bias if bias is not None else scores
    n_groups = int(config.get("n_group", 1) or 1)
    topk_groups = int(config.get("topk_group", 1) or 1)
    if n_groups > 1:
        grouped = choice_scores.view(choice_scores.shape[0], n_groups, -1)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1) if bias is not None else grouped.amax(dim=-1)
        keep_groups = group_scores.topk(max(1, min(topk_groups, n_groups)), dim=-1)[1]
        mask = grouped.new_ones(choice_scores.shape[0], n_groups, dtype=torch.bool).scatter_(1, keep_groups, False)
        choice_scores = grouped.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
    top_k = max(1, min(int(config.get("num_experts_per_tok", 1) or 1), choice_scores.shape[-1]))
    indices = torch.topk(choice_scores, top_k, dim=-1)[1]
    weights = scores.gather(1, indices)
    if scoring == "sigmoid":
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = weights * float(config.get("routed_scaling_factor", 1.0) or 1.0)
    return weights, indices


def _moe_layer(
    store: RemoteDeepSeekStore,
    layer_index: int,
    hidden: torch.Tensor,
    route_weights: torch.Tensor,
    route_indices: torch.Tensor,
    *,
    config: dict[str, Any],
    dtype: torch.dtype,
) -> torch.Tensor:
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_output = torch.zeros_like(flat_hidden, dtype=dtype)
    for expert_index in sorted({int(v) for v in route_indices.reshape(-1).tolist()}):
        row_idx, top_idx = torch.where(route_indices == expert_index)
        if row_idx.numel() == 0:
            continue
        expert_hidden = flat_hidden[row_idx]
        expert_out = _mlp_prefix(
            store,
            f"model.layers.{layer_index}.mlp.experts.{expert_index}",
            expert_hidden,
            dtype=dtype,
        )
        flat_output[row_idx] += expert_out.reshape_as(expert_hidden) * route_weights[row_idx, top_idx].view(-1, 1).to(dtype=dtype)
    if int(config.get("n_shared_experts", 0) or 0) > 0:
        flat_output += _mlp_prefix(store, f"model.layers.{layer_index}.mlp.shared_experts", flat_hidden, dtype=dtype).reshape_as(
            flat_output
        )
    return flat_output.reshape_as(hidden).contiguous()


def _mlp_prefix(store: RemoteDeepSeekStore, prefix: str, hidden: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    gate = _fp8_linear(store, f"{prefix}.gate_proj.weight", hidden, dtype=dtype)
    up = _fp8_linear(store, f"{prefix}.up_proj.weight", hidden, dtype=dtype)
    act = F.silu(gate.float()) * up.float()
    return _fp8_linear(store, f"{prefix}.down_proj.weight", act, dtype=dtype)


def _fp8_linear(store: RemoteDeepSeekStore, name: str, hidden: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    weight = _dequant_pair(store, name, dtype=dtype)
    out = F.linear(hidden.reshape(-1, hidden.shape[-1]).float(), weight.float())
    return out.reshape(*hidden.shape[:-1], weight.shape[0]).to(dtype=dtype).contiguous()


def _dequant_pair(store: RemoteDeepSeekStore, name: str, *, dtype: torch.dtype) -> torch.Tensor:
    fp8_bytes, scale = store.fp8_pair(name)
    return _dequantize_fp8_block_scaled(fp8_bytes, scale, dtype=dtype)


def _dequantize_fp8_block_scaled(fp8_bytes: torch.Tensor, scale_inv: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn.")
    rows, cols = int(fp8_bytes.shape[0]), int(fp8_bytes.shape[1])
    expanded = scale_inv.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:rows, :cols]
    return (fp8_bytes.contiguous().view(torch.float8_e4m3fn).to(torch.float32) * expanded).to(dtype=dtype).contiguous()


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out_dtype = hidden.dtype
    weight = weight.to(device=hidden.device)
    normed = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (normed * weight.float().view(*([1] * (hidden.ndim - 1)), -1)).to(dtype=out_dtype)


def _apply_rope(x: torch.Tensor, *, config: dict[str, Any]) -> torch.Tensor:
    dim = int(x.shape[-1])
    if dim == 0 or dim % 2 != 0:
        return x
    freqs = 1.0 / (float(config.get("rope_theta", 10000.0)) ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    cos = torch.cos(freqs * 0.0).view(*([1] * (x.ndim - 1)), -1)
    sin = torch.sin(freqs * 0.0).view(*([1] * (x.ndim - 1)), -1)
    pair = x.float().reshape(*x.shape[:-1], dim // 2, 2)
    return torch.stack((pair[..., 0] * cos - pair[..., 1] * sin, pair[..., 0] * sin + pair[..., 1] * cos), dim=-1).flatten(-2).to(
        dtype=x.dtype
    )


def _softmax_scale(config: dict[str, Any]) -> float:
    qk_nope = int(config["qk_nope_head_dim"])
    qk_rope = int(config["qk_rope_head_dim"])
    scale = (qk_nope + qk_rope) ** -0.5
    rope_scaling = config.get("rope_scaling") or {}
    factor = float(rope_scaling.get("factor", 1.0) or 1.0)
    original = int(rope_scaling.get("original_max_position_embeddings", 4096) or 4096)
    max_position = int(config.get("max_position_embeddings", original) or original)
    if max_position > original and factor > 1.0:
        mscale = 0.1 * float(rope_scaling.get("mscale_all_dim", 1.0) or 1.0) * math.log(factor) + 1.0
        scale *= mscale * mscale
    return float(scale)


def _tensor_from_safetensors_bytes(raw: bytes, dtype: str, shape: list[int], *, device: torch.device) -> torch.Tensor:
    dtype = dtype.upper()
    if dtype in {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"}:
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(tuple(shape))
    else:
        tensor = torch.frombuffer(bytearray(raw), dtype=_torch_dtype(dtype)).reshape(tuple(shape)).clone()
    return tensor.to(device=device).contiguous()


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "BF16":
        return torch.bfloat16
    if dtype == "F16":
        return torch.float16
    if dtype == "F32":
        return torch.float32
    if dtype in {"I32", "U32"}:
        return torch.int32
    raise ValueError(f"Unsupported safetensors dtype {dtype}.")


def _dtype_size(dtype: str) -> int:
    dtype = dtype.upper()
    if dtype in {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ", "U8", "I8", "BOOL"}:
        return 1
    return torch.empty((), dtype=_torch_dtype(dtype)).element_size()


def _failed_remote_probe(
    *,
    repo_id: str,
    revision: str,
    layer_index: int,
    token_id: int,
    torch_version: str,
    note: str,
) -> RemoteDeepSeekLayerProbeResult:
    return RemoteDeepSeekLayerProbeResult(
        passed=False,
        cuda_available=False,
        device="cpu",
        device_name="cpu",
        torch_version=torch_version,
        repo_id=repo_id,
        revision=revision,
        layer_index=int(layer_index),
        token_id=int(token_id),
        config_hidden_layers=0,
        layers_executed=0,
        selected_experts=[],
        output_shape=[],
        checksum=0.0,
        max_abs_value=0.0,
        bytes_downloaded=0,
        tensors_downloaded=0,
        elapsed_seconds=0.0,
        attention_elapsed_seconds=0.0,
        router_elapsed_seconds=0.0,
        moe_elapsed_seconds=0.0,
        projected_config_layers_seconds_per_token=0.0,
        note=note,
    )


def _hf_url(repo_id: str, revision: str, filename: str) -> str:
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    quoted_file = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    return f"https://huggingface.co/{quoted_repo}/resolve/{urllib.parse.quote(revision, safe='')}/{quoted_file}"


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers=_hf_headers())
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


@lru_cache(maxsize=2048)
def _http_range(url: str, start: int, end: int) -> bytes:
    headers = _hf_headers()
    headers["Range"] = f"bytes={int(start)}-{int(end)}"
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            if attempt == 2:
                raise
            time.sleep(1.0 + attempt)
    raise RuntimeError("unreachable")


def _hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "pcketlm-plm13-gpu-validation"}
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers
