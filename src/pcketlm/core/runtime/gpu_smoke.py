"""Tiny GPU smoke path for cloud notebook validation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GpuSmokeResult:
    passed: bool
    cuda_available: bool
    device: str
    torch_version: str
    checksum: float
    max_abs_repeat_diff: float
    output_shape: list[int]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeepSeekGpuProbeResult:
    passed: bool
    cuda_available: bool
    device: str
    torch_version: str
    layers_executed: int
    benchmark_layers: int
    iterations: int
    hidden_size: int
    intermediate_size: int
    routed_experts: int
    top_k: int
    cache_len: int
    benchmark_seconds: float
    seconds_per_probe_token: float
    seconds_per_layer: float
    projected_62_layer_seconds_per_token: float
    projected_k64_effective_seconds_per_position: float
    checksum: float
    max_abs_repeat_diff: float
    output_shape: list[int]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fp8_bytes(values: torch.Tensor) -> torch.Tensor:
    return values.detach().cpu().to(torch.float8_e4m3fn).view(torch.uint8).contiguous()


def _block_scale(rows: int, cols: int, *, value: float, device: torch.device) -> torch.Tensor:
    return torch.full(((rows + 127) // 128, (cols + 127) // 128), value, dtype=torch.float32, device=device)


def _dequantize_block_scaled(fp8_bytes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = fp8_bytes.view(torch.float8_e4m3fn).to(torch.float32)
    rows, cols = values.shape
    expanded = scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:rows, :cols]
    return values * expanded


def _make_fp8_weight(rows: int, cols: int, *, start: float, end: float, scale: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.linspace(start, end, steps=rows * cols, dtype=torch.float32).reshape(rows, cols)
    return _fp8_bytes(values).to(device), _block_scale(rows, cols, value=scale, device=device)


def _fp8_linear(hidden: torch.Tensor, fp8_weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden, _dequantize_block_scaled(fp8_weight, scales))


def _fp8_mlp(
    hidden: torch.Tensor,
    gate: tuple[torch.Tensor, torch.Tensor],
    up: tuple[torch.Tensor, torch.Tensor],
    down: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    activated = F.silu(_fp8_linear(hidden, *gate)) * _fp8_linear(hidden, *up)
    return _fp8_linear(activated, *down)


def _moe_block(hidden: torch.Tensor) -> torch.Tensor:
    hidden_size = int(hidden.shape[-1])
    intermediate = 32
    router_weight = torch.linspace(-0.3, 0.3, steps=4 * hidden_size, device=hidden.device).reshape(4, hidden_size)
    route_scores = F.linear(hidden, router_weight)
    route_probs = torch.softmax(route_scores.float(), dim=-1)
    route_weights, expert_ids = torch.topk(route_probs, k=2, dim=-1)

    expert_outputs: list[torch.Tensor] = []
    for expert_id in expert_ids.reshape(-1).tolist():
        offset = float(expert_id) * 0.05
        gate = _make_fp8_weight(intermediate, hidden_size, start=-0.6 + offset, end=0.6 + offset, scale=1.0, device=hidden.device)
        up = _make_fp8_weight(intermediate, hidden_size, start=0.5 - offset, end=-0.5 - offset, scale=0.875, device=hidden.device)
        down = _make_fp8_weight(hidden_size, intermediate, start=-0.4, end=0.4, scale=1.125, device=hidden.device)
        expert_outputs.append(_fp8_mlp(hidden, gate, up, down))

    stacked = torch.stack(expert_outputs, dim=1)
    return (stacked * route_weights.reshape(1, -1, 1)).sum(dim=1)


def _mla_attention(hidden: torch.Tensor) -> torch.Tensor:
    heads = 2
    nope_dim = 8
    rope_dim = 4
    value_dim = 8
    cache_len = 6
    hidden_size = int(hidden.shape[-1])
    device = hidden.device

    q_nope_weight = torch.linspace(-0.2, 0.2, steps=heads * nope_dim * hidden_size, device=device).reshape(
        heads * nope_dim, hidden_size
    )
    q_pe_weight = torch.linspace(0.25, -0.25, steps=heads * rope_dim * hidden_size, device=device).reshape(
        heads * rope_dim, hidden_size
    )
    q_nope = F.linear(hidden, q_nope_weight).reshape(heads, nope_dim)
    q_pe = F.linear(hidden, q_pe_weight).reshape(heads, rope_dim)
    kv_cache = torch.linspace(-0.4, 0.4, steps=cache_len * nope_dim, device=device).reshape(cache_len, nope_dim)
    pe_cache = torch.linspace(0.3, -0.3, steps=cache_len * rope_dim, device=device).reshape(cache_len, rope_dim)
    value_cache = torch.linspace(-0.5, 0.5, steps=cache_len * value_dim, device=device).reshape(cache_len, value_dim)

    scores = (torch.einsum("hd,td->ht", q_nope, kv_cache) + torch.einsum("hr,tr->ht", q_pe, pe_cache)) / (
        nope_dim + rope_dim
    ) ** 0.5
    probs = torch.softmax(scores.float(), dim=-1)
    return torch.einsum("ht,td->hd", probs, value_cache).reshape(1, heads * value_dim)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _SyntheticDeepSeekGpuBlock:
    def __init__(
        self,
        *,
        device: torch.device,
        hidden_size: int,
        intermediate_size: int,
        routed_experts: int,
        top_k: int,
        cache_len: int,
        heads: int,
        nope_dim: int,
        rope_dim: int,
        value_dim: int,
    ) -> None:
        self.device = device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.routed_experts = routed_experts
        self.top_k = top_k
        self.cache_len = cache_len
        self.heads = heads
        self.nope_dim = nope_dim
        self.rope_dim = rope_dim
        self.value_dim = value_dim
        self.router_weight = torch.linspace(-0.25, 0.25, steps=routed_experts * hidden_size, device=device).reshape(
            routed_experts, hidden_size
        )
        self.gate: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.up: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.down: list[tuple[torch.Tensor, torch.Tensor]] = []
        for expert_id in range(routed_experts):
            offset = expert_id * 0.015
            self.gate.append(
                _make_fp8_weight(
                    intermediate_size,
                    hidden_size,
                    start=-0.45 + offset,
                    end=0.45 + offset,
                    scale=0.9375,
                    device=device,
                )
            )
            self.up.append(
                _make_fp8_weight(
                    intermediate_size,
                    hidden_size,
                    start=0.35 - offset,
                    end=-0.35 - offset,
                    scale=1.0625,
                    device=device,
                )
            )
            self.down.append(
                _make_fp8_weight(
                    hidden_size,
                    intermediate_size,
                    start=-0.30,
                    end=0.30,
                    scale=1.0,
                    device=device,
                )
            )

        self.q_nope_weight = torch.linspace(-0.11, 0.11, steps=heads * nope_dim * hidden_size, device=device).reshape(
            heads * nope_dim, hidden_size
        )
        self.q_pe_weight = torch.linspace(0.09, -0.09, steps=heads * rope_dim * hidden_size, device=device).reshape(
            heads * rope_dim, hidden_size
        )
        self.kv_cache = torch.linspace(-0.25, 0.25, steps=cache_len * nope_dim, device=device).reshape(cache_len, nope_dim)
        self.pe_cache = torch.linspace(0.20, -0.20, steps=cache_len * rope_dim, device=device).reshape(cache_len, rope_dim)
        self.value_cache = torch.linspace(-0.18, 0.18, steps=cache_len * value_dim, device=device).reshape(cache_len, value_dim)
        self.attn_project = torch.linspace(-0.08, 0.08, steps=hidden_size * heads * value_dim, device=device).reshape(
            hidden_size, heads * value_dim
        )
        self.norm_weight = torch.linspace(0.85, 1.15, steps=hidden_size, device=device)

    def _attention(self, hidden: torch.Tensor) -> torch.Tensor:
        q_nope = F.linear(hidden, self.q_nope_weight).reshape(self.heads, self.nope_dim)
        q_pe = F.linear(hidden, self.q_pe_weight).reshape(self.heads, self.rope_dim)
        scores = (
            torch.einsum("hd,td->ht", q_nope, self.kv_cache)
            + torch.einsum("hr,tr->ht", q_pe, self.pe_cache)
        ) / (self.nope_dim + self.rope_dim) ** 0.5
        probs = torch.softmax(scores.float(), dim=-1)
        context = torch.einsum("ht,td->hd", probs, self.value_cache).reshape(1, self.heads * self.value_dim)
        return F.linear(context, self.attn_project)

    def _moe(self, hidden: torch.Tensor) -> torch.Tensor:
        route_probs = torch.softmax(F.linear(hidden, self.router_weight).float(), dim=-1)
        route_weights, expert_ids = torch.topk(route_probs, k=self.top_k, dim=-1)
        outputs: list[torch.Tensor] = []
        for expert_id in expert_ids.reshape(-1).tolist():
            outputs.append(_fp8_mlp(hidden, self.gate[expert_id], self.up[expert_id], self.down[expert_id]))
        stacked = torch.stack(outputs, dim=1)
        return (stacked * route_weights.reshape(1, -1, 1)).sum(dim=1)

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = F.layer_norm(hidden, (self.hidden_size,), weight=self.norm_weight)
        return hidden + self._attention(normed) + self._moe(normed)


def run_deepseek_gpu_probe(
    *,
    require_cuda: bool = False,
    device: str | None = None,
    hidden_size: int = 768,
    intermediate_size: int = 1536,
    routed_experts: int = 8,
    top_k: int = 2,
    cache_len: int = 384,
    benchmark_layers: int = 4,
    iterations: int = 3,
) -> DeepSeekGpuProbeResult:
    cuda_available = torch.cuda.is_available()
    if device is None:
        if cuda_available:
            device = "cuda"
        elif require_cuda:
            return DeepSeekGpuProbeResult(
                passed=False,
                cuda_available=False,
                device="cpu",
                torch_version=torch.__version__,
                layers_executed=0,
                benchmark_layers=benchmark_layers,
                iterations=iterations,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                routed_experts=routed_experts,
                top_k=top_k,
                cache_len=cache_len,
                benchmark_seconds=0.0,
                seconds_per_probe_token=0.0,
                seconds_per_layer=0.0,
                projected_62_layer_seconds_per_token=0.0,
                projected_k64_effective_seconds_per_position=0.0,
                checksum=0.0,
                max_abs_repeat_diff=0.0,
                output_shape=[],
                note="CUDA is required for the DeepSeek GPU probe but is not available.",
            )
        else:
            device = "cpu"

    torch_device = torch.device(device)
    heads = 8
    nope_dim = 64
    rope_dim = 32
    value_dim = 64
    if hidden_size < heads * value_dim:
        heads = max(1, hidden_size // value_dim)
    block = _SyntheticDeepSeekGpuBlock(
        device=torch_device,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        routed_experts=routed_experts,
        top_k=top_k,
        cache_len=cache_len,
        heads=heads,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        value_dim=value_dim,
    )
    hidden = torch.linspace(-0.5, 0.5, steps=hidden_size, dtype=torch.float32, device=torch_device).reshape(1, hidden_size)

    def token_forward() -> torch.Tensor:
        out = hidden
        for _ in range(benchmark_layers):
            out = block(out)
        return out

    first = token_forward()
    second = token_forward()
    _synchronize(torch_device)
    diff = float((first - second).abs().max().detach().cpu().item())

    start = time.perf_counter()
    last = first
    with torch.no_grad():
        for _ in range(iterations):
            last = token_forward()
        _synchronize(torch_device)
    elapsed = time.perf_counter() - start

    seconds_per_probe_token = elapsed / max(1, iterations)
    seconds_per_layer = seconds_per_probe_token / max(1, benchmark_layers)
    projected_full = seconds_per_layer * 62
    projected_k64 = projected_full / 65
    checksum = float(last.sum().detach().cpu().item())
    output_shape = list(last.shape)
    passed = bool(
        torch.isfinite(last).all().item()
        and diff == 0.0
        and output_shape == [1, hidden_size]
        and benchmark_layers > 0
        and iterations > 0
    )
    return DeepSeekGpuProbeResult(
        passed=passed,
        cuda_available=cuda_available,
        device=str(torch_device),
        torch_version=torch.__version__,
        layers_executed=62,
        benchmark_layers=benchmark_layers,
        iterations=iterations,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        routed_experts=routed_experts,
        top_k=top_k,
        cache_len=cache_len,
        benchmark_seconds=elapsed,
        seconds_per_probe_token=seconds_per_probe_token,
        seconds_per_layer=seconds_per_layer,
        projected_62_layer_seconds_per_token=projected_full,
        projected_k64_effective_seconds_per_position=projected_k64,
        checksum=checksum,
        max_abs_repeat_diff=diff,
        output_shape=output_shape,
        note=(
            "Synthetic DeepSeek-shaped FP8 GPU probe. This is not a full DeepSeek run; "
            "it measures CUDA FP8 dequant, routed MoE, MLA-shaped attention, and a 62-layer projection."
        ),
    )


def run_gpu_smoke(*, require_cuda: bool = False, device: str | None = None) -> GpuSmokeResult:
    cuda_available = torch.cuda.is_available()
    if device is None:
        if cuda_available:
            device = "cuda"
        elif require_cuda:
            return GpuSmokeResult(
                passed=False,
                cuda_available=False,
                device="cpu",
                torch_version=torch.__version__,
                checksum=0.0,
                max_abs_repeat_diff=0.0,
                output_shape=[],
                note="CUDA is required for this smoke run but is not available.",
            )
        else:
            device = "cpu"

    torch_device = torch.device(device)
    torch.manual_seed(14)
    hidden = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float32, device=torch_device).reshape(1, 64)

    def forward() -> torch.Tensor:
        moe = _moe_block(hidden)
        attn = _mla_attention(hidden)
        attn_project = torch.linspace(-0.2, 0.2, steps=64 * attn.shape[-1], device=torch_device).reshape(64, attn.shape[-1])
        return hidden + moe + F.linear(attn, attn_project)

    first = forward()
    second = forward()
    diff = float((first - second).abs().max().detach().cpu().item())
    checksum = float(first.sum().detach().cpu().item())
    passed = bool(torch.isfinite(first).all().item() and diff == 0.0 and list(first.shape) == [1, 64])
    _synchronize(torch_device)
    return GpuSmokeResult(
        passed=passed,
        cuda_available=cuda_available,
        device=str(torch_device),
        torch_version=torch.__version__,
        checksum=checksum,
        max_abs_repeat_diff=diff,
        output_shape=list(first.shape),
        note="FP8 dequant, MLP, MoE combine, and MLA-shaped attention smoke completed.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pcketlm synthetic FP8 GPU smoke path.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail when CUDA is not available.")
    parser.add_argument("--device", default=None, help="Torch device override, for example cuda or cpu.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--deepseek-probe", action="store_true", help="Also run the synthetic DeepSeek-shaped GPU timing probe.")
    args = parser.parse_args(argv)

    result = run_gpu_smoke(require_cuda=args.require_cuda, device=args.device)
    probe: DeepSeekGpuProbeResult | None = None
    if args.deepseek_probe:
        probe = run_deepseek_gpu_probe(require_cuda=args.require_cuda, device=args.device)
        payload: dict[str, Any] = {"smoke": result.to_dict(), "deepseek_gpu_probe": probe.to_dict()}
    else:
        payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"pcketlm GPU smoke: passed={result.passed} device={result.device} "
            f"cuda_available={result.cuda_available} checksum={result.checksum:.6f}"
        )
        if probe is not None:
            print(
                "DeepSeek GPU probe: "
                f"passed={probe.passed} projected_full={probe.projected_62_layer_seconds_per_token:.3f}s/token "
                f"projected_k64={probe.projected_k64_effective_seconds_per_position:.3f}s/position"
            )
    if args.require_cuda and not result.cuda_available:
        return 2
    passed = result.passed and (probe.passed if probe is not None else True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
