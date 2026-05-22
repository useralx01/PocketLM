"""Tiny GPU smoke path for cloud notebook validation."""

from __future__ import annotations

import argparse
import json
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
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
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
    args = parser.parse_args(argv)

    result = run_gpu_smoke(require_cuda=args.require_cuda, device=args.device)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"pcketlm GPU smoke: passed={result.passed} device={result.device} "
            f"cuda_available={result.cuda_available} checksum={result.checksum:.6f}"
        )
    if args.require_cuda and not result.cuda_available:
        return 2
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
