import torch
import torch.nn.functional as F


def _reference_moe(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    top_k: int,
    normalize_topk: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_f = hidden.float()
    router_logits = F.linear(hidden_f, router_weight.float())
    router_probs = torch.softmax(router_logits, dim=-1)
    weights, experts = torch.topk(router_probs, k=top_k, dim=-1)
    if normalize_topk:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    output = torch.zeros_like(hidden_f)
    for token in range(hidden.shape[0]):
        for rank in range(top_k):
            expert = int(experts[token, rank].item())
            gate = F.linear(hidden_f[token : token + 1], gate_weight[expert].float())
            up = F.linear(hidden_f[token : token + 1], up_weight[expert].float())
            down = F.linear(F.silu(gate) * up, down_weight[expert].float())
            output[token] += weights[token, rank] * down.squeeze(0)
    return output.to(torch.float16), experts.to(torch.int64), weights.to(torch.float32)


def test_native_moe_forward_matches_python_reference() -> None:
    from pcketlm.native import moe_forward_fp16, native_fp16_moe_available

    assert native_fp16_moe_available() is True
    torch.manual_seed(777)
    seq_len = 3
    hidden_size = 8
    num_experts = 5
    intermediate_size = 12
    top_k = 2
    hidden = (torch.randn((seq_len, hidden_size), dtype=torch.float32) * 0.2).to(torch.float16)
    router = (torch.randn((num_experts, hidden_size), dtype=torch.float32) * 0.2).to(torch.float16)
    gate = (torch.randn((num_experts, intermediate_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)
    up = (torch.randn((num_experts, intermediate_size, hidden_size), dtype=torch.float32) * 0.15).to(torch.float16)
    down = (torch.randn((num_experts, hidden_size, intermediate_size), dtype=torch.float32) * 0.15).to(torch.float16)

    native, native_experts, native_weights = moe_forward_fp16(
        hidden,
        router,
        gate,
        up,
        down,
        top_k=top_k,
        normalize_topk=True,
    )
    expected, expected_experts, expected_weights = _reference_moe(
        hidden,
        router,
        gate,
        up,
        down,
        top_k,
        True,
    )

    assert torch.equal(native_experts, expected_experts)
    assert torch.allclose(native_weights, expected_weights, atol=1e-6, rtol=1e-6)
    assert torch.allclose(native.float(), expected.float(), atol=1e-3, rtol=1e-3)


def test_native_moe_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_MOE", "1")
    from pcketlm.native import moe_forward_fp16

    hidden = torch.ones((1, 4), dtype=torch.float16)
    try:
        moe_forward_fp16(
            hidden,
            torch.ones((2, 4), dtype=torch.float16),
            torch.ones((2, 4, 4), dtype=torch.float16),
            torch.ones((2, 4, 4), dtype=torch.float16),
            torch.ones((2, 4, 4), dtype=torch.float16),
            top_k=1,
            normalize_topk=True,
        )
    except RuntimeError as exc:
        assert "unavailable" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("PCKETLM_DISABLE_NATIVE_MOE did not disable MoE")
