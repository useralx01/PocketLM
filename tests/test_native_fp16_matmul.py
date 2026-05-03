import os
import time

import torch


def _ordered_fp16_bits(tensor: torch.Tensor) -> torch.Tensor:
    bits = tensor.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, 0x8000 - bits, bits)


def _max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((_ordered_fp16_bits(a) - _ordered_fp16_bits(b)).abs().max().item())


def test_native_fp16_matmul_matches_torch_small_shapes() -> None:
    from pcketlm.native import fp16_matmul, native_fp16_matmul_available

    assert native_fp16_matmul_available() is True
    torch.manual_seed(123)
    shapes = [(1, 8, 8), (3, 9, 5), (8, 16, 8), (17, 31, 13)]
    for m, k, n in shapes:
        a = (torch.randn((m, k), dtype=torch.float32) * 0.25).to(torch.float16)
        b = (torch.randn((k, n), dtype=torch.float32) * 0.25).to(torch.float16)
        native = fp16_matmul(a, b)
        expected = (a.float() @ b.float()).to(torch.float16)
        assert _max_ulp(native, expected) <= 1


def test_native_fp16_matmul_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_MATMUL", "1")
    from pcketlm.native import fp16_matmul

    try:
        fp16_matmul(torch.ones((1, 1), dtype=torch.float16), torch.ones((1, 1), dtype=torch.float16))
    except RuntimeError as exc:
        assert "unavailable" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("PCKETLM_DISABLE_NATIVE_MATMUL did not disable the native matmul path")


def test_native_fp16_matmul_blas_kill_switch_keeps_handwritten_fallback(monkeypatch) -> None:
    from pcketlm.native import fp16_matmul

    monkeypatch.setenv("PCKETLM_DISABLE_BLAS_GEMM", "1")
    torch.manual_seed(321)
    a = (torch.randn((5, 7), dtype=torch.float32) * 0.2).to(torch.float16)
    b = (torch.randn((7, 3), dtype=torch.float32) * 0.2).to(torch.float16)
    native = fp16_matmul(a, b)
    expected = (a.float() @ b.float()).to(torch.float16)
    assert _max_ulp(native, expected) <= 1


def test_native_fp16_matmul_microbench_smoke() -> None:
    from pcketlm.native import fp16_matmul

    old_threads = os.environ.get("PCKETLM_NATIVE_THREADS")
    try:
        os.environ["PCKETLM_NATIVE_THREADS"] = "2"
        torch.manual_seed(456)
        a = torch.randn((128, 128), dtype=torch.float16)
        b = torch.randn((128, 128), dtype=torch.float16)
        start = time.perf_counter()
        out = fp16_matmul(a, b)
        elapsed = time.perf_counter() - start
        assert out.shape == (128, 128)
        assert elapsed < 2.0
    finally:
        if old_threads is None:
            os.environ.pop("PCKETLM_NATIVE_THREADS", None)
        else:
            os.environ["PCKETLM_NATIVE_THREADS"] = old_threads


def test_native_lm_head_topk_matches_torch_for_fp16_and_bf16() -> None:
    from pcketlm.native import lm_head_topk_u16

    torch.manual_seed(789)
    for dtype in (torch.float16, torch.bfloat16):
        hidden = (torch.randn((32,), dtype=torch.float32) * 0.2).to(dtype)
        weight = (torch.randn((41, 32), dtype=torch.float32) * 0.2).to(dtype)
        native_logits, native_ids = lm_head_topk_u16(hidden, weight, top_k=5, token_offset=100)
        expected = torch.mv(weight.float(), hidden.float())
        expected_logits, expected_offsets = torch.topk(expected, k=5)
        assert torch.equal(native_ids, expected_offsets.to(torch.int64) + 100)
        assert torch.allclose(native_logits, expected_logits, atol=1e-4, rtol=1e-4)
