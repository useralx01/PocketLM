import os
import time

import torch


def test_native_packed_gemv_matches_torch_for_fp16_and_bf16() -> None:
    from pcketlm.native import native_fp16_packed_gemv_available, packed_gemv_rows8, pack_weight_rows8

    assert native_fp16_packed_gemv_available() is True
    torch.manual_seed(20260504)
    for dtype in (torch.float16, torch.bfloat16):
        for rows, cols in ((1, 7), (8, 16), (17, 31), (32, 64)):
            hidden = (torch.randn((cols,), dtype=torch.float32) * 0.2).to(dtype)
            weight = (torch.randn((rows, cols), dtype=torch.float32) * 0.2).to(dtype)
            packed = pack_weight_rows8(weight)
            native = packed_gemv_rows8(hidden, packed, rows=rows, cols=cols)
            expected = torch.mv(weight.float(), hidden.float())
            assert torch.allclose(native, expected, atol=1e-5, rtol=1e-5)


def test_native_packed_gemv_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_PACKED_GEMV", "1")
    from pcketlm.native import pack_weight_rows8

    try:
        pack_weight_rows8(torch.ones((8, 8), dtype=torch.float16))
    except RuntimeError as exc:
        assert "unavailable" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("PCKETLM_DISABLE_NATIVE_PACKED_GEMV did not disable packed GEMV")


def test_native_packed_gemv_microbench_smoke() -> None:
    from pcketlm.native import packed_gemv_rows8, pack_weight_rows8

    old_threads = os.environ.get("PCKETLM_NATIVE_THREADS")
    try:
        os.environ["PCKETLM_NATIVE_THREADS"] = "1"
        torch.manual_seed(9876)
        rows = 1024
        cols = 1024
        hidden = torch.randn((cols,), dtype=torch.float16)
        weight = torch.randn((rows, cols), dtype=torch.float16)
        packed = pack_weight_rows8(weight)
        start = time.perf_counter()
        native = packed_gemv_rows8(hidden, packed, rows=rows, cols=cols)
        elapsed = time.perf_counter() - start
        expected = torch.mv(weight.float(), hidden.float())
        assert torch.allclose(native, expected, atol=5e-3, rtol=5e-3)
        assert elapsed < 1.0
    finally:
        if old_threads is None:
            os.environ.pop("PCKETLM_NATIVE_THREADS", None)
        else:
            os.environ["PCKETLM_NATIVE_THREADS"] = old_threads
