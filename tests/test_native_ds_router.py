from pathlib import Path

import torch

import pcketlm.native as native


def test_ds_router_topk_matches_python_reference_on_synthetic_vectors() -> None:
    generator = torch.Generator().manual_seed(20260519)
    router = torch.randn((11, 16), generator=generator, dtype=torch.float32).mul(0.25).to(torch.float16)
    hidden_rows = torch.randn((5, 16), generator=generator, dtype=torch.float32).mul(0.5).to(torch.float16)

    for hidden in hidden_rows:
        native_ids, native_weights = native.ds_router_topk(hidden, router, k=4)
        python_ids, python_weights = native._python_ds_router_topk(hidden, router, 4)

        assert torch.equal(native_ids, python_ids)
        assert torch.allclose(native_weights, python_weights, atol=1e-6, rtol=1e-6)


def test_ds_router_topk_kill_switch_falls_back_to_python(monkeypatch) -> None:
    router = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float16,
    )
    hidden = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float16)
    monkeypatch.setenv("PCKETLM_DISABLE_NATIVE_DS_ROUTER", "1")

    ids, weights = native.ds_router_topk(hidden, router, k=2)
    expected_ids, expected_weights = native._python_ds_router_topk(hidden, router, 2)

    assert native.native_ds_router_available() is False
    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)


def test_ds_router_topk_matches_real_deepseek_router_ids() -> None:
    from pcketlm.core.runtime.fp8_source import _load_regular_tensor

    router, blockers = _load_regular_tensor("deepseek-v3", "model.layers.3.mlp.gate.weight")
    assert not blockers
    assert router is not None
    hidden = torch.linspace(-0.5, 0.5, int(router.shape[1]), dtype=torch.float32).to(torch.float16)
    router_fp16 = router.to(torch.float16)

    native_ids, native_weights = native.ds_router_topk(hidden, router_fp16, k=8)
    python_ids, python_weights = native._python_ds_router_topk(hidden, router_fp16, 8)

    assert torch.equal(native_ids, python_ids)
    assert torch.allclose(native_weights, python_weights, atol=1e-6, rtol=1e-6)
    assert native_ids.numel() == 8
