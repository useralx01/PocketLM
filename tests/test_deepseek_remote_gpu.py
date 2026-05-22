from types import SimpleNamespace

import torch

from pcketlm.core.runtime.deepseek_remote_gpu import (
    _ResidentRemoteDeepSeekLayerPager,
    _hf_url,
    _tensor_from_safetensors_bytes,
)


class _DummyResidentLayer:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = int(nbytes)
        self.released = False

    def resident_nbytes(self) -> int:
        return 0 if self.released else self.nbytes

    def release(self) -> None:
        self.released = True


def test_hf_url_escapes_repo_revision_and_filename() -> None:
    assert _hf_url("deepseek-ai/DeepSeek-V3", "main", "model.safetensors.index.json") == (
        "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/model.safetensors.index.json"
    )


def test_tensor_from_fp8_safetensors_bytes_keeps_raw_uint8() -> None:
    tensor = _tensor_from_safetensors_bytes(bytes([1, 2, 3, 4]), "F8_E4M3", [2, 2], device=torch.device("cpu"))

    assert tensor.dtype == torch.uint8
    assert tensor.shape == (2, 2)
    assert tensor.tolist() == [[1, 2], [3, 4]]


def test_tensor_from_bf16_safetensors_bytes_materializes_dtype() -> None:
    source = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    tensor = _tensor_from_safetensors_bytes(source.view(torch.uint8).numpy().tobytes(), "BF16", [1, 2], device=torch.device("cpu"))

    assert tensor.dtype == torch.bfloat16
    assert torch.equal(tensor, source)


def test_resident_layer_pager_evicts_lru_without_touching_protected_layer() -> None:
    pager = _ResidentRemoteDeepSeekLayerPager(
        SimpleNamespace(device=torch.device("cpu")),
        config={},
        dtype=torch.float16,
        max_resident_bytes=100,
    )
    pager.layers = {
        3: _DummyResidentLayer(70),  # type: ignore[assignment]
        4: _DummyResidentLayer(70),  # type: ignore[assignment]
    }
    pager._lru = [3, 4]

    pager.enforce_budget(protected_layer=4)

    assert sorted(pager.layers) == [4]
    assert pager.evictions == 1
    assert pager.resident_nbytes() == 70
