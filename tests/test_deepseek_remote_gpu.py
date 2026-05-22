import torch

from pcketlm.core.runtime.deepseek_remote_gpu import _hf_url, _tensor_from_safetensors_bytes


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
