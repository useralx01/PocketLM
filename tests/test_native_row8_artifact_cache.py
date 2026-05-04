from pathlib import Path

import ctypes
import torch

from tools.pack_weights_row8 import pack_rows8_tensor


def test_native_row8_artifact_handle_loads_stable_bytes(tmp_path: Path) -> None:
    from pcketlm.native import load_native_row8_tensor, map_native_row8_file, native_row8_artifact_available

    assert native_row8_artifact_available()
    packed = torch.arange(32, dtype=torch.int32).to(torch.uint16)
    path = tmp_path / "packed.bin"
    path.write_bytes(packed.numpy().tobytes())

    handle = load_native_row8_tensor(path, 0, packed.numel() * packed.element_size())
    try:
        payload = ctypes.string_at(handle.ptr, handle.nbytes)
        assert payload == path.read_bytes()
    finally:
        handle.close()

    mapped = map_native_row8_file(path)
    try:
        ptr = mapped.data_ptr(0, packed.numel() * packed.element_size())
        assert ctypes.string_at(ptr, packed.numel() * packed.element_size()) == path.read_bytes()
    finally:
        mapped.close()


def test_dense_layer_decode_packed_rows8_ptrs_matches_tensor_path(tmp_path: Path) -> None:
    from pcketlm.native import NativeKvSession, load_native_row8_tensor

    hidden_size = 8
    intermediate_size = 16
    num_heads = 2
    num_kv_heads = 1
    head_dim = hidden_size // num_heads
    kv_width = num_kv_heads * head_dim
    torch.manual_seed(5588)
    hidden = torch.randn((hidden_size,), dtype=torch.float32).to(torch.bfloat16)
    input_norm = torch.ones((hidden_size,), dtype=torch.bfloat16)
    post_norm = torch.ones((hidden_size,), dtype=torch.bfloat16)
    q = (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16)
    k = (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16)
    v = (torch.randn((kv_width, hidden_size)) * 0.1).to(torch.bfloat16)
    o = (torch.randn((hidden_size, hidden_size)) * 0.1).to(torch.bfloat16)
    gate = (torch.randn((intermediate_size, hidden_size)) * 0.1).to(torch.bfloat16)
    up = (torch.randn((intermediate_size, hidden_size)) * 0.1).to(torch.bfloat16)
    down = (torch.randn((hidden_size, intermediate_size)) * 0.1).to(torch.bfloat16)
    packed = [pack_rows8_tensor(tensor) for tensor in (q, k, v, o, gate, up, down)]
    handles = []
    for index, packed_tensor in enumerate(packed):
        path = tmp_path / f"packed-{index}.bin"
        path.write_bytes(packed_tensor.numpy().tobytes())
        handles.append(load_native_row8_tensor(path, 0, packed_tensor.numel() * packed_tensor.element_size()))

    tensor_session = NativeKvSession(layer_count=1, max_seq_len=8, kv_width=kv_width, dtype=torch.bfloat16)
    ptr_session = NativeKvSession(layer_count=1, max_seq_len=8, kv_width=kv_width, dtype=torch.bfloat16)
    try:
        tensor_out = tensor_session.dense_layer_decode_packed_rows8(
            0,
            hidden,
            input_norm,
            post_norm,
            *packed,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_eps=1e-6,
            rope_theta=10000.0,
        )
        ptr_out = ptr_session.dense_layer_decode_packed_rows8_ptrs(
            0,
            hidden,
            input_norm,
            post_norm,
            *(handle.ptr for handle in handles),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_eps=1e-6,
            rope_theta=10000.0,
        )
        assert torch.equal(ptr_out, tensor_out)
    finally:
        tensor_session.close()
        ptr_session.close()
        for handle in handles:
            handle.close()
