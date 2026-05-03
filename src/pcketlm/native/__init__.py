"""ctypes bindings for pcketlm native helper kernels."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch


_NATIVE_DIR = Path(__file__).resolve().parent
_Q4_DLL = _NATIVE_DIR / "q4_dequant.dll"
_FP16_LOADER_DLL = _NATIVE_DIR / "fp16_loader.dll"
_FP16_MATMUL_DLL = _NATIVE_DIR / "fp16_matmul.dll"
_FP16_ATTENTION_DLL = _NATIVE_DIR / "fp16_attention.dll"
_FP16_MOE_DLL = _NATIVE_DIR / "fp16_moe.dll"
_Q4_LIB: ctypes.CDLL | None = None
_Q4_LOAD_ERROR: Exception | None = None
_FP16_LOADER_LIB: ctypes.CDLL | None = None
_FP16_LOADER_ERROR: Exception | None = None
_FP16_MATMUL_LIB: ctypes.CDLL | None = None
_FP16_MATMUL_ERROR: Exception | None = None
_FP16_ATTENTION_LIB: ctypes.CDLL | None = None
_FP16_ATTENTION_ERROR: Exception | None = None
_FP16_MOE_LIB: ctypes.CDLL | None = None
_FP16_MOE_ERROR: Exception | None = None


def _native_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_Q4", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_fp16_load_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_FP16_LOAD", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_matmul_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_MATMUL", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_attention_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_ATTENTION", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_moe_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_MOE", "").strip().lower() in {"1", "true", "yes", "on"}


def _load_q4_lib() -> ctypes.CDLL | None:
    global _Q4_LIB, _Q4_LOAD_ERROR
    if _native_disabled():
        return None
    if _Q4_LIB is not None:
        return _Q4_LIB
    if not _Q4_DLL.exists():
        _Q4_LOAD_ERROR = FileNotFoundError(str(_Q4_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_Q4_DLL))
        lib.q4_dequant_to_fp16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]
        lib.q4_dequant_to_fp16.restype = None
        lib.q4_cpu_has_avx2_f16c.argtypes = []
        lib.q4_cpu_has_avx2_f16c.restype = ctypes.c_int
    except Exception as exc:  # pragma: no cover - defensive platform path
        _Q4_LOAD_ERROR = exc
        return None
    _Q4_LIB = lib
    _Q4_LOAD_ERROR = None
    return lib


def native_q4_available() -> bool:
    return _load_q4_lib() is not None


def native_q4_has_avx2_f16c() -> bool:
    lib = _load_q4_lib()
    return bool(lib and lib.q4_cpu_has_avx2_f16c())


def native_q4_load_error() -> Exception | None:
    _load_q4_lib()
    return _Q4_LOAD_ERROR


def _load_fp16_loader_lib() -> ctypes.CDLL | None:
    global _FP16_LOADER_LIB, _FP16_LOADER_ERROR
    if _native_fp16_load_disabled():
        return None
    if _FP16_LOADER_LIB is not None:
        return _FP16_LOADER_LIB
    if not _FP16_LOADER_DLL.exists():
        _FP16_LOADER_ERROR = FileNotFoundError(str(_FP16_LOADER_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_FP16_LOADER_DLL))
        lib.native_read_tensor_bytes.argtypes = [
            ctypes.c_char_p,
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
            ctypes.c_void_p,
        ]
        lib.native_read_tensor_bytes.restype = ctypes.c_int
        lib.native_copy_tensor_bytes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulonglong,
            ctypes.c_void_p,
        ]
        lib.native_copy_tensor_bytes.restype = ctypes.c_int
    except Exception as exc:  # pragma: no cover - defensive platform path
        _FP16_LOADER_ERROR = exc
        return None
    _FP16_LOADER_LIB = lib
    _FP16_LOADER_ERROR = None
    return lib


def native_fp16_loader_available() -> bool:
    return _load_fp16_loader_lib() is not None


def native_fp16_loader_error() -> Exception | None:
    _load_fp16_loader_lib()
    return _FP16_LOADER_ERROR


def _load_fp16_matmul_lib() -> ctypes.CDLL | None:
    global _FP16_MATMUL_LIB, _FP16_MATMUL_ERROR
    if _native_matmul_disabled():
        return None
    if _FP16_MATMUL_LIB is not None:
        return _FP16_MATMUL_LIB
    if not _FP16_MATMUL_DLL.exists():
        _FP16_MATMUL_ERROR = FileNotFoundError(str(_FP16_MATMUL_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_FP16_MATMUL_DLL))
        lib.native_fp16_matmul.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]
        lib.native_fp16_matmul.restype = ctypes.c_int
    except Exception as exc:  # pragma: no cover - defensive platform path
        _FP16_MATMUL_ERROR = exc
        return None
    _FP16_MATMUL_LIB = lib
    _FP16_MATMUL_ERROR = None
    return lib


def native_fp16_matmul_available() -> bool:
    return _load_fp16_matmul_lib() is not None


def native_fp16_matmul_error() -> Exception | None:
    _load_fp16_matmul_lib()
    return _FP16_MATMUL_ERROR


def fp16_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lib = _load_fp16_matmul_lib()
    if lib is None:
        reason = "disabled" if _native_matmul_disabled() else _FP16_MATMUL_ERROR
        raise RuntimeError(f"Native fp16 matmul is unavailable: {reason}")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("fp16_matmul requires torch.float16 inputs")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("fp16_matmul requires 2D matrices")
    if int(a.shape[1]) != int(b.shape[0]):
        raise ValueError("fp16_matmul input shapes are incompatible")
    a_cpu = a.detach().cpu().contiguous()
    b_cpu = b.detach().cpu().contiguous()
    out = torch.empty((int(a_cpu.shape[0]), int(b_cpu.shape[1])), dtype=torch.float16)
    code = lib.native_fp16_matmul(
        ctypes.c_void_p(int(a_cpu.data_ptr())),
        ctypes.c_void_p(int(b_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(int(a_cpu.shape[0])),
        ctypes.c_longlong(int(b_cpu.shape[1])),
        ctypes.c_longlong(int(a_cpu.shape[1])),
    )
    if code != 0:
        raise RuntimeError(f"native_fp16_matmul failed with code {code}")
    return out


def _load_fp16_attention_lib() -> ctypes.CDLL | None:
    global _FP16_ATTENTION_LIB, _FP16_ATTENTION_ERROR
    if _native_attention_disabled():
        return None
    if _FP16_ATTENTION_LIB is not None:
        return _FP16_ATTENTION_LIB
    if not _FP16_ATTENTION_DLL.exists():
        _FP16_ATTENTION_ERROR = FileNotFoundError(str(_FP16_ATTENTION_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_FP16_ATTENTION_DLL))
        lib.native_attention_prefill_fp16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_float,
        ]
        lib.native_attention_prefill_fp16.restype = ctypes.c_int
    except Exception as exc:  # pragma: no cover - defensive platform path
        _FP16_ATTENTION_ERROR = exc
        return None
    _FP16_ATTENTION_LIB = lib
    _FP16_ATTENTION_ERROR = None
    return lib


def native_fp16_attention_available() -> bool:
    return _load_fp16_attention_lib() is not None


def native_fp16_attention_error() -> Exception | None:
    _load_fp16_attention_lib()
    return _FP16_ATTENTION_ERROR


def attention_prefill_fp16(
    hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    position_offset: int = 0,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    lib = _load_fp16_attention_lib()
    if lib is None:
        reason = "disabled" if _native_attention_disabled() else _FP16_ATTENTION_ERROR
        raise RuntimeError(f"Native fp16 attention is unavailable: {reason}")
    tensors = [hidden, q_weight, k_weight, v_weight, o_weight]
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise TypeError("attention_prefill_fp16 requires torch.float16 tensors")
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [seq_len, hidden_size]")
    seq_len = int(hidden.shape[0])
    hidden_size = int(hidden.shape[1])
    hidden_cpu = hidden.detach().cpu().contiguous()
    q_cpu = q_weight.detach().cpu().contiguous()
    k_cpu = k_weight.detach().cpu().contiguous()
    v_cpu = v_weight.detach().cpu().contiguous()
    o_cpu = o_weight.detach().cpu().contiguous()
    out = torch.empty((seq_len, hidden_size), dtype=torch.float16)
    code = lib.native_attention_prefill_fp16(
        ctypes.c_void_p(int(hidden_cpu.data_ptr())),
        ctypes.c_void_p(int(q_cpu.data_ptr())),
        ctypes.c_void_p(int(k_cpu.data_ptr())),
        ctypes.c_void_p(int(v_cpu.data_ptr())),
        ctypes.c_void_p(int(o_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(seq_len),
        ctypes.c_longlong(hidden_size),
        ctypes.c_longlong(int(num_attention_heads)),
        ctypes.c_longlong(int(num_key_value_heads)),
        ctypes.c_longlong(int(position_offset)),
        ctypes.c_float(float(rope_theta)),
    )
    if code != 0:
        raise RuntimeError(f"native_attention_prefill_fp16 failed with code {code}")
    return out


def _load_fp16_moe_lib() -> ctypes.CDLL | None:
    global _FP16_MOE_LIB, _FP16_MOE_ERROR
    if _native_moe_disabled():
        return None
    if _FP16_MOE_LIB is not None:
        return _FP16_MOE_LIB
    if not _FP16_MOE_DLL.exists():
        _FP16_MOE_ERROR = FileNotFoundError(str(_FP16_MOE_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_FP16_MOE_DLL))
        lib.native_moe_forward_fp16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_int,
        ]
        lib.native_moe_forward_fp16.restype = ctypes.c_int
    except Exception as exc:  # pragma: no cover - defensive platform path
        _FP16_MOE_ERROR = exc
        return None
    _FP16_MOE_LIB = lib
    _FP16_MOE_ERROR = None
    return lib


def native_fp16_moe_available() -> bool:
    return _load_fp16_moe_lib() is not None


def native_fp16_moe_error() -> Exception | None:
    _load_fp16_moe_lib()
    return _FP16_MOE_ERROR


def moe_forward_fp16(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    top_k: int,
    normalize_topk: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lib = _load_fp16_moe_lib()
    if lib is None:
        reason = "disabled" if _native_moe_disabled() else _FP16_MOE_ERROR
        raise RuntimeError(f"Native fp16 MoE is unavailable: {reason}")
    tensors = [hidden, router_weight, gate_weight, up_weight, down_weight]
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise TypeError("moe_forward_fp16 requires torch.float16 tensors")
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [seq_len, hidden_size]")
    if gate_weight.ndim != 3 or up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("expert weights must be rank-3 [num_experts, rows, cols]")
    seq_len = int(hidden.shape[0])
    hidden_size = int(hidden.shape[1])
    num_experts = int(router_weight.shape[0])
    intermediate_size = int(gate_weight.shape[1])
    hidden_cpu = hidden.detach().cpu().contiguous()
    router_cpu = router_weight.detach().cpu().contiguous()
    gate_cpu = gate_weight.detach().cpu().contiguous()
    up_cpu = up_weight.detach().cpu().contiguous()
    down_cpu = down_weight.detach().cpu().contiguous()
    out = torch.empty((seq_len, hidden_size), dtype=torch.float16)
    selected_experts = torch.empty((seq_len, int(top_k)), dtype=torch.int64)
    selected_weights = torch.empty((seq_len, int(top_k)), dtype=torch.float32)
    code = lib.native_moe_forward_fp16(
        ctypes.c_void_p(int(hidden_cpu.data_ptr())),
        ctypes.c_void_p(int(router_cpu.data_ptr())),
        ctypes.c_void_p(int(gate_cpu.data_ptr())),
        ctypes.c_void_p(int(up_cpu.data_ptr())),
        ctypes.c_void_p(int(down_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_void_p(int(selected_experts.data_ptr())),
        ctypes.c_void_p(int(selected_weights.data_ptr())),
        ctypes.c_longlong(seq_len),
        ctypes.c_longlong(hidden_size),
        ctypes.c_longlong(num_experts),
        ctypes.c_longlong(int(top_k)),
        ctypes.c_longlong(intermediate_size),
        ctypes.c_int(1 if normalize_topk else 0),
    )
    if code != 0:
        raise RuntimeError(f"native_moe_forward_fp16 failed with code {code}")
    return out, selected_experts, selected_weights


def native_read_tensor_bytes(path: str | Path, absolute_offset: int, nbytes: int, out: torch.Tensor) -> None:
    lib = _load_fp16_loader_lib()
    if lib is None:
        reason = "disabled" if _native_fp16_load_disabled() else _FP16_LOADER_ERROR
        raise RuntimeError(f"Native fp16 loader is unavailable: {reason}")
    if not out.is_contiguous():
        raise ValueError("native_read_tensor_bytes requires a contiguous output tensor")
    if out.nelement() * out.element_size() != int(nbytes):
        raise ValueError("output tensor byte size does not match requested native read size")
    path_bytes = str(Path(path)).encode("utf-8")
    code = lib.native_read_tensor_bytes(
        ctypes.c_char_p(path_bytes),
        ctypes.c_ulonglong(int(absolute_offset)),
        ctypes.c_ulonglong(int(nbytes)),
        ctypes.c_void_p(int(out.data_ptr())),
    )
    if code != 0:
        raise OSError(f"native_read_tensor_bytes failed with code {code} for {path}")


def native_read_bytes(path: str | Path, absolute_offset: int, nbytes: int) -> bytearray:
    lib = _load_fp16_loader_lib()
    if lib is None:
        reason = "disabled" if _native_fp16_load_disabled() else _FP16_LOADER_ERROR
        raise RuntimeError(f"Native fp16 loader is unavailable: {reason}")
    raw = bytearray(int(nbytes))
    raw_view = (ctypes.c_uint8 * int(nbytes)).from_buffer(raw)
    path_bytes = str(Path(path)).encode("utf-8")
    code = lib.native_read_tensor_bytes(
        ctypes.c_char_p(path_bytes),
        ctypes.c_ulonglong(int(absolute_offset)),
        ctypes.c_ulonglong(int(nbytes)),
        ctypes.c_void_p(ctypes.addressof(raw_view)),
    )
    if code != 0:
        raise OSError(f"native_read_tensor_bytes failed with code {code} for {path}")
    return raw


def native_copy_tensor_bytes(source: bytes | bytearray | memoryview, out: torch.Tensor) -> None:
    lib = _load_fp16_loader_lib()
    if lib is None:
        reason = "disabled" if _native_fp16_load_disabled() else _FP16_LOADER_ERROR
        raise RuntimeError(f"Native fp16 loader is unavailable: {reason}")
    if not out.is_contiguous():
        raise ValueError("native_copy_tensor_bytes requires a contiguous output tensor")
    nbytes = out.nelement() * out.element_size()
    if len(source) != int(nbytes):
        raise ValueError("source byte size does not match output tensor byte size")
    if isinstance(source, bytearray):
        source_view = (ctypes.c_uint8 * int(nbytes)).from_buffer(source)
        source_ptr = ctypes.addressof(source_view)
    else:
        source_bytes = bytes(source)
        source_ptr = ctypes.cast(ctypes.c_char_p(source_bytes), ctypes.c_void_p).value
    code = lib.native_copy_tensor_bytes(
        ctypes.c_void_p(int(source_ptr)),
        ctypes.c_ulonglong(int(nbytes)),
        ctypes.c_void_p(int(out.data_ptr())),
    )
    if code != 0:
        raise OSError(f"native_copy_tensor_bytes failed with code {code}")


def q4_dequant_to_fp16(
    packed: torch.Tensor,
    scales: torch.Tensor,
    num_channels: int,
    channel_size: int,
) -> torch.Tensor:
    lib = _load_q4_lib()
    if lib is None:
        reason = "disabled" if _native_disabled() else _Q4_LOAD_ERROR
        raise RuntimeError(f"Native Q4 dequant is unavailable: {reason}")
    packed_cpu = packed.detach().cpu().contiguous().to(torch.uint8)
    scales_cpu = scales.detach().cpu().contiguous().to(torch.float16)
    out = torch.empty((int(num_channels) * int(channel_size),), dtype=torch.float16)
    lib.q4_dequant_to_fp16(
        ctypes.c_void_p(int(packed_cpu.data_ptr())),
        ctypes.c_void_p(int(scales_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(int(num_channels)),
        ctypes.c_longlong(int(channel_size)),
    )
    return out
