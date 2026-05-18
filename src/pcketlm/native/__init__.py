"""ctypes bindings for pcketlm native helper kernels."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from collections import OrderedDict

import torch


_NATIVE_DIR = Path(__file__).resolve().parent
_Q4_DLL = _NATIVE_DIR / "q4_dequant.dll"
_FP16_LOADER_DLL = _NATIVE_DIR / "fp16_loader.dll"
_FP16_MATMUL_DLL = _NATIVE_DIR / "fp16_matmul.dll"
_FP16_ATTENTION_DLL = _NATIVE_DIR / "fp16_attention.dll"
_FP16_MOE_DLL = _NATIVE_DIR / "fp16_moe.dll"
_FP16_KV_DLL = _NATIVE_DIR / "fp16_kv_cache.dll"
_FP16_PACKED_GEMV_DLL = _NATIVE_DIR / "fp16_packed_gemv.dll"
_ROW8_ARTIFACT_DLL = _NATIVE_DIR / "row8_artifact_cache.dll"
_FP8_LINEAR_DLL = _NATIVE_DIR / "fp8_linear.dll"
_PCKETLM_FORWARD_DLL = _NATIVE_DIR / "pcketlm_forward.dll"
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
_FP16_KV_LIB: ctypes.CDLL | None = None
_FP16_KV_ERROR: Exception | None = None
_FP16_PACKED_GEMV_LIB: ctypes.CDLL | None = None
_FP16_PACKED_GEMV_ERROR: Exception | None = None
_ROW8_ARTIFACT_LIB: ctypes.CDLL | None = None
_ROW8_ARTIFACT_ERROR: Exception | None = None
_FP8_LINEAR_LIB: ctypes.CDLL | None = None
_FP8_LINEAR_ERROR: Exception | None = None
_PCKETLM_FORWARD_LIB: ctypes.CDLL | None = None
_PCKETLM_FORWARD_ERROR: Exception | None = None
_PACKED_GEMV_CACHE: "OrderedDict[str, torch.Tensor]" = OrderedDict()
_PACKED_GEMV_CACHE_BYTES = 0
_PACKED_GEMV_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "stores": 0,
    "evictions": 0,
}


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


def _native_kv_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_KV", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_packed_gemv_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_PACKED_GEMV", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_row8_artifact_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_ROW8_ARTIFACT", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_fp8_linear_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_FP8_LINEAR", "").strip().lower() in {"1", "true", "yes", "on"}


def _monolithic_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_MONOLITHIC", "").strip().lower() in {"1", "true", "yes", "on"}


def _u16_storage_dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    raise TypeError("native KV kernels require torch.float16 or torch.bfloat16 tensors")


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
        if hasattr(lib, "q4_dequant_many_to_fp16"):
            lib.q4_dequant_many_to_fp16.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.c_longlong,
            ]
            lib.q4_dequant_many_to_fp16.restype = ctypes.c_int
        if hasattr(lib, "q4_moe_selected_forward_u16"):
            lib.q4_moe_selected_forward_u16.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
            ]
            lib.q4_moe_selected_forward_u16.restype = ctypes.c_int
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


def _load_fp8_linear_lib() -> ctypes.CDLL | None:
    global _FP8_LINEAR_LIB, _FP8_LINEAR_ERROR
    if _native_fp8_linear_disabled():
        return None
    if _FP8_LINEAR_LIB is not None:
        return _FP8_LINEAR_LIB
    if not _FP8_LINEAR_DLL.exists():
        _FP8_LINEAR_ERROR = FileNotFoundError(str(_FP8_LINEAR_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_FP8_LINEAR_DLL))
        lib.fp8_e4m3_block_linear_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]
        lib.fp8_e4m3_block_linear_f32.restype = ctypes.c_int
        if hasattr(lib, "fp8_e4m3_block_dual_linear_f32"):
            lib.fp8_e4m3_block_dual_linear_f32.argtypes = [
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
            ]
            lib.fp8_e4m3_block_dual_linear_f32.restype = ctypes.c_int
        if hasattr(lib, "fp8_e4m3_block_mlp_f32"):
            lib.fp8_e4m3_block_mlp_f32.argtypes = [
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
            ]
            lib.fp8_e4m3_block_mlp_f32.restype = ctypes.c_int
        if hasattr(lib, "fp8_e4m3_block_mlp_many_f32"):
            lib.fp8_e4m3_block_mlp_many_f32.argtypes = [
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
                ctypes.c_longlong,
            ]
            lib.fp8_e4m3_block_mlp_many_f32.restype = ctypes.c_int
    except Exception as exc:  # pragma: no cover - defensive platform path
        _FP8_LINEAR_ERROR = exc
        return None
    _FP8_LINEAR_LIB = lib
    _FP8_LINEAR_ERROR = None
    return lib


def native_fp8_linear_available() -> bool:
    return _load_fp8_linear_lib() is not None


def native_fp8_linear_error() -> Exception | None:
    _load_fp8_linear_lib()
    return _FP8_LINEAR_ERROR


def native_fp8_dual_linear_available() -> bool:
    lib = _load_fp8_linear_lib()
    return bool(lib is not None and hasattr(lib, "fp8_e4m3_block_dual_linear_f32"))


def native_fp8_mlp_available() -> bool:
    lib = _load_fp8_linear_lib()
    return bool(lib is not None and hasattr(lib, "fp8_e4m3_block_mlp_f32"))


def native_fp8_mlp_many_available() -> bool:
    lib = _load_fp8_linear_lib()
    return bool(lib is not None and hasattr(lib, "fp8_e4m3_block_mlp_many_f32"))


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
        if hasattr(lib, "native_lm_head_topk_u16"):
            lib.native_lm_head_topk_u16.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_int,
                ctypes.c_longlong,
            ]
            lib.native_lm_head_topk_u16.restype = ctypes.c_int
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


def _load_fp16_packed_gemv_lib() -> ctypes.CDLL | None:
    global _FP16_PACKED_GEMV_LIB, _FP16_PACKED_GEMV_ERROR
    if _native_packed_gemv_disabled():
        return None
    if _FP16_PACKED_GEMV_LIB is not None:
        return _FP16_PACKED_GEMV_LIB
    if not _FP16_PACKED_GEMV_DLL.exists():
        _FP16_PACKED_GEMV_ERROR = FileNotFoundError(str(_FP16_PACKED_GEMV_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_FP16_PACKED_GEMV_DLL))
        lib.native_pack_u16_rows8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]
        lib.native_pack_u16_rows8.restype = ctypes.c_int
        lib.native_packed_gemv_rows8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_int,
        ]
        lib.native_packed_gemv_rows8.restype = ctypes.c_int
    except Exception as exc:  # pragma: no cover - defensive platform path
        _FP16_PACKED_GEMV_ERROR = exc
        return None
    _FP16_PACKED_GEMV_LIB = lib
    _FP16_PACKED_GEMV_ERROR = None
    return lib


def native_fp16_packed_gemv_available() -> bool:
    return _load_fp16_packed_gemv_lib() is not None


def native_fp16_packed_gemv_error() -> Exception | None:
    _load_fp16_packed_gemv_lib()
    return _FP16_PACKED_GEMV_ERROR


def _load_row8_artifact_lib() -> ctypes.CDLL | None:
    global _ROW8_ARTIFACT_LIB, _ROW8_ARTIFACT_ERROR
    if _native_row8_artifact_disabled():
        return None
    if _ROW8_ARTIFACT_LIB is not None:
        return _ROW8_ARTIFACT_LIB
    if not _ROW8_ARTIFACT_DLL.exists():
        _ROW8_ARTIFACT_ERROR = FileNotFoundError(str(_ROW8_ARTIFACT_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_ROW8_ARTIFACT_DLL))
        lib.row8_artifact_load_tensor.argtypes = [
            ctypes.c_char_p,
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
        ]
        lib.row8_artifact_load_tensor.restype = ctypes.c_void_p
        lib.row8_artifact_free_tensor.argtypes = [ctypes.c_void_p]
        lib.row8_artifact_free_tensor.restype = None
        lib.row8_artifact_tensor_data.argtypes = [ctypes.c_void_p]
        lib.row8_artifact_tensor_data.restype = ctypes.c_void_p
        lib.row8_artifact_tensor_nbytes.argtypes = [ctypes.c_void_p]
        lib.row8_artifact_tensor_nbytes.restype = ctypes.c_ulonglong
        lib.row8_artifact_map_file.argtypes = [ctypes.c_char_p]
        lib.row8_artifact_map_file.restype = ctypes.c_void_p
        lib.row8_artifact_unmap_file.argtypes = [ctypes.c_void_p]
        lib.row8_artifact_unmap_file.restype = None
        lib.row8_artifact_mapped_data.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
        ]
        lib.row8_artifact_mapped_data.restype = ctypes.c_void_p
        lib.row8_artifact_mapped_nbytes.argtypes = [ctypes.c_void_p]
        lib.row8_artifact_mapped_nbytes.restype = ctypes.c_ulonglong
    except Exception as exc:  # pragma: no cover - defensive platform path
        _ROW8_ARTIFACT_ERROR = exc
        return None
    _ROW8_ARTIFACT_LIB = lib
    _ROW8_ARTIFACT_ERROR = None
    return lib


def native_row8_artifact_available() -> bool:
    return _load_row8_artifact_lib() is not None


def native_row8_artifact_error() -> Exception | None:
    _load_row8_artifact_lib()
    return _ROW8_ARTIFACT_ERROR


def _load_pcketlm_forward_lib() -> ctypes.CDLL | None:
    global _PCKETLM_FORWARD_LIB, _PCKETLM_FORWARD_ERROR
    if _monolithic_disabled():
        return None
    if _PCKETLM_FORWARD_LIB is not None:
        return _PCKETLM_FORWARD_LIB
    if not _PCKETLM_FORWARD_DLL.exists():
        _PCKETLM_FORWARD_ERROR = FileNotFoundError(str(_PCKETLM_FORWARD_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_PCKETLM_FORWARD_DLL))
        lib.pcketlm_cpu_has_avx2.argtypes = []
        lib.pcketlm_cpu_has_avx2.restype = ctypes.c_int
        lib.pcketlm_session_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.pcketlm_session_create.restype = ctypes.c_void_p
        lib.pcketlm_session_destroy.argtypes = [ctypes.c_void_p]
        lib.pcketlm_session_destroy.restype = None
        lib.pcketlm_session_committed_length.argtypes = [ctypes.c_void_p]
        lib.pcketlm_session_committed_length.restype = ctypes.c_longlong
        lib.pcketlm_session_tentative_length.argtypes = [ctypes.c_void_p]
        lib.pcketlm_session_tentative_length.restype = ctypes.c_longlong
        lib.pcketlm_layers_executed.argtypes = [ctypes.c_void_p]
        lib.pcketlm_layers_executed.restype = ctypes.c_longlong
        lib.pcketlm_session_kernels_ready.argtypes = [ctypes.c_void_p]
        lib.pcketlm_session_kernels_ready.restype = ctypes.c_int
        lib.pcketlm_session_kernel_error.argtypes = [ctypes.c_void_p]
        lib.pcketlm_session_kernel_error.restype = ctypes.c_char_p
        lib.pcketlm_session_call_count.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        lib.pcketlm_session_call_count.restype = ctypes.c_longlong
        lib.pcketlm_global_call_count.argtypes = [ctypes.c_longlong]
        lib.pcketlm_global_call_count.restype = ctypes.c_longlong
        lib.pcketlm_reset_global_call_counts.argtypes = []
        lib.pcketlm_reset_global_call_counts.restype = None
        lib.pcketlm_session_clear_tensors.argtypes = [ctypes.c_void_p]
        lib.pcketlm_session_clear_tensors.restype = ctypes.c_int
        lib.pcketlm_session_register_u16_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
        ]
        lib.pcketlm_session_register_u16_tensor.restype = ctypes.c_int
        lib.pcketlm_session_register_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]
        lib.pcketlm_session_register_tensor.restype = ctypes.c_int
        lib.pcketlm_session_tensor_count.argtypes = [ctypes.c_void_p]
        lib.pcketlm_session_tensor_count.restype = ctypes.c_longlong
        lib.pcketlm_session_tensor_nitems.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.pcketlm_session_tensor_nitems.restype = ctypes.c_longlong
        lib.pcketlm_session_tensor_data_ptr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.pcketlm_session_tensor_data_ptr.restype = ctypes.c_ulonglong
        lib.pcketlm_forward_prefill.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
        ]
        lib.pcketlm_forward_prefill.restype = ctypes.c_int
        lib.pcketlm_forward_decode.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p]
        lib.pcketlm_forward_decode.restype = ctypes.c_int
        lib.pcketlm_forward_verify.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
        ]
        lib.pcketlm_forward_verify.restype = ctypes.c_int
        lib.pcketlm_session_commit.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        lib.pcketlm_session_commit.restype = ctypes.c_int
        lib.pcketlm_session_rollback.argtypes = [ctypes.c_void_p]
        lib.pcketlm_session_rollback.restype = ctypes.c_int
    except Exception as exc:  # pragma: no cover - defensive platform path
        _PCKETLM_FORWARD_ERROR = exc
        return None
    _PCKETLM_FORWARD_LIB = lib
    _PCKETLM_FORWARD_ERROR = None
    return lib


def native_monolithic_available() -> bool:
    return _load_pcketlm_forward_lib() is not None


def native_monolithic_error() -> Exception | None:
    _load_pcketlm_forward_lib()
    return _PCKETLM_FORWARD_ERROR


def native_monolithic_has_avx2() -> bool:
    lib = _load_pcketlm_forward_lib()
    return bool(lib and lib.pcketlm_cpu_has_avx2())


def monolithic_call_count(call_type: str = "all") -> int:
    lib = _load_pcketlm_forward_lib()
    if lib is None:
        return 0
    mapping = {"prefill": 0, "decode": 1, "verify": 2, "all": 3}
    return int(lib.pcketlm_global_call_count(ctypes.c_longlong(mapping[call_type])))


def reset_monolithic_call_counts() -> None:
    lib = _load_pcketlm_forward_lib()
    if lib is not None:
        lib.pcketlm_reset_global_call_counts()


class MonolithicForwardSession:
    """Opaque ctypes session for the native monolithic forward boundary.

    This wrapper intentionally exposes only token/logit and commit/rollback
    operations; Python never reads or writes the native session internals.
    """

    def __init__(self, model_config: dict | str, weight_source: str = ""):
        lib = _load_pcketlm_forward_lib()
        if lib is None:
            reason = "disabled" if _monolithic_disabled() else _PCKETLM_FORWARD_ERROR
            raise RuntimeError(f"Native monolithic forward is unavailable: {reason}")
        if isinstance(model_config, str):
            config_text = model_config
            config = json.loads(model_config)
        else:
            config = dict(model_config)
            config_text = json.dumps(config, separators=(",", ":"))
        handle = lib.pcketlm_session_create(
            ctypes.c_char_p(config_text.encode("utf-8")),
            ctypes.c_char_p(str(weight_source).encode("utf-8")),
        )
        if not handle:
            raise RuntimeError("pcketlm_session_create returned a null handle")
        self._lib = lib
        self._handle = ctypes.c_void_p(handle)
        self.config = config
        self.vocab_size = int(config.get("vocab_size", 0) or 0)
        self._tensor_keepalive: dict[str, torch.Tensor] = {}
        if self.vocab_size <= 0:
            self.close()
            raise ValueError("model_config must include a positive vocab_size")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.pcketlm_session_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "MonolithicForwardSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def committed_length(self) -> int:
        return int(self._lib.pcketlm_session_committed_length(self._handle))

    def tentative_length(self) -> int:
        return int(self._lib.pcketlm_session_tentative_length(self._handle))

    def layers_executed(self) -> int:
        return int(self._lib.pcketlm_layers_executed(self._handle))

    def kernels_ready(self) -> bool:
        return bool(self._lib.pcketlm_session_kernels_ready(self._handle))

    def kernel_error(self) -> str:
        raw = self._lib.pcketlm_session_kernel_error(self._handle)
        return "" if not raw else raw.decode("utf-8", errors="replace")

    def call_count(self, call_type: str = "all") -> int:
        mapping = {"prefill": 0, "decode": 1, "verify": 2, "all": 3}
        return int(self._lib.pcketlm_session_call_count(self._handle, ctypes.c_longlong(mapping[call_type])))

    def clear_tensors(self) -> None:
        code = self._lib.pcketlm_session_clear_tensors(self._handle)
        if code != 0:
            raise RuntimeError(f"pcketlm_session_clear_tensors failed with code {code}")
        self._tensor_keepalive.clear()

    def register_u16_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.dtype not in {torch.float16, torch.bfloat16, torch.uint16}:
            raise TypeError("register_u16_tensor requires float16, bfloat16, or uint16 storage")
        cpu = tensor.detach().cpu().contiguous().view(torch.uint16).reshape(-1)
        self._tensor_keepalive[f"name:{name}"] = cpu
        code = self._lib.pcketlm_session_register_u16_tensor(
            self._handle,
            ctypes.c_char_p(str(name).encode("utf-8")),
            ctypes.c_void_p(int(cpu.data_ptr())),
            ctypes.c_longlong(int(cpu.numel())),
        )
        if code != 0:
            raise RuntimeError(f"pcketlm_session_register_u16_tensor failed with code {code}")

    def register_tensor(
        self,
        *,
        layer_idx: int,
        tensor_role: int,
        tensor: torch.Tensor,
        dtype_code: int | None = None,
    ) -> None:
        if tensor.dtype not in {torch.float16, torch.bfloat16, torch.uint16}:
            raise TypeError("register_tensor requires float16, bfloat16, or uint16 storage")
        cpu = tensor.detach().cpu().contiguous()
        if cpu.ndim == 0:
            rows, cols = 1, 1
        elif cpu.ndim == 1:
            rows, cols = int(cpu.shape[0]), 1
        else:
            rows, cols = int(cpu.shape[0]), int(cpu.reshape(cpu.shape[0], -1).shape[1])
        if dtype_code is None:
            dtype_code = 1 if tensor.dtype == torch.bfloat16 else 0
        storage = cpu.view(torch.uint16).reshape(-1)
        self._tensor_keepalive[f"role:{int(layer_idx)}:{int(tensor_role)}"] = storage
        code = self._lib.pcketlm_session_register_tensor(
            self._handle,
            ctypes.c_longlong(int(layer_idx)),
            ctypes.c_longlong(int(tensor_role)),
            ctypes.c_void_p(int(storage.data_ptr())),
            ctypes.c_longlong(rows),
            ctypes.c_longlong(cols),
            ctypes.c_longlong(int(dtype_code)),
        )
        if code != 0:
            raise RuntimeError(f"pcketlm_session_register_tensor failed with code {code}")

    def tensor_count(self) -> int:
        return int(self._lib.pcketlm_session_tensor_count(self._handle))

    def tensor_nitems(self, name: str) -> int:
        return int(
            self._lib.pcketlm_session_tensor_nitems(
                self._handle,
                ctypes.c_char_p(str(name).encode("utf-8")),
            )
        )

    def tensor_data_ptr(self, name: str) -> int:
        return int(
            self._lib.pcketlm_session_tensor_data_ptr(
                self._handle,
                ctypes.c_char_p(str(name).encode("utf-8")),
            )
        )

    def prefill(self, token_ids: torch.Tensor | list[int]) -> torch.Tensor:
        tokens = torch.as_tensor(token_ids, dtype=torch.int64).detach().cpu().contiguous().reshape(-1)
        out = torch.empty((self.vocab_size,), dtype=torch.float16)
        code = self._lib.pcketlm_forward_prefill(
            self._handle,
            ctypes.c_void_p(int(tokens.data_ptr())),
            ctypes.c_longlong(int(tokens.numel())),
            ctypes.c_void_p(int(out.data_ptr())),
        )
        if code != 0:
            raise RuntimeError(f"pcketlm_forward_prefill failed with code {code}")
        return out

    def decode(self, token_id: int) -> torch.Tensor:
        out = torch.empty((self.vocab_size,), dtype=torch.float16)
        code = self._lib.pcketlm_forward_decode(
            self._handle,
            ctypes.c_longlong(int(token_id)),
            ctypes.c_void_p(int(out.data_ptr())),
        )
        if code != 0:
            raise RuntimeError(f"pcketlm_forward_decode failed with code {code}")
        return out

    def verify(self, candidate_token_ids: torch.Tensor | list[int]) -> torch.Tensor:
        tokens = torch.as_tensor(candidate_token_ids, dtype=torch.int64).detach().cpu().contiguous().reshape(-1)
        out = torch.empty((int(tokens.numel()) + 1, self.vocab_size), dtype=torch.float16)
        code = self._lib.pcketlm_forward_verify(
            self._handle,
            ctypes.c_void_p(int(tokens.data_ptr())),
            ctypes.c_longlong(int(tokens.numel())),
            ctypes.c_void_p(int(out.data_ptr())),
        )
        if code != 0:
            raise RuntimeError(f"pcketlm_forward_verify failed with code {code}")
        return out

    def commit(self, count: int) -> None:
        code = self._lib.pcketlm_session_commit(self._handle, ctypes.c_longlong(int(count)))
        if code != 0:
            raise RuntimeError(f"pcketlm_session_commit failed with code {code}")

    def rollback(self) -> None:
        code = self._lib.pcketlm_session_rollback(self._handle)
        if code != 0:
            raise RuntimeError(f"pcketlm_session_rollback failed with code {code}")


class NativeRow8Tensor:
    def __init__(self, path: str | Path, offset: int, nbytes: int):
        lib = _load_row8_artifact_lib()
        if lib is None:
            reason = "disabled" if _native_row8_artifact_disabled() else _ROW8_ARTIFACT_ERROR
            raise RuntimeError(f"Native row8 artifact cache is unavailable: {reason}")
        path_bytes = str(Path(path)).encode("utf-8")
        handle = lib.row8_artifact_load_tensor(
            ctypes.c_char_p(path_bytes),
            ctypes.c_ulonglong(int(offset)),
            ctypes.c_ulonglong(int(nbytes)),
        )
        if not handle:
            raise OSError(f"row8_artifact_load_tensor failed for {path}")
        self._lib = lib
        self._handle = ctypes.c_void_p(handle)
        self.nbytes = int(lib.row8_artifact_tensor_nbytes(self._handle))
        ptr = lib.row8_artifact_tensor_data(self._handle)
        if not ptr or self.nbytes != int(nbytes):
            self.close()
            raise OSError(f"row8_artifact_tensor_data failed for {path}")
        self.ptr = int(ptr)
        self.path = str(path)
        self.offset = int(offset)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.row8_artifact_free_tensor(self._handle)
            self._handle = None
            self.ptr = 0

    def __del__(self):  # pragma: no cover - GC safety net
        try:
            self.close()
        except Exception:
            pass


def load_native_row8_tensor(path: str | Path, offset: int, nbytes: int) -> NativeRow8Tensor:
    return NativeRow8Tensor(path, offset, nbytes)


class NativeRow8MappedFile:
    def __init__(self, path: str | Path):
        lib = _load_row8_artifact_lib()
        if lib is None:
            reason = "disabled" if _native_row8_artifact_disabled() else _ROW8_ARTIFACT_ERROR
            raise RuntimeError(f"Native row8 artifact cache is unavailable: {reason}")
        path_bytes = str(Path(path)).encode("utf-8")
        handle = lib.row8_artifact_map_file(ctypes.c_char_p(path_bytes))
        if not handle:
            raise OSError(f"row8_artifact_map_file failed for {path}")
        self._lib = lib
        self._handle = ctypes.c_void_p(handle)
        self.nbytes = int(lib.row8_artifact_mapped_nbytes(self._handle))
        self.path = str(path)

    def data_ptr(self, offset: int, nbytes: int) -> int:
        ptr = self._lib.row8_artifact_mapped_data(
            self._handle,
            ctypes.c_ulonglong(int(offset)),
            ctypes.c_ulonglong(int(nbytes)),
        )
        if not ptr:
            raise ValueError("row8 mapped tensor range is invalid")
        return int(ptr)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.row8_artifact_unmap_file(self._handle)
            self._handle = None

    def __del__(self):  # pragma: no cover - GC safety net
        try:
            self.close()
        except Exception:
            pass


class NativeRow8MappedTensor:
    def __init__(self, mapped_file: NativeRow8MappedFile, offset: int, nbytes: int):
        self._mapped_file = mapped_file
        self.offset = int(offset)
        self.nbytes = int(nbytes)
        self.ptr = mapped_file.data_ptr(offset, nbytes)


def map_native_row8_file(path: str | Path) -> NativeRow8MappedFile:
    return NativeRow8MappedFile(path)


def _packed_gemv_cache_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_PACKED_GEMV_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}


def _packed_gemv_cache_budget_bytes() -> int:
    raw = os.environ.get("PCKETLM_PACKED_GEMV_CACHE_MB", "").strip()
    if not raw:
        return 0
    try:
        mb = int(raw)
    except ValueError:
        return 0
    return max(0, mb) * 1024 * 1024


def reset_packed_gemv_cache() -> None:
    global _PACKED_GEMV_CACHE_BYTES
    _PACKED_GEMV_CACHE.clear()
    _PACKED_GEMV_CACHE_BYTES = 0
    for key in _PACKED_GEMV_CACHE_STATS:
        _PACKED_GEMV_CACHE_STATS[key] = 0


def packed_gemv_cache_stats() -> dict[str, int | float]:
    return {
        **_PACKED_GEMV_CACHE_STATS,
        "resident_count": len(_PACKED_GEMV_CACHE),
        "resident_bytes": _PACKED_GEMV_CACHE_BYTES,
        "resident_mb": round(_PACKED_GEMV_CACHE_BYTES / 1024 / 1024, 4),
        "budget_bytes": _packed_gemv_cache_budget_bytes(),
    }


def pack_weight_rows8(weight: torch.Tensor) -> torch.Tensor:
    lib = _load_fp16_packed_gemv_lib()
    if lib is None:
        reason = "disabled" if _native_packed_gemv_disabled() else _FP16_PACKED_GEMV_ERROR
        raise RuntimeError(f"Native packed GEMV is unavailable: {reason}")
    if weight.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("pack_weight_rows8 requires torch.float16 or torch.bfloat16 weight")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [rows, cols]")
    weight_cpu = weight.detach().cpu().contiguous()
    rows = int(weight_cpu.shape[0])
    cols = int(weight_cpu.shape[1])
    row_blocks = (rows + 7) // 8
    packed_u16 = torch.empty((row_blocks * cols * 8,), dtype=torch.uint16)
    code = lib.native_pack_u16_rows8(
        ctypes.c_void_p(int(weight_cpu.view(torch.uint16).data_ptr())),
        ctypes.c_void_p(int(packed_u16.data_ptr())),
        ctypes.c_longlong(rows),
        ctypes.c_longlong(cols),
    )
    if code != 0:
        raise RuntimeError(f"native_pack_u16_rows8 failed with code {code}")
    return packed_u16


def cached_pack_weight_rows8(cache_key: str, weight: torch.Tensor) -> torch.Tensor:
    global _PACKED_GEMV_CACHE_BYTES
    if _packed_gemv_cache_disabled():
        _PACKED_GEMV_CACHE_STATS["misses"] += 1
        return pack_weight_rows8(weight)
    budget = _packed_gemv_cache_budget_bytes()
    if budget <= 0:
        _PACKED_GEMV_CACHE_STATS["misses"] += 1
        return pack_weight_rows8(weight)
    key = str(cache_key)
    cached = _PACKED_GEMV_CACHE.get(key)
    if cached is not None:
        _PACKED_GEMV_CACHE.move_to_end(key)
        _PACKED_GEMV_CACHE_STATS["hits"] += 1
        return cached
    _PACKED_GEMV_CACHE_STATS["misses"] += 1
    packed = pack_weight_rows8(weight)
    packed_bytes = int(packed.numel() * packed.element_size())
    if packed_bytes > budget:
        return packed
    while _PACKED_GEMV_CACHE and _PACKED_GEMV_CACHE_BYTES + packed_bytes > budget:
        _old_key, old_value = _PACKED_GEMV_CACHE.popitem(last=False)
        _PACKED_GEMV_CACHE_BYTES -= int(old_value.numel() * old_value.element_size())
        _PACKED_GEMV_CACHE_STATS["evictions"] += 1
    _PACKED_GEMV_CACHE[key] = packed
    _PACKED_GEMV_CACHE_BYTES += packed_bytes
    _PACKED_GEMV_CACHE_STATS["stores"] += 1
    return packed


def packed_gemv_rows8(hidden: torch.Tensor, packed_weight: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    lib = _load_fp16_packed_gemv_lib()
    if lib is None:
        reason = "disabled" if _native_packed_gemv_disabled() else _FP16_PACKED_GEMV_ERROR
        raise RuntimeError(f"Native packed GEMV is unavailable: {reason}")
    if hidden.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("packed_gemv_rows8 requires torch.float16 or torch.bfloat16 hidden")
    if packed_weight.dtype != torch.uint16:
        raise TypeError("packed_weight must be a torch.uint16 tensor from pack_weight_rows8")
    hidden_cpu = hidden.detach().cpu().contiguous().reshape(-1)
    if hidden_cpu.numel() != int(cols):
        raise ValueError("hidden length does not match packed weight column count")
    packed_cpu = packed_weight.detach().cpu().contiguous()
    expected_count = ((int(rows) + 7) // 8) * int(cols) * 8
    if packed_cpu.numel() != expected_count:
        raise ValueError("packed weight length does not match rows/cols")
    out = torch.empty((int(rows),), dtype=torch.float32)
    code = lib.native_packed_gemv_rows8(
        ctypes.c_void_p(int(hidden_cpu.view(torch.uint16).data_ptr())),
        ctypes.c_void_p(int(packed_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(int(rows)),
        ctypes.c_longlong(int(cols)),
        ctypes.c_int(_u16_storage_dtype_code(hidden_cpu.dtype)),
    )
    if code != 0:
        raise RuntimeError(f"native_packed_gemv_rows8 failed with code {code}")
    return out


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


def lm_head_topk_u16(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    top_k: int,
    token_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _load_fp16_matmul_lib()
    if lib is None or not hasattr(lib, "native_lm_head_topk_u16"):
        reason = "disabled" if _native_matmul_disabled() else _FP16_MATMUL_ERROR
        raise RuntimeError(f"Native lm_head top-k is unavailable: {reason}")
    if hidden.dtype != weight.dtype:
        raise TypeError("lm_head_topk_u16 requires hidden and weight to share dtype")
    dtype_code = _u16_storage_dtype_code(hidden.dtype)
    hidden_cpu = hidden.detach().cpu().contiguous().reshape(-1)
    weight_cpu = weight.detach().cpu().contiguous()
    if weight_cpu.ndim != 2:
        raise ValueError("weight must have shape [rows, hidden_size]")
    rows = int(weight_cpu.shape[0])
    hidden_size = int(weight_cpu.shape[1])
    if hidden_cpu.numel() != hidden_size:
        raise ValueError("hidden size does not match lm_head weight")
    ids = torch.empty((int(top_k),), dtype=torch.int64)
    logits = torch.empty((int(top_k),), dtype=torch.float32)
    code = lib.native_lm_head_topk_u16(
        ctypes.c_void_p(int(hidden_cpu.data_ptr())),
        ctypes.c_void_p(int(weight_cpu.data_ptr())),
        ctypes.c_void_p(int(ids.data_ptr())),
        ctypes.c_void_p(int(logits.data_ptr())),
        ctypes.c_longlong(rows),
        ctypes.c_longlong(hidden_size),
        ctypes.c_longlong(int(top_k)),
        ctypes.c_int(dtype_code),
        ctypes.c_longlong(int(token_offset)),
    )
    if code != 0:
        raise RuntimeError(f"native_lm_head_topk_u16 failed with code {code}")
    order = torch.argsort(logits, descending=True)
    return logits[order], ids[order]


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
        lib.native_moe_selected_forward_u16.argtypes = [
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
            ctypes.c_int,
        ]
        lib.native_moe_selected_forward_u16.restype = ctypes.c_int
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


def moe_selected_forward_u16(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    lib = _load_fp16_moe_lib()
    if lib is None:
        reason = "disabled" if _native_moe_disabled() else _FP16_MOE_ERROR
        raise RuntimeError(f"Native fp16 MoE is unavailable: {reason}")
    dtype_code = _u16_storage_dtype_code(hidden.dtype)
    tensors = [hidden, gate_weight, up_weight, down_weight]
    if any(tensor.dtype != hidden.dtype for tensor in tensors):
        raise TypeError(f"moe_selected_forward_u16 requires all model tensors to share dtype {hidden.dtype}")
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [seq_len, hidden_size]")
    if gate_weight.ndim != 3 or up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("selected expert weights must be rank-3")
    seq_len = int(hidden.shape[0])
    hidden_size = int(hidden.shape[1])
    selected_count = int(gate_weight.shape[0])
    intermediate_size = int(gate_weight.shape[1])
    if route_weights.shape != (seq_len, selected_count):
        raise ValueError("route_weights must have shape [seq_len, selected_count]")
    hidden_cpu = hidden.detach().cpu().contiguous()
    gate_cpu = gate_weight.detach().cpu().contiguous()
    up_cpu = up_weight.detach().cpu().contiguous()
    down_cpu = down_weight.detach().cpu().contiguous()
    route_cpu = route_weights.detach().cpu().contiguous().to(torch.float32)
    out = torch.empty((seq_len, hidden_size), dtype=hidden.dtype)
    code = lib.native_moe_selected_forward_u16(
        ctypes.c_void_p(int(hidden_cpu.data_ptr())),
        ctypes.c_void_p(int(gate_cpu.data_ptr())),
        ctypes.c_void_p(int(up_cpu.data_ptr())),
        ctypes.c_void_p(int(down_cpu.data_ptr())),
        ctypes.c_void_p(int(route_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(seq_len),
        ctypes.c_longlong(hidden_size),
        ctypes.c_longlong(selected_count),
        ctypes.c_longlong(intermediate_size),
        ctypes.c_int(dtype_code),
    )
    if code != 0:
        raise RuntimeError(f"native_moe_selected_forward_u16 failed with code {code}")
    return out


def _load_fp16_kv_lib() -> ctypes.CDLL | None:
    global _FP16_KV_LIB, _FP16_KV_ERROR
    if _native_kv_disabled():
        return None
    if _FP16_KV_LIB is not None:
        return _FP16_KV_LIB
    if not _FP16_KV_DLL.exists():
        _FP16_KV_ERROR = FileNotFoundError(str(_FP16_KV_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_FP16_KV_DLL))
        lib.kv_prefill_init.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong]
        lib.kv_prefill_init.restype = ctypes.c_void_p
        lib.kv_prefill_init_typed.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
        lib.kv_prefill_init_typed.restype = ctypes.c_void_p
        lib.kv_free.argtypes = [ctypes.c_void_p]
        lib.kv_free.restype = None
        for name in ("kv_append_committed", "kv_append_tentative"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
            fn.restype = ctypes.c_int
        lib.kv_commit.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        lib.kv_commit.restype = ctypes.c_int
        lib.kv_rollback.argtypes = [ctypes.c_void_p]
        lib.kv_rollback.restype = ctypes.c_int
        lib.kv_committed_length.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        lib.kv_committed_length.restype = ctypes.c_longlong
        lib.kv_tentative_length.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        lib.kv_tentative_length.restype = ctypes.c_longlong
        lib.kv_copy_layer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_int,
        ]
        lib.kv_copy_layer.restype = ctypes.c_int
        lib.kv_attention_decode_fp16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_float,
        ]
        lib.kv_attention_decode_fp16.restype = ctypes.c_int
        lib.kv_attention_decode_u16_ext.argtypes = [
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
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
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.kv_attention_decode_u16_ext.restype = ctypes.c_int
        if hasattr(lib, "kv_attention_decode_u16_ext_hd"):
            lib.kv_attention_decode_u16_ext_hd.argtypes = [
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
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
                ctypes.c_float,
                ctypes.c_float,
            ]
            lib.kv_attention_decode_u16_ext_hd.restype = ctypes.c_int
        lib.kv_dense_layer_decode_fp16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
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
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.kv_dense_layer_decode_fp16.restype = ctypes.c_int
        if hasattr(lib, "kv_dense_layer_prefill_fp16"):
            lib.kv_dense_layer_prefill_fp16.argtypes = [
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
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
                ctypes.c_float,
                ctypes.c_float,
            ]
            lib.kv_dense_layer_prefill_fp16.restype = ctypes.c_int
        lib.kv_dense_layer_decode_u16_ext.argtypes = [
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
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
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.kv_dense_layer_decode_u16_ext.restype = ctypes.c_int
        if hasattr(lib, "kv_dense_layer_decode_u16_ext_packed_rows8"):
            lib.kv_dense_layer_decode_u16_ext_packed_rows8.argtypes = lib.kv_dense_layer_decode_u16_ext.argtypes
            lib.kv_dense_layer_decode_u16_ext_packed_rows8.restype = ctypes.c_int
        if hasattr(lib, "kv_dense_layer_prefill_u16_ext"):
            lib.kv_dense_layer_prefill_u16_ext.argtypes = [
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
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
                ctypes.c_float,
                ctypes.c_float,
            ]
            lib.kv_dense_layer_prefill_u16_ext.restype = ctypes.c_int
    except Exception as exc:  # pragma: no cover - defensive platform path
        _FP16_KV_ERROR = exc
        return None
    _FP16_KV_LIB = lib
    _FP16_KV_ERROR = None
    return lib


def native_fp16_kv_available() -> bool:
    return _load_fp16_kv_lib() is not None


def native_fp16_kv_error() -> Exception | None:
    _load_fp16_kv_lib()
    return _FP16_KV_ERROR


class NativeKvSession:
    def __init__(self, layer_count: int, max_seq_len: int, kv_width: int, dtype: torch.dtype = torch.float16):
        lib = _load_fp16_kv_lib()
        if lib is None:
            reason = "disabled" if _native_kv_disabled() else _FP16_KV_ERROR
            raise RuntimeError(f"Native fp16 KV cache is unavailable: {reason}")
        dtype_code = _u16_storage_dtype_code(dtype)
        handle = lib.kv_prefill_init_typed(
            ctypes.c_longlong(int(layer_count)),
            ctypes.c_longlong(int(max_seq_len)),
            ctypes.c_longlong(int(kv_width)),
            ctypes.c_int(dtype_code),
        )
        if not handle:
            raise RuntimeError("kv_prefill_init failed")
        self._lib = lib
        self._handle = ctypes.c_void_p(handle)
        self.layer_count = int(layer_count)
        self.max_seq_len = int(max_seq_len)
        self.kv_width = int(kv_width)
        self.dtype = dtype

    @property
    def handle(self) -> int:
        if not self._handle:
            raise RuntimeError("NativeKvSession is closed")
        return int(self._handle.value)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.kv_free(self._handle)
            self._handle = None

    def __del__(self):  # pragma: no cover - GC safety net
        try:
            self.close()
        except Exception:
            pass

    def _append(self, fn_name: str, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, count: int) -> None:
        if k_new.dtype != self.dtype or v_new.dtype != self.dtype:
            raise TypeError(f"KV cache append requires {self.dtype} tensors")
        k_cpu = k_new.detach().cpu().contiguous().reshape(-1)
        v_cpu = v_new.detach().cpu().contiguous().reshape(-1)
        expected = int(count) * self.kv_width
        if k_cpu.numel() != expected or v_cpu.numel() != expected:
            raise ValueError("KV tensor size does not match count * kv_width")
        code = getattr(self._lib, fn_name)(
            self._handle,
            ctypes.c_longlong(int(layer)),
            ctypes.c_void_p(int(k_cpu.data_ptr())),
            ctypes.c_void_p(int(v_cpu.data_ptr())),
            ctypes.c_longlong(int(count)),
        )
        if code != 0:
            raise RuntimeError(f"{fn_name} failed with code {code}")

    def append_committed(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, count: int) -> None:
        self._append("kv_append_committed", layer, k_new, v_new, count)

    def append_tentative(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, count: int) -> None:
        self._append("kv_append_tentative", layer, k_new, v_new, count)

    def commit(self, count: int) -> None:
        code = self._lib.kv_commit(self._handle, ctypes.c_longlong(int(count)))
        if code != 0:
            raise RuntimeError(f"kv_commit failed with code {code}")

    def rollback(self) -> None:
        code = self._lib.kv_rollback(self._handle)
        if code != 0:
            raise RuntimeError(f"kv_rollback failed with code {code}")

    def committed_length(self, layer: int) -> int:
        return int(self._lib.kv_committed_length(self._handle, ctypes.c_longlong(int(layer))))

    def tentative_length(self, layer: int) -> int:
        return int(self._lib.kv_tentative_length(self._handle, ctypes.c_longlong(int(layer))))

    def copy_layer(self, layer: int, *, include_tentative: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        count = self.committed_length(layer) + (self.tentative_length(layer) if include_tentative else 0)
        k_out = torch.empty((count, self.kv_width), dtype=self.dtype)
        v_out = torch.empty((count, self.kv_width), dtype=self.dtype)
        code = self._lib.kv_copy_layer(
            self._handle,
            ctypes.c_longlong(int(layer)),
            ctypes.c_void_p(int(k_out.data_ptr())),
            ctypes.c_void_p(int(v_out.data_ptr())),
            ctypes.c_longlong(int(count)),
            ctypes.c_int(1 if include_tentative else 0),
        )
        if code != 0:
            raise RuntimeError(f"kv_copy_layer failed with code {code}")
        return k_out, v_out

    def attention_decode_fp16(
        self,
        layer: int,
        hidden: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_theta: float,
        head_dim: int | None = None,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        q_norm_weight: torch.Tensor | None = None,
        k_norm_weight: torch.Tensor | None = None,
        rms_eps: float = 1e-6,
    ) -> torch.Tensor:
        tensors = [hidden, q_weight, k_weight, v_weight, o_weight]
        for optional in (q_bias, k_bias, v_bias, q_norm_weight, k_norm_weight):
            if optional is not None:
                tensors.append(optional)
        if any(tensor.dtype != self.dtype for tensor in tensors):
            raise TypeError(f"attention_decode_fp16 requires {self.dtype} tensors")
        hidden_cpu = hidden.detach().cpu().contiguous().reshape(-1)
        q_cpu = q_weight.detach().cpu().contiguous()
        k_cpu = k_weight.detach().cpu().contiguous()
        v_cpu = v_weight.detach().cpu().contiguous()
        o_cpu = o_weight.detach().cpu().contiguous()
        q_bias_cpu = None if q_bias is None else q_bias.detach().cpu().contiguous().reshape(-1)
        k_bias_cpu = None if k_bias is None else k_bias.detach().cpu().contiguous().reshape(-1)
        v_bias_cpu = None if v_bias is None else v_bias.detach().cpu().contiguous().reshape(-1)
        q_norm_cpu = None if q_norm_weight is None else q_norm_weight.detach().cpu().contiguous().reshape(-1)
        k_norm_cpu = None if k_norm_weight is None else k_norm_weight.detach().cpu().contiguous().reshape(-1)
        hidden_size = int(hidden_cpu.numel())
        out = torch.empty((hidden_size,), dtype=self.dtype)
        explicit_head_dim = int(head_dim or 0)
        fn = self._lib.kv_attention_decode_u16_ext
        args = [
            self._handle,
            ctypes.c_longlong(int(layer)),
            ctypes.c_void_p(int(hidden_cpu.data_ptr())),
            ctypes.c_void_p(int(q_cpu.data_ptr())),
            ctypes.c_void_p(int(k_cpu.data_ptr())),
            ctypes.c_void_p(int(v_cpu.data_ptr())),
            ctypes.c_void_p(int(o_cpu.data_ptr())),
            ctypes.c_void_p(0 if q_bias_cpu is None else int(q_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if k_bias_cpu is None else int(k_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if v_bias_cpu is None else int(v_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if q_norm_cpu is None else int(q_norm_cpu.data_ptr())),
            ctypes.c_void_p(0 if k_norm_cpu is None else int(k_norm_cpu.data_ptr())),
            ctypes.c_void_p(int(out.data_ptr())),
            ctypes.c_longlong(hidden_size),
            ctypes.c_longlong(int(num_attention_heads)),
            ctypes.c_longlong(int(num_key_value_heads)),
        ]
        if explicit_head_dim > 0 and hasattr(self._lib, "kv_attention_decode_u16_ext_hd"):
            fn = self._lib.kv_attention_decode_u16_ext_hd
            args.append(ctypes.c_longlong(explicit_head_dim))
        args.extend([
            ctypes.c_float(float(rope_theta)),
            ctypes.c_float(float(rms_eps)),
        ])
        code = fn(*args)
        if code != 0:
            raise RuntimeError(f"kv_attention_decode_fp16 failed with code {code}")
        return out

    def dense_layer_decode_fp16(
        self,
        layer: int,
        hidden: torch.Tensor,
        input_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        *,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_eps: float,
        rope_theta: float,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        q_norm_weight: torch.Tensor | None = None,
        k_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tensors = [
            hidden,
            input_norm_weight,
            post_norm_weight,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            gate_weight,
            up_weight,
            down_weight,
        ]
        for optional in (q_bias, k_bias, v_bias, q_norm_weight, k_norm_weight):
            if optional is not None:
                tensors.append(optional)
        if any(tensor.dtype != self.dtype for tensor in tensors):
            raise TypeError(f"dense_layer_decode_fp16 requires {self.dtype} tensors")
        cpu_tensors = [tensor.detach().cpu().contiguous() for tensor in tensors]
        q_bias_cpu = None if q_bias is None else q_bias.detach().cpu().contiguous().reshape(-1)
        k_bias_cpu = None if k_bias is None else k_bias.detach().cpu().contiguous().reshape(-1)
        v_bias_cpu = None if v_bias is None else v_bias.detach().cpu().contiguous().reshape(-1)
        q_norm_cpu = None if q_norm_weight is None else q_norm_weight.detach().cpu().contiguous().reshape(-1)
        k_norm_cpu = None if k_norm_weight is None else k_norm_weight.detach().cpu().contiguous().reshape(-1)
        hidden_size = int(cpu_tensors[0].numel())
        out = torch.empty((hidden_size,), dtype=self.dtype)
        code = self._lib.kv_dense_layer_decode_u16_ext(
            self._handle,
            ctypes.c_longlong(int(layer)),
            ctypes.c_void_p(int(cpu_tensors[0].reshape(-1).data_ptr())),
            ctypes.c_void_p(int(cpu_tensors[1].reshape(-1).data_ptr())),
            ctypes.c_void_p(int(cpu_tensors[2].reshape(-1).data_ptr())),
            ctypes.c_void_p(int(cpu_tensors[3].data_ptr())),
            ctypes.c_void_p(int(cpu_tensors[4].data_ptr())),
            ctypes.c_void_p(int(cpu_tensors[5].data_ptr())),
            ctypes.c_void_p(int(cpu_tensors[6].data_ptr())),
            ctypes.c_void_p(int(cpu_tensors[7].data_ptr())),
            ctypes.c_void_p(int(cpu_tensors[8].data_ptr())),
            ctypes.c_void_p(int(cpu_tensors[9].data_ptr())),
            ctypes.c_void_p(0 if q_bias_cpu is None else int(q_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if k_bias_cpu is None else int(k_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if v_bias_cpu is None else int(v_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if q_norm_cpu is None else int(q_norm_cpu.data_ptr())),
            ctypes.c_void_p(0 if k_norm_cpu is None else int(k_norm_cpu.data_ptr())),
            ctypes.c_void_p(int(out.data_ptr())),
            ctypes.c_longlong(hidden_size),
            ctypes.c_longlong(int(intermediate_size)),
            ctypes.c_longlong(int(num_attention_heads)),
            ctypes.c_longlong(int(num_key_value_heads)),
            ctypes.c_float(float(rms_eps)),
            ctypes.c_float(float(rope_theta)),
        )
        if code != 0:
            raise RuntimeError(f"kv_dense_layer_decode_fp16 failed with code {code}")
        return out

    def dense_layer_decode_packed_rows8(
        self,
        layer: int,
        hidden: torch.Tensor,
        input_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        q_weight_packed: torch.Tensor,
        k_weight_packed: torch.Tensor,
        v_weight_packed: torch.Tensor,
        o_weight_packed: torch.Tensor,
        gate_weight_packed: torch.Tensor,
        up_weight_packed: torch.Tensor,
        down_weight_packed: torch.Tensor,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_eps: float,
        rope_theta: float,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        q_norm_weight: torch.Tensor | None = None,
        k_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not hasattr(self._lib, "kv_dense_layer_decode_u16_ext_packed_rows8"):
            raise RuntimeError("kv_dense_layer_decode_u16_ext_packed_rows8 is unavailable")
        raw_tensors = [hidden, input_norm_weight, post_norm_weight]
        if any(tensor.dtype != self.dtype for tensor in raw_tensors):
            raise TypeError(f"dense_layer_decode_packed_rows8 requires {self.dtype} raw tensors")
        packed_tensors = [
            q_weight_packed,
            k_weight_packed,
            v_weight_packed,
            o_weight_packed,
            gate_weight_packed,
            up_weight_packed,
            down_weight_packed,
        ]
        if any(tensor.dtype != torch.uint16 for tensor in packed_tensors):
            raise TypeError("packed dense layer weights must be torch.uint16")
        raw_cpu = [tensor.detach().cpu().contiguous() for tensor in raw_tensors]
        packed_cpu = [tensor.detach().cpu().contiguous() for tensor in packed_tensors]
        q_bias_cpu = None if q_bias is None else q_bias.detach().cpu().contiguous().reshape(-1)
        k_bias_cpu = None if k_bias is None else k_bias.detach().cpu().contiguous().reshape(-1)
        v_bias_cpu = None if v_bias is None else v_bias.detach().cpu().contiguous().reshape(-1)
        q_norm_cpu = None if q_norm_weight is None else q_norm_weight.detach().cpu().contiguous().reshape(-1)
        k_norm_cpu = None if k_norm_weight is None else k_norm_weight.detach().cpu().contiguous().reshape(-1)
        out = torch.empty((int(hidden_size),), dtype=self.dtype)
        code = self._lib.kv_dense_layer_decode_u16_ext_packed_rows8(
            self._handle,
            ctypes.c_longlong(int(layer)),
            ctypes.c_void_p(int(raw_cpu[0].reshape(-1).data_ptr())),
            ctypes.c_void_p(int(raw_cpu[1].reshape(-1).data_ptr())),
            ctypes.c_void_p(int(raw_cpu[2].reshape(-1).data_ptr())),
            ctypes.c_void_p(int(packed_cpu[0].data_ptr())),
            ctypes.c_void_p(int(packed_cpu[1].data_ptr())),
            ctypes.c_void_p(int(packed_cpu[2].data_ptr())),
            ctypes.c_void_p(int(packed_cpu[3].data_ptr())),
            ctypes.c_void_p(int(packed_cpu[4].data_ptr())),
            ctypes.c_void_p(int(packed_cpu[5].data_ptr())),
            ctypes.c_void_p(int(packed_cpu[6].data_ptr())),
            ctypes.c_void_p(0 if q_bias_cpu is None else int(q_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if k_bias_cpu is None else int(k_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if v_bias_cpu is None else int(v_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if q_norm_cpu is None else int(q_norm_cpu.data_ptr())),
            ctypes.c_void_p(0 if k_norm_cpu is None else int(k_norm_cpu.data_ptr())),
            ctypes.c_void_p(int(out.data_ptr())),
            ctypes.c_longlong(int(hidden_size)),
            ctypes.c_longlong(int(intermediate_size)),
            ctypes.c_longlong(int(num_heads)),
            ctypes.c_longlong(int(num_kv_heads)),
            ctypes.c_float(float(rms_eps)),
            ctypes.c_float(float(rope_theta)),
        )
        if code != 0:
            raise RuntimeError(f"kv_dense_layer_decode_u16_ext_packed_rows8 failed with code {code}")
        return out

    def dense_layer_decode_packed_rows8_ptrs(
        self,
        layer: int,
        hidden: torch.Tensor,
        input_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        q_weight_packed_ptr: int,
        k_weight_packed_ptr: int,
        v_weight_packed_ptr: int,
        o_weight_packed_ptr: int,
        gate_weight_packed_ptr: int,
        up_weight_packed_ptr: int,
        down_weight_packed_ptr: int,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_eps: float,
        rope_theta: float,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        q_norm_weight: torch.Tensor | None = None,
        k_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del head_dim
        if not hasattr(self._lib, "kv_dense_layer_decode_u16_ext_packed_rows8"):
            raise RuntimeError("kv_dense_layer_decode_u16_ext_packed_rows8 is unavailable")
        raw_tensors = [hidden, input_norm_weight, post_norm_weight]
        if any(tensor.dtype != self.dtype for tensor in raw_tensors):
            raise TypeError(f"dense_layer_decode_packed_rows8_ptrs requires {self.dtype} raw tensors")
        ptrs = [
            q_weight_packed_ptr,
            k_weight_packed_ptr,
            v_weight_packed_ptr,
            o_weight_packed_ptr,
            gate_weight_packed_ptr,
            up_weight_packed_ptr,
            down_weight_packed_ptr,
        ]
        if any(int(ptr) == 0 for ptr in ptrs):
            raise ValueError("packed row8 pointer must be non-zero")
        raw_cpu = [tensor.detach().cpu().contiguous() for tensor in raw_tensors]
        q_bias_cpu = None if q_bias is None else q_bias.detach().cpu().contiguous().reshape(-1)
        k_bias_cpu = None if k_bias is None else k_bias.detach().cpu().contiguous().reshape(-1)
        v_bias_cpu = None if v_bias is None else v_bias.detach().cpu().contiguous().reshape(-1)
        q_norm_cpu = None if q_norm_weight is None else q_norm_weight.detach().cpu().contiguous().reshape(-1)
        k_norm_cpu = None if k_norm_weight is None else k_norm_weight.detach().cpu().contiguous().reshape(-1)
        out = torch.empty((int(hidden_size),), dtype=self.dtype)
        code = self._lib.kv_dense_layer_decode_u16_ext_packed_rows8(
            self._handle,
            ctypes.c_longlong(int(layer)),
            ctypes.c_void_p(int(raw_cpu[0].reshape(-1).data_ptr())),
            ctypes.c_void_p(int(raw_cpu[1].reshape(-1).data_ptr())),
            ctypes.c_void_p(int(raw_cpu[2].reshape(-1).data_ptr())),
            ctypes.c_void_p(int(q_weight_packed_ptr)),
            ctypes.c_void_p(int(k_weight_packed_ptr)),
            ctypes.c_void_p(int(v_weight_packed_ptr)),
            ctypes.c_void_p(int(o_weight_packed_ptr)),
            ctypes.c_void_p(int(gate_weight_packed_ptr)),
            ctypes.c_void_p(int(up_weight_packed_ptr)),
            ctypes.c_void_p(int(down_weight_packed_ptr)),
            ctypes.c_void_p(0 if q_bias_cpu is None else int(q_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if k_bias_cpu is None else int(k_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if v_bias_cpu is None else int(v_bias_cpu.data_ptr())),
            ctypes.c_void_p(0 if q_norm_cpu is None else int(q_norm_cpu.data_ptr())),
            ctypes.c_void_p(0 if k_norm_cpu is None else int(k_norm_cpu.data_ptr())),
            ctypes.c_void_p(int(out.data_ptr())),
            ctypes.c_longlong(int(hidden_size)),
            ctypes.c_longlong(int(intermediate_size)),
            ctypes.c_longlong(int(num_heads)),
            ctypes.c_longlong(int(num_kv_heads)),
            ctypes.c_float(float(rms_eps)),
            ctypes.c_float(float(rope_theta)),
        )
        if code != 0:
            raise RuntimeError(f"kv_dense_layer_decode_u16_ext_packed_rows8 failed with code {code}")
        return out

    def dense_layer_prefill_fp16(
        self,
        layer: int,
        hidden: torch.Tensor,
        input_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        *,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_eps: float,
        rope_theta: float,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        q_norm_weight: torch.Tensor | None = None,
        k_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tensors = [
            hidden,
            input_norm_weight,
            post_norm_weight,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            gate_weight,
            up_weight,
            down_weight,
        ]
        for optional in (q_bias, k_bias, v_bias, q_norm_weight, k_norm_weight):
            if optional is not None:
                tensors.append(optional)
        if any(tensor.dtype != self.dtype for tensor in tensors):
            raise TypeError(f"dense_layer_prefill_fp16 requires {self.dtype} tensors")
        if hidden.ndim != 2:
            raise ValueError("hidden must have shape [seq_len, hidden_size]")
        cpu_tensors = [tensor.detach().cpu().contiguous() for tensor in tensors[:10]]
        q_bias_cpu = None if q_bias is None else q_bias.detach().cpu().contiguous().reshape(-1)
        k_bias_cpu = None if k_bias is None else k_bias.detach().cpu().contiguous().reshape(-1)
        v_bias_cpu = None if v_bias is None else v_bias.detach().cpu().contiguous().reshape(-1)
        q_norm_cpu = None if q_norm_weight is None else q_norm_weight.detach().cpu().contiguous().reshape(-1)
        k_norm_cpu = None if k_norm_weight is None else k_norm_weight.detach().cpu().contiguous().reshape(-1)
        seq_len = int(cpu_tensors[0].shape[0])
        hidden_size = int(cpu_tensors[0].shape[1])
        out = torch.empty((seq_len, hidden_size), dtype=self.dtype)
        if hasattr(self._lib, "kv_dense_layer_prefill_u16_ext"):
            code = self._lib.kv_dense_layer_prefill_u16_ext(
                self._handle,
                ctypes.c_longlong(int(layer)),
                ctypes.c_void_p(int(cpu_tensors[0].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[1].reshape(-1).data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[2].reshape(-1).data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[3].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[4].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[5].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[6].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[7].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[8].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[9].data_ptr())),
                ctypes.c_void_p(0 if q_bias_cpu is None else int(q_bias_cpu.data_ptr())),
                ctypes.c_void_p(0 if k_bias_cpu is None else int(k_bias_cpu.data_ptr())),
                ctypes.c_void_p(0 if v_bias_cpu is None else int(v_bias_cpu.data_ptr())),
                ctypes.c_void_p(0 if q_norm_cpu is None else int(q_norm_cpu.data_ptr())),
                ctypes.c_void_p(0 if k_norm_cpu is None else int(k_norm_cpu.data_ptr())),
                ctypes.c_void_p(int(out.data_ptr())),
                ctypes.c_longlong(seq_len),
                ctypes.c_longlong(hidden_size),
                ctypes.c_longlong(int(intermediate_size)),
                ctypes.c_longlong(int(num_attention_heads)),
                ctypes.c_longlong(int(num_key_value_heads)),
                ctypes.c_float(float(rms_eps)),
                ctypes.c_float(float(rope_theta)),
            )
        elif hasattr(self._lib, "kv_dense_layer_prefill_fp16"):
            code = self._lib.kv_dense_layer_prefill_fp16(
                self._handle,
                ctypes.c_longlong(int(layer)),
                ctypes.c_void_p(int(cpu_tensors[0].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[1].reshape(-1).data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[2].reshape(-1).data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[3].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[4].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[5].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[6].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[7].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[8].data_ptr())),
                ctypes.c_void_p(int(cpu_tensors[9].data_ptr())),
                ctypes.c_void_p(int(out.data_ptr())),
                ctypes.c_longlong(seq_len),
                ctypes.c_longlong(hidden_size),
                ctypes.c_longlong(int(intermediate_size)),
                ctypes.c_longlong(int(num_attention_heads)),
                ctypes.c_longlong(int(num_key_value_heads)),
                ctypes.c_float(float(rms_eps)),
                ctypes.c_float(float(rope_theta)),
            )
        else:
            raise RuntimeError("kv_dense_layer_prefill_fp16 is unavailable")
        if code != 0:
            raise RuntimeError(f"kv_dense_layer_prefill_fp16 failed with code {code}")
        return out


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


def fp8_e4m3_block_linear_f32(fp8_weight: torch.Tensor, scale_inv: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    lib = _load_fp8_linear_lib()
    if lib is None:
        reason = "disabled" if _native_fp8_linear_disabled() else _FP8_LINEAR_ERROR
        raise RuntimeError(f"Native FP8 linear is unavailable: {reason}")
    if fp8_weight.ndim != 2:
        raise ValueError("fp8_weight must have shape [out_rows, in_cols]")
    if scale_inv.ndim != 2:
        raise ValueError("scale_inv must have shape [scale_rows, scale_cols]")
    hidden_cpu = hidden.detach().cpu().contiguous().reshape(-1, int(hidden.shape[-1])).to(torch.float32)
    weight_cpu = fp8_weight.detach().cpu().contiguous().to(torch.uint8)
    scale_cpu = scale_inv.detach().cpu().contiguous().to(torch.float32)
    out_rows = int(weight_cpu.shape[0])
    in_cols = int(weight_cpu.shape[1])
    if int(hidden_cpu.shape[1]) != in_cols:
        raise ValueError("hidden input size does not match FP8 weight columns")
    expected_scale_rows = (out_rows + 127) // 128
    expected_scale_cols = (in_cols + 127) // 128
    if int(scale_cpu.shape[0]) != expected_scale_rows or int(scale_cpu.shape[1]) != expected_scale_cols:
        raise ValueError("scale_inv shape does not match FP8 128x128 block layout")

    out = torch.empty((int(hidden_cpu.shape[0]), out_rows), dtype=torch.float32)
    code = lib.fp8_e4m3_block_linear_f32(
        ctypes.c_void_p(int(weight_cpu.data_ptr())),
        ctypes.c_void_p(int(scale_cpu.data_ptr())),
        ctypes.c_void_p(int(hidden_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(int(hidden_cpu.shape[0])),
        ctypes.c_longlong(out_rows),
        ctypes.c_longlong(in_cols),
        ctypes.c_longlong(expected_scale_cols),
    )
    if code != 0:
        raise RuntimeError(f"fp8_e4m3_block_linear_f32 failed with code {code}")
    return out


def fp8_e4m3_block_dual_linear_f32(
    fp8_weight_a: torch.Tensor,
    scale_inv_a: torch.Tensor,
    fp8_weight_b: torch.Tensor,
    scale_inv_b: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _load_fp8_linear_lib()
    if lib is None or not hasattr(lib, "fp8_e4m3_block_dual_linear_f32"):
        reason = "disabled" if _native_fp8_linear_disabled() else _FP8_LINEAR_ERROR
        raise RuntimeError(f"Native FP8 dual linear is unavailable: {reason}")
    if fp8_weight_a.ndim != 2 or fp8_weight_b.ndim != 2:
        raise ValueError("fp8 weights must have shape [out_rows, in_cols]")
    weight_a_cpu = fp8_weight_a.detach().cpu().contiguous().to(torch.uint8)
    weight_b_cpu = fp8_weight_b.detach().cpu().contiguous().to(torch.uint8)
    if tuple(weight_a_cpu.shape) != tuple(weight_b_cpu.shape):
        raise ValueError("dual FP8 weights must have matching shapes")
    scale_a_cpu = scale_inv_a.detach().cpu().contiguous().to(torch.float32)
    scale_b_cpu = scale_inv_b.detach().cpu().contiguous().to(torch.float32)
    if tuple(scale_a_cpu.shape) != tuple(scale_b_cpu.shape):
        raise ValueError("dual FP8 scales must have matching shapes")
    hidden_cpu = hidden.detach().cpu().contiguous().reshape(-1, int(hidden.shape[-1])).to(torch.float32)
    out_rows = int(weight_a_cpu.shape[0])
    in_cols = int(weight_a_cpu.shape[1])
    if int(hidden_cpu.shape[1]) != in_cols:
        raise ValueError("hidden input size does not match FP8 weight columns")
    expected_scale_rows = (out_rows + 127) // 128
    expected_scale_cols = (in_cols + 127) // 128
    if (
        int(scale_a_cpu.shape[0]) != expected_scale_rows
        or int(scale_a_cpu.shape[1]) != expected_scale_cols
    ):
        raise ValueError("scale_inv shape does not match FP8 128x128 block layout")

    out_a = torch.empty((int(hidden_cpu.shape[0]), out_rows), dtype=torch.float32)
    out_b = torch.empty((int(hidden_cpu.shape[0]), out_rows), dtype=torch.float32)
    code = lib.fp8_e4m3_block_dual_linear_f32(
        ctypes.c_void_p(int(weight_a_cpu.data_ptr())),
        ctypes.c_void_p(int(scale_a_cpu.data_ptr())),
        ctypes.c_void_p(int(weight_b_cpu.data_ptr())),
        ctypes.c_void_p(int(scale_b_cpu.data_ptr())),
        ctypes.c_void_p(int(hidden_cpu.data_ptr())),
        ctypes.c_void_p(int(out_a.data_ptr())),
        ctypes.c_void_p(int(out_b.data_ptr())),
        ctypes.c_longlong(int(hidden_cpu.shape[0])),
        ctypes.c_longlong(out_rows),
        ctypes.c_longlong(in_cols),
        ctypes.c_longlong(expected_scale_cols),
    )
    if code != 0:
        raise RuntimeError(f"fp8_e4m3_block_dual_linear_f32 failed with code {code}")
    return out_a, out_b


def fp8_e4m3_block_mlp_f32(
    gate_weight: torch.Tensor,
    gate_scale_inv: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale_inv: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale_inv: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    lib = _load_fp8_linear_lib()
    if lib is None or not hasattr(lib, "fp8_e4m3_block_mlp_f32"):
        reason = "disabled" if _native_fp8_linear_disabled() else _FP8_LINEAR_ERROR
        raise RuntimeError(f"Native FP8 MLP is unavailable: {reason}")
    if gate_weight.ndim != 2 or up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("FP8 MLP weights must be 2D")
    gate_cpu = gate_weight.detach().cpu().contiguous().to(torch.uint8)
    up_cpu = up_weight.detach().cpu().contiguous().to(torch.uint8)
    down_cpu = down_weight.detach().cpu().contiguous().to(torch.uint8)
    if tuple(gate_cpu.shape) != tuple(up_cpu.shape):
        raise ValueError("FP8 MLP gate and up weights must have matching shapes")
    intermediate_rows = int(gate_cpu.shape[0])
    hidden_cols = int(gate_cpu.shape[1])
    if tuple(down_cpu.shape) != (hidden_cols, intermediate_rows):
        raise ValueError("FP8 MLP down weight must have shape [hidden, intermediate]")
    gate_scale_cpu = gate_scale_inv.detach().cpu().contiguous().to(torch.float32)
    up_scale_cpu = up_scale_inv.detach().cpu().contiguous().to(torch.float32)
    down_scale_cpu = down_scale_inv.detach().cpu().contiguous().to(torch.float32)
    if tuple(gate_scale_cpu.shape) != tuple(up_scale_cpu.shape):
        raise ValueError("FP8 MLP gate and up scales must have matching shapes")
    expected_gate_scale = ((intermediate_rows + 127) // 128, (hidden_cols + 127) // 128)
    expected_down_scale = ((hidden_cols + 127) // 128, (intermediate_rows + 127) // 128)
    if tuple(gate_scale_cpu.shape) != expected_gate_scale:
        raise ValueError("FP8 MLP gate scale shape does not match 128x128 block layout")
    if tuple(down_scale_cpu.shape) != expected_down_scale:
        raise ValueError("FP8 MLP down scale shape does not match 128x128 block layout")
    hidden_cpu = hidden.detach().cpu().contiguous().reshape(-1, int(hidden.shape[-1])).to(torch.float32)
    if int(hidden_cpu.shape[1]) != hidden_cols:
        raise ValueError("hidden input size does not match FP8 MLP input size")

    out = torch.empty((int(hidden_cpu.shape[0]), hidden_cols), dtype=torch.float32)
    code = lib.fp8_e4m3_block_mlp_f32(
        ctypes.c_void_p(int(gate_cpu.data_ptr())),
        ctypes.c_void_p(int(gate_scale_cpu.data_ptr())),
        ctypes.c_void_p(int(up_cpu.data_ptr())),
        ctypes.c_void_p(int(up_scale_cpu.data_ptr())),
        ctypes.c_void_p(int(down_cpu.data_ptr())),
        ctypes.c_void_p(int(down_scale_cpu.data_ptr())),
        ctypes.c_void_p(int(hidden_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(int(hidden_cpu.shape[0])),
        ctypes.c_longlong(intermediate_rows),
        ctypes.c_longlong(hidden_cols),
        ctypes.c_longlong(expected_gate_scale[1]),
        ctypes.c_longlong(expected_down_scale[1]),
    )
    if code != 0:
        raise RuntimeError(f"fp8_e4m3_block_mlp_f32 failed with code {code}")
    return out


def fp8_e4m3_block_mlp_many_f32(
    items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    hidden: torch.Tensor,
) -> torch.Tensor:
    lib = _load_fp8_linear_lib()
    if lib is None or not hasattr(lib, "fp8_e4m3_block_mlp_many_f32"):
        reason = "disabled" if _native_fp8_linear_disabled() else _FP8_LINEAR_ERROR
        raise RuntimeError(f"Native FP8 many-MLP is unavailable: {reason}")
    if not items:
        raise ValueError("items must not be empty")
    prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    first_gate = items[0][0].detach().cpu().contiguous().to(torch.uint8)
    intermediate_rows = int(first_gate.shape[0])
    hidden_cols = int(first_gate.shape[1])
    for gate, gate_scale, up, up_scale, down, down_scale in items:
        gate_cpu = gate.detach().cpu().contiguous().to(torch.uint8)
        up_cpu = up.detach().cpu().contiguous().to(torch.uint8)
        down_cpu = down.detach().cpu().contiguous().to(torch.uint8)
        gate_scale_cpu = gate_scale.detach().cpu().contiguous().to(torch.float32)
        up_scale_cpu = up_scale.detach().cpu().contiguous().to(torch.float32)
        down_scale_cpu = down_scale.detach().cpu().contiguous().to(torch.float32)
        if tuple(gate_cpu.shape) != (intermediate_rows, hidden_cols):
            raise ValueError("all gate weights must have matching shape")
        if tuple(up_cpu.shape) != (intermediate_rows, hidden_cols):
            raise ValueError("all up weights must have matching shape")
        if tuple(down_cpu.shape) != (hidden_cols, intermediate_rows):
            raise ValueError("all down weights must have matching shape")
        prepared.append((gate_cpu, gate_scale_cpu, up_cpu, up_scale_cpu, down_cpu, down_scale_cpu))

    expected_gate_scale = ((intermediate_rows + 127) // 128, (hidden_cols + 127) // 128)
    expected_down_scale = ((hidden_cols + 127) // 128, (intermediate_rows + 127) // 128)
    for gate, gate_scale, up, up_scale, down, down_scale in prepared:
        if tuple(gate_scale.shape) != expected_gate_scale or tuple(up_scale.shape) != expected_gate_scale:
            raise ValueError("gate/up scale shapes do not match 128x128 block layout")
        if tuple(down_scale.shape) != expected_down_scale:
            raise ValueError("down scale shape does not match 128x128 block layout")

    hidden_cpu = hidden.detach().cpu().contiguous().reshape(-1, int(hidden.shape[-1])).to(torch.float32)
    if int(hidden_cpu.shape[1]) != hidden_cols:
        raise ValueError("hidden input size does not match FP8 MLP input size")
    ptr_dtype = torch.int64
    gate_ptrs = torch.tensor([int(item[0].data_ptr()) for item in prepared], dtype=ptr_dtype)
    gate_scale_ptrs = torch.tensor([int(item[1].data_ptr()) for item in prepared], dtype=ptr_dtype)
    up_ptrs = torch.tensor([int(item[2].data_ptr()) for item in prepared], dtype=ptr_dtype)
    up_scale_ptrs = torch.tensor([int(item[3].data_ptr()) for item in prepared], dtype=ptr_dtype)
    down_ptrs = torch.tensor([int(item[4].data_ptr()) for item in prepared], dtype=ptr_dtype)
    down_scale_ptrs = torch.tensor([int(item[5].data_ptr()) for item in prepared], dtype=ptr_dtype)
    out = torch.empty((len(prepared), int(hidden_cpu.shape[0]), hidden_cols), dtype=torch.float32)
    code = lib.fp8_e4m3_block_mlp_many_f32(
        ctypes.c_void_p(int(gate_ptrs.data_ptr())),
        ctypes.c_void_p(int(gate_scale_ptrs.data_ptr())),
        ctypes.c_void_p(int(up_ptrs.data_ptr())),
        ctypes.c_void_p(int(up_scale_ptrs.data_ptr())),
        ctypes.c_void_p(int(down_ptrs.data_ptr())),
        ctypes.c_void_p(int(down_scale_ptrs.data_ptr())),
        ctypes.c_void_p(int(hidden_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(len(prepared)),
        ctypes.c_longlong(int(hidden_cpu.shape[0])),
        ctypes.c_longlong(intermediate_rows),
        ctypes.c_longlong(hidden_cols),
        ctypes.c_longlong(expected_gate_scale[1]),
        ctypes.c_longlong(expected_down_scale[1]),
    )
    if code != 0:
        raise RuntimeError(f"fp8_e4m3_block_mlp_many_f32 failed with code {code}")
    return out


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


def q4_dequant_many_to_fp16(
    items: list[tuple[torch.Tensor, torch.Tensor, int, int]],
) -> list[torch.Tensor]:
    lib = _load_q4_lib()
    if lib is None or not hasattr(lib, "q4_dequant_many_to_fp16"):
        reason = "disabled" if _native_disabled() else _Q4_LOAD_ERROR
        raise RuntimeError(f"Native Q4 batch dequant is unavailable: {reason}")
    if not items:
        return []

    packed_tensors: list[torch.Tensor] = []
    scale_tensors: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    packed_ptrs = (ctypes.c_void_p * len(items))()
    scale_ptrs = (ctypes.c_void_p * len(items))()
    out_ptrs = (ctypes.c_void_p * len(items))()
    channels = (ctypes.c_longlong * len(items))()
    channel_sizes = (ctypes.c_longlong * len(items))()
    for index, (packed, scales, num_channels, channel_size) in enumerate(items):
        packed_cpu = packed.detach().cpu().contiguous().to(torch.uint8)
        scales_cpu = scales.detach().cpu().contiguous().to(torch.float16)
        out = torch.empty((int(num_channels) * int(channel_size),), dtype=torch.float16)
        packed_tensors.append(packed_cpu)
        scale_tensors.append(scales_cpu)
        outputs.append(out)
        packed_ptrs[index] = ctypes.c_void_p(int(packed_cpu.data_ptr()))
        scale_ptrs[index] = ctypes.c_void_p(int(scales_cpu.data_ptr()))
        out_ptrs[index] = ctypes.c_void_p(int(out.data_ptr()))
        channels[index] = ctypes.c_longlong(int(num_channels))
        channel_sizes[index] = ctypes.c_longlong(int(channel_size))

    code = lib.q4_dequant_many_to_fp16(
        packed_ptrs,
        scale_ptrs,
        out_ptrs,
        channels,
        channel_sizes,
        ctypes.c_longlong(len(items)),
    )
    if code != 0:
        raise RuntimeError(f"Native Q4 batch dequant failed with code {code}")
    return outputs


def q4_moe_selected_forward_u16(
    hidden: torch.Tensor,
    experts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    route_weights: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> torch.Tensor:
    lib = _load_q4_lib()
    if lib is None or not hasattr(lib, "q4_moe_selected_forward_u16"):
        reason = "disabled" if _native_disabled() else _Q4_LOAD_ERROR
        raise RuntimeError(f"Native Q4 MoE selected forward is unavailable: {reason}")
    if not experts:
        raise ValueError("experts must not be empty")
    hidden_cpu = hidden.detach().cpu().contiguous().reshape(-1).to(torch.float16)
    if hidden_cpu.numel() != int(hidden_size):
        raise ValueError("hidden size mismatch")
    route_cpu = route_weights.detach().cpu().contiguous().reshape(-1).to(torch.float32)
    if route_cpu.numel() != len(experts):
        raise ValueError("route weight count must match selected experts")

    selected_count = len(experts)
    gate_packed_ptrs = (ctypes.c_void_p * selected_count)()
    gate_scale_ptrs = (ctypes.c_void_p * selected_count)()
    up_packed_ptrs = (ctypes.c_void_p * selected_count)()
    up_scale_ptrs = (ctypes.c_void_p * selected_count)()
    down_packed_ptrs = (ctypes.c_void_p * selected_count)()
    down_scale_ptrs = (ctypes.c_void_p * selected_count)()
    keepalive: list[torch.Tensor] = [hidden_cpu, route_cpu]
    for index, (gate_packed, gate_scales, up_packed, up_scales, down_packed, down_scales) in enumerate(experts):
        tensors = [
            gate_packed.detach().cpu().contiguous().to(torch.uint8),
            gate_scales.detach().cpu().contiguous().to(torch.float16),
            up_packed.detach().cpu().contiguous().to(torch.uint8),
            up_scales.detach().cpu().contiguous().to(torch.float16),
            down_packed.detach().cpu().contiguous().to(torch.uint8),
            down_scales.detach().cpu().contiguous().to(torch.float16),
        ]
        keepalive.extend(tensors)
        gate_packed_ptrs[index] = ctypes.c_void_p(int(tensors[0].data_ptr()))
        gate_scale_ptrs[index] = ctypes.c_void_p(int(tensors[1].data_ptr()))
        up_packed_ptrs[index] = ctypes.c_void_p(int(tensors[2].data_ptr()))
        up_scale_ptrs[index] = ctypes.c_void_p(int(tensors[3].data_ptr()))
        down_packed_ptrs[index] = ctypes.c_void_p(int(tensors[4].data_ptr()))
        down_scale_ptrs[index] = ctypes.c_void_p(int(tensors[5].data_ptr()))

    out = torch.empty((int(hidden_size),), dtype=torch.float16)
    code = lib.q4_moe_selected_forward_u16(
        ctypes.c_void_p(int(hidden_cpu.data_ptr())),
        gate_packed_ptrs,
        gate_scale_ptrs,
        up_packed_ptrs,
        up_scale_ptrs,
        down_packed_ptrs,
        down_scale_ptrs,
        ctypes.c_void_p(int(route_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(int(hidden_size)),
        ctypes.c_longlong(int(intermediate_size)),
        ctypes.c_longlong(selected_count),
    )
    if code != 0:
        raise RuntimeError(f"Native Q4 MoE selected forward failed with code {code}")
    return out
