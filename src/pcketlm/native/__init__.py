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
_FP8_DEQUANT_DLL = _NATIVE_DIR / "fp8_dequant.dll"
_FP8_LINEAR_DLL = _NATIVE_DIR / "fp8_linear.dll"
_DS_FORWARD_DLL = _NATIVE_DIR / "ds_forward.dll"
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
_FP8_DEQUANT_LIB: ctypes.CDLL | None = None
_FP8_DEQUANT_ERROR: Exception | None = None
_FP8_LINEAR_LIB: ctypes.CDLL | None = None
_FP8_LINEAR_ERROR: Exception | None = None
_DS_FORWARD_LIB: ctypes.CDLL | None = None
_DS_FORWARD_ERROR: Exception | None = None
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


def _native_fp8_dequant_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_FP8_DEQUANT", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_ds_router_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_DS_ROUTER", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_ds_moe_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_DS_MOE", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_ds_attention_bridge_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_DS_ATTENTION_BRIDGE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _native_flash_mla_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_NATIVE_FLASH_MLA", "").strip().lower() in {"1", "true", "yes", "on"}


def _native_fused_ds_attention_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_FUSED_DS_ATTENTION", "").strip().lower() in {"1", "true", "yes", "on"}


def _monolithic_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_MONOLITHIC", "").strip().lower() in {"1", "true", "yes", "on"}


def _ds_monolithic_disabled() -> bool:
    return os.environ.get("PCKETLM_DISABLE_DS_MONOLITHIC", "").strip().lower() in {"1", "true", "yes", "on"}


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


def _load_fp8_dequant_lib() -> ctypes.CDLL | None:
    global _FP8_DEQUANT_LIB, _FP8_DEQUANT_ERROR
    if _native_fp8_dequant_disabled():
        return None
    if _FP8_DEQUANT_LIB is not None:
        return _FP8_DEQUANT_LIB
    if not _FP8_DEQUANT_DLL.exists():
        _FP8_DEQUANT_ERROR = FileNotFoundError(str(_FP8_DEQUANT_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_FP8_DEQUANT_DLL))
        lib.fp8_dequant_cpu_has_avx2_f16c.argtypes = []
        lib.fp8_dequant_cpu_has_avx2_f16c.restype = ctypes.c_int
        lib.fp8_e4m3_dequant_to_fp16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]
        lib.fp8_e4m3_dequant_to_fp16.restype = ctypes.c_int
        if hasattr(lib, "fp8_e4m3_dequant_to_bf16"):
            lib.fp8_e4m3_dequant_to_bf16.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
            ]
            lib.fp8_e4m3_dequant_to_bf16.restype = ctypes.c_int
        if lib.fp8_dequant_cpu_has_avx2_f16c() != 1:
            _FP8_DEQUANT_ERROR = RuntimeError("CPU does not report AVX2+F16C support.")
            return None
    except Exception as exc:  # pragma: no cover - defensive platform path
        _FP8_DEQUANT_ERROR = exc
        return None
    _FP8_DEQUANT_LIB = lib
    _FP8_DEQUANT_ERROR = None
    return lib


def native_fp8_dequant_available() -> bool:
    return _load_fp8_dequant_lib() is not None


def native_fp8_dequant_error() -> Exception | None:
    _load_fp8_dequant_lib()
    return _FP8_DEQUANT_ERROR


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


_DS_TENSOR_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_longlong),
)
_DS_ATTENTION_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_longlong,
    ctypes.c_void_p,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_longlong),
)
_DS_DECODE_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_longlong),
)
_DS_PREFILL_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_longlong),
)
_DS_VERIFY_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_longlong),
)


def _load_ds_forward_lib() -> ctypes.CDLL | None:
    global _DS_FORWARD_LIB, _DS_FORWARD_ERROR
    if _ds_monolithic_disabled():
        return None
    if _DS_FORWARD_LIB is not None:
        return _DS_FORWARD_LIB
    if not _DS_FORWARD_DLL.exists():
        _DS_FORWARD_ERROR = FileNotFoundError(str(_DS_FORWARD_DLL))
        return None
    try:
        lib = ctypes.CDLL(str(_DS_FORWARD_DLL))
        lib.ds_cpu_has_avx2.argtypes = []
        lib.ds_cpu_has_avx2.restype = ctypes.c_int
        lib.ds_session_create.argtypes = [
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.ds_session_create.restype = ctypes.c_void_p
        lib.ds_session_destroy.argtypes = [ctypes.c_void_p]
        lib.ds_session_destroy.restype = None
        lib.ds_session_register_layer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            _DS_TENSOR_CALLBACK,
            _DS_TENSOR_CALLBACK,
        ]
        lib.ds_session_register_layer.restype = ctypes.c_int
        lib.ds_monolithic_call_counter.argtypes = [ctypes.c_void_p]
        lib.ds_monolithic_call_counter.restype = ctypes.c_longlong
        lib.ds_callback_invocation_count.argtypes = [ctypes.c_void_p]
        lib.ds_callback_invocation_count.restype = ctypes.c_longlong
        lib.ds_expert_invocation_count.argtypes = [ctypes.c_void_p]
        lib.ds_expert_invocation_count.restype = ctypes.c_longlong
        lib.ds_attention_invocation_count.argtypes = [ctypes.c_void_p]
        lib.ds_attention_invocation_count.restype = ctypes.c_longlong
        if hasattr(lib, "ds_fused_attention_invocation_count"):
            lib.ds_fused_attention_invocation_count.argtypes = [ctypes.c_void_p]
            lib.ds_fused_attention_invocation_count.restype = ctypes.c_longlong
        lib.ds_layers_executed_count.argtypes = [ctypes.c_void_p]
        lib.ds_layers_executed_count.restype = ctypes.c_longlong
        lib.ds_registered_layer_count.argtypes = [ctypes.c_void_p]
        lib.ds_registered_layer_count.restype = ctypes.c_longlong
        lib.ds_registered_fp8_nbytes.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        lib.ds_registered_fp8_nbytes.restype = ctypes.c_longlong
        lib.ds_registered_scale_nbytes.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        lib.ds_registered_scale_nbytes.restype = ctypes.c_longlong
        lib.ds_router_topk_u16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.ds_router_topk_u16.restype = ctypes.c_int
        lib.ds_moe_layer_forward_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_void_p,
        ]
        lib.ds_moe_layer_forward_f32.restype = ctypes.c_int
        lib.ds_attention_layer_forward_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_longlong,
            _DS_ATTENTION_CALLBACK,
            ctypes.c_void_p,
        ]
        lib.ds_attention_layer_forward_f32.restype = ctypes.c_int
        lib.ds_forward_decode_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_longlong,
            _DS_DECODE_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_longlong,
        ]
        lib.ds_forward_decode_f32.restype = ctypes.c_int
        lib.ds_forward_prefill_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            _DS_PREFILL_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_longlong,
        ]
        lib.ds_forward_prefill_f32.restype = ctypes.c_int
        lib.ds_forward_verify_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
            _DS_VERIFY_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_longlong,
        ]
        lib.ds_forward_verify_f32.restype = ctypes.c_int
        if hasattr(lib, "ds_mla_attention_flash_forward"):
            lib.ds_mla_attention_flash_forward.argtypes = [
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
                ctypes.c_float,
            ]
            lib.ds_mla_attention_flash_forward.restype = ctypes.c_int
        if hasattr(lib, "ds_attention_block_forward_f32"):
            lib.ds_attention_block_forward_f32.argtypes = [
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
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_float,
                ctypes.c_float,
            ]
            lib.ds_attention_block_forward_f32.restype = ctypes.c_int
        lib.ds_moe_layer_forward_fp8_f32.argtypes = [
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
            ctypes.c_longlong,
            ctypes.c_void_p,
        ]
        lib.ds_moe_layer_forward_fp8_f32.restype = ctypes.c_int
    except Exception as exc:  # pragma: no cover - defensive platform path
        _DS_FORWARD_ERROR = exc
        return None
    _DS_FORWARD_LIB = lib
    _DS_FORWARD_ERROR = None
    return lib


def native_ds_forward_available() -> bool:
    return _load_ds_forward_lib() is not None


def native_ds_forward_error() -> Exception | None:
    _load_ds_forward_lib()
    return _DS_FORWARD_ERROR


def native_ds_forward_has_avx2() -> bool:
    lib = _load_ds_forward_lib()
    return bool(lib and lib.ds_cpu_has_avx2())


def native_ds_router_available() -> bool:
    lib = _load_ds_forward_lib()
    return bool(lib is not None and not _native_ds_router_disabled() and hasattr(lib, "ds_router_topk_u16"))


def ds_router_topk(
    hidden: torch.Tensor,
    router_weights: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DeepSeek router top-k, falling back to the Python reference when disabled."""
    hidden_cpu = hidden.detach().cpu().contiguous().reshape(-1)
    weights_cpu = router_weights.detach().cpu().contiguous()
    if weights_cpu.ndim != 2:
        raise ValueError("router_weights must be rank-2 [num_experts, hidden_dim]")
    if hidden_cpu.numel() != weights_cpu.shape[1]:
        raise ValueError("hidden size must match router weight columns")
    effective_k = max(1, min(int(k), int(weights_cpu.shape[0])))
    if not native_ds_router_available():
        return _python_ds_router_topk(hidden_cpu, weights_cpu, effective_k)
    if hidden_cpu.dtype not in {torch.float16, torch.bfloat16}:
        hidden_cpu = hidden_cpu.to(torch.float16)
    if weights_cpu.dtype != hidden_cpu.dtype:
        weights_cpu = weights_cpu.to(hidden_cpu.dtype)
    ids = torch.empty((effective_k,), dtype=torch.int64)
    route_weights = torch.empty((effective_k,), dtype=torch.float32)
    lib = _load_ds_forward_lib()
    assert lib is not None
    code = lib.ds_router_topk_u16(
        ctypes.c_void_p(int(hidden_cpu.view(torch.uint16).data_ptr())),
        ctypes.c_void_p(int(weights_cpu.view(torch.uint16).data_ptr())),
        ctypes.c_longlong(int(weights_cpu.shape[1])),
        ctypes.c_longlong(int(weights_cpu.shape[0])),
        ctypes.c_longlong(effective_k),
        ctypes.c_longlong(_u16_storage_dtype_code(hidden_cpu.dtype)),
        ctypes.c_void_p(int(ids.data_ptr())),
        ctypes.c_void_p(int(route_weights.data_ptr())),
    )
    if code != 0:
        raise RuntimeError(f"ds_router_topk_u16 failed with code {code}")
    return ids, route_weights


def _python_ds_router_topk(
    hidden: torch.Tensor,
    router_weights: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.nn.functional.linear(hidden.reshape(1, -1).float(), router_weights.float()).reshape(-1)
    probs = torch.softmax(scores, dim=-1)
    weights, ids = torch.topk(probs, int(k), dim=-1)
    weights = weights / weights.sum().clamp_min(1e-12)
    return ids.to(torch.int64).contiguous(), weights.to(torch.float32).contiguous()


def native_ds_moe_available() -> bool:
    lib = _load_ds_forward_lib()
    return bool(lib is not None and not _native_ds_moe_disabled() and hasattr(lib, "ds_moe_layer_forward_f32"))


def native_ds_attention_bridge_available() -> bool:
    lib = _load_ds_forward_lib()
    return bool(
        lib is not None
        and not _native_ds_attention_bridge_disabled()
        and hasattr(lib, "ds_attention_layer_forward_f32")
    )


def native_ds_decode_available() -> bool:
    lib = _load_ds_forward_lib()
    return bool(lib is not None and hasattr(lib, "ds_forward_decode_f32"))


def native_flash_mla_available() -> bool:
    lib = _load_ds_forward_lib()
    return bool(
        lib is not None
        and not _native_flash_mla_disabled()
        and hasattr(lib, "ds_mla_attention_flash_forward")
    )


def native_fused_ds_attention_available() -> bool:
    lib = _load_ds_forward_lib()
    return bool(
        lib is not None
        and not _native_fused_ds_attention_disabled()
        and hasattr(lib, "ds_attention_block_forward_f32")
    )


def _python_mla_attention_flash_reference(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    wkv_b: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    q_nope_f = q_nope.detach().cpu().contiguous().to(torch.float32)
    q_pe_f = q_pe.detach().cpu().contiguous().to(torch.float32)
    kv_f = kv_cache.detach().cpu().contiguous().to(torch.float32)
    pe_f = pe_cache.detach().cpu().contiguous().to(torch.float32)
    wkv_f = wkv_b.detach().cpu().contiguous().to(torch.float32)
    qk_nope = int(q_nope_f.shape[-1])
    v_head_dim = int(wkv_f.shape[1]) - qk_nope
    q_abs = torch.einsum("hd,hdc->hc", q_nope_f, wkv_f[:, :qk_nope])
    scores = (torch.einsum("hc,tc->ht", q_abs, kv_f) + torch.einsum("hr,tr->ht", q_pe_f, pe_f)) * float(
        softmax_scale
    )
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    latent = torch.einsum("ht,tc->hc", probs, kv_f)
    return torch.einsum("hc,hdc->hd", latent, wkv_f[:, qk_nope : qk_nope + v_head_dim]).contiguous()


def ds_mla_attention_flash_forward(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    wkv_b: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Run DeepSeek compressed-KV MLA attention core with an online-softmax native kernel."""
    q_nope_cpu = q_nope.detach().cpu().contiguous().to(torch.float32)
    q_pe_cpu = q_pe.detach().cpu().contiguous().to(torch.float32)
    kv_cpu = kv_cache.detach().cpu().contiguous().to(torch.float32)
    pe_cpu = pe_cache.detach().cpu().contiguous().to(torch.float32)
    wkv_cpu = wkv_b.detach().cpu().contiguous().to(torch.float32)
    if q_nope_cpu.ndim != 2 or q_pe_cpu.ndim != 2:
        raise ValueError("q_nope and q_pe must be [heads, dim]")
    if kv_cpu.ndim != 2 or pe_cpu.ndim != 2:
        raise ValueError("kv_cache and pe_cache must be [cache_len, dim]")
    if wkv_cpu.ndim != 3:
        raise ValueError("wkv_b must be [heads, qk_nope + v_head_dim, kv_lora_rank]")
    num_heads = int(q_nope_cpu.shape[0])
    qk_nope_dim = int(q_nope_cpu.shape[1])
    qk_rope_dim = int(q_pe_cpu.shape[1])
    cache_len = int(kv_cpu.shape[0])
    kv_lora_rank = int(kv_cpu.shape[1])
    if int(q_pe_cpu.shape[0]) != num_heads or int(wkv_cpu.shape[0]) != num_heads:
        raise ValueError("head count mismatch")
    if int(pe_cpu.shape[0]) != cache_len or int(pe_cpu.shape[1]) != qk_rope_dim:
        raise ValueError("pe_cache shape mismatch")
    if int(wkv_cpu.shape[2]) != kv_lora_rank or int(wkv_cpu.shape[1]) <= qk_nope_dim:
        raise ValueError("wkv_b shape mismatch")
    v_head_dim = int(wkv_cpu.shape[1]) - qk_nope_dim
    if not native_flash_mla_available():
        return _python_mla_attention_flash_reference(
            q_nope_cpu,
            q_pe_cpu,
            kv_cpu,
            pe_cpu,
            wkv_cpu,
            softmax_scale=float(softmax_scale),
        )

    out = torch.empty((num_heads, v_head_dim), dtype=torch.float32)
    lib = _load_ds_forward_lib()
    assert lib is not None
    code = lib.ds_mla_attention_flash_forward(
        ctypes.c_void_p(int(q_nope_cpu.data_ptr())),
        ctypes.c_void_p(int(q_pe_cpu.data_ptr())),
        ctypes.c_void_p(int(kv_cpu.data_ptr())),
        ctypes.c_void_p(int(pe_cpu.data_ptr())),
        ctypes.c_void_p(int(wkv_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(num_heads),
        ctypes.c_longlong(cache_len),
        ctypes.c_longlong(qk_nope_dim),
        ctypes.c_longlong(qk_rope_dim),
        ctypes.c_longlong(kv_lora_rank),
        ctypes.c_longlong(v_head_dim),
        ctypes.c_float(float(softmax_scale)),
    )
    if code != 0:
        raise RuntimeError(f"ds_mla_attention_flash_forward failed with code {code}")
    return out


def ds_attention_block_forward(
    session: "DeepSeekNativeSession",
    hidden: torch.Tensor,
    q_a_weight: torch.Tensor,
    q_b_weight: torch.Tensor,
    kv_a_weight: torch.Tensor,
    kv_b_weight: torch.Tensor,
    o_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    *,
    previous_kv_cache: torch.Tensor | None,
    previous_pe_cache: torch.Tensor | None,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    q_lora_rank: int,
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_dim: int,
    qk_rope_dim: int,
    v_head_dim: int,
    rms_eps: float,
    softmax_scale: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Run a one-token DeepSeek attention block as one native C call."""
    hidden_cpu = hidden.detach().cpu().contiguous().reshape(-1).to(torch.float32)
    if int(hidden_cpu.numel()) <= 0:
        raise ValueError("hidden must not be empty")
    hidden_dim = int(hidden_cpu.numel())
    q_a_cpu = q_a_weight.detach().cpu().contiguous().to(torch.float32)
    q_b_cpu = q_b_weight.detach().cpu().contiguous().to(torch.float32)
    kv_a_cpu = kv_a_weight.detach().cpu().contiguous().to(torch.float32)
    kv_b_cpu = kv_b_weight.detach().cpu().contiguous().to(torch.float32)
    o_cpu = o_weight.detach().cpu().contiguous().to(torch.float32)
    q_norm_cpu = q_norm_weight.detach().cpu().contiguous().reshape(-1).to(torch.float32)
    kv_norm_cpu = kv_norm_weight.detach().cpu().contiguous().reshape(-1).to(torch.float32)
    cos_cpu = rope_cos.detach().cpu().contiguous().reshape(-1).to(torch.float32)
    sin_cpu = rope_sin.detach().cpu().contiguous().reshape(-1).to(torch.float32)

    q_rank = int(q_lora_rank)
    kv_rank = int(kv_lora_rank)
    heads = int(num_heads)
    qk_nope = int(qk_nope_dim)
    qk_rope = int(qk_rope_dim)
    v_dim = int(v_head_dim)
    if tuple(q_a_cpu.shape) != (q_rank, hidden_dim):
        raise ValueError("q_a_weight must be [q_lora_rank, hidden_dim]")
    if tuple(q_b_cpu.shape) != (heads * (qk_nope + qk_rope), q_rank):
        raise ValueError("q_b_weight shape does not match DeepSeek attention config")
    if tuple(kv_a_cpu.shape) != (kv_rank + qk_rope, hidden_dim):
        raise ValueError("kv_a_weight shape does not match DeepSeek attention config")
    if tuple(kv_b_cpu.shape) != (heads * (qk_nope + v_dim), kv_rank):
        raise ValueError("kv_b_weight shape does not match DeepSeek attention config")
    if tuple(o_cpu.shape) != (hidden_dim, heads * v_dim):
        raise ValueError("o_weight shape does not match DeepSeek attention config")
    if int(q_norm_cpu.numel()) != q_rank or int(kv_norm_cpu.numel()) != kv_rank:
        raise ValueError("norm weight sizes do not match ranks")
    if int(cos_cpu.numel()) != max(0, qk_rope // 2) or int(sin_cpu.numel()) != max(0, qk_rope // 2):
        raise ValueError("rope cos/sin sizes must be qk_rope_dim / 2")

    if previous_kv_cache is None:
        prev_kv_cpu = torch.empty((0, kv_rank), dtype=torch.float32)
    else:
        prev_kv_cpu = previous_kv_cache.detach().cpu().contiguous().reshape(-1, kv_rank).to(torch.float32)
    previous_len = int(prev_kv_cpu.shape[0])
    if previous_pe_cache is None:
        prev_pe_cpu = torch.empty((0, qk_rope), dtype=torch.float32)
    else:
        prev_pe_cpu = previous_pe_cache.detach().cpu().contiguous().reshape(-1, qk_rope).to(torch.float32)
    if int(prev_pe_cpu.shape[0]) != previous_len:
        raise ValueError("previous KV and PE cache lengths must match")

    if not native_fused_ds_attention_available():
        raise RuntimeError("Native fused DeepSeek attention is unavailable")

    next_kv = torch.empty((previous_len + 1, kv_rank), dtype=torch.float32)
    next_pe = torch.empty((previous_len + 1, qk_rope), dtype=torch.float32)
    out = torch.empty((hidden_dim,), dtype=torch.float32)
    lib = _load_ds_forward_lib()
    assert lib is not None
    prev_kv_ptr = 0 if previous_len == 0 else int(prev_kv_cpu.data_ptr())
    prev_pe_ptr = 0 if previous_len == 0 else int(prev_pe_cpu.data_ptr())
    code = lib.ds_attention_block_forward_f32(
        session._handle,
        ctypes.c_void_p(int(hidden_cpu.data_ptr())),
        ctypes.c_void_p(int(q_a_cpu.data_ptr())),
        ctypes.c_void_p(int(q_b_cpu.data_ptr())),
        ctypes.c_void_p(int(kv_a_cpu.data_ptr())),
        ctypes.c_void_p(int(kv_b_cpu.data_ptr())),
        ctypes.c_void_p(int(o_cpu.data_ptr())),
        ctypes.c_void_p(int(q_norm_cpu.data_ptr())),
        ctypes.c_void_p(int(kv_norm_cpu.data_ptr())),
        ctypes.c_void_p(prev_kv_ptr),
        ctypes.c_void_p(prev_pe_ptr),
        ctypes.c_void_p(int(cos_cpu.data_ptr())),
        ctypes.c_void_p(int(sin_cpu.data_ptr())),
        ctypes.c_void_p(int(next_kv.data_ptr())),
        ctypes.c_void_p(int(next_pe.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(hidden_dim),
        ctypes.c_longlong(q_rank),
        ctypes.c_longlong(kv_rank),
        ctypes.c_longlong(heads),
        ctypes.c_longlong(qk_nope),
        ctypes.c_longlong(qk_rope),
        ctypes.c_longlong(v_dim),
        ctypes.c_longlong(previous_len),
        ctypes.c_float(float(rms_eps)),
        ctypes.c_float(float(softmax_scale)),
    )
    session._tensor_keepalive.extend(
        [
            hidden_cpu,
            q_a_cpu,
            q_b_cpu,
            kv_a_cpu,
            kv_b_cpu,
            o_cpu,
            q_norm_cpu,
            kv_norm_cpu,
            prev_kv_cpu,
            prev_pe_cpu,
            cos_cpu,
            sin_cpu,
            next_kv,
            next_pe,
            out,
        ]
    )
    if code != 0:
        raise RuntimeError(f"ds_attention_block_forward_f32 failed with code {code}")
    return out, (next_kv, next_pe)


def ds_forward_decode(
    session: "DeepSeekNativeSession",
    input_token_id: int,
    decode_fn,
    *,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Run the DeepSeek token-level forward boundary and copy callback logits through C."""
    flat_first = None
    if vocab_size is None:
        first = decode_fn(int(input_token_id)).detach().cpu().contiguous().to(torch.float32)
        flat_first = first.reshape(-1)
        effective_vocab = int(flat_first.numel())
    else:
        effective_vocab = int(vocab_size)
    if not native_ds_decode_available():
        if flat_first is None:
            flat_first = decode_fn(int(input_token_id)).detach().cpu().contiguous().to(torch.float32).reshape(-1)
        if int(flat_first.numel()) != effective_vocab:
            raise ValueError("vocab_size must match decode_fn output size")
        return flat_first.clone()

    keepalive = [] if flat_first is None else [flat_first]

    def callback(c_input_token_id, out_ptr, out_count):  # noqa: ANN001
        logits = decode_fn(int(c_input_token_id)).detach().cpu().contiguous().to(torch.float32).reshape(-1)
        if int(logits.numel()) != effective_vocab:
            return 21
        keepalive.append(logits)
        out_ptr[0] = ctypes.c_void_p(int(logits.data_ptr()))
        out_count[0] = int(logits.numel())
        return 0

    c_callback = _DS_DECODE_CALLBACK(callback)
    session._callback_keepalive.append(c_callback)
    out = torch.empty((effective_vocab,), dtype=torch.float32)
    code = session._lib.ds_forward_decode_f32(
        session._handle,
        ctypes.c_longlong(int(input_token_id)),
        c_callback,
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(effective_vocab),
    )
    session._tensor_keepalive.extend(keepalive)
    if code != 0:
        raise RuntimeError(f"ds_forward_decode_f32 failed with code {code}")
    return out


def ds_forward_prefill(
    session: "DeepSeekNativeSession",
    input_token_ids: list[int] | torch.Tensor,
    prefill_fn,
    *,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Run the DeepSeek prefill boundary and copy final logits through C."""
    token_tensor = torch.as_tensor([int(value) for value in input_token_ids], dtype=torch.int64).contiguous()
    if int(token_tensor.numel()) <= 0:
        raise ValueError("input_token_ids must not be empty")
    flat_first = None
    if vocab_size is None:
        first = prefill_fn([int(value) for value in token_tensor.tolist()]).detach().cpu().contiguous().to(torch.float32)
        flat_first = first.reshape(-1)
        effective_vocab = int(flat_first.numel())
    else:
        effective_vocab = int(vocab_size)
    if not native_ds_decode_available():
        if flat_first is None:
            flat_first = prefill_fn([int(value) for value in token_tensor.tolist()]).detach().cpu().contiguous().to(
                torch.float32
            ).reshape(-1)
        if int(flat_first.numel()) != effective_vocab:
            raise ValueError("vocab_size must match prefill_fn output size")
        return flat_first.clone()

    keepalive = [] if flat_first is None else [flat_first]

    def callback(tokens_ptr, n_tokens, out_ptr, out_count):  # noqa: ANN001
        count = int(n_tokens)
        array_type = ctypes.c_longlong * count
        tokens = [int(value) for value in array_type.from_address(int(tokens_ptr))]
        logits = prefill_fn(tokens).detach().cpu().contiguous().to(torch.float32).reshape(-1)
        if int(logits.numel()) != effective_vocab:
            return 21
        keepalive.append(logits)
        out_ptr[0] = ctypes.c_void_p(int(logits.data_ptr()))
        out_count[0] = int(logits.numel())
        return 0

    c_callback = _DS_PREFILL_CALLBACK(callback)
    session._callback_keepalive.append(c_callback)
    out = torch.empty((effective_vocab,), dtype=torch.float32)
    code = session._lib.ds_forward_prefill_f32(
        session._handle,
        ctypes.c_void_p(int(token_tensor.data_ptr())),
        ctypes.c_longlong(int(token_tensor.numel())),
        c_callback,
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(effective_vocab),
    )
    session._tensor_keepalive.extend(keepalive)
    if code != 0:
        raise RuntimeError(f"ds_forward_prefill_f32 failed with code {code}")
    return out


def ds_forward_verify(
    session: "DeepSeekNativeSession",
    candidate_token_ids: list[int] | torch.Tensor,
    verify_fn,
    *,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Run the DeepSeek speculative verify boundary and copy per-position logits through C."""
    token_tensor = torch.as_tensor([int(value) for value in candidate_token_ids], dtype=torch.int64).contiguous()
    if int(token_tensor.numel()) <= 0:
        raise ValueError("candidate_token_ids must not be empty")
    flat_first = None
    if vocab_size is None:
        first = verify_fn([int(value) for value in token_tensor.tolist()]).detach().cpu().contiguous().to(torch.float32)
        if first.ndim == 1:
            first = first.reshape(1, -1)
        flat_first = first.reshape(-1)
        effective_vocab = int(first.shape[-1])
    else:
        effective_vocab = int(vocab_size)
    expected_count = int(token_tensor.numel()) * effective_vocab
    if not native_ds_decode_available():
        if flat_first is None:
            first = verify_fn([int(value) for value in token_tensor.tolist()]).detach().cpu().contiguous().to(
                torch.float32
            )
            if first.ndim == 1:
                first = first.reshape(1, -1)
            flat_first = first.reshape(-1)
        if expected_count != int(flat_first.numel()):
            raise ValueError("verify_fn output must be [k, vocab_size]")
        return flat_first.reshape(int(token_tensor.numel()), effective_vocab).clone()

    keepalive = [] if flat_first is None else [flat_first]

    def callback(tokens_ptr, k, out_ptr, out_count):  # noqa: ANN001
        count = int(k)
        array_type = ctypes.c_longlong * count
        tokens = [int(value) for value in array_type.from_address(int(tokens_ptr))]
        logits = verify_fn(tokens).detach().cpu().contiguous().to(torch.float32)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        logits = logits.reshape(-1)
        if int(logits.numel()) != expected_count:
            return 21
        keepalive.append(logits)
        out_ptr[0] = ctypes.c_void_p(int(logits.data_ptr()))
        out_count[0] = int(logits.numel())
        return 0

    c_callback = _DS_VERIFY_CALLBACK(callback)
    session._callback_keepalive.append(c_callback)
    out = torch.empty((expected_count,), dtype=torch.float32)
    code = session._lib.ds_forward_verify_f32(
        session._handle,
        ctypes.c_void_p(int(token_tensor.data_ptr())),
        ctypes.c_longlong(int(token_tensor.numel())),
        c_callback,
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(effective_vocab),
    )
    session._tensor_keepalive.extend(keepalive)
    if code != 0:
        raise RuntimeError(f"ds_forward_verify_f32 failed with code {code}")
    return out.reshape(int(token_tensor.numel()), effective_vocab)


def ds_attention_layer_forward(
    session: "DeepSeekNativeSession",
    layer_idx: int,
    hidden: torch.Tensor,
    attention_fn,
) -> torch.Tensor:
    """Call the existing Python MLA attention path through the C-side DeepSeek bridge."""
    hidden_cpu = hidden.detach().cpu().contiguous().to(torch.float32)
    original_shape = tuple(hidden_cpu.shape)
    flat = hidden_cpu.reshape(-1, int(hidden_cpu.shape[-1]))
    if not native_ds_attention_bridge_available():
        return attention_fn(int(layer_idx), hidden_cpu).detach().cpu().contiguous().to(torch.float32)

    keepalive: list[torch.Tensor] = []

    def callback(c_layer_idx, hidden_ptr, batch, hidden_dim, out_ptr, out_count):  # noqa: ANN001
        count = int(batch) * int(hidden_dim)
        array_type = ctypes.c_float * count
        incoming = torch.tensor(array_type.from_address(int(hidden_ptr)), dtype=torch.float32).reshape(
            *original_shape
        )
        output = attention_fn(int(c_layer_idx), incoming).detach().cpu().contiguous().to(torch.float32)
        if tuple(output.shape) != original_shape:
            output = output.reshape(*original_shape)
        output_flat = output.reshape(-1)
        keepalive.append(output_flat)
        out_ptr[0] = ctypes.c_void_p(int(output_flat.data_ptr()))
        out_count[0] = int(output_flat.numel())
        return 0

    c_callback = _DS_ATTENTION_CALLBACK(callback)
    session._callback_keepalive.append(c_callback)
    out = torch.empty_like(flat)
    code = session._lib.ds_attention_layer_forward_f32(
        session._handle,
        ctypes.c_longlong(int(layer_idx)),
        ctypes.c_void_p(int(flat.data_ptr())),
        ctypes.c_longlong(int(flat.shape[0])),
        ctypes.c_longlong(int(flat.shape[1])),
        c_callback,
        ctypes.c_void_p(int(out.data_ptr())),
    )
    session._tensor_keepalive.extend(keepalive)
    if code != 0:
        raise RuntimeError(f"ds_attention_layer_forward_f32 failed with code {code}")
    return out.reshape(*original_shape)


def ds_moe_layer_forward(
    session: "DeepSeekNativeSession",
    expert_outputs: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    """Combine selected expert outputs through the DeepSeek C-side MoE dispatch boundary."""
    outputs_cpu = expert_outputs.detach().cpu().contiguous().to(torch.float32)
    if outputs_cpu.ndim == 2:
        outputs_cpu = outputs_cpu.unsqueeze(1)
    if outputs_cpu.ndim != 3:
        raise ValueError("expert_outputs must be [selected, batch, hidden]")
    weights_cpu = route_weights.detach().cpu().contiguous().reshape(-1).to(torch.float32)
    if int(weights_cpu.numel()) != int(outputs_cpu.shape[0]):
        raise ValueError("route weight count must match selected expert count")
    if not native_ds_moe_available():
        return _python_ds_moe_layer_forward(outputs_cpu, weights_cpu)
    out = torch.empty((int(outputs_cpu.shape[1]), int(outputs_cpu.shape[2])), dtype=torch.float32)
    code = session._lib.ds_moe_layer_forward_f32(
        session._handle,
        ctypes.c_void_p(int(outputs_cpu.data_ptr())),
        ctypes.c_void_p(int(weights_cpu.data_ptr())),
        ctypes.c_longlong(int(outputs_cpu.shape[0])),
        ctypes.c_longlong(int(outputs_cpu.shape[1])),
        ctypes.c_longlong(int(outputs_cpu.shape[2])),
        ctypes.c_void_p(int(out.data_ptr())),
    )
    if code != 0:
        raise RuntimeError(f"ds_moe_layer_forward_f32 failed with code {code}")
    return out


def ds_moe_layer_forward_fp8(
    session: "DeepSeekNativeSession",
    items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    hidden: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    """Run selected FP8 expert MLPs through PLM-2's native kernel, then combine in C."""
    if not native_ds_moe_available():
        expert_outputs = fp8_e4m3_block_mlp_many_f32(items, hidden)
        return _python_ds_moe_layer_forward(expert_outputs, route_weights)
    if not items:
        raise ValueError("items must not be empty")
    prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    first_gate = items[0][0].detach().cpu().contiguous().to(torch.uint8)
    intermediate_rows = int(first_gate.shape[0])
    hidden_cols = int(first_gate.shape[1])
    for gate, gate_scale, up, up_scale, down, down_scale in items:
        tensors = (
            gate.detach().cpu().contiguous().to(torch.uint8),
            gate_scale.detach().cpu().contiguous().to(torch.float32),
            up.detach().cpu().contiguous().to(torch.uint8),
            up_scale.detach().cpu().contiguous().to(torch.float32),
            down.detach().cpu().contiguous().to(torch.uint8),
            down_scale.detach().cpu().contiguous().to(torch.float32),
        )
        if tuple(tensors[0].shape) != (intermediate_rows, hidden_cols):
            raise ValueError("all gate weights must have matching shape")
        if tuple(tensors[2].shape) != (intermediate_rows, hidden_cols):
            raise ValueError("all up weights must have matching shape")
        if tuple(tensors[4].shape) != (hidden_cols, intermediate_rows):
            raise ValueError("all down weights must have matching shape")
        prepared.append(tensors)

    gate_scale_cols = (hidden_cols + 127) // 128
    down_scale_cols = (intermediate_rows + 127) // 128
    hidden_cpu = hidden.detach().cpu().contiguous().reshape(-1, int(hidden.shape[-1])).to(torch.float32)
    route_cpu = route_weights.detach().cpu().contiguous().reshape(-1).to(torch.float32)
    if int(hidden_cpu.shape[1]) != hidden_cols:
        raise ValueError("hidden input size does not match FP8 MLP input size")
    if int(route_cpu.numel()) != len(prepared):
        raise ValueError("route weight count must match selected experts")
    ptr_dtype = torch.int64
    gate_ptrs = torch.tensor([int(item[0].data_ptr()) for item in prepared], dtype=ptr_dtype)
    gate_scale_ptrs = torch.tensor([int(item[1].data_ptr()) for item in prepared], dtype=ptr_dtype)
    up_ptrs = torch.tensor([int(item[2].data_ptr()) for item in prepared], dtype=ptr_dtype)
    up_scale_ptrs = torch.tensor([int(item[3].data_ptr()) for item in prepared], dtype=ptr_dtype)
    down_ptrs = torch.tensor([int(item[4].data_ptr()) for item in prepared], dtype=ptr_dtype)
    down_scale_ptrs = torch.tensor([int(item[5].data_ptr()) for item in prepared], dtype=ptr_dtype)
    out = torch.empty((int(hidden_cpu.shape[0]), hidden_cols), dtype=torch.float32)
    code = session._lib.ds_moe_layer_forward_fp8_f32(
        session._handle,
        ctypes.c_void_p(int(gate_ptrs.data_ptr())),
        ctypes.c_void_p(int(gate_scale_ptrs.data_ptr())),
        ctypes.c_void_p(int(up_ptrs.data_ptr())),
        ctypes.c_void_p(int(up_scale_ptrs.data_ptr())),
        ctypes.c_void_p(int(down_ptrs.data_ptr())),
        ctypes.c_void_p(int(down_scale_ptrs.data_ptr())),
        ctypes.c_void_p(int(hidden_cpu.data_ptr())),
        ctypes.c_void_p(int(route_cpu.data_ptr())),
        ctypes.c_longlong(len(prepared)),
        ctypes.c_longlong(int(hidden_cpu.shape[0])),
        ctypes.c_longlong(intermediate_rows),
        ctypes.c_longlong(hidden_cols),
        ctypes.c_longlong(gate_scale_cols),
        ctypes.c_longlong(down_scale_cols),
        ctypes.c_void_p(int(out.data_ptr())),
    )
    if code != 0:
        raise RuntimeError(f"ds_moe_layer_forward_fp8_f32 failed with code {code}")
    return out


def _python_ds_moe_layer_forward(expert_outputs: torch.Tensor, route_weights: torch.Tensor) -> torch.Tensor:
    outputs_cpu = expert_outputs.detach().cpu().contiguous().to(torch.float32)
    if outputs_cpu.ndim == 2:
        outputs_cpu = outputs_cpu.unsqueeze(1)
    weights_cpu = route_weights.detach().cpu().contiguous().reshape(-1).to(torch.float32)
    return (outputs_cpu * weights_cpu.view(-1, 1, 1)).sum(dim=0)


class DeepSeekNativeSession:
    """ctypes wrapper for the DeepSeek C-side session and pack callbacks."""

    def __init__(
        self,
        *,
        num_layers: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        fp8_pack_callback: object | None = None,
        kv_session_handle: int | None = None,
    ):
        lib = _load_ds_forward_lib()
        if lib is None:
            reason = "disabled" if _ds_monolithic_disabled() else _DS_FORWARD_ERROR
            raise RuntimeError(f"DeepSeek native session is unavailable: {reason}")
        self._lib = lib
        self._callback_keepalive: list[object] = []
        self._tensor_keepalive: list[object] = []
        pack_ptr = 0 if fp8_pack_callback is None else int(ctypes.cast(fp8_pack_callback, ctypes.c_void_p).value or 0)
        kv_ptr = 0 if kv_session_handle is None else int(kv_session_handle)
        handle = lib.ds_session_create(
            ctypes.c_longlong(int(num_layers)),
            ctypes.c_longlong(int(hidden_dim)),
            ctypes.c_longlong(int(num_experts)),
            ctypes.c_longlong(int(top_k)),
            ctypes.c_void_p(pack_ptr),
            ctypes.c_void_p(kv_ptr),
        )
        if not handle:
            raise RuntimeError("ds_session_create returned a null handle")
        self._handle = ctypes.c_void_p(handle)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.ds_session_destroy(self._handle)
            self._handle = None
            self._callback_keepalive.clear()
            self._tensor_keepalive.clear()

    def __enter__(self) -> "DeepSeekNativeSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def tensor_callback(fn):
        return _DS_TENSOR_CALLBACK(fn)

    def register_layer(
        self,
        *,
        layer_idx: int,
        layer_kind: int,
        weight_role: int,
        fp8_callback,
        scale_callback,
    ) -> None:
        fp8_cb = fp8_callback if isinstance(fp8_callback, _DS_TENSOR_CALLBACK) else _DS_TENSOR_CALLBACK(fp8_callback)
        scale_cb = (
            scale_callback
            if isinstance(scale_callback, _DS_TENSOR_CALLBACK)
            else _DS_TENSOR_CALLBACK(scale_callback)
        )
        self._callback_keepalive.extend([fp8_cb, scale_cb])
        code = self._lib.ds_session_register_layer(
            self._handle,
            ctypes.c_longlong(int(layer_idx)),
            ctypes.c_longlong(int(layer_kind)),
            ctypes.c_longlong(int(weight_role)),
            fp8_cb,
            scale_cb,
        )
        if code != 0:
            raise RuntimeError(f"ds_session_register_layer failed with code {code}")

    def keep_tensor_alive(self, tensor: object) -> None:
        self._tensor_keepalive.append(tensor)

    def monolithic_call_count(self) -> int:
        return int(self._lib.ds_monolithic_call_counter(self._handle))

    def callback_invocation_count(self) -> int:
        return int(self._lib.ds_callback_invocation_count(self._handle))

    def expert_invocation_count(self) -> int:
        return int(self._lib.ds_expert_invocation_count(self._handle))

    def attention_invocation_count(self) -> int:
        return int(self._lib.ds_attention_invocation_count(self._handle))

    def fused_attention_invocation_count(self) -> int:
        if not hasattr(self._lib, "ds_fused_attention_invocation_count"):
            return 0
        return int(self._lib.ds_fused_attention_invocation_count(self._handle))

    def layers_executed_count(self) -> int:
        return int(self._lib.ds_layers_executed_count(self._handle))

    def registered_layer_count(self) -> int:
        return int(self._lib.ds_registered_layer_count(self._handle))

    def registered_fp8_nbytes(self, index: int) -> int:
        return int(self._lib.ds_registered_fp8_nbytes(self._handle, ctypes.c_longlong(int(index))))

    def registered_scale_nbytes(self, index: int) -> int:
        return int(self._lib.ds_registered_scale_nbytes(self._handle, ctypes.c_longlong(int(index))))


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


def fp8_e4m3_dequant_to_fp16(fp8_weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    lib = _load_fp8_dequant_lib()
    if lib is None:
        reason = "disabled" if _native_fp8_dequant_disabled() else _FP8_DEQUANT_ERROR
        raise RuntimeError(f"Native FP8 dequant is unavailable: {reason}")
    if fp8_weight.ndim != 2:
        raise ValueError("fp8_weight must have shape [rows, cols]")
    if scale_inv.ndim != 2:
        raise ValueError("scale_inv must have shape [scale_rows, scale_cols]")
    weight_cpu = fp8_weight.detach().cpu().contiguous().to(torch.uint8)
    scale_cpu = scale_inv.detach().cpu().contiguous().to(torch.float32)
    rows = int(weight_cpu.shape[0])
    cols = int(weight_cpu.shape[1])
    expected_scale_rows = (rows + 127) // 128
    expected_scale_cols = (cols + 127) // 128
    if tuple(scale_cpu.shape) != (expected_scale_rows, expected_scale_cols):
        raise ValueError("scale_inv shape does not match FP8 128x128 block layout")

    out = torch.empty((rows, cols), dtype=torch.float16)
    code = lib.fp8_e4m3_dequant_to_fp16(
        ctypes.c_void_p(int(weight_cpu.data_ptr())),
        ctypes.c_void_p(int(scale_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(rows),
        ctypes.c_longlong(cols),
        ctypes.c_longlong(expected_scale_cols),
    )
    if code != 0:
        raise RuntimeError(f"fp8_e4m3_dequant_to_fp16 failed with code {code}")
    return out


def fp8_e4m3_dequant_to_bf16(fp8_weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    lib = _load_fp8_dequant_lib()
    if lib is None or not hasattr(lib, "fp8_e4m3_dequant_to_bf16"):
        reason = "disabled" if _native_fp8_dequant_disabled() else _FP8_DEQUANT_ERROR
        raise RuntimeError(f"Native FP8 BF16 dequant is unavailable: {reason}")
    if fp8_weight.ndim != 2:
        raise ValueError("fp8_weight must have shape [rows, cols]")
    if scale_inv.ndim != 2:
        raise ValueError("scale_inv must have shape [scale_rows, scale_cols]")
    weight_cpu = fp8_weight.detach().cpu().contiguous().to(torch.uint8)
    scale_cpu = scale_inv.detach().cpu().contiguous().to(torch.float32)
    rows = int(weight_cpu.shape[0])
    cols = int(weight_cpu.shape[1])
    expected_scale_rows = (rows + 127) // 128
    expected_scale_cols = (cols + 127) // 128
    if tuple(scale_cpu.shape) != (expected_scale_rows, expected_scale_cols):
        raise ValueError("scale_inv shape does not match FP8 128x128 block layout")

    out = torch.empty((rows, cols), dtype=torch.bfloat16)
    code = lib.fp8_e4m3_dequant_to_bf16(
        ctypes.c_void_p(int(weight_cpu.data_ptr())),
        ctypes.c_void_p(int(scale_cpu.data_ptr())),
        ctypes.c_void_p(int(out.data_ptr())),
        ctypes.c_longlong(rows),
        ctypes.c_longlong(cols),
        ctypes.c_longlong(expected_scale_cols),
    )
    if code != 0:
        raise RuntimeError(f"fp8_e4m3_dequant_to_bf16 failed with code {code}")
    return out


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
