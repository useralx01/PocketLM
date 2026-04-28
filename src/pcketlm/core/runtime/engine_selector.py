"""Runtime engine selection diagnostics."""

from __future__ import annotations

import importlib.util
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch

from pcketlm.core.runtime.load_attempt import _memory_snapshot
from pcketlm.core.runtime.gguf_backend import build_gguf_backend_status
from pcketlm.core.storage.paths import artifacts_root, original_model_root


MIN_CUDA_ACCELERATION_VRAM_GB = 8.0
MIN_GGUF_CONVERSION_DISK_GB = 8.0


@dataclass(slots=True)
class RuntimeBackendCandidate:
    """One possible backend path for Pocket LLM."""

    backend_id: str
    label: str
    status: str
    available_now: bool
    implemented_now: bool
    recommended: bool = False
    package_available: bool = False
    hardware_available: bool = False
    requires_model_conversion: bool = False
    summary: str = ""
    blockers: list[str] = field(default_factory=list)
    next_action: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "backend_id": self.backend_id,
            "label": self.label,
            "status": self.status,
            "available_now": self.available_now,
            "implemented_now": self.implemented_now,
            "recommended": self.recommended,
            "package_available": self.package_available,
            "hardware_available": self.hardware_available,
            "requires_model_conversion": self.requires_model_conversion,
            "summary": self.summary,
            "blockers": list(self.blockers),
            "next_action": self.next_action,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class RuntimeBackendReport:
    """Machine-local backend capability report."""

    model_id: str
    platform: str
    system_free_gb: float | None
    recommended_backend_id: str
    recommended_summary: str
    candidates: list[RuntimeBackendCandidate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "platform": self.platform,
            "system_free_gb": self.system_free_gb,
            "recommended_backend_id": self.recommended_backend_id,
            "recommended_summary": self.recommended_summary,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(slots=True)
class RuntimeEngineDecision:
    """Plain status for the currently selected runtime engine."""

    model_id: str
    selected_engine: str
    selected_backend: str
    hardware_acceleration_available: bool
    cpu_available: bool = True
    system_free_gb: float | None = None
    cuda_device_count: int = 0
    cuda_devices: list[dict] = field(default_factory=list)
    recommended_backend_id: str = "direct-cpu"
    blockers: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "selected_engine": self.selected_engine,
            "selected_backend": self.selected_backend,
            "hardware_acceleration_available": self.hardware_acceleration_available,
            "cpu_available": self.cpu_available,
            "system_free_gb": self.system_free_gb,
            "cuda_device_count": self.cuda_device_count,
            "cuda_devices": list(self.cuda_devices),
            "recommended_backend_id": self.recommended_backend_id,
            "blockers": list(self.blockers),
            "summary": self.summary,
        }


def _cuda_devices() -> list[dict]:
    if not torch.cuda.is_available():
        return []
    devices: list[dict] = []
    for index in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_vram_gb": round(int(props.total_memory) / (1024**3), 2),
                    "backend": "cuda",
                }
            )
        except Exception:
            continue
    return devices


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _disk_free_gb(path: Path) -> float | None:
    try:
        target = path if path.exists() else path.parent
        return round(int(shutil.disk_usage(target).free) / (1024**3), 2)
    except OSError:
        return None


def _gguf_files(model_id: str) -> list[Path]:
    roots = [original_model_root(model_id), artifacts_root(model_id)]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.glob("*.gguf")))
    return files


def build_runtime_backend_report(model_id: str) -> RuntimeBackendReport:
    """Build a local report for CPU, GPU, and GGUF-style backend options."""
    try:
        memory = _memory_snapshot()
        free_gb = memory.free_gb
    except Exception:
        free_gb = None

    cuda_devices = _cuda_devices()
    largest_cuda_vram = max((float(device.get("total_vram_gb", 0.0)) for device in cuda_devices), default=0.0)
    torch_cuda_ready = bool(cuda_devices)
    directml_package = _module_available("torch_directml")
    gguf_status = build_gguf_backend_status(model_id)
    llama_cpp_package = gguf_status.package_available
    gguf_files = [model_file.path for model_file in gguf_status.model_files]
    conversion_disk_free_gb = _disk_free_gb(artifacts_root(model_id))
    conversion_disk_ready = conversion_disk_free_gb is not None and conversion_disk_free_gb >= MIN_GGUF_CONVERSION_DISK_GB

    candidates = [
        RuntimeBackendCandidate(
            backend_id="direct-cpu",
            label="Direct CPU",
            status="active",
            available_now=True,
            implemented_now=True,
            package_available=True,
            hardware_available=True,
            summary="Current Pocket LLM Python/Torch CPU runtime. Stable and used as the fallback path.",
            next_action="Keep as the safe baseline while faster engines are tested.",
            details={"torch_version": torch.__version__},
        ),
        RuntimeBackendCandidate(
            backend_id="cuda",
            label="CUDA GPU",
            status="planned" if largest_cuda_vram >= MIN_CUDA_ACCELERATION_VRAM_GB else "blocked",
            available_now=torch_cuda_ready and largest_cuda_vram >= MIN_CUDA_ACCELERATION_VRAM_GB,
            implemented_now=False,
            package_available=torch_cuda_ready,
            hardware_available=bool(cuda_devices),
            summary=(
                "CUDA hardware is visible with enough VRAM, but Pocket LLM has not implemented CUDA execution yet."
                if largest_cuda_vram >= MIN_CUDA_ACCELERATION_VRAM_GB
                else "CUDA is not a usable Queen backend on this runtime right now."
            ),
            blockers=(
                ["CUDA execution is not implemented in Pocket LLM yet."]
                if largest_cuda_vram >= MIN_CUDA_ACCELERATION_VRAM_GB
                else ["No CUDA device with enough VRAM is visible to the current Torch runtime."]
            ),
            next_action="Add a CUDA execution backend only after a GPU with enough VRAM is visible.",
            details={"cuda_devices": cuda_devices, "minimum_vram_gb": MIN_CUDA_ACCELERATION_VRAM_GB},
        ),
        RuntimeBackendCandidate(
            backend_id="directml",
            label="DirectML",
            status="planned" if directml_package else "missing-package",
            available_now=directml_package,
            implemented_now=False,
            package_available=directml_package,
            hardware_available=platform.system().lower() == "windows",
            summary=(
                "DirectML package is installed, but Pocket LLM has not wired a DirectML execution path yet."
                if directml_package
                else "DirectML could be useful on Windows GPUs, but the required package is not installed."
            ),
            blockers=[] if directml_package else ["Python package torch-directml is not installed."],
            next_action="Prototype DirectML only after package install and a small tensor smoke prove the adapter is usable.",
            details={"platform": platform.system()},
        ),
        RuntimeBackendCandidate(
            backend_id="llama-cpp-gguf",
            label="llama.cpp / GGUF",
            status="ready-to-prototype" if llama_cpp_package and gguf_files else "conversion-needed",
            available_now=llama_cpp_package and bool(gguf_files),
            implemented_now=False,
            package_available=llama_cpp_package,
            hardware_available=True,
            requires_model_conversion=not bool(gguf_files),
            summary=(
                "A GGUF runtime package and GGUF model file are present, so this is ready for a local prototype."
                if llama_cpp_package and gguf_files
                else "This is the best next speed target for weak hardware, but it needs a GGUF runtime package and/or converted model file."
            ),
            blockers=[
                blocker
                for blocker in [
                    None if llama_cpp_package else "Python package llama-cpp-python is not installed.",
                    None if gguf_files else "No GGUF model file is present under the model or artifact folders.",
                    None if conversion_disk_ready else "Free disk space may be too low for a safe conversion artifact.",
                ]
                if blocker is not None
            ],
            next_action="Prepare a GGUF conversion/prototype only after the user approves the conversion/download cost.",
            details={
                "gguf_files": [str(path) for path in gguf_files],
                "conversion_disk_free_gb": conversion_disk_free_gb,
                "minimum_conversion_disk_gb": MIN_GGUF_CONVERSION_DISK_GB,
                "adapter_ready": gguf_status.ready,
            },
        ),
    ]

    recommended_id = "llama-cpp-gguf"
    recommended_summary = (
        "The next serious speed prototype should be llama.cpp/GGUF, because current safetensors repacking gave only small gains and this path is proven for weak hardware."
    )
    if largest_cuda_vram >= MIN_CUDA_ACCELERATION_VRAM_GB:
        recommended_id = "cuda"
        recommended_summary = "CUDA is visible with enough VRAM, so a CUDA prototype would be the most direct acceleration target after the CPU fallback."
    elif directml_package:
        recommended_id = "directml"
        recommended_summary = "DirectML is installed on Windows, so it is the next GPU-style prototype target before asking for a model conversion."

    for candidate in candidates:
        candidate.recommended = candidate.backend_id == recommended_id

    return RuntimeBackendReport(
        model_id=model_id,
        platform=platform.platform(),
        system_free_gb=free_gb,
        recommended_backend_id=recommended_id,
        recommended_summary=recommended_summary,
        candidates=candidates,
    )


def select_runtime_engine(model_id: str) -> RuntimeEngineDecision:
    """Choose the safe runtime engine for the current machine."""
    report = build_runtime_backend_report(model_id)
    free_gb = report.system_free_gb

    cuda_devices = _cuda_devices()
    if cuda_devices:
        largest = max(cuda_devices, key=lambda item: float(item.get("total_vram_gb", 0.0)))
        vram_gb = float(largest.get("total_vram_gb", 0.0))
        if vram_gb >= 8.0:
            return RuntimeEngineDecision(
                model_id=model_id,
                selected_engine="planned-gpu",
                selected_backend="cuda",
                hardware_acceleration_available=True,
                system_free_gb=free_gb,
                cuda_device_count=len(cuda_devices),
                cuda_devices=cuda_devices,
                recommended_backend_id=report.recommended_backend_id,
                blockers=["CUDA backend execution is detected but not implemented in Pocket LLM direct runtime yet."],
                summary="A CUDA GPU is visible, but Pocket LLM is still using the safe direct CPU runtime until GPU execution is implemented.",
            )
        return RuntimeEngineDecision(
            model_id=model_id,
            selected_engine="direct-cpu",
            selected_backend="torch-cpu",
            hardware_acceleration_available=True,
            system_free_gb=free_gb,
            cuda_device_count=len(cuda_devices),
            cuda_devices=cuda_devices,
            recommended_backend_id=report.recommended_backend_id,
            blockers=[f"Largest CUDA GPU has only {vram_gb} GB VRAM, below the current 8 GB acceleration target."],
            summary="Pocket LLM selected the direct CPU runtime because visible GPU memory is too small for the current Queen path.",
        )

    return RuntimeEngineDecision(
        model_id=model_id,
        selected_engine="direct-cpu",
        selected_backend="torch-cpu",
        hardware_acceleration_available=False,
        system_free_gb=free_gb,
        cuda_device_count=0,
        cuda_devices=[],
        recommended_backend_id=report.recommended_backend_id,
        blockers=["No CUDA GPU backend is available to the current Python runtime."],
        summary="Pocket LLM selected the direct CPU runtime because no supported GPU backend is available yet.",
    )
