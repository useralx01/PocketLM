"""Privacy-preserving installation health reports for PocketLM."""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pcketlm.core.model_compatibility import build_compatibility_matrix
from pcketlm.core.registry.catalog import build_model_catalog
from pcketlm.core.runtime.gguf_backend import llama_server_path
from pcketlm.core.storage.paths import state_root


SUPERVISOR_SCHEMA = "pocketlm.installation-health.v1"


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    total_bytes: int | None
    available_bytes: int | None


def _supervisor_root(root: Path | None = None) -> Path:
    return (state_root() if root is None else Path(root)) / "supervisor"


def installation_id(root: Path | None = None) -> str:
    """Return a stable random ID without using machine-identifying hardware values."""
    path = _supervisor_root(root) / "installation.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = str(payload.get("installation_id") or "")
        uuid.UUID(value)
        return value
    except (OSError, ValueError, json.JSONDecodeError):
        value = str(uuid.uuid4())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"installation_id": value}, indent=2), encoding="utf-8")
        return value


def _memory_snapshot() -> MemorySnapshot:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return MemorySnapshot(int(status.total_physical), int(status.available_physical))
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return MemorySnapshot(
            page_size * int(os.sysconf("SC_PHYS_PAGES")),
            page_size * int(os.sysconf("SC_AVPHYS_PAGES")),
        )
    except (AttributeError, OSError, ValueError):
        return MemorySnapshot(None, None)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "device_count": 0, "devices": []}
    try:
        import torch

        if not torch.cuda.is_available():
            return result
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "name": str(properties.name),
                    "memory_gb": round(int(properties.total_memory) / (1024**3), 2),
                    "capability": f"{properties.major}.{properties.minor}",
                }
            )
        return {"available": True, "device_count": len(devices), "devices": devices}
    except Exception as exc:
        result["probe_error"] = type(exc).__name__
        return result


def _gb(value: int | None) -> float | None:
    return None if value is None else round(value / (1024**3), 2)


def build_supervisor_report(*, root: Path | None = None) -> dict[str, Any]:
    """Build a sanitized report suitable for UI, CI artifacts, or sharing."""
    memory = _memory_snapshot()
    catalog = build_model_catalog()
    compatibility = build_compatibility_matrix()
    ready_models = [entry for entry in catalog if entry.runnable]
    supported_families = [entry for entry in compatibility if entry.local_ready]
    python_supported = sys.version_info >= (3, 12)
    package_installed = _package_version("pcketlm") is not None
    runtime_binary = llama_server_path().exists()

    blockers: list[str] = []
    warnings: list[str] = []
    if not python_supported:
        blockers.append("Python 3.12 or newer is required.")
    if not package_installed:
        warnings.append("PocketLM is running from source instead of an installed package.")
    if not ready_models:
        warnings.append("No complete local model is currently runnable.")
    if not runtime_binary:
        warnings.append("The optional llama.cpp server runtime is not installed.")

    status = "blocked" if blockers else "ready" if ready_models else "partial"
    return {
        "schema": SUPERVISOR_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "installation_id": installation_id(root),
        "status": status,
        "privacy": {
            "local_only": True,
            "outbound_telemetry": False,
            "contains_paths": False,
            "contains_prompts": False,
        },
        "system": {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "cpu": platform.processor() or "Unknown",
            "logical_cpu_count": os.cpu_count(),
            "memory_total_gb": _gb(memory.total_bytes),
            "memory_available_gb": _gb(memory.available_bytes),
            "python_version": platform.python_version(),
        },
        "gpu": _cuda_snapshot(),
        "software": {
            "pocketlm_version": _package_version("pcketlm") or "source",
            "torch_version": _package_version("torch"),
            "transformers_version": _package_version("transformers"),
            "llama_server_installed": runtime_binary,
        },
        "models": {
            "registered": len(catalog),
            "runnable": len(ready_models),
            "entries": [
                {
                    "model_id": entry.model_id,
                    "family": entry.family,
                    "capability": entry.capability,
                    "backend": entry.backend,
                    "source_status": entry.source_status,
                    "runnable": entry.runnable,
                }
                for entry in catalog
            ],
        },
        "compatibility": [
            {
                "key": entry.profile.key,
                "label": entry.profile.label,
                "capability": entry.profile.capability,
                "implementation_status": entry.profile.implementation_status,
                "local_ready": entry.local_ready,
                "blockers": list(entry.blockers),
            }
            for entry in compatibility
        ],
        "summary": {
            "ready_family_count": len(supported_families),
            "known_family_count": len(compatibility),
            "runnable_model_count": len(ready_models),
            "blockers": blockers,
            "warnings": warnings,
        },
    }


def save_supervisor_report(
    report: dict[str, Any] | None = None,
    *,
    output: Path | None = None,
    root: Path | None = None,
) -> Path:
    """Persist a report as a proof artifact and return its path."""
    payload = build_supervisor_report(root=root) if report is None else report
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = _supervisor_root(root) / "reports" / f"{stamp}-{payload['installation_id'][:8]}.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output
