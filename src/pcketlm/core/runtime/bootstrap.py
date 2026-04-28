"""Runtime bootstrap and preflight helpers."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from pcketlm.core.runtime.session import RuntimeSession
from pcketlm.core.runtime.source import RuntimeSourceDescriptor, describe_runtime_source


@dataclass(slots=True)
class RuntimeDependencyStatus:
    """Runtime dependency availability for the first loader pass."""

    transformers: bool
    tokenizers: bool
    safetensors: bool
    torch: bool

    def to_dict(self) -> dict:
        """Serialize dependency availability."""
        return {
            "transformers": self.transformers,
            "tokenizers": self.tokenizers,
            "safetensors": self.safetensors,
            "torch": self.torch,
        }


@dataclass(slots=True)
class RuntimeComponentCheck:
    """Result of checking loadable runtime components."""

    config_loadable: bool = False
    tokenizer_loadable: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize component check status."""
        return {
            "config_loadable": self.config_loadable,
            "tokenizer_loadable": self.tokenizer_loadable,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RuntimeBootstrapResult:
    """Preflight result for a runtime launch attempt."""

    model_id: str
    model_dir: Path
    profile_id: str | None
    source: RuntimeSourceDescriptor
    dependencies: RuntimeDependencyStatus
    components: RuntimeComponentCheck
    can_attempt_load: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plain_english_summary: str = ""

    def to_dict(self) -> dict:
        """Serialize the bootstrap result."""
        return {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "profile_id": self.profile_id,
            "source": self.source.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "components": self.components.to_dict(),
            "can_attempt_load": self.can_attempt_load,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "plain_english_summary": self.plain_english_summary,
        }


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def probe_runtime_dependencies() -> RuntimeDependencyStatus:
    """Probe dependency availability for a first runtime attempt."""
    return RuntimeDependencyStatus(
        transformers=_module_available("transformers"),
        tokenizers=_module_available("tokenizers"),
        safetensors=_module_available("safetensors"),
        torch=_module_available("torch"),
    )


def _load_runtime_components(model_dir: Path, dependencies: RuntimeDependencyStatus) -> RuntimeComponentCheck:
    check = RuntimeComponentCheck()

    if not dependencies.transformers:
        check.warnings.append("`transformers` is not installed, so config/tokenizer loading was not checked.")
        return check

    try:
        from transformers import AutoConfig, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive import guard
        check.warnings.append(f"Transformers import failed during runtime preflight: {exc}")
        return check

    try:
        AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        check.config_loadable = True
    except Exception as exc:
        check.warnings.append(f"Config preflight failed: {exc}")

    try:
        AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        check.tokenizer_loadable = True
    except Exception as exc:
        check.warnings.append(f"Tokenizer preflight failed: {exc}")

    return check


def _build_bootstrap_summary(can_attempt_load: bool, blockers: list[str], source: RuntimeSourceDescriptor) -> str:
    if can_attempt_load:
        return "The runtime preflight passed, so pcketlm can attempt the first loader pass."
    if blockers:
        return "The runtime preflight is blocked right now: " + "; ".join(blockers)
    return source.plain_english_summary


def build_runtime_bootstrap(model_id: str, model_dir: Path, profile_id: str | None = None) -> RuntimeBootstrapResult:
    """Build a runtime preflight result for one model source folder."""
    source = describe_runtime_source(model_dir)
    dependencies = probe_runtime_dependencies()
    components = _load_runtime_components(model_dir, dependencies)

    blockers: list[str] = []
    warnings = list(source.warnings)
    warnings.extend(components.warnings)

    if not source.ready:
        blockers.append(source.plain_english_summary)
    if not dependencies.transformers:
        blockers.append("`transformers` is not installed.")
    if not dependencies.tokenizers:
        blockers.append("`tokenizers` is not installed.")
    if not dependencies.safetensors:
        blockers.append("`safetensors` is not installed.")
    if not dependencies.torch:
        blockers.append("`torch` is not installed yet, so model weights cannot be loaded.")
    if dependencies.transformers and not components.config_loadable:
        blockers.append("The config could not be loaded through the runtime preflight.")
    if dependencies.transformers and not components.tokenizer_loadable:
        blockers.append("The tokenizer could not be loaded through the runtime preflight.")

    can_attempt_load = source.ready and not blockers
    return RuntimeBootstrapResult(
        model_id=model_id,
        model_dir=model_dir,
        profile_id=profile_id,
        source=source,
        dependencies=dependencies,
        components=components,
        can_attempt_load=can_attempt_load,
        blockers=blockers,
        warnings=warnings,
        plain_english_summary=_build_bootstrap_summary(can_attempt_load, blockers, source),
    )


def create_runtime_session(model_id: str, model_dir: Path, profile_id: str | None = None) -> RuntimeSession:
    """Create a runtime session from a bootstrap result."""
    bootstrap = build_runtime_bootstrap(model_id, model_dir, profile_id=profile_id)
    session_status = "ready" if bootstrap.can_attempt_load else "blocked"
    return RuntimeSession(
        session_id=f"session-{uuid4().hex[:8]}",
        model_id=model_id,
        source_path=model_dir,
        profile_id=profile_id,
        status=session_status,
        source_ready=bootstrap.source.ready,
        can_attempt_load=bootstrap.can_attempt_load,
        blockers=list(bootstrap.blockers),
        warnings=list(bootstrap.warnings),
        summary=bootstrap.plain_english_summary,
    )
