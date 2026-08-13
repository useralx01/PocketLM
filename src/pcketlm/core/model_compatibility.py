"""Truthful model compatibility profiles and live readiness evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.model_families import normalize_family_key
from pcketlm.core.registry.models import ModelRecord
from pcketlm.core.registry.repository import load_model_registry
from pcketlm.core.runtime.gguf_backend import find_gguf_model_files, llama_server_path
from pcketlm.core.storage.paths import state_root


@dataclass(frozen=True, slots=True)
class CompatibilityProfile:
    key: str
    label: str
    family: str
    capability: str
    backend: str
    implementation_status: str
    chat_status: str
    target: str
    evidence: str
    default_path: bool = False

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "capability": self.capability,
            "backend": self.backend,
            "implementation_status": self.implementation_status,
            "chat_status": self.chat_status,
            "target": self.target,
            "evidence": self.evidence,
            "default_path": self.default_path,
        }


@dataclass(slots=True)
class CompatibilityResult:
    profile: CompatibilityProfile
    status: str
    local_ready: bool
    source_present: bool
    artifact_present: bool
    blockers: list[str] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            **self.profile.to_dict(),
            "status": self.status,
            "local_ready": self.local_ready,
            "source_present": self.source_present,
            "artifact_present": self.artifact_present,
            "blockers": list(self.blockers),
            "model_ids": list(self.model_ids),
        }


COMPATIBILITY_PROFILES: tuple[CompatibilityProfile, ...] = (
    CompatibilityProfile("qwen", "Qwen dense", "qwen", "chat", "llama.cpp / GGUF", "Proven", "Proven", "Qwen 2.5 / Qwen 3 text", "Measured local GGUF and direct-runtime proofs exist.", True),
    CompatibilityProfile("qwen-moe", "Qwen MoE", "qwen-moe", "chat", "Pocket paged MoE", "Proven", "Proven", "Qwen3 MoE", "Bit-exact tiny oracle and local runtime proofs exist."),
    CompatibilityProfile("mixtral", "Mixtral", "mixtral", "chat", "Pocket paged MoE", "Experimental", "Experimental", "Mixtral MoE", "Bit-exact tiny Mixtral oracle passes; full customer path remains experimental."),
    CompatibilityProfile("deepseek-v3", "DeepSeek V3", "deepseek", "chat", "Pocket FP8 exact", "Proven", "Proven", "DeepSeek V3 FP8", "Exact CPU path has measured visible-token proof; CUDA path has synthetic validation."),
    CompatibilityProfile("kimi-k2", "Kimi K2", "kimi", "chat", "llama.cpp / GGUF", "Experimental", "Experimental", "moonshotai/Kimi-K2-Instruct GGUF", "llama.cpp Kimi K2 template route and Pocket request contract are fixture-verified.", True),
    CompatibilityProfile("gemma-3", "Gemma 3", "gemma", "chat", "llama.cpp / GGUF", "Experimental", "Experimental", "Gemma 3 text GGUF", "llama.cpp Gemma template route and Pocket request contract are fixture-verified.", True),
    CompatibilityProfile("kronos", "Kronos", "kronos", "forecast", "Kronos CPU adapter", "Experimental", "Unsupported", "NeoQuasar/Kronos-small + tokenizer", "Forecast adapter contract is fixture-verified; Kronos does not accept chat prompts."),
)


def profile_for_family(family: str | None, model_id: str = "") -> CompatibilityProfile | None:
    normalized_id = model_id.strip().lower().replace("_", "-")
    if "deepseek" in normalized_id and "v3" in normalized_id:
        return next(profile for profile in COMPATIBILITY_PROFILES if profile.key == "deepseek-v3")
    for marker, key in (
        ("kronos", "kronos"),
        ("kronk", "kronos"),
        ("kimi", "kimi-k2"),
        ("gemma", "gemma-3"),
        ("mixtral", "mixtral"),
    ):
        if marker in normalized_id:
            return next(profile for profile in COMPATIBILITY_PROFILES if profile.key == key)
    if "qwen" in normalized_id:
        key = "qwen-moe" if "moe" in normalized_id or "a3b" in normalized_id else "qwen"
        return next(profile for profile in COMPATIBILITY_PROFILES if profile.key == key)
    family_key = normalize_family_key(family)
    return next((profile for profile in COMPATIBILITY_PROFILES if profile.family == family_key), None)


def _record_files(record: ModelRecord) -> tuple[bool, bool]:
    source_present = record.source_path.exists() and any(record.source_path.rglob("*"))
    artifact_present = any(path.exists() for path in record.artifact_paths)
    if not artifact_present:
        artifact_present = any(record.source_path.rglob("*.gguf")) if record.source_path.exists() else False
    if not artifact_present:
        artifact_present = bool(find_gguf_model_files(record.model_id))
    return source_present, artifact_present


def build_compatibility_matrix(records: dict[str, ModelRecord] | None = None) -> list[CompatibilityResult]:
    """Build one live support row per target family."""
    registry = load_model_registry() if records is None else records
    results: list[CompatibilityResult] = []
    runtime_present = llama_server_path().exists()

    for profile in COMPATIBILITY_PROFILES:
        matching = [
            record for record in registry.values()
            if profile_for_family(record.family, record.model_id) == profile
        ]
        source_present = False
        artifact_present = False
        structurally_runnable = False
        for record in matching:
            record_source, record_artifact = _record_files(record)
            source_present = source_present or record_source
            artifact_present = artifact_present or record_artifact
            structurally_runnable = structurally_runnable or (record.runnable and record_source)

        blockers: list[str] = []
        if profile.backend == "llama.cpp / GGUF":
            local_ready = runtime_present and artifact_present
            if not artifact_present:
                blockers.append("No local GGUF artifact is installed.")
            if not runtime_present:
                blockers.append("llama-server.exe is not installed.")
        elif profile.family == "kronos":
            local_ready = structurally_runnable
            if not source_present:
                blockers.append("Kronos model and tokenizer sources are not installed.")
        else:
            local_ready = structurally_runnable or artifact_present
            if not local_ready:
                blockers.append("No complete local source or runnable artifact is installed.")

        status = profile.implementation_status if local_ready else "Missing"
        results.append(
            CompatibilityResult(
                profile=profile,
                status=status,
                local_ready=local_ready,
                source_present=source_present,
                artifact_present=artifact_present,
                blockers=blockers,
                model_ids=sorted(record.model_id for record in matching),
            )
        )
    return results


def compatibility_proof_path() -> Path:
    return state_root() / "model-compatibility-proof.json"
