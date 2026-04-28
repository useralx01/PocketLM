"""Saved optimization profile records."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OptimizationProfile:
    """A saved per-model profile target before optimized artifacts exist."""

    profile_id: str
    model_id: str
    label: str
    runtime_mode: str
    summary: str = ""
    template_id: str | None = None
    capability_priorities: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    artifact_ready: bool = False
    profile_path: str | None = None

    def to_dict(self) -> dict:
        """Serialize the saved profile."""
        return {
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "label": self.label,
            "runtime_mode": self.runtime_mode,
            "summary": self.summary,
            "template_id": self.template_id,
            "capability_priorities": list(self.capability_priorities),
            "tradeoffs": list(self.tradeoffs),
            "settings": dict(self.settings),
            "artifact_ready": self.artifact_ready,
            "profile_path": self.profile_path,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OptimizationProfile":
        """Deserialize a saved profile."""
        return cls(
            profile_id=str(payload.get("profile_id") or ""),
            model_id=str(payload.get("model_id") or ""),
            label=str(payload.get("label") or ""),
            runtime_mode=str(payload.get("runtime_mode") or "Quality"),
            summary=str(payload.get("summary") or ""),
            template_id=None if payload.get("template_id") is None else str(payload.get("template_id")),
            capability_priorities=[str(value) for value in payload.get("capability_priorities", [])],
            tradeoffs=[str(value) for value in payload.get("tradeoffs", [])],
            settings=dict(payload.get("settings") or {}),
            artifact_ready=bool(payload.get("artifact_ready", False)),
            profile_path=None if payload.get("profile_path") is None else str(payload.get("profile_path")),
        )
