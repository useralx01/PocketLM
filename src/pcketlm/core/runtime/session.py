"""Runtime session helpers."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RuntimeSession:
    """Minimal runtime session state."""

    session_id: str
    model_id: str
    source_path: Path
    profile_id: str | None = None
    status: str = "created"
    source_ready: bool = False
    can_attempt_load: bool = False
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        """Serialize the runtime session."""
        return {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "source_path": str(self.source_path),
            "profile_id": self.profile_id,
            "status": self.status,
            "source_ready": self.source_ready,
            "can_attempt_load": self.can_attempt_load,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "summary": self.summary,
        }
