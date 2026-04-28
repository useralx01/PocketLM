"""Built-in optimization profile templates."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ProfileTemplate:
    """A safe high-level profile target before artifact generation exists."""

    template_id: str
    label: str
    summary: str
    capability_priorities: tuple[str, ...] = field(default_factory=tuple)
    tradeoffs: tuple[str, ...] = field(default_factory=tuple)
    artifact_ready: bool = False

    def to_dict(self) -> dict:
        """Serialize the profile template."""
        return {
            "template_id": self.template_id,
            "label": self.label,
            "summary": self.summary,
            "capability_priorities": list(self.capability_priorities),
            "tradeoffs": list(self.tradeoffs),
            "artifact_ready": self.artifact_ready,
        }


PROFILE_TEMPLATES: tuple[ProfileTemplate, ...] = (
    ProfileTemplate(
        template_id="balanced-local",
        label="Balanced Local",
        summary="General local chat profile that keeps quality first while using the safer streamed runtime path.",
        capability_priorities=("plain English chat", "planning", "stability"),
        tradeoffs=("Not the fastest path yet", "Does not create a smaller artifact yet"),
    ),
    ProfileTemplate(
        template_id="agent-coder",
        label="Agent Coder",
        summary="Agent-oriented target for coding, tool-use readiness, planning, and concise follow-through.",
        capability_priorities=("coding", "tool use", "planning", "stability"),
        tradeoffs=("Creativity and multilingual behavior are lower priority", "Needs benchmark proof before release"),
    ),
    ProfileTemplate(
        template_id="low-memory",
        label="Low Memory",
        summary="Weak-hardware target that favors lower RAM pressure and safer streaming over speed.",
        capability_priorities=("memory savings", "stability", "plain English chat"),
        tradeoffs=("May run slower", "Quality must be measured against the balanced profile"),
    ),
)


def list_profile_templates() -> list[ProfileTemplate]:
    """Return built-in profile templates in product order."""
    return list(PROFILE_TEMPLATES)


def profile_templates_summary() -> str:
    """Return a compact plain-English summary of built-in profile templates."""
    lines: list[str] = []
    for template in PROFILE_TEMPLATES:
        priorities = ", ".join(template.capability_priorities)
        tradeoffs = "; ".join(template.tradeoffs)
        lines.append(f"{template.label}: {template.summary}\nPriorities: {priorities}\nTradeoffs: {tradeoffs}")
    return "\n\n".join(lines)
