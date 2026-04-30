"""Model-family metadata for product and runtime planning."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ModelFamily:
    """Product-level metadata for a model family."""

    key: str
    label: str
    priority: int
    runtime_status: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        key="qwen",
        label="Qwen",
        priority=1,
        runtime_status="active",
        aliases=("qwen2", "qwen2.5", "qwen3"),
        notes="First dense text runtime target.",
    ),
    ModelFamily(
        key="qwen-moe",
        label="Qwen MoE",
        priority=2,
        runtime_status="active",
        aliases=("qwen3-moe", "qwen3_moe", "qwen-moe", "moe-qwen"),
        notes="Paged MoE runtime target for huge sparse models.",
    ),
    ModelFamily(
        key="kimi",
        label="Kimi",
        priority=3,
        runtime_status="planned",
        aliases=("moonshot",),
        notes="Next priority after Qwen when a local open-weight path is selected.",
    ),
    ModelFamily(
        key="kronos",
        label="Kronos/Kronk",
        priority=4,
        runtime_status="planned",
        aliases=("kronk",),
        notes="Planned if a concrete supported local model target exists.",
    ),
    ModelFamily(
        key="gemma",
        label="Gemma",
        priority=5,
        runtime_status="planned",
        aliases=("gemma2", "gemma3"),
        notes="Planned after the earlier priority families.",
    ),
)

_FAMILIES_BY_KEY = {family.key: family for family in MODEL_FAMILIES}
_ALIASES = {
    alias: family.key
    for family in MODEL_FAMILIES
    for alias in (family.key, *family.aliases)
}


def normalize_family_key(value: str | None) -> str:
    """Normalize a user, registry, or config family value."""
    if not value:
        return "unknown"
    cleaned = value.strip().lower().replace("_", "-")
    return _ALIASES.get(cleaned, cleaned)


def get_model_family(value: str | None) -> ModelFamily | None:
    """Return known family metadata for a family-like value."""
    return _FAMILIES_BY_KEY.get(normalize_family_key(value))


def family_label(value: str | None) -> str:
    """Return a user-facing family label."""
    family = get_model_family(value)
    if family:
        return family.label
    if not value:
        return "Unknown"
    return value.strip().replace("_", " ").replace("-", " ").title()


def family_runtime_status(value: str | None) -> str:
    """Return the current pcketlm runtime support status for a family."""
    family = get_model_family(value)
    return family.runtime_status if family else "unverified"


def family_priority_summary() -> list[dict]:
    """Return the current product priority order for known families."""
    return [
        {
            "key": family.key,
            "label": family.label,
            "priority": family.priority,
            "runtime_status": family.runtime_status,
            "notes": family.notes,
        }
        for family in sorted(MODEL_FAMILIES, key=lambda item: item.priority)
    ]
