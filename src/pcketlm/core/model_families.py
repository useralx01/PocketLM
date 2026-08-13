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
        key="mixtral",
        label="Mixtral",
        priority=3,
        runtime_status="active",
        aliases=("mistral-moe", "mixtral-moe", "mistralai"),
        notes="Cross-family MoE validation target for paged expert runtime.",
    ),
    ModelFamily(
        key="kimi",
        label="Kimi K2",
        priority=4,
        runtime_status="experimental",
        aliases=("moonshot", "kimi-k2", "kimi_k2"),
        notes="Storage-light contract verified; local GGUF weights are still required.",
    ),
    ModelFamily(
        key="kronos",
        label="Kronos/Kronk",
        priority=5,
        runtime_status="experimental",
        aliases=("kronk",),
        notes="CPU forecasting adapter implemented; this is not a chat model.",
    ),
    ModelFamily(
        key="gemma",
        label="Gemma",
        priority=6,
        runtime_status="experimental",
        aliases=("gemma2", "gemma3", "gemma3-text", "gemma3_text"),
        notes="Storage-light GGUF chat contract verified; local weights are still required.",
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
