"""Persisted per-model profile records."""

from __future__ import annotations

import json
from pathlib import Path

from pcketlm.core.profiles.models import OptimizationProfile
from pcketlm.core.profiles.templates import ProfileTemplate, list_profile_templates
from pcketlm.core.storage.paths import profiles_root


def _profile_path(model_id: str, profile_id: str) -> Path:
    return profiles_root(model_id) / f"{profile_id}.profile.json"


def _runtime_mode_for_template(template: ProfileTemplate) -> str:
    return "Quality"


def _settings_for_template(template: ProfileTemplate) -> dict:
    if template.template_id == "agent-coder":
        return {
            "response_style": "concise",
            "tool_readiness": "planned",
            "default_max_new_tokens": 6,
            "runtime_mode": "Quality",
        }
    if template.template_id == "low-memory":
        return {
            "response_style": "short",
            "memory_priority": "high",
            "default_max_new_tokens": 3,
            "runtime_mode": "Quality",
        }
    return {
        "response_style": "balanced",
        "default_max_new_tokens": 4,
        "runtime_mode": "Quality",
    }


def profile_from_template(model_id: str, template: ProfileTemplate) -> OptimizationProfile:
    """Build a saved profile record from a built-in template."""
    profile_id = template.template_id
    return OptimizationProfile(
        profile_id=profile_id,
        model_id=model_id,
        label=template.label,
        runtime_mode=_runtime_mode_for_template(template),
        summary=template.summary,
        template_id=template.template_id,
        capability_priorities=list(template.capability_priorities),
        tradeoffs=list(template.tradeoffs),
        settings=_settings_for_template(template),
        artifact_ready=template.artifact_ready,
        profile_path=str(_profile_path(model_id, profile_id)),
    )


def write_profile(profile: OptimizationProfile) -> OptimizationProfile:
    """Persist one profile record."""
    path = _profile_path(profile.model_id, profile.profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.profile_path = str(path)
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return profile


def ensure_default_profiles(model_id: str) -> list[OptimizationProfile]:
    """Create the three free profile records if they are missing."""
    saved: list[OptimizationProfile] = []
    for template in list_profile_templates():
        path = _profile_path(model_id, template.template_id)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                profile = OptimizationProfile.from_dict(payload)
                profile.profile_path = str(path)
                current_default = profile_from_template(model_id, template)
                if profile.template_id == template.template_id and not profile.artifact_ready:
                    if profile.runtime_mode != current_default.runtime_mode or profile.settings != current_default.settings:
                        profile = write_profile(current_default)
                saved.append(profile)
                continue
            except (OSError, json.JSONDecodeError):
                pass
        saved.append(write_profile(profile_from_template(model_id, template)))
    return saved


def list_saved_profiles(model_id: str) -> list[OptimizationProfile]:
    """Return saved profile records, seeding the free defaults first."""
    ensure_default_profiles(model_id)
    root = profiles_root(model_id)
    profiles: list[OptimizationProfile] = []
    for path in sorted(root.glob("*.profile.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        profile = OptimizationProfile.from_dict(payload)
        if not profile.profile_id:
            continue
        profile.profile_path = str(path)
        profiles.append(profile)
    order = {template.template_id: index for index, template in enumerate(list_profile_templates())}
    return sorted(profiles, key=lambda profile: (order.get(profile.profile_id, 999), profile.label.lower()))


def get_saved_profile(model_id: str, profile_id: str) -> OptimizationProfile | None:
    """Return one saved profile by id, seeding defaults first."""
    normalized = profile_id.strip()
    if not normalized:
        return None
    for profile in list_saved_profiles(model_id):
        if profile.profile_id == normalized:
            return profile
    return None


def build_profile_compare_summary(model_id: str, latest_benchmark: dict | None = None) -> dict:
    """Build a plain comparison surface from saved profiles and latest benchmark data."""
    profiles = list_saved_profiles(model_id)
    cases = (latest_benchmark or {}).get("cases", []) if latest_benchmark else []
    quality_case = next((case for case in cases if case.get("label") == "Quality"), None)
    balanced_case = next((case for case in cases if case.get("label") == "Balanced"), None)
    default_time = quality_case.get("elapsed_seconds") if quality_case else None
    rows: list[dict] = []
    for profile in profiles:
        reference_case = balanced_case if profile.runtime_mode == "Balanced" else quality_case
        rows.append(
            {
                "profile_id": profile.profile_id,
                "label": profile.label,
                "runtime_mode": profile.runtime_mode,
                "summary": profile.summary,
                "artifact_ready": profile.artifact_ready,
                "reference_elapsed_seconds": None if reference_case is None else reference_case.get("elapsed_seconds"),
                "reference_output": "" if reference_case is None else str(reference_case.get("generated_text") or ""),
                "difference_from_default": (
                    "Uses the current Quality baseline."
                    if profile.runtime_mode == "Quality"
                    else "Uses the current Balanced speed-preview path; quality must be checked before release."
                ),
            }
        )
    return {
        "model_id": model_id,
        "default_runtime_mode": "Quality",
        "default_elapsed_seconds": default_time,
        "profile_count": len(profiles),
        "profiles": rows,
    }
