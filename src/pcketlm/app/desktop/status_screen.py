"""Desktop status-screen view model helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pcketlm.core.acquisition import build_acquisition_snapshot
from pcketlm.core.model_families import family_label, family_runtime_status
from pcketlm.core.registry.repository import load_model_registry
from pcketlm.core.runtime import (
    attempt_real_model_load,
    build_runtime_bootstrap,
    build_streaming_control_state,
)
from pcketlm.core.storage.paths import original_model_root


DEFAULT_MODEL_ID = "qwen2.5-14b-instruct"
DEFAULT_MODEL_LABEL = "Qwen2.5-14B-Instruct"


@dataclass(slots=True)
class StatusScreenModel:
    """Desktop-ready status view for one model."""

    model_id: str
    model_label: str
    family_label: str
    family_runtime_status: str
    format_label: str
    source_label: str
    model_dir: Path
    acquisition_status: str
    acquisition_progress_pct: float
    acquisition_progress_text: str
    acquisition_download_text: str
    acquisition_shards_text: str
    acquisition_summary: str
    runtime_status: str
    config_status: str
    tokenizer_status: str
    weights_status: str
    runtime_libs_status: str
    runtime_summary: str
    load_status: str
    blocker_category: str | None
    blocker_severity: str | None
    recommended_action: str
    streaming_status: str
    streaming_action_label: str
    streaming_summary: str
    streaming_rotation_text: str
    streaming_cache_text: str
    streaming_window_text: str
    model_actions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_step: str = ""
    details: dict = field(default_factory=dict)


@dataclass(slots=True)
class StatusScreenOption:
    """One desktop-selectable model option."""

    model_id: str
    model_label: str
    family_key: str
    family_label: str
    family_runtime_status: str
    source_label: str
    model_dir: Path
    registered: bool = False

    @property
    def selection_label(self) -> str:
        suffix = "registered" if self.registered else "detected local source"
        return f"{self.model_label} ({self.model_id})  [{suffix}]"


def _status_title(value: str) -> str:
    return value.replace("_", " ").title()


def _download_text(bytes_on_disk_gb: float, expected_bytes_gb: float | None) -> str:
    if expected_bytes_gb is None:
        return f"{bytes_on_disk_gb} GB downloaded"
    return f"{bytes_on_disk_gb} GB / {expected_bytes_gb} GB"


def _runtime_libs_status(bootstrap: dict) -> str:
    deps = bootstrap["dependencies"]
    return "Ready" if all(deps.values()) else "Missing"


def _blockers(bootstrap: dict) -> list[str]:
    return list(bootstrap["blockers"]) if bootstrap["blockers"] else ["No blockers right now."]


def _model_actions(runtime_ready: bool, streaming_ready: bool, family_status: str) -> list[str]:
    """Return the plain-English model home actions for the selected model."""
    chat_status = "available" if runtime_ready or streaming_ready else "blocked until source/runtime readiness improves"
    support_note = "active runtime family" if family_status == "active" else f"{family_status} runtime family"
    return [
        f"Chat test: {chat_status}.",
        "Personalize: planned profile flow for memory, speed, capability, and agent-role tradeoffs.",
        "Compare: planned original-vs-profile behavior and performance view.",
        "Inspect: available now through readiness, blockers, streaming, and details views.",
        "Benchmark: planned repeatable quality, speed, memory, and stability checks.",
        f"Family support: {support_note}.",
    ]


def list_status_screen_options() -> list[StatusScreenOption]:
    """List desktop-selectable model options from the registry, with a safe local fallback."""
    records = load_model_registry()
    options: list[StatusScreenOption] = []

    for record in sorted(records.values(), key=lambda item: item.model_id):
        options.append(
            StatusScreenOption(
                model_id=record.model_id,
                model_label=record.label,
                family_key=record.family,
                family_label=family_label(record.family),
                family_runtime_status=family_runtime_status(record.family),
                source_label=f"{record.source_origin.title()} registry entry",
                model_dir=record.source_path,
                registered=True,
            )
        )

    if options:
        return options

    default_dir = original_model_root(DEFAULT_MODEL_ID)
    if default_dir.exists():
        options.append(
            StatusScreenOption(
                model_id=DEFAULT_MODEL_ID,
                model_label=DEFAULT_MODEL_LABEL,
                family_key="qwen",
                family_label="Qwen",
                family_runtime_status=family_runtime_status("qwen"),
                source_label="Detected local source",
                model_dir=default_dir,
                registered=False,
            )
        )

    return options


def resolve_status_screen_option(model_id: str | None = None) -> StatusScreenOption:
    """Resolve one desktop model option by id or fall back to the first available option."""
    options = list_status_screen_options()
    if not options:
        return StatusScreenOption(
            model_id=DEFAULT_MODEL_ID,
            model_label=DEFAULT_MODEL_LABEL,
            family_key="qwen",
            family_label="Qwen",
            family_runtime_status=family_runtime_status("qwen"),
            source_label="Expected local source",
            model_dir=original_model_root(DEFAULT_MODEL_ID),
            registered=False,
        )

    if model_id:
        for option in options:
            if option.model_id == model_id:
                return option

    return options[0]


def build_status_screen_model(
    model_id: str = DEFAULT_MODEL_ID,
    model_label: str | None = None,
    model_dir: Path | None = None,
) -> StatusScreenModel:
    """Build a desktop-ready status model from live pcketlm state."""
    option = resolve_status_screen_option(model_id) if model_dir is None else None
    resolved_label = model_label or (option.model_label if option else DEFAULT_MODEL_LABEL)
    source_dir = model_dir or (option.model_dir if option else original_model_root(model_id))
    family_label_value = option.family_label if option else "Qwen"
    family_runtime_status_value = option.family_runtime_status if option else family_runtime_status("qwen")
    source_label = option.source_label if option else "Official local import"
    acquisition = build_acquisition_snapshot(source_dir)
    bootstrap_result = build_runtime_bootstrap(model_id, source_dir)
    load_result = attempt_real_model_load(model_id, source_dir)
    streaming_control = build_streaming_control_state(model_id)
    bootstrap = bootstrap_result.to_dict()
    load = load_result.to_dict()
    source = bootstrap["source"]

    progress_pct = acquisition.progress_pct or 0.0
    progress_text = (
        f"{acquisition.progress_bar} {progress_pct}%"
        if acquisition.progress_pct is not None
        else f"{acquisition.progress_bar} unknown"
    )

    config_status = "Ready" if bootstrap["components"]["config_loadable"] else "Blocked"
    tokenizer_status = "Ready" if bootstrap["components"]["tokenizer_loadable"] else "Blocked"
    weights_status = "Ready" if source["ready"] else "Incomplete"
    runtime_status = "Ready" if bootstrap["can_attempt_load"] else "Blocked"
    load_status = "Loaded" if load["load_succeeded"] else ("Attempted" if load["load_attempted"] else "Blocked")
    recommended_action = load["recommended_action"] or acquisition.recommended_next_step
    streaming_status = "Ready" if streaming_control.ready else "Blocked"
    if streaming_control.action_key == "rotate-forward":
        streaming_status = "Advancing"
    elif streaming_control.action_key == "repair-cache-window":
        streaming_status = "Repair Needed"
    elif streaming_control.action_key == "hold-position":
        streaming_status = "Holding"

    if streaming_control.ready:
        recommended_action = streaming_control.summary

    model_actions = _model_actions(
        runtime_ready=runtime_status == "Ready",
        streaming_ready=streaming_control.ready,
        family_status=family_runtime_status_value,
    )

    return StatusScreenModel(
        model_id=model_id,
        model_label=resolved_label,
        family_label=family_label_value,
        family_runtime_status=family_runtime_status_value,
        format_label=source["format_name"],
        source_label=source_label,
        model_dir=source_dir,
        acquisition_status=_status_title(acquisition.status),
        acquisition_progress_pct=progress_pct,
        acquisition_progress_text=progress_text,
        acquisition_download_text=_download_text(
            acquisition.bytes_on_disk_gb,
            acquisition.expected_bytes_gb,
        ),
        acquisition_shards_text=f"{acquisition.present_shards} / {acquisition.expected_shards}",
        acquisition_summary=acquisition.plain_english_summary,
        runtime_status=runtime_status,
        config_status=config_status,
        tokenizer_status=tokenizer_status,
        weights_status=weights_status,
        runtime_libs_status=_runtime_libs_status(bootstrap),
        runtime_summary=bootstrap["plain_english_summary"],
        load_status=load_status,
        blocker_category=load["blocker_category"],
        blocker_severity=load["blocker_severity"],
        recommended_action=recommended_action,
        streaming_status=streaming_status,
        streaming_action_label=streaming_control.action_label,
        streaming_summary=streaming_control.summary,
        streaming_rotation_text=(
            f"Step {streaming_control.rotation_step}  |  Last event: {streaming_control.last_event}"
        ),
        streaming_cache_text=(
            f"Hits {streaming_control.cache_hit_count} (+{streaming_control.last_cache_hit_delta})  |  "
            f"Misses {streaming_control.cache_miss_count} (+{streaming_control.last_cache_miss_delta})  |  "
            f"Refills {streaming_control.refill_count}"
        ),
        streaming_window_text=(
            f"Hot: {', '.join(streaming_control.current_hot_unit_ids) or 'none'}\n"
            f"Warm: {', '.join(streaming_control.current_warm_unit_ids) or 'none'}\n"
            f"Overflow head: {streaming_control.overflow_head_unit_id or 'none'}"
        ),
        model_actions=model_actions,
        blockers=load["blockers"] if load["blockers"] else _blockers(bootstrap),
        warnings=list(dict.fromkeys(list(source["warnings"]) + list(bootstrap["warnings"]) + list(load["warnings"]))),
        next_step=recommended_action,
        details={
            "acquisition": acquisition.to_dict(),
            "bootstrap": bootstrap,
            "load_attempt": load,
            "streaming_control": streaming_control.to_dict(),
            "selection": {
                "model_id": model_id,
                "model_label": resolved_label,
                "family_label": family_label_value,
                "family_runtime_status": family_runtime_status_value,
                "source_label": source_label,
                "registered": option.registered if option else False,
            },
        },
    )
