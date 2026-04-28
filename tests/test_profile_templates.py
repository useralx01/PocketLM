import json

from pcketlm.core.profiles import (
    build_profile_compare_summary,
    ensure_default_profiles,
    list_profile_templates,
    list_saved_profiles,
    profile_templates_summary,
)


def test_profile_templates_include_core_product_targets() -> None:
    templates = list_profile_templates()

    assert [template.template_id for template in templates] == [
        "balanced-local",
        "agent-coder",
        "low-memory",
    ]
    assert templates[0].artifact_ready is False
    assert "coding" in templates[1].capability_priorities
    assert "memory savings" in templates[2].capability_priorities


def test_profile_templates_summary_is_plain_english() -> None:
    summary = profile_templates_summary()

    assert "Balanced Local" in summary
    assert "Agent Coder" in summary
    assert "Low Memory" in summary
    assert "Tradeoffs:" in summary


def test_default_profiles_are_saved_per_model(tmp_path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    profiles = ensure_default_profiles("qwen-test")
    saved = list_saved_profiles("qwen-test")

    assert [profile.profile_id for profile in profiles] == ["balanced-local", "agent-coder", "low-memory"]
    assert [profile.profile_id for profile in saved] == ["balanced-local", "agent-coder", "low-memory"]
    assert saved[0].model_id == "qwen-test"
    assert saved[0].profile_path is not None
    assert (tmp_path / "models" / "qwen-test" / "profiles" / "balanced-local.profile.json").exists()


def test_profile_compare_summary_uses_latest_benchmark(tmp_path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    summary = build_profile_compare_summary(
        "qwen-test",
        {
            "cases": [
                {"label": "Balanced", "elapsed_seconds": 39.7, "generated_text": "rough"},
                {"label": "Quality", "elapsed_seconds": 42.7, "generated_text": "Hello!"},
            ]
        },
    )

    assert summary["profile_count"] == 3
    assert summary["default_elapsed_seconds"] == 42.7
    low_memory = next(profile for profile in summary["profiles"] if profile["profile_id"] == "low-memory")
    assert low_memory["runtime_mode"] == "Quality"
    assert low_memory["reference_elapsed_seconds"] == 42.7


def test_default_profiles_refresh_built_in_runtime_defaults(tmp_path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    profiles = ensure_default_profiles("qwen-test")
    low_memory = next(profile for profile in profiles if profile.profile_id == "low-memory")
    low_memory.runtime_mode = "Balanced"
    low_memory.settings["runtime_mode"] = "Balanced"
    (tmp_path / "models" / "qwen-test" / "profiles" / "low-memory.profile.json").write_text(
        json.dumps(low_memory.to_dict()),
        encoding="utf-8",
    )

    refreshed = ensure_default_profiles("qwen-test")
    saved_low_memory = next(profile for profile in refreshed if profile.profile_id == "low-memory")

    assert saved_low_memory.runtime_mode == "Quality"
    assert saved_low_memory.settings["runtime_mode"] == "Quality"
