from pathlib import Path

from pcketlm.app.desktop.status_screen import build_status_screen_model, list_status_screen_options


def test_build_status_screen_model(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.app import desktop

    monkeypatch.setattr(
        desktop.status_screen,
        "build_acquisition_snapshot",
        lambda model_dir: type(
            "Acq",
            (),
            {
                "status": "downloading",
                "progress_pct": 55.5,
                "progress_bar": "[#####-----]",
                "bytes_on_disk_gb": 10.0,
                "expected_bytes_gb": 20.0,
                "present_shards": 1,
                "expected_shards": 8,
                "plain_english_summary": "Download is active.",
                "recommended_next_step": "Let the download continue.",
                "to_dict": lambda self: {"status": "downloading"},
            },
        )(),
    )
    monkeypatch.setattr(
        desktop.status_screen,
        "build_runtime_bootstrap",
        lambda model_id, model_dir: type(
            "Bootstrap",
            (),
            {
                "to_dict": lambda self: {
                    "dependencies": {
                        "transformers": True,
                        "tokenizers": True,
                        "safetensors": True,
                        "torch": True,
                    },
                    "components": {
                        "config_loadable": True,
                        "tokenizer_loadable": True,
                    },
                    "can_attempt_load": False,
                    "blockers": ["Need more shards."],
                    "warnings": [],
                    "plain_english_summary": "Blocked by missing shards.",
                    "source": {
                        "format_name": "safetensors-sharded",
                        "ready": False,
                        "warnings": [],
                    },
                }
            },
        )(),
    )
    monkeypatch.setattr(
        desktop.status_screen,
        "attempt_real_model_load",
        lambda model_id, model_dir: type(
            "Load",
            (),
            {
                "to_dict": lambda self: {
                    "load_attempted": False,
                    "load_succeeded": False,
                    "blockers": ["Need more shards."],
                    "blocker_category": "source-files",
                    "blocker_severity": "high",
                    "recommended_action": "Wait for the shard set to complete.",
                    "warnings": [],
                }
            },
        )(),
    )
    monkeypatch.setattr(
        desktop.status_screen,
        "build_streaming_control_state",
        lambda model_id: type(
            "Control",
            (),
            {
                "action_key": "materialize-current-window",
                "action_label": "Materialize Current Window",
                "summary": "Materialize the current streaming window first.",
                "rotation_step": 0,
                "last_event": "missing",
                "current_hot_unit_ids": [],
                "current_warm_unit_ids": [],
                "overflow_head_unit_id": None,
                "refill_count": 0,
                "cache_hit_count": 0,
                "cache_miss_count": 0,
                "last_cache_hit_delta": 0,
                "last_cache_miss_delta": 0,
                "ready": True,
                "to_dict": lambda self: {"action_key": "materialize-current-window"},
            },
        )(),
    )

    model = build_status_screen_model(model_dir=tmp_path)

    assert model.model_label == "Qwen2.5-14B-Instruct"
    assert model.family_runtime_status == "active"
    assert model.acquisition_progress_text.endswith("55.5%")
    assert model.runtime_status == "Blocked"
    assert model.load_status == "Blocked"
    assert model.blocker_category == "source-files"
    assert model.recommended_action == "Materialize the current streaming window first."
    assert model.runtime_libs_status == "Ready"
    assert model.blockers == ["Need more shards."]
    assert model.streaming_action_label == "Materialize Current Window"
    assert model.streaming_summary == "Materialize the current streaming window first."
    assert "Chat test: available." in model.model_actions
    assert any(action.startswith("Personalize: planned") for action in model.model_actions)


def test_list_status_screen_options_uses_registry_when_present(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.app import desktop
    from pcketlm.core.model_import.inspect import ConfigSummary
    from pcketlm.core.registry.models import ModelRecord

    monkeypatch.setattr(
        desktop.status_screen,
        "load_model_registry",
        lambda: {
            "qwen-test": ModelRecord(
                model_id="qwen-test",
                label="Qwen Test",
                family="qwen",
                model_type="dense",
                source_path=tmp_path / "registered-source",
                original_path=tmp_path / "models" / "qwen-test" / "original",
                source_origin="huggingface",
                format_name="safetensors-sharded",
                config=ConfigSummary(model_type="qwen2"),
            )
        },
    )

    options = list_status_screen_options()

    assert len(options) == 1
    assert options[0].registered is True
    assert options[0].model_id == "qwen-test"
    assert options[0].family_label == "Qwen"
    assert options[0].family_runtime_status == "active"
    assert "registered" in options[0].selection_label
