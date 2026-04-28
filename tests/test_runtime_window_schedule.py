import json
from pathlib import Path

from pcketlm.core.runtime.window_schedule import build_window_schedule, write_window_schedule


def test_build_window_schedule_assigns_units_to_hot_and_warm_windows(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    cache_root = tmp_path / "state" / "streaming" / "qwen-test"
    hot = cache_root / "hot-window"
    warm = cache_root / "warm-window"
    hot.mkdir(parents=True)
    warm.mkdir(parents=True)
    (cache_root / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "qwen-test",
                "model_dir": str(tmp_path / "model"),
                "summary": "streaming summary",
                "viability": "viable",
                "working_window_bytes": 2500,
                "prefetch_window_bytes": 1500,
                "estimated_chunks": 2,
                "chunk_plan": [
                    {"chunk_id": "hot-window", "target_bytes": 2500, "target_gb": 0.0, "purpose": "active"}
                ],
                "cache_layout": {
                    "cache_root": str(cache_root),
                    "manifest_path": str(cache_root / "manifest.json"),
                    "hot_window_dir": str(hot),
                    "warm_window_dir": str(warm),
                },
            }
        ),
        encoding="utf-8",
    )
    (cache_root / "units.json").write_text(
        json.dumps(
            {
                "model_id": "qwen-test",
                "model_dir": str(tmp_path / "model"),
                "manifest_path": str(cache_root / "manifest.json"),
                "unit_map_path": str(cache_root / "units.json"),
                "units": [
                    {
                        "unit_id": "unit-0001-seg-001",
                        "shard_name": "a.safetensors",
                        "shard_path": str(tmp_path / "model" / "a.safetensors"),
                        "shard_bytes": 2000,
                        "segment_index": 0,
                        "segment_offset_bytes": 0,
                        "segment_bytes": 2000,
                        "segment_gb": 0.0,
                        "total_segments": 1,
                        "target_window": "hot-window",
                    },
                    {
                        "unit_id": "unit-0002-seg-001",
                        "shard_name": "b.safetensors",
                        "shard_path": str(tmp_path / "model" / "b.safetensors"),
                        "shard_bytes": 1000,
                        "segment_index": 0,
                        "segment_offset_bytes": 0,
                        "segment_bytes": 1000,
                        "segment_gb": 0.0,
                        "total_segments": 1,
                        "target_window": "warm-window",
                    },
                ],
                "blockers": [],
                "ready": True,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runtime.window_schedule,
        "load_streaming_manifest",
        lambda model_id: type(
            "Manifest",
            (),
            {
                "blockers": [],
                "ready": True,
                "working_window_bytes": 2500,
                "prefetch_window_bytes": 1500,
            },
        )(),
    )
    monkeypatch.setattr(
        runtime.window_schedule,
        "build_streaming_unit_map",
        lambda model_id, model_dir: type(
            "UnitMap",
            (),
            {
                "blockers": [],
                "ready": True,
                "units": [
                    type(
                        "Unit",
                        (),
                        {
                            "unit_id": "unit-0001-seg-001",
                            "shard_name": "a.safetensors",
                            "segment_bytes": 2000,
                            "segment_gb": 0.0,
                            "segment_index": 0,
                            "total_segments": 1,
                            "target_window": "hot-window",
                        },
                    )(),
                    type(
                        "Unit",
                        (),
                        {
                            "unit_id": "unit-0002-seg-001",
                            "shard_name": "b.safetensors",
                            "segment_bytes": 1000,
                            "segment_gb": 0.0,
                            "segment_index": 0,
                            "total_segments": 1,
                            "target_window": "warm-window",
                        },
                    )(),
                ],
            },
        )(),
    )

    schedule = build_window_schedule("qwen-test", tmp_path / "model")

    assert schedule.ready is True
    assert schedule.hot_window_units[0].unit_id == "unit-0001-seg-001"
    assert schedule.warm_window_units[0].unit_id == "unit-0002-seg-001"


def test_write_window_schedule_persists_schedule_json(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime, storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime.window_schedule,
        "build_window_schedule",
        lambda model_id, model_dir: type(
            "Schedule",
            (),
            {
                "schedule_path": tmp_path / "state" / "streaming" / model_id / "schedule.json",
                "to_dict": lambda self: {"model_id": model_id, "ready": True},
            },
        )(),
    )

    path = write_window_schedule("qwen-test", tmp_path / "model")

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ready"] is True
