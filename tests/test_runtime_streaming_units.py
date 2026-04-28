import json
from pathlib import Path

from pcketlm.core.runtime.streaming_units import build_streaming_unit_map, write_streaming_unit_map


def test_build_streaming_unit_map_reads_actual_shards(tmp_path: Path, monkeypatch) -> None:
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
                "working_window_bytes": 1024,
                "prefetch_window_bytes": 512,
                "segment_budget_bytes": 512,
                "estimated_chunks": 2,
                "chunk_plan": [
                    {"chunk_id": "hot-window", "target_bytes": 1024, "target_gb": 0.0, "purpose": "active"}
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

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 2048)
    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"x" * 1024)

    unit_map = build_streaming_unit_map("qwen-test", model_dir)

    assert unit_map.ready is True
    assert len(unit_map.units) == 6
    assert unit_map.units[0].total_segments == 4
    assert unit_map.units[0].segment_bytes == 512
    assert unit_map.units[0].target_window == "hot-window"
    assert unit_map.units[1].target_window == "warm-window"
    assert unit_map.units[-1].target_window == "warm-window"


def test_write_streaming_unit_map_persists_units_json(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

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
                "working_window_bytes": 1024,
                "prefetch_window_bytes": 512,
                "segment_budget_bytes": 512,
                "estimated_chunks": 1,
                "chunk_plan": [
                    {
                        "chunk_id": "hot-window",
                        "target_bytes": 1024,
                        "target_gb": 0.0,
                        "purpose": "active",
                    }
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
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1024)

    path = write_streaming_unit_map("qwen-test", model_dir)

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert payload["units"][0]["shard_name"] == "model-00001-of-00001.safetensors"
    assert payload["units"][0]["segment_bytes"] == 512
