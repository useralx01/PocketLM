import json
from pathlib import Path

from pcketlm.core.runtime.streaming_reader import load_streaming_manifest


def test_load_streaming_manifest_reads_manifest_and_dirs(tmp_path: Path, monkeypatch) -> None:
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
                "model_dir": "C:/model",
                "summary": "streaming summary",
                "viability": "viable",
                "working_window_bytes": 10,
                "prefetch_window_bytes": 5,
                "segment_budget_bytes": 5,
                "estimated_chunks": 3,
                "chunk_plan": [
                    {
                        "chunk_id": "hot-window",
                        "target_bytes": 10,
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

    manifest = load_streaming_manifest("qwen-test")

    assert manifest.ready is True
    assert manifest.estimated_chunks == 3
    assert manifest.segment_budget_bytes == 5
    assert manifest.chunks[0].chunk_id == "hot-window"
    assert manifest.blockers == []


def test_load_streaming_manifest_reports_missing_manifest(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    manifest = load_streaming_manifest("missing-model")

    assert manifest.ready is False
    assert any("does not exist" in blocker for blocker in manifest.blockers)


def test_load_streaming_manifest_relocates_stale_paths(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    cache_root = tmp_path / "state" / "streaming" / "qwen-test"
    hot = cache_root / "hot-window"
    warm = cache_root / "warm-window"
    model_dir = tmp_path / "models" / "qwen-test" / "original"
    hot.mkdir(parents=True)
    warm.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    manifest_path = cache_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": "qwen-test",
                "model_dir": "C:/missing-old-root/models/qwen-test/original",
                "summary": "streaming summary",
                "viability": "viable",
                "working_window_bytes": 10,
                "prefetch_window_bytes": 5,
                "segment_budget_bytes": 5,
                "estimated_chunks": 3,
                "chunk_plan": [
                    {
                        "chunk_id": "hot-window",
                        "target_bytes": 10,
                        "target_gb": 0.0,
                        "purpose": "active",
                    }
                ],
                "cache_layout": {
                    "cache_root": "C:/missing-old-root/state/streaming/qwen-test",
                    "manifest_path": "C:/missing-old-root/state/streaming/qwen-test/manifest.json",
                    "hot_window_dir": "C:/missing-old-root/state/streaming/qwen-test/hot-window",
                    "warm_window_dir": "C:/missing-old-root/state/streaming/qwen-test/warm-window",
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_streaming_manifest("qwen-test")

    assert manifest.ready is True
    assert manifest.model_dir == model_dir
    assert manifest.cache_root == cache_root
    assert manifest.hot_window_dir == hot
    assert manifest.warm_window_dir == warm

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["model_dir"] == str(model_dir)
    assert payload["cache_layout"]["cache_root"] == str(cache_root)
