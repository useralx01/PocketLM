import json
from pathlib import Path

from pcketlm.core.runtime.streaming_state import bootstrap_streaming_state


def test_bootstrap_streaming_state_writes_manifest_and_directories(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime
    from pcketlm.core.runtime.load_attempt import MemorySnapshot

    monkeypatch.setattr(
        runtime.streaming_state,
        "plan_staged_disk_streaming",
        lambda model_id, model_dir: type(
            "Plan",
            (),
            {
                "model_id": model_id,
                "model_dir": model_dir,
                "summary": "test summary",
                "viability": "viable",
                "memory": MemorySnapshot(total_bytes=16, free_bytes=8),
                "model_bytes": 10,
                "model_gb": 0.0,
                "working_window_bytes": 4,
                "working_window_gb": 0.0,
                "prefetch_window_bytes": 2,
                "prefetch_window_gb": 0.0,
                "segment_budget_bytes": 2,
                "segment_budget_gb": 0.0,
                "estimated_chunks": 3,
                "chunk_plan": [],
                "blockers": [],
                "cache_layout": type(
                    "CacheLayout",
                    (),
                    {
                        "cache_root": tmp_path / "streaming" / model_id,
                        "manifest_path": tmp_path / "streaming" / model_id / "manifest.json",
                        "hot_window_dir": tmp_path / "streaming" / model_id / "hot-window",
                        "warm_window_dir": tmp_path / "streaming" / model_id / "warm-window",
                        "to_dict": lambda self: {
                            "cache_root": str(self.cache_root),
                            "manifest_path": str(self.manifest_path),
                            "hot_window_dir": str(self.hot_window_dir),
                            "warm_window_dir": str(self.warm_window_dir),
                        },
                    },
                )(),
            },
        )(),
    )

    result = bootstrap_streaming_state("qwen-test", Path("C:/model"))

    assert result.created is True
    assert result.manifest_path.exists()
    assert result.hot_window_dir.exists()
    assert result.warm_window_dir.exists()

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["model_id"] == "qwen-test"
    assert payload["summary"] == "test summary"
