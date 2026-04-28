from pathlib import Path

from pcketlm.core.runtime.streaming_materialize import materialize_window_schedule, verify_materialized_cache
from pcketlm.core.runtime.streaming_units import StreamableWeightUnit


def test_materialize_window_schedule_writes_hot_and_warm_cache_files(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime

    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"abcdefghij")

    hot_dir = tmp_path / "streaming" / "qwen-test" / "hot-window"
    warm_dir = tmp_path / "streaming" / "qwen-test" / "warm-window"
    hot_dir.mkdir(parents=True)
    warm_dir.mkdir(parents=True)

    monkeypatch.setattr(
        runtime.streaming_materialize,
        "load_streaming_manifest",
        lambda model_id: type(
            "Manifest",
            (),
            {
                "ready": True,
                "blockers": [],
                "cache_root": tmp_path / "streaming" / model_id,
                "hot_window_dir": hot_dir,
                "warm_window_dir": warm_dir,
            },
        )(),
    )
    monkeypatch.setattr(
        runtime.streaming_materialize,
        "build_window_schedule",
        lambda model_id, model_dir: type(
            "Schedule",
            (),
            {
                "ready": True,
                "blockers": [],
                "hot_window_units": [type("Scheduled", (), {"unit_id": "unit-hot"})()],
                "warm_window_units": [type("Scheduled", (), {"unit_id": "unit-warm"})()],
                "overflow_units": [],
            },
        )(),
    )
    monkeypatch.setattr(
        runtime.streaming_materialize,
        "build_streaming_unit_map",
        lambda model_id, model_dir: type(
            "UnitMap",
            (),
            {
                "ready": True,
                "blockers": [],
                "units": [
                    StreamableWeightUnit(
                        unit_id="unit-hot",
                        shard_name=shard_path.name,
                        shard_path=shard_path,
                        shard_bytes=10,
                        segment_index=0,
                        segment_offset_bytes=0,
                        segment_bytes=4,
                        segment_gb=0.0,
                        total_segments=2,
                        target_window="hot-window",
                    ),
                    StreamableWeightUnit(
                        unit_id="unit-warm",
                        shard_name=shard_path.name,
                        shard_path=shard_path,
                        shard_bytes=10,
                        segment_index=1,
                        segment_offset_bytes=4,
                        segment_bytes=3,
                        segment_gb=0.0,
                        total_segments=2,
                        target_window="warm-window",
                    ),
                ],
            },
        )(),
    )

    result = materialize_window_schedule("qwen-test", tmp_path / "model")

    assert result.ready is True
    assert (hot_dir / "unit-hot.bin").read_bytes() == b"abcd"
    assert (warm_dir / "unit-warm.bin").read_bytes() == b"efg"
    assert result.cache_index_path.exists()
    assert result.hot_segments[0].checksum_sha256
    assert result.hot_segments[0].verified is True


def test_materialize_window_schedule_reports_short_reads(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime

    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"abc")

    hot_dir = tmp_path / "streaming" / "qwen-test" / "hot-window"
    warm_dir = tmp_path / "streaming" / "qwen-test" / "warm-window"
    hot_dir.mkdir(parents=True)
    warm_dir.mkdir(parents=True)

    monkeypatch.setattr(
        runtime.streaming_materialize,
        "load_streaming_manifest",
        lambda model_id: type(
            "Manifest",
            (),
            {
                "ready": True,
                "blockers": [],
                "cache_root": tmp_path / "streaming" / model_id,
                "hot_window_dir": hot_dir,
                "warm_window_dir": warm_dir,
            },
        )(),
    )
    monkeypatch.setattr(
        runtime.streaming_materialize,
        "build_window_schedule",
        lambda model_id, model_dir: type(
            "Schedule",
            (),
            {
                "ready": True,
                "blockers": [],
                "hot_window_units": [type("Scheduled", (), {"unit_id": "unit-hot"})()],
                "warm_window_units": [],
                "overflow_units": [],
            },
        )(),
    )
    monkeypatch.setattr(
        runtime.streaming_materialize,
        "build_streaming_unit_map",
        lambda model_id, model_dir: type(
            "UnitMap",
            (),
            {
                "ready": True,
                "blockers": [],
                "units": [
                    StreamableWeightUnit(
                        unit_id="unit-hot",
                        shard_name=shard_path.name,
                        shard_path=shard_path,
                        shard_bytes=3,
                        segment_index=0,
                        segment_offset_bytes=0,
                        segment_bytes=5,
                        segment_gb=0.0,
                        total_segments=1,
                        target_window="hot-window",
                    ),
                ],
            },
        )(),
    )

    result = materialize_window_schedule("qwen-test", tmp_path / "model")

    assert result.ready is False
    assert any("expected 5 bytes" in blocker for blocker in result.blockers)


def test_verify_materialized_cache_reports_checksum_mismatch(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime

    cache_root = tmp_path / "streaming" / "qwen-test"
    hot_dir = cache_root / "hot-window"
    warm_dir = cache_root / "warm-window"
    hot_dir.mkdir(parents=True)
    warm_dir.mkdir(parents=True)
    cache_path = hot_dir / "unit-hot.bin"
    cache_path.write_bytes(b"tampered")
    (cache_root / "cache-index.json").write_text(
        """{
  "model_id": "qwen-test",
  "cache_root": "ignored",
  "cache_index_path": "ignored",
  "hot_window_dir": "ignored",
  "warm_window_dir": "ignored",
  "hot_segments": [
    {
      "unit_id": "unit-hot",
      "window": "hot-window",
      "cache_path": "%s",
      "shard_name": "model-00001-of-00001.safetensors",
      "segment_index": 0,
      "bytes_written": 4,
      "checksum_sha256": "bad",
      "verified": true
    }
  ],
  "warm_segments": [],
  "verification_mode": "write-boundary",
  "blockers": [],
  "ready": true
}""" % str(cache_path).replace("\\", "\\\\"),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runtime.streaming_materialize,
        "load_streaming_manifest",
        lambda model_id: type(
            "Manifest",
            (),
            {
                "blockers": [],
                "cache_root": cache_root,
                "hot_window_dir": hot_dir,
                "warm_window_dir": warm_dir,
            },
        )(),
    )

    result = verify_materialized_cache("qwen-test")

    assert result.ready is False
    assert any("checksum does not match" in blocker for blocker in result.blockers)
