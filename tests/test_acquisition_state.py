from pathlib import Path

from pcketlm.core.acquisition.state import build_acquisition_snapshot


def test_build_acquisition_snapshot_partial(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    snapshot = build_acquisition_snapshot(tmp_path)

    assert snapshot.status == "partial"
    assert snapshot.bytes_on_disk_gb >= 0
    assert snapshot.progress_bar.startswith("[")
    assert "not usable yet" in snapshot.plain_english_summary
    assert "recheck acquisition state" in snapshot.recommended_next_step


def test_build_acquisition_snapshot_missing(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    snapshot = build_acquisition_snapshot(missing_dir)

    assert snapshot.status == "missing"
    assert snapshot.progress_bar == "[" + ("?" * 24) + "]"
    assert "does not exist yet" in snapshot.plain_english_summary
