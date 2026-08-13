"""Installation supervisor contract tests."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from types import SimpleNamespace
import urllib.request

from pcketlm.core import supervisor


def _catalog_entry(*, runnable: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        model_id="fixture-model",
        family="qwen",
        capability="chat",
        backend="fixture",
        source_status="ready" if runnable else "missing",
        runnable=runnable,
    )


def _compatibility_entry(*, ready: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        profile=SimpleNamespace(
            key="qwen",
            label="Qwen dense",
            capability="chat",
            implementation_status="Proven",
        ),
        local_ready=ready,
        blockers=[] if ready else ["No local model is installed."],
    )


def _patch_report_inputs(monkeypatch, tmp_path: Path, *, runnable: bool = True) -> None:
    monkeypatch.setattr(supervisor, "build_model_catalog", lambda: [_catalog_entry(runnable=runnable)])
    monkeypatch.setattr(supervisor, "build_compatibility_matrix", lambda: [_compatibility_entry(ready=runnable)])
    monkeypatch.setattr(supervisor, "llama_server_path", lambda: tmp_path / "missing-llama-server.exe")
    monkeypatch.setattr(supervisor, "_cuda_snapshot", lambda: {"available": False, "device_count": 0, "devices": []})
    monkeypatch.setattr(supervisor, "_memory_snapshot", lambda: supervisor.MemorySnapshot(16 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(supervisor, "_package_version", lambda name: "0.1.0" if name == "pcketlm" else "test")


def test_installation_id_is_random_and_stable(tmp_path) -> None:
    first = supervisor.installation_id(tmp_path)
    second = supervisor.installation_id(tmp_path)

    assert first == second
    assert len(first) == 36
    assert json.loads((tmp_path / "supervisor" / "installation.json").read_text())["installation_id"] == first


def test_supervisor_report_is_ready_and_sanitized(monkeypatch, tmp_path) -> None:
    _patch_report_inputs(monkeypatch, tmp_path)

    report = supervisor.build_supervisor_report(root=tmp_path)
    serialized = json.dumps(report).lower()

    assert report["status"] == "ready"
    assert report["summary"]["runnable_model_count"] == 1
    assert report["privacy"]["outbound_telemetry"] is False
    assert "hostname" not in serialized
    assert "username" not in serialized
    assert "project_root" not in serialized
    assert str(tmp_path).lower().replace("\\", "\\\\") not in serialized


def test_supervisor_report_is_partial_without_a_model(monkeypatch, tmp_path) -> None:
    _patch_report_inputs(monkeypatch, tmp_path, runnable=False)

    report = supervisor.build_supervisor_report(root=tmp_path)

    assert report["status"] == "partial"
    assert report["summary"]["runnable_model_count"] == 0
    assert "No complete local model is currently runnable." in report["summary"]["warnings"]


def test_save_supervisor_report_writes_proof(monkeypatch, tmp_path) -> None:
    _patch_report_inputs(monkeypatch, tmp_path)
    output = tmp_path / "proof.json"

    saved = supervisor.save_supervisor_report(root=tmp_path, output=output)

    assert saved == output
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == supervisor.SUPERVISOR_SCHEMA


def test_local_supervisor_api_returns_sanitized_health(monkeypatch, tmp_path) -> None:
    from pcketlm.app.web import main as web_main

    _patch_report_inputs(monkeypatch, tmp_path)
    report = supervisor.build_supervisor_report(root=tmp_path)
    monkeypatch.setattr(web_main, "build_supervisor_report", lambda: report)
    server = web_main.create_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urllib.request.urlopen(f"http://{host}:{port}/api/supervisor/health", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert payload["schema"] == supervisor.SUPERVISOR_SCHEMA
        assert payload["privacy"]["local_only"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
