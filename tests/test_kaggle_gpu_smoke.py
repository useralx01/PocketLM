from __future__ import annotations

import json

from tools.kaggle_gpu_smoke import kaggle_credentials, prepare_kernel


def test_prepare_kaggle_kernel_embeds_gpu_smoke(tmp_path) -> None:
    kernel_dir = prepare_kernel(
        username="tester",
        slug="pcketlm-gpu-smoke-test",
        title="PocketLM GPU Smoke Test",
        out_dir=tmp_path,
        repo_url="https://github.com/example/pcketlm.git",
        branch="plm-14",
        commit="abc1234",
    )

    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    script = (kernel_dir / "gpu_smoke_kaggle.py").read_text(encoding="utf-8")

    assert metadata["id"] == "tester/pcketlm-gpu-smoke-test"
    assert metadata["enable_gpu"] is True
    assert metadata["is_private"] is True
    assert metadata["code_file"] == "gpu_smoke_kaggle.py"
    assert "run_gpu_smoke(require_cuda=True)" in script
    assert "SOURCE_COMMIT = 'abc1234'" in script
    assert "PCKETLM_GPU_SMOKE_JSON_START" in script


def test_kaggle_credentials_reports_missing_without_secret(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    creds = kaggle_credentials()

    assert creds.available is False
    assert creds.username is None
    assert "missing" in creds.message.lower()
