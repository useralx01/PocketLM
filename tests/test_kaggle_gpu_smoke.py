from __future__ import annotations

import json

from tools.kaggle_gpu_smoke import kaggle_credentials, kernel_id_from_submit_output, prepare_kernel
from tools.kaggle_deepseek_gpu_validate import prepare_kernel as prepare_deepseek_kernel


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
    notebook = json.loads((kernel_dir / "gpu_smoke_kaggle.ipynb").read_text(encoding="utf-8"))
    script = "".join(notebook["cells"][0]["source"])

    assert metadata["id"] == "tester/pcketlm-gpu-smoke-test"
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == "true"
    assert metadata["is_private"] == "true"
    assert metadata["kernel_type"] == "notebook"
    assert metadata["code_file"] == "gpu_smoke_kaggle.ipynb"
    assert notebook["cells"][0]["id"] == "gpu-smoke"
    assert "PCKETLM_NOTEBOOK_PYTHON_PROBE" in script
    assert "PCKETLM_NOTEBOOK_PYTHON_SELECTED" in script
    assert "/opt/conda/bin/python" in script
    assert "run_gpu_smoke(require_cuda=True)" in script
    assert "SOURCE_COMMIT = " in script
    assert "abc1234" in script
    assert "PCKETLM_GPU_SMOKE_JSON_START" in script
    assert "PCKETLM_TORCH_PROBE" in script
    assert "PCKETLM_TORCH_REPAIR_FAILED" in script
    assert "download.pytorch.org/whl/cu121" in script
    assert "from __future__ import annotations" not in script
    assert "raise SystemExit(main())" not in script


def test_prepare_deepseek_gpu_validate_kernel_embeds_remote_validator(tmp_path) -> None:
    kernel_dir = prepare_deepseek_kernel(
        username="tester",
        slug="pcketlm-deepseek-test",
        title="pcketlm-deepseek-test",
        out_dir=tmp_path,
        resident_layers=3,
        resident_iterations=2,
        paged_layers=4,
        paged_budget_gb=6.0,
    )

    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((kernel_dir / "deepseek_gpu_validate.ipynb").read_text(encoding="utf-8"))
    script = "".join(notebook["cells"][0]["source"])

    assert metadata["id"] == "tester/pcketlm-deepseek-test"
    assert metadata["kernel_type"] == "notebook"
    assert metadata["enable_gpu"] == "true"
    assert "deepseek_remote_gpu_embedded.py" in script
    assert "run_remote_deepseek_resident_decode_probe" in script
    assert "run_remote_deepseek_paged_decode_probe" in script
    assert "PCKETLM_DEEPSEEK_VALIDATE_OUTPUT_START" in script
    assert "layer_count=3" in script
    assert "layer_count=4" in script


def test_prepare_deepseek_gpu_validate_kernel_can_attach_kaggle_model_source(tmp_path) -> None:
    kernel_dir = prepare_deepseek_kernel(
        username="tester",
        slug="pcketlm-deepseek-local-test",
        title="pcketlm-deepseek-local-test",
        out_dir=tmp_path,
        model_source="deepseek-ai/deepseek-v3/transformers/deepseek-v3/2",
        local_decode_layers=4,
        local_budget_gb=12.0,
        local_prefetch_window=1,
        local_max_new_tokens=1,
        repo_url="https://github.com/example/pcketlm.git",
        branch="plm-13",
        commit="abc1234",
    )

    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((kernel_dir / "deepseek_gpu_validate.ipynb").read_text(encoding="utf-8"))
    script = "".join(notebook["cells"][0]["source"])

    assert metadata["model_sources"] == ["deepseek-ai/deepseek-v3/transformers/deepseek-v3/2"]
    assert "run_local_deepseek_paged_decode_loop" in script
    assert "build_tensor_catalog" in script
    assert "LOCAL_DECODE_LAYERS = 4" in script
    assert "SOURCE_COMMIT = 'abc1234'" in script
    assert "/kaggle/input/deepseek-v3/transformers/deepseek-v3/2" in script


def test_kaggle_credentials_reports_missing_without_secret(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    creds = kaggle_credentials()

    assert creds.available is False
    assert creds.username is None
    assert "missing" in creds.message.lower()


def test_kaggle_credentials_accepts_access_token(monkeypatch, tmp_path) -> None:
    token_dir = tmp_path / ".kaggle"
    token_dir.mkdir()
    (token_dir / "access_token").write_text("KGAT_test", encoding="utf-8")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    creds = kaggle_credentials()

    assert creds.available is True
    assert creds.username == "api-token"
    assert creds.source.endswith("access_token")


def test_kernel_id_from_submit_output_uses_returned_url() -> None:
    output = (
        "Kernel version 1 successfully pushed.  Please check progress at "
        "https://www.kaggle.com/code/lichtnicht/pocketlm-gpu-smoke-notebook\n"
    )

    assert kernel_id_from_submit_output(output, "lichtnicht/pocketlm-gpu-smoke-nb") == (
        "lichtnicht/pocketlm-gpu-smoke-notebook"
    )
