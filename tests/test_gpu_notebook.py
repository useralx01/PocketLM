from __future__ import annotations

import json
from pathlib import Path


def test_gpu_notebook_runs_repo_smoke_commands() -> None:
    notebook = json.loads(Path("notebooks/gpu_test.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "REPO_URL" in source
    assert "git" in source
    assert "tools/gpu_smoke.py" in source
    assert "--require-cuda" in source
    assert "--deepseek-probe" in source
    assert "plm-13-gpu-effective-speed" in source
    assert "pytest" in source
    assert "tests/gpu" in source
