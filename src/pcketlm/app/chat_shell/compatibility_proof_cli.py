"""Generate a storage-light proof for Pocket LLM model compatibility contracts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pcketlm.core.model_compatibility import build_compatibility_matrix, compatibility_proof_path
from pcketlm.core.model_import.inspect import inspect_model_source
from pcketlm.core.runtime.kronos_backend import run_kronos_forecast
from pcketlm.core.validation.files import validate_model_source


class _FixtureLoadable:
    @classmethod
    def from_pretrained(cls, _path: str):
        return cls()

    def eval(self):
        return self


class _FixturePredictor:
    def __init__(self, _model, _tokenizer, **kwargs):
        if kwargs.get("device") != "cpu":
            raise ValueError("Kronos fixture must use CPU.")

    def predict(self, **_kwargs):
        return [{"open": 12.0, "high": 13.5, "low": 11.5, "close": 13.0}]


def _run_contract_tests() -> dict:
    tests = [
        "tests/test_model_families.py",
        "tests/test_model_import_inspect.py",
        "tests/test_validation_files.py",
        "tests/test_download_state.py",
        "tests/test_registry_catalog.py",
        "tests/test_model_compatibility.py",
        "tests/test_kronos_backend.py",
        "tests/test_gguf_backend.py",
        "tests/test_web_main.py",
    ]
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    summary = next((line for line in reversed(output.splitlines()) if " passed" in line), output[-500:])
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "summary": summary,
    }


def build_proof(*, test_verification: dict | None = None) -> dict:
    targets: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="pcketlm-compat-") as raw_root:
        root = Path(raw_root)
        gemma = root / "gemma"
        gemma.mkdir()
        (gemma / "config.json").write_text(
            '{"model_type":"gemma3_text","architectures":["Gemma3ForCausalLM"]}',
            encoding="utf-8",
        )
        (gemma / "tokenizer.model").write_bytes(b"fixture")
        (gemma / "model.safetensors").write_bytes(b"fixture")
        gemma_inspection = inspect_model_source(gemma)
        gemma_validation = validate_model_source(gemma, family="gemma")
        targets["gemma-3"] = {
            "contract_passed": gemma_validation.result == "ok" and gemma_inspection.config.model_type == "gemma3_text",
            "route": "llama.cpp / GGUF chat template",
            "full_weights_tested": False,
        }

        kimi = root / "kimi"
        kimi.mkdir()
        (kimi / "kimi-k2-fixture.gguf").write_bytes(b"GGUF")
        kimi_validation = validate_model_source(kimi, family="kimi")
        targets["kimi-k2"] = {
            "contract_passed": kimi_validation.result == "ok" and inspect_model_source(kimi).format_name == "gguf",
            "route": "llama.cpp / GGUF chat template",
            "full_weights_tested": False,
        }

        forecast = run_kronos_forecast(
            {
                "rows": [
                    {"open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0},
                    {"open": 11.0, "high": 13.0, "low": 10.0, "close": 12.0},
                ],
                "timestamps": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
                "future_timestamps": ["2026-01-01T02:00:00Z"],
            },
            model_path=root / "kronos-model",
            tokenizer_path=root / "kronos-tokenizer",
            model_factory=_FixtureLoadable,
            tokenizer_factory=_FixtureLoadable,
            predictor_factory=_FixturePredictor,
            dataframe_factory=lambda rows: rows,
            series_factory=lambda values: values,
        )
        targets["kronos"] = {
            "contract_passed": forecast.ready and forecast.predictions[0]["close"] == 13.0,
            "route": "official Kronos-compatible CPU forecast adapter",
            "chat_status": "Unsupported",
            "full_weights_tested": False,
        }

    matrix = [row.to_dict() for row in build_compatibility_matrix()]
    overall_pass = all(item["contract_passed"] for item in targets.values())
    if test_verification is not None:
        overall_pass = overall_pass and bool(test_verification.get("passed"))
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "proof_type": "storage-light compatibility contract",
        "overall_pass": overall_pass,
        "targets": targets,
        "test_verification": test_verification,
        "live_compatibility_matrix": matrix,
        "limitations": [
            "No Kimi, Gemma, or Kronos production weights were downloaded.",
            "Production inference remains unproven until each local weight set is installed and benchmarked.",
        ],
    }


def main() -> int:
    proof = build_proof(test_verification=_run_contract_tests())
    path = compatibility_proof_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps({"proof": str(path), "overall_pass": proof["overall_pass"]}, indent=2))
    return 0 if proof["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
