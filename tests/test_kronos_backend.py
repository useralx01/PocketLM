from pathlib import Path

import pytest

from pcketlm.core.runtime.kronos_backend import run_kronos_forecast, validate_forecast_request


def _payload() -> dict:
    return {
        "rows": [
            {"open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 100.0},
            {"open": 11.0, "high": 13.0, "low": 10.0, "close": 12.0, "volume": 120.0},
        ],
        "timestamps": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
        "future_timestamps": ["2026-01-01T02:00:00Z"],
    }


def test_validate_forecast_request_rejects_chat_text() -> None:
    with pytest.raises(ValueError, match="rows"):
        validate_forecast_request({"prompt": "Will BTC go up?"})


def test_kronos_forecast_adapter_contract_runs_on_cpu_fixture(tmp_path: Path) -> None:
    captured = {}

    class FakeLoadable:
        @classmethod
        def from_pretrained(cls, path):
            captured.setdefault("paths", []).append(path)
            return cls()

        def eval(self):
            return self

    class FakePredictor:
        def __init__(self, model, tokenizer, **kwargs):
            captured["device"] = kwargs["device"]

        def predict(self, **kwargs):
            captured["pred_len"] = kwargs["pred_len"]
            return [{"open": 12.0, "high": 13.5, "low": 11.5, "close": 13.0}]

    result = run_kronos_forecast(
        _payload(),
        model_path=tmp_path / "model",
        tokenizer_path=tmp_path / "tokenizer",
        model_factory=FakeLoadable,
        tokenizer_factory=FakeLoadable,
        predictor_factory=FakePredictor,
        dataframe_factory=lambda rows: rows,
        series_factory=lambda values: values,
    )

    assert result.ready is True
    assert result.predictions[0]["close"] == 13.0
    assert captured["device"] == "cpu"
    assert captured["pred_len"] == 1
