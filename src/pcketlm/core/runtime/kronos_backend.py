"""Optional CPU adapter for Kronos financial time-series forecasting."""

from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pcketlm.core.storage.paths import original_model_root


@dataclass(slots=True)
class KronosBackendStatus:
    model_id: str
    ready: bool
    package_available: bool
    pandas_available: bool
    model_path: Path
    tokenizer_path: Path
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "ready": self.ready,
            "package_available": self.package_available,
            "pandas_available": self.pandas_available,
            "model_path": str(self.model_path),
            "tokenizer_path": str(self.tokenizer_path),
            "blockers": list(self.blockers),
            "capability": "forecast",
            "chat_status": "Unsupported",
        }


@dataclass(slots=True)
class KronosForecastResult:
    ready: bool
    model_id: str
    predictions: list[dict] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    backend: str = "kronos-cpu"

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "model_id": self.model_id,
            "predictions": list(self.predictions),
            "blockers": list(self.blockers),
            "backend": self.backend,
            "capability": "forecast",
        }


def _default_tokenizer_path() -> Path:
    configured = os.environ.get("PCKETLM_KRONOS_TOKENIZER_PATH")
    return Path(configured) if configured else original_model_root("kronos-tokenizer-base")


def build_kronos_backend_status(
    model_id: str = "kronos-small",
    *,
    model_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
) -> KronosBackendStatus:
    resolved_model = Path(model_path) if model_path else original_model_root(model_id)
    resolved_tokenizer = Path(tokenizer_path) if tokenizer_path else _default_tokenizer_path()
    package_available = importlib.util.find_spec("model") is not None
    pandas_available = importlib.util.find_spec("pandas") is not None
    blockers: list[str] = []
    if not resolved_model.exists():
        blockers.append("Kronos model source is missing.")
    if not resolved_tokenizer.exists():
        blockers.append("Kronos tokenizer source is missing.")
    if not package_available:
        blockers.append("The official Kronos 'model' package is not installed.")
    if not pandas_available:
        blockers.append("pandas is not installed.")
    return KronosBackendStatus(
        model_id=model_id,
        ready=not blockers,
        package_available=package_available,
        pandas_available=pandas_available,
        model_path=resolved_model,
        tokenizer_path=resolved_tokenizer,
        blockers=blockers,
    )


def validate_forecast_request(payload: dict) -> dict:
    """Validate and normalize a storage-light Kronos forecast request."""
    rows = payload.get("rows")
    timestamps = payload.get("timestamps")
    future_timestamps = payload.get("future_timestamps")
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list of OHLC records.")
    if not isinstance(timestamps, list) or len(timestamps) != len(rows):
        raise ValueError("timestamps must contain one value per OHLC row.")
    if not isinstance(future_timestamps, list) or not future_timestamps:
        raise ValueError("future_timestamps must be a non-empty list.")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or any(key not in row for key in ("open", "high", "low", "close")):
            raise ValueError(f"row {index} must include open, high, low, and close.")
        if any(not isinstance(row[key], (int, float)) for key in ("open", "high", "low", "close")):
            raise ValueError(f"row {index} OHLC values must be numeric.")
    return {
        "rows": rows,
        "timestamps": timestamps,
        "future_timestamps": future_timestamps,
        "pred_len": len(future_timestamps),
        "temperature": float(payload.get("temperature", 1.0)),
        "top_p": float(payload.get("top_p", 0.9)),
        "sample_count": max(1, int(payload.get("sample_count", 1))),
    }


def run_kronos_forecast(
    payload: dict,
    *,
    model_id: str = "kronos-small",
    model_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    model_factory: Any | None = None,
    tokenizer_factory: Any | None = None,
    predictor_factory: Callable[..., Any] | None = None,
    dataframe_factory: Callable[[list[dict]], Any] | None = None,
    series_factory: Callable[[list], Any] | None = None,
) -> KronosForecastResult:
    """Run the official Kronos predictor on CPU, with injectable fixture factories."""
    request = validate_forecast_request(payload)
    status = build_kronos_backend_status(model_id, model_path=model_path, tokenizer_path=tokenizer_path)
    fixture_mode = all(item is not None for item in (model_factory, tokenizer_factory, predictor_factory, dataframe_factory, series_factory))
    if not status.ready and not fixture_mode:
        return KronosForecastResult(False, model_id, blockers=status.blockers)

    if not fixture_mode:
        kronos_module = importlib.import_module("model")
        pandas = importlib.import_module("pandas")
        model_factory = kronos_module.Kronos
        tokenizer_factory = kronos_module.KronosTokenizer
        predictor_factory = kronos_module.KronosPredictor
        dataframe_factory = pandas.DataFrame
        series_factory = pandas.Series

    model = model_factory.from_pretrained(str(status.model_path))
    tokenizer = tokenizer_factory.from_pretrained(str(status.tokenizer_path))
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(tokenizer, "eval"):
        tokenizer.eval()
    predictor = predictor_factory(model, tokenizer, device="cpu", max_context=512)
    prediction = predictor.predict(
        df=dataframe_factory(request["rows"]),
        x_timestamp=series_factory(request["timestamps"]),
        y_timestamp=series_factory(request["future_timestamps"]),
        pred_len=request["pred_len"],
        T=request["temperature"],
        top_p=request["top_p"],
        sample_count=request["sample_count"],
        verbose=False,
    )
    if hasattr(prediction, "to_dict"):
        predictions = prediction.to_dict(orient="records")
    else:
        predictions = list(prediction)
    return KronosForecastResult(True, model_id, predictions=predictions)
