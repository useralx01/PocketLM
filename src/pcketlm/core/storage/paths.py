"""Common project paths."""

from pathlib import Path


def project_root() -> Path:
    """Return the pcketlm project root."""
    return Path(__file__).resolve().parents[4]


def models_root() -> Path:
    """Return the models directory."""
    return project_root() / "models"


def state_root() -> Path:
    """Return the state directory."""
    return project_root() / "state"


def registry_root() -> Path:
    """Return the registry metadata directory."""
    return state_root() / "registry"


def registry_file() -> Path:
    """Return the model registry file path."""
    return registry_root() / "models.json"


def streaming_state_root() -> Path:
    """Return the root directory for staged streaming state."""
    return state_root() / "streaming"


def streaming_model_root(model_id: str) -> Path:
    """Return the staged streaming state directory for one model."""
    return streaming_state_root() / model_id


def model_root(model_id: str) -> Path:
    """Return the root directory for one model."""
    return models_root() / model_id


def original_model_root(model_id: str) -> Path:
    """Return the immutable original directory for one model."""
    return model_root(model_id) / "original"


def artifacts_root(model_id: str) -> Path:
    """Return the artifacts directory for one model."""
    return model_root(model_id) / "artifacts"


def profiles_root(model_id: str) -> Path:
    """Return the profiles directory for one model."""
    return model_root(model_id) / "profiles"


def benchmarks_root(model_id: str) -> Path:
    """Return the benchmarks directory for one model."""
    return model_root(model_id) / "benchmarks"


def ensure_base_directories() -> None:
    """Create the base project directories if they do not exist."""
    for path in (models_root(), state_root(), registry_root(), streaming_state_root()):
        path.mkdir(parents=True, exist_ok=True)
