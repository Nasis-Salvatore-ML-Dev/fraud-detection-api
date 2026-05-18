"""Load and validate the XGBoost fraud detection model bundle."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
from xgboost import XGBClassifier

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "xgboost_fraud_v1.pkl"

_REQUIRED_KEYS = {"model", "feature_names", "threshold", "version"}


@dataclass
class ModelBundle:
    model: XGBClassifier
    feature_names: list[str]
    threshold: float
    version: str


def load_model_bundle() -> ModelBundle:
    """Load the model bundle from disk and return a validated ModelBundle.

    The path is resolved from the MODEL_PATH environment variable; if unset,
    defaults to models/xgboost_fraud_v1.pkl relative to the repo root.

    Raises:
        RuntimeError: If the file is missing, unpicklable, or the bundle is
                      missing any of the required keys.
    """
    raw_path = os.environ.get("MODEL_PATH", str(_DEFAULT_MODEL_PATH))
    model_path = Path(raw_path)

    if not model_path.is_absolute():
        model_path = _REPO_ROOT / model_path

    log.info("Loading model bundle from %s", model_path)

    try:
        bundle: dict = joblib.load(model_path)
    except FileNotFoundError:
        raise RuntimeError(
            f"Model file not found: {model_path}. "
            "Set the MODEL_PATH environment variable to the correct path."
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to deserialize model bundle from {model_path}: {exc}") from exc

    missing = _REQUIRED_KEYS - set(bundle.keys())
    if missing:
        raise RuntimeError(
            f"Model bundle at {model_path} is missing required keys: {sorted(missing)}. "
            f"Found keys: {sorted(bundle.keys())}"
        )

    result = ModelBundle(
        model=bundle["model"],
        feature_names=bundle["feature_names"],
        threshold=bundle["threshold"],
        version=bundle["version"],
    )

    log.info(
        "Model loaded: version=%s  features=%d  threshold=%.4f",
        result.version,
        len(result.feature_names),
        result.threshold,
    )

    return result


def inverse_transform_probability(probability: float) -> float:
    """Return the probability unchanged.

    Identity function kept for API consistency with the car-valuation project
    pattern, where this step applies a log-target inverse transform.
    Fraud probability is already in [0, 1] and needs no transformation.
    """
    return probability
