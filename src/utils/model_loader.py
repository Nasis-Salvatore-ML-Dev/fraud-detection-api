"""Load and validate the XGBoost fraud detection model bundle.

At Lambda cold-start the model and baseline are not baked into the image.
Both are downloaded from S3 on first call if not already present locally.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
from xgboost import XGBClassifier

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = "/tmp/xgboost_fraud_v1.pkl"

_DEFAULT_S3_BUCKET = "fraud-model-artifacts-209998132741"
_DEFAULT_S3_KEY = "models/xgboost_fraud_v1.pkl"

_LOCAL_BASELINE_PATH = _REPO_ROOT / "data" / "baselines" / "training_baseline.json"
_TMP_BASELINE_PATH = Path("/tmp/training_baseline.json")
_S3_BASELINE_KEY = "baselines/training_baseline.json"

_REQUIRED_KEYS = {"model", "feature_names", "threshold", "version"}


@dataclass
class ModelBundle:
    model: XGBClassifier
    feature_names: list[str]
    threshold: float
    version: str


def _download_from_s3(bucket: str, key: str, dest: str) -> None:
    import boto3

    region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
    s3 = boto3.client("s3", region_name=region)
    log.info("Downloading s3://%s/%s → %s", bucket, key, dest)
    s3.download_file(bucket, key, dest)
    log.info("Download complete: %s", dest)


def _ensure_baseline() -> None:
    """Download the training baseline from S3 to /tmp/ if not available locally."""
    if os.environ.get("BASELINE_PATH"):
        log.info("BASELINE_PATH already set: %s", os.environ["BASELINE_PATH"])
        return

    if _LOCAL_BASELINE_PATH.exists():
        log.info("Baseline found at repo path: %s", _LOCAL_BASELINE_PATH)
        return

    if _TMP_BASELINE_PATH.exists():
        log.info("Baseline already in /tmp: %s", _TMP_BASELINE_PATH)
        os.environ["BASELINE_PATH"] = str(_TMP_BASELINE_PATH)
        return

    bucket = os.environ.get("MODEL_S3_BUCKET", _DEFAULT_S3_BUCKET)
    try:
        _download_from_s3(bucket, _S3_BASELINE_KEY, str(_TMP_BASELINE_PATH))
        os.environ["BASELINE_PATH"] = str(_TMP_BASELINE_PATH)
        log.info("BASELINE_PATH set to %s", _TMP_BASELINE_PATH)
    except Exception as exc:
        log.warning("Could not download baseline from S3 (non-fatal): %s", exc)


def load_model_bundle() -> ModelBundle:
    """Load the model bundle, downloading from S3 if not present locally.

    Resolution order:
    1. MODEL_PATH env var (if set and file exists)
    2. /tmp/xgboost_fraud_v1.pkl (already downloaded in a prior invocation)
    3. S3 download → /tmp/xgboost_fraud_v1.pkl

    Also ensures the training baseline is available, downloading from S3 if
    needed and setting BASELINE_PATH so DriftMonitor can locate it.

    Raises:
        RuntimeError: If the model cannot be loaded from any source.
    """
    _ensure_baseline()

    raw_path = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
    model_path = Path(raw_path)

    if not model_path.is_absolute():
        model_path = _REPO_ROOT / model_path

    if model_path.exists():
        log.info("Loading model bundle from local path: %s", model_path)
    else:
        bucket = os.environ.get("MODEL_S3_BUCKET", _DEFAULT_S3_BUCKET)
        key = os.environ.get("MODEL_S3_KEY", _DEFAULT_S3_KEY)
        dest = _DEFAULT_MODEL_PATH
        try:
            _download_from_s3(bucket, key, dest)
        except Exception as exc:
            raise RuntimeError(
                f"Model not found at {model_path} and S3 download failed: {exc}"
            ) from exc
        model_path = Path(dest)
        os.environ["MODEL_PATH"] = dest

    try:
        bundle: dict = joblib.load(model_path)
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
