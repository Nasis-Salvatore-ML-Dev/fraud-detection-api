"""Transform a PredictionRequest into a model-ready DataFrame.

Feature engineering mirrors scripts/train.py exactly. Column order is
validated against the saved model bundle at module import time so
mismatches surface immediately rather than silently corrupting predictions.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.api.schemas import PredictionRequest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = _REPO_ROOT / "models" / "xgboost_fraud_v1.pkl"

# Fixed training-set statistics for amount_zscore (must match train.py ddof=0)
_AMOUNT_MEAN: float = 90.8249
_AMOUNT_STD: float = 250.5032


# ---------------------------------------------------------------------------
# Load model feature order once at import time
# ---------------------------------------------------------------------------
def _load_model_feature_names(model_path: Path) -> list[str]:
    """Return the feature_names list stored inside the model bundle."""
    bundle = joblib.load(model_path)
    names: list[str] = bundle["feature_names"]
    return names


try:
    MODEL_FEATURE_NAMES: list[str] = _load_model_feature_names(_MODEL_PATH)
except Exception as exc:
    raise RuntimeError(f"Failed to load model bundle from {_MODEL_PATH}: {exc}") from exc

# Expected feature order, derived from the loaded model (authoritative source).
# Shown here for documentation; do NOT hard-code a separate list to compare against.
_EXPECTED_N_FEATURES = len(MODEL_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_feature_dataframe(request: PredictionRequest) -> pd.DataFrame:
    """Convert a PredictionRequest into a single-row DataFrame for inference.

    Applies the same feature engineering as scripts/train.py:
      - amount_log    = log1p(Amount)
      - amount_zscore = (Amount - 90.8249) / 250.5032   (training-set stats)
      - hour_of_day   = (Time % 86400) / 3600

    Returns a DataFrame whose columns are in the exact order the model expects.

    Raises:
        ValueError: If the constructed feature set does not match the model's
                    expected feature names (number or order mismatch).
    """
    amount = request.amount
    time = request.time

    # Map request fields to model column names (V1-V28 are uppercase in model)
    raw: dict[str, float] = {
        "V1": request.v1,
        "V2": request.v2,
        "V3": request.v3,
        "V4": request.v4,
        "V5": request.v5,
        "V6": request.v6,
        "V7": request.v7,
        "V8": request.v8,
        "V9": request.v9,
        "V10": request.v10,
        "V11": request.v11,
        "V12": request.v12,
        "V13": request.v13,
        "V14": request.v14,
        "V15": request.v15,
        "V16": request.v16,
        "V17": request.v17,
        "V18": request.v18,
        "V19": request.v19,
        "V20": request.v20,
        "V21": request.v21,
        "V22": request.v22,
        "V23": request.v23,
        "V24": request.v24,
        "V25": request.v25,
        "V26": request.v26,
        "V27": request.v27,
        "V28": request.v28,
        "Amount": amount,
        "amount_log": float(np.log1p(amount)),
        "amount_zscore": float((amount - _AMOUNT_MEAN) / _AMOUNT_STD),
        "hour_of_day": float((time % 86400) / 3600),
    }

    # Reorder to match the model's expected feature sequence
    try:
        ordered = {col: raw[col] for col in MODEL_FEATURE_NAMES}
    except KeyError as missing:
        raise ValueError(
            f"Engineered features are missing column(s) expected by the model: {missing}. "
            f"Model expects: {MODEL_FEATURE_NAMES}"
        ) from missing

    df = pd.DataFrame([ordered])

    # Validate column count and names exactly
    if list(df.columns) != MODEL_FEATURE_NAMES:
        raise ValueError(
            f"Feature mismatch after construction.\n"
            f"  Got:      {list(df.columns)}\n"
            f"  Expected: {MODEL_FEATURE_NAMES}"
        )

    return df
