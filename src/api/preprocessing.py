"""Transform a PredictionRequest into a model-ready DataFrame.

Feature engineering mirrors scripts/train.py exactly. Column order is
validated against the feature_names passed in at request time.
"""

import numpy as np
import pandas as pd

from src.api.schemas import PredictionRequest

# Fixed training-set statistics for amount_zscore (must match train.py ddof=0)
_AMOUNT_MEAN: float = 90.8249
_AMOUNT_STD: float = 250.5032


def build_feature_dataframe(
    request: PredictionRequest,
    feature_names: list[str],
) -> pd.DataFrame:
    """Convert a PredictionRequest into a single-row DataFrame for inference.

    Applies the same feature engineering as scripts/train.py:
      - amount_log    = log1p(Amount)
      - amount_zscore = (Amount - 90.8249) / 250.5032   (training-set stats)
      - hour_of_day   = (Time % 86400) / 3600

    Returns a DataFrame whose columns are in the exact order the model expects.

    Raises:
        ValueError: If the constructed feature set does not match feature_names.
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
        ordered = {col: raw[col] for col in feature_names}
    except KeyError as missing:
        raise ValueError(
            f"Engineered features are missing column(s) expected by the model: {missing}. "
            f"Model expects: {feature_names}"
        ) from missing

    df = pd.DataFrame([ordered])

    if list(df.columns) != feature_names:
        raise ValueError(
            f"Feature mismatch after construction.\n"
            f"  Got:      {list(df.columns)}\n"
            f"  Expected: {feature_names}"
        )

    return df
