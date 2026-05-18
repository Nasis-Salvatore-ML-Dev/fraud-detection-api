"""Pydantic v2 request/response schemas for the fraud detection API."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Input features for a single transaction fraud prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "time": 406.0,
                "amount": 149.62,
                "v1": -1.3598071,
                "v2": -0.0727812,
                "v3": 2.5363467,
                "v4": 1.3781553,
                "v5": -0.3383207,
                "v6": 0.4623879,
                "v7": 0.2395986,
                "v8": 0.0986979,
                "v9": 0.3637870,
                "v10": 0.0907941,
                "v11": -0.5515995,
                "v12": -0.6178009,
                "v13": -0.9913898,
                "v14": -0.3111694,
                "v15": 1.4681770,
                "v16": -0.4704005,
                "v17": 0.2079708,
                "v18": 0.0257905,
                "v19": 0.4039936,
                "v20": 0.2514121,
                "v21": -0.0183068,
                "v22": 0.2778376,
                "v23": -0.1104739,
                "v24": 0.0669280,
                "v25": 0.1285394,
                "v26": -0.1891148,
                "v27": 0.1336559,
                "v28": -0.0210531,
            }
        }
    )

    time: float = Field(description="Seconds elapsed since the first transaction in the dataset")
    amount: float = Field(ge=0, description="Transaction amount in dollars")

    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float


class PredictionResponse(BaseModel):
    """Result of a fraud prediction for a single transaction."""

    prediction_id: str
    is_fraud: bool
    fraud_probability: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    shap_values: dict[str, float]
    model_version: str
    processing_time_ms: float
    flagged_for_review: bool = Field(
        description="True when fraud_probability is in [0.3, 0.7] (uncertain region)"
    )
    threshold_used: float


class OverrideRequest(BaseModel):
    """Request to override a model prediction with a human decision."""

    prediction_id: str
    reason: Optional[str] = None
    notes: Optional[str] = None


class OverrideResponse(BaseModel):
    """Confirmation of a prediction override."""

    prediction_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    """API and model health status."""

    status: str
    model_version: str
    model_loaded: bool
    timestamp: str


class BiasReportResponse(BaseModel):
    """Fairness / bias audit report for the deployed model."""

    model_version: str
    overall_auprc: float
    overall_recall: float
    overall_fpr: float
    bias_segments: list[dict]
    computed_at: str
    recommendation: Optional[str] = None


class DriftReportResponse(BaseModel):
    """Feature and prediction drift report."""

    computed_at: str
    n_recent_predictions: int
    overall_status: str = Field(
        description="One of: stable, monitor, action_required"
    )
    features: list[dict]
    recommendation: str
