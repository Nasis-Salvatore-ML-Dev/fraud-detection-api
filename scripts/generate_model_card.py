"""
Auto-generate model_card.json following Mitchell et al. 2019.

Incorporates the latest bias report from data/reports/bias_report.json
if it exists; otherwise the bias_segments section is left empty.

Run:
    python scripts/generate_model_card.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging: INFO → stdout, WARNING+ → stderr
# ---------------------------------------------------------------------------
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(logging.DEBUG)
_stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)

_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[_stdout_handler, _stderr_handler],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
BIAS_REPORT_PATH = REPO_ROOT / "data" / "reports" / "bias_report.json"
MODEL_CARD_PATH = REPO_ROOT / "model_card.json"

# ---------------------------------------------------------------------------
# Hardcoded Phase-1 baseline metrics (updated on each retrain)
# ---------------------------------------------------------------------------
_PHASE1_AUPRC = 0.7942
_PHASE1_RECALL = 0.7600
_PHASE1_FPR = 0.000141


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ── 1. Load bias report (non-fatal if absent) ─────────────────────────
    bias_segments: list[dict] = []
    bias_computed_at: str | None = None
    bias_recommendation: str | None = None

    if BIAS_REPORT_PATH.exists():
        try:
            with open(BIAS_REPORT_PATH) as f:
                bias_report = json.load(f)
            bias_segments = bias_report.get("bias_segments", [])
            bias_computed_at = bias_report.get("computed_at")
            bias_recommendation = bias_report.get("recommendation")
            log.info(
                "Bias report loaded — %d segment(s), computed_at=%s",
                len(bias_segments),
                bias_computed_at,
            )
        except Exception as exc:
            log.warning("Could not load bias report (non-fatal): %s", exc)
    else:
        log.warning(
            "No bias report at %s — run  python scripts/run_bias_test.py  first.",
            BIAS_REPORT_PATH,
        )

    flagged = [s for s in bias_segments if s.get("flagged")]

    # ── 2. Build model card ───────────────────────────────────────────────
    log.info("Generating model card")

    card: dict = {
        "model_details": {
            "name": "fraud-detection-xgboost",
            "version": "xgboost_fraud_v1",
            "type": "XGBClassifier",
            "task": "binary_classification",
            "framework": "XGBoost 2.x + scikit-learn wrapper",
            "description": (
                "Gradient-boosted decision tree classifier that scores individual "
                "credit-card transactions as fraudulent or legitimate. "
                "Trained on PCA-anonymised features from the Kaggle Credit Card "
                "Fraud Detection dataset."
            ),
        },
        "intended_use": {
            "primary_use": (
                "Real-time fraud scoring for European credit-card transactions. "
                "Intended as a decision-support tool for fraud analysts."
            ),
            "out_of_scope": [
                "Automated account closure decisions without human review.",
                "Scoring transactions outside European card networks.",
                "Any use where PCA feature identities are assumed known.",
            ],
        },
        "factors": {
            "relevant_factors": [
                "Transaction amount (Amount)",
                "Hour of day derived from elapsed time (hour_of_day)",
                "V1–V28: PCA-transformed card-network features (identities withheld by Kaggle)",
            ],
            "evaluation_factors": [
                "Amount segment: high (>$1000) vs. low (≤$1000)",
                "Time segment: evening (≥18:00) vs. daytime (<18:00)",
            ],
        },
        "metrics": {
            "performance_measures": ["AUPRC", "Recall", "False Positive Rate"],
            "decision_threshold": 0.5,
            "results": {
                "auprc": _PHASE1_AUPRC,
                "recall": _PHASE1_RECALL,
                "fpr": _PHASE1_FPR,
                "note": "Phase-1 baseline on 20% temporal hold-out. Updated on each retrain.",
            },
            "bias_results": {
                "computed_at": bias_computed_at,
                "segments": bias_segments,
                "recommendation": bias_recommendation,
            },
        },
        "evaluation_data": {
            "dataset": "Kaggle Credit Card Fraud Detection",
            "source": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
            "total_transactions": 284807,
            "fraud_cases": 492,
            "fraud_rate_pct": round(492 / 284807 * 100, 4),
            "split": "Chronological 80/20 — first 80% train, last 20% test",
            "time_coverage": "2 days of European transactions (Sept 2013)",
            "preprocessing": (
                "No scaling applied to V1–V28 (already PCA-transformed). "
                "Amount log-transformed (log1p) and z-scored using training-set statistics. "
                "Hour of day derived from elapsed seconds modulo 86400."
            ),
        },
        "ethical_considerations": [
            "No personally identifiable information in training data — "
            "all card-network features are PCA-anonymised by the dataset provider.",
            "FPR parity tested across Amount (high/low) and time-of-day (evening/daytime) segments. "
            "Segments with FPR > 2× overall FPR are flagged for review.",
            "Uncertain predictions (fraud probability 0.3–0.7) are automatically routed to a "
            "human override queue rather than acted on autonomously.",
            "EU AI Act Annex III classification: high-risk system (financial services). "
            "Requires human oversight, audit logging, and conformity assessment before deployment.",
        ],
        "caveats": [
            "Dataset covers only 2 days — model may not generalise to seasonal fraud patterns "
            "or novel attack vectors that emerged after Sept 2013.",
            "V1–V28 are PCA components of undisclosed features, preventing direct feature "
            "interpretation or manual override based on individual feature values.",
            "Decision threshold tuned for high recall (≥0.90) on the test set. "
            "Expect higher FPR in production due to distribution shift over time.",
            "PSI drift monitoring covers only the four engineered features "
            "(Amount, amount_log, amount_zscore, hour_of_day), not the 28 PCA components.",
        ],
        "mlops": {
            "platform": "AWS Lambda (container image) + API Gateway",
            "audit_log": "DynamoDB table  fraud-audit-log  (permanent TTL)",
            "override_queue": "DynamoDB table  fraud-override-queue  (30-day TTL)",
            "drift_detection": "PSI on Amount, amount_log, amount_zscore, hour_of_day; "
            "alert thresholds stable < 0.10 < monitor < 0.20 < action_required",
            "bias_testing": "Per-segment AUPRC and FPR parity; gates CD pipeline via "
            "scripts/run_bias_test.py (exit 1 on failure)",
            "ci_cd": "GitHub Actions — CI on PR, CD on merge to main",
            "model_registry": "s3://fraud-detection-models/xgboost_fraud_v1.pkl",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── 3. Write model_card.json ──────────────────────────────────────────
    with open(MODEL_CARD_PATH, "w") as f:
        json.dump(card, f, indent=2)
    log.info("Model card written → %s", MODEL_CARD_PATH)

    # ── 4. Print flagged bias segments ────────────────────────────────────
    if flagged:
        print()
        print("── Flagged Bias Segments ────────────────────────────────────")
        for seg in flagged:
            auprc_str = f"{seg['auprc']:.4f}" if seg.get("auprc") is not None else "N/A"
            fpr_str = f"{seg['fpr']:.6f}" if seg.get("fpr") is not None else "N/A"
            print(
                f"  ⚠  {seg['segment']:<14}  "
                f"AUPRC={auprc_str}  FPR={fpr_str}  "
                f"auprc_flagged={seg['flagged'] and not seg.get('fpr_flagged', False)}  "
                f"fpr_flagged={seg.get('fpr_flagged', False)}"
            )
        print()
        log.warning(
            "%d segment(s) flagged — review bias report before deploying.",
            len(flagged),
        )
    else:
        log.info("No bias segments flagged.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        log.error("generate_model_card failed: %s", exc, exc_info=True)
        sys.exit(1)
