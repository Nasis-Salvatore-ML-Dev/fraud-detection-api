# fraud-detection-api

Real-time credit-card fraud scoring API — XGBoost on AWS Lambda with DynamoDB audit logging, PSI drift monitoring, and an EU AI Act–aligned bias testing gate.

---

## Live endpoints

**Base URL (staging):** `https://4mhpswg272.execute-api.eu-central-1.amazonaws.com`

> **Availability note:** This is a portfolio project deployed on AWS free tier. The staging endpoint is live as of May 2026. If you are reading this significantly later, the endpoint may have been taken down to avoid post-free-tier costs. To run locally, see [Local development](#local-development).

| Method | Path            | Description                                                                                                           |
| ------ | --------------- | --------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/health`       | Liveness check — returns model version and load status                                                                |
| `POST` | `/predict`      | Fraud probability, confidence score, and review flag                                                                  |
| `GET`  | `/explain/{id}` | SHAP breakdown (architecture ready; disabled in Lambda runtime — see [Engineering decisions](#engineering-decisions)) |
| `POST` | `/override`     | Flag a prediction for human review; writes to override queue                                                          |
| `GET`  | `/drift`        | PSI drift report across the 500 most recent predictions                                                               |
| `GET`  | `/metrics`      | Per-segment bias report and FPR parity results                                                                        |

---

## Quick start

### Predict

```bash
curl -X POST https://4mhpswg272.execute-api.eu-central-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time": 0.0,
    "amount": 149.62,
    "v1": -1.3598071, "v2": -0.0727812, "v3": 2.5363467,
    "v4": 1.3781553,  "v5": -0.3383207, "v6": 0.4623879,
    "v7": 0.2395986,  "v8": 0.0986979,  "v9": 0.3637870,
    "v10": 0.0907941, "v11": -0.5515995, "v12": -0.6178009,
    "v13": -0.9913898, "v14": -0.3111694, "v15": 1.4681770,
    "v16": -0.4704005, "v17": 0.2079708,  "v18": 0.0257905,
    "v19": 0.4039936,  "v20": 0.2514121,  "v21": -0.0183068,
    "v22": 0.2778376,  "v23": -0.1104739, "v24": 0.0669280,
    "v25": 0.1285394,  "v26": -0.1891148, "v27": 0.1336559,
    "v28": -0.0210531
  }'
```

**Response**

```json
{
  "prediction_id": "3f7a1c2e-84b0-4d9a-a3e1-0f52c8d6e291",
  "is_fraud": false,
  "fraud_probability": 0.0008,
  "confidence_score": 0.9,
  "shap_values": {},
  "model_version": "xgboost_fraud_v1",
  "processing_time_ms": 4.2,
  "flagged_for_review": false,
  "threshold_used": 0.5
}
```

> `confidence_score` is `0.9` when `fraud_probability` is outside the uncertain band [0.3, 0.7] and `0.4` when inside it. Predictions inside the band are also `flagged_for_review: true` and routed to the human override queue.

### Override a flagged prediction

```bash
curl -X POST https://4mhpswg272.execute-api.eu-central-1.amazonaws.com/override \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "3f7a1c2e-84b0-4d9a-a3e1-0f52c8d6e291",
    "reason": "Customer confirmed transaction",
    "notes": "Called cardholder — verified purchase"
  }'
```

---

## Architecture

### Deployment pipeline

```
GitHub push
  └── CI (GitHub Actions)
        ├── ruff lint + format check
        ├── mypy type check
        └── pytest unit + integration tests
              └── merge to main → CD (concurrency: one deployment at a time, cancel-in-progress: false)
                    ├── docker build + push to ECR (tagged git SHA + latest)
                    ├── staging Lambda update
                    ├── staging smoke test (GET /health + POST /predict → exit 1 on fail)
                    ├── bias gate (run_bias_test.py → exit 1 if any segment flagged)
                    ├── generate_model_card.py → model_card.json → S3
                    ├── production Lambda update (only runs if all above pass)
                    └── production smoke test (GET /health → exit 1 on fail)
```

### Runtime path

```
Client
  └── API Gateway (REST)
        └── Lambda (FastAPI + Mangum, container image)
              ├── XGBoost predict_proba
              ├── DynamoDB  fraud-audit-log     (permanent, compliance)
              └── DynamoDB  fraud-override-queue (30-day TTL, human review)

CloudWatch ← PSI metrics (FraudDetection/FraudPSI, per feature)
```

**Cold-start sequence** (once per container): load model bundle from `MODEL_PATH` → init `AuditLogger` → init `DriftMonitor` (loads PSI baseline) → init `BiasTestSuite` (non-fatal).

---

## Model

| Metric    | Value        | Notes                                                                                                               |
| --------- | ------------ | ------------------------------------------------------------------------------------------------------------------- |
| AUPRC     | **0.7942**   | Area under precision-recall curve on hold-out set                                                                   |
| Recall    | **0.7600**   | Fraction of fraudulent transactions correctly caught                                                                |
| FPR       | **0.000141** | False positive rate — legitimate transactions incorrectly flagged                                                   |
| Threshold | 0.5          | Default threshold; recall and FPR measured at this value. Optuna tuning in progress to push AUPRC toward 0.85–0.90. |

**Dataset:** Kaggle Credit Card Fraud Detection — 284,807 transactions, 492 frauds (0.17%), two days of European card activity, September 2013. Split chronologically: first 80% for training, last 20% for evaluation. V1–V28 are PCA-transformed card-network features; the original feature identities are withheld by the dataset provider.

**Known limitations:**

- Two days of data — model has not seen seasonal fraud patterns or attacks that post-date September 2013.
- V1–V28 cannot be interpreted directly; SHAP values explain the PCA components, not the original transaction attributes.
- FPR measured on the test set; expect higher FPR in production as the transaction distribution drifts.

### Bias testing

Four segments are tested on every CD run:

| Segment       | Definition                 | Gate condition                              |
| ------------- | -------------------------- | ------------------------------------------- |
| `high_amount` | Amount > $1,000            | AUPRC < 70% of overall, or FPR > 2× overall |
| `low_amount`  | Amount ≤ $1,000            | same                                        |
| `high_hour`   | hour_of_day ≥ 18 (evening) | same                                        |
| `low_hour`    | hour_of_day < 18 (daytime) | same                                        |

A segment failure blocks deployment (`run_bias_test.py` exits 1).

---

## MLOps pipeline

### CI — runs on every pull request

1. `ruff check` + `ruff format --check` — style and import order
2. `mypy src/` — static type checking
3. `pytest tests/unit/` with coverage report

### CD — runs on merge to `main`

1. `docker build` with `GIT_SHA` build arg → push to ECR (tagged `{sha}` + `latest`)
2. Staging Lambda update (`update-function-code`)
3. Staging smoke test: `GET /health` must return `model_loaded: true` + `POST /predict` must return valid response
4. Bias gate: `PYTHONPATH=. python scripts/run_bias_test.py` — exits 1 and blocks if any segment is flagged
5. `python scripts/generate_model_card.py` — regenerates `model_card.json` → uploads to S3
6. PSI baseline verified: `python scripts/compute_baseline.py`
7. Production Lambda update — only runs if all prior steps pass
8. Production smoke test: `GET /health` must return 200

**Drift monitoring:** `GET /drift` computes PSI across the last 500 predictions against the training baseline. CloudWatch receives one `FraudPSI` metric per feature per call. PSI thresholds: < 0.10 stable, 0.10–0.20 monitor, ≥ 0.20 action required.

---

## Engineering decisions

**XGBoost over neural approaches** — gradient-boosted trees match or exceed neural networks on tabular fraud data while training in minutes, supporting SHAP exact values, and producing calibrated probabilities without a separate calibration step.

**AWS Lambda over ECS** — fraud scoring is bursty and latency-tolerant at the p50 level. Lambda's per-request billing eliminates the cost of idle capacity, and the container image runtime removes the 250 MB deployment package limit. Cold-start latency (~800 ms) is acceptable for a fraud-review workflow that is not synchronous with the card authorisation path.

**DynamoDB over RDS for audit logs** — the audit log is append-only with UUID primary keys and no relational queries. DynamoDB's on-demand capacity mode, point-in-time recovery, and per-item TTL satisfy the compliance requirement without schema migrations. `fraud-audit-log` has no TTL (permanent); `fraud-override-queue` expires items after 30 days.

**PSI for drift detection** — Population Stability Index is symmetric, bounded below by zero, and has established industry thresholds (0.10, 0.20) that map cleanly to operational alert levels. KS-test and MMD were considered; PSI was chosen for its interpretability to non-ML stakeholders and its suitability for monitoring univariate marginals without needing a reference label.

**SHAP disabled in Lambda runtime** — the SHAP package requires C extensions compiled against the exact CPython ABI of the execution environment. The Lambda container image uses a stripped runtime whose ABI differs from the locally-built wheel. The `SHAPExplainer` class and `src/explainability/` module are fully implemented and tested locally; re-enabling requires building SHAP inside the container image and setting `SHAP_ENABLED=1`. The `/explain/{id}` endpoint returns the stored audit record — when SHAP is enabled, this includes per-feature attribution values.

**FPR parity testing** — a model with good overall recall can still disproportionately flag legitimate transactions in a specific population segment (high-value purchases, evening transactions). FPR > 2× the overall rate in any segment triggers the CD gate. This threshold is conservative by design; EU AI Act Annex III classifies real-time credit scoring as high-risk.

---

## EU AI Act compliance

| Article                           | Requirement                                      | Implementation                                                                             |
| --------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Art. 9 — Risk management          | Systematic testing before deployment             | Bias gate in CD pipeline; `run_bias_test.py` exits 1 on failure                            |
| Art. 10 — Data governance         | Fairness across relevant population segments     | FPR parity tested across Amount and time-of-day segments                                   |
| Art. 11 — Technical documentation | Documented model characteristics and performance | Auto-generated `model_card.json` on every deployment                                       |
| Art. 12 — Record keeping          | Audit trail for all automated decisions          | DynamoDB `fraud-audit-log` — permanent, append-only, UUID-keyed                            |
| Art. 13 — Transparency            | Explainable outputs                              | SHAP architecture implemented; per-prediction attribution stored in audit log              |
| Art. 14 — Human oversight         | Humans can review and override                   | Override queue (`fraud-override-queue`) + confidence gating for uncertain predictions      |
| Art. 15 — Accuracy and robustness | Monitoring for performance degradation           | PSI drift monitoring with CloudWatch metrics; rule-based fallback via threshold adjustment |

---

## Dataset citation

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.  
_Calibrating Probability with Undersampling for Unbalanced Classification._  
In Proceedings of the IEEE Symposium Series on Computational Intelligence (SSCI), 2015.

Yann-Aël Le Borgne and Gianluca Bontempi.  
_Reproducible Machine Learning for Credit Card Fraud Detection — Practical Handbook._  
Université Libre de Bruxelles, 2022.

Dataset hosted by the Machine Learning Group, Université Libre de Bruxelles (ULB), in collaboration with Worldline.  
Available at: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Local development

```bash
# Install all dependencies
make install

# Format, lint, type-check, and test in one step
make all

# Individual steps
make format        # ruff format + ruff check --fix
make lint          # ruff check + ruff format --check
make type-check    # mypy src/
make test          # pytest tests/unit/ with coverage

# ML pipeline
make train         # train XGBoost, save model bundle + PSI baseline
make baseline      # verify PSI baseline exists and print summary
make bias-test     # run bias gate (PYTHONPATH=. required)
make model-card    # regenerate model_card.json
make export-onnx   # convert to ONNX and benchmark (requires skl2onnx)

# Docker
make docker-build  # build Lambda container image
make docker-run    # run locally on port 9000

# Integration and load tests
make test-integration
# locust -f tests/load/locustfile.py --host=https://4mhpswg272.execute-api.eu-central-1.amazonaws.com
```

**Environment variables**

| Variable             | Default                       | Description                                 |
| -------------------- | ----------------------------- | ------------------------------------------- |
| `MODEL_PATH`         | `models/xgboost_fraud_v1.pkl` | Path to model bundle                        |
| `MODEL_VERSION`      | `xgboost_fraud_v1`            | Version label logged with each prediction   |
| `AUDIT_TABLE`        | `fraud-audit-log`             | DynamoDB table for prediction audit records |
| `AWS_DEFAULT_REGION` | `eu-central-1`                | AWS region for DynamoDB and CloudWatch      |




# Attribution License 1.0

Copyright (c) 2026 Salvatore Nasisi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the **"Software"**), to use, study, copy, modify, merge, publish, distribute, and sublicense the Software, subject to the following conditions:

---

## 1. Attribution Required

All copies or substantial portions of the Software, including modified or derivative works, must retain:

- the original copyright notice,
- this license text,
- and clear attribution to the original author: **Salvo**.

---

## 2. No False Authorship Claims

You may not claim that the original Software was created entirely by you.

Modified versions must clearly indicate that changes were made and must not misrepresent the origin of the original work.

---

## 3. Redistribution Conditions

Any public redistribution of the Software, whether modified or unmodified, must include visible acknowledgment of the original author in:

- source code,
- documentation,
- or repository metadata.

### Example acknowledgment

> "Based on original work by Salvo."

---

## 4. Personal and Private Use

Private, personal, or internal use without redistribution does not require public attribution.

---

## 5. Commercial Use

Commercial use is permitted provided attribution requirements are preserved and authorship is not misrepresented.

---

## 6. Warranty Disclaimer

THE SOFTWARE IS PROVIDED **"AS IS"**, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.

IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 7. Termination

Any violation of this license automatically terminates the rights granted under it.

---

By using, copying, modifying, or distributing this Software, you agree to the terms of this license.
