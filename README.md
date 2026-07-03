# Credit Card Fraud Detection — LSTM + FastAPI on AWS

End-to-end fraud detection system: from exploratory analysis and model benchmarking on **1.85M credit card transactions**, to a **sequence-based LSTM model**, a **cost-optimised decision threshold**, and a **containerised real-time API deployed on AWS EC2**.

**Final model (LSTM, hold-out test set):**

| Metric | Score |
|---|---|
| ROC-AUC | 0.9993 |
| PR-AUC | 0.9865 |
| Precision (F1-optimal threshold, t = 0.28) | 0.98 |
| Recall (F1-optimal threshold, t = 0.28) | 0.95 |
| Recall (cost-optimal threshold, t = 0.011) | 0.975 |

---

## Repository Structure

```
├── NoteBooks/
│   ├── Logistic_models.ipynb          # Baseline logistic regression + LASSO feature selection
│   ├── Xg_Boost_models.ipynb          # Tuned XGBoost (TimeSeriesSplit CV) + leakage audit
│   ├── Deep_models.ipynb              # Feed-forward NN + LSTM sequence model
│   └── Estimating_Business_Cost.ipynb # Cost-based threshold optimisation + sensitivity analysis
├── Model_Deployment/
│   ├── Fraud_analysis.ipynb           # EDA and feature engineering
│   ├── service_raw.py                 # FastAPI inference service
│   ├── Dockerfile                     # Container build
│   ├── requirements.txt               # Pinned dependencies
│   ├── AWS_Deployment_Report.pdf      # Deployment write-up
│   └── Artifacts/                     # Trained model + preprocessing artifacts
│       ├── lstm_fraud_model.keras
│       ├── preprocessor.joblib        # Fitted scaler + one-hot encoder
│       ├── schema.json                # Raw input feature schema
│       ├── windowing.json             # Sequence window config (T=32, F=17)
│       ├── threshold.json             # Cost-optimal decision threshold
│       └── model_card.json            # Model metadata
└── Fraud Detection Report.pdf         # Full modelling report
```

## Dataset

[Kaggle — Simulated Credit Card Transactions](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (`fraudTrain.csv` + `fraudTest.csv`, combined for a custom time-based split): **1,852,385 transactions**, of which ~0.5% are fraudulent.

## Feature Engineering

EDA revealed that fraud commonly appears as **burst behaviour** — a small "test" payment followed shortly by a large transaction on the same card. Two engineered features capture this:

- `time_since_last` — seconds since the card's previous transaction
- `last_amt` — the card's previous transaction amount

Final feature set: `amt`, `trans_hour`, `time_since_last`, `last_amt`, `category` (one-hot encoded → 17 model features). A dedicated leakage audit in `Xg_Boost_models.ipynb` recomputes both engineered features from past-only information and verifies zero look-ahead mismatches.

## Modelling Progression

All models are evaluated on held-out test data — a chronological split for the tabular models (random shuffling would leak future behaviour into training), and a grouped-by-card split for the LSTM (each card's transaction history stays within one split).

| Model | PR-AUC | Precision / Recall (best threshold) | Notes |
|---|---|---|---|
| Logistic regression (baseline) | — | 0.02 / 0.76 | Class-weighted; establishes floor |
| + engineered features | — | 0.06 / 0.86 | Big lift from `time_since_last`, `last_amt` |
| LASSO (L1) feature selection | — | — | 128/765 features retained; confirms category + engineered features dominate |
| Feed-forward NN | 0.89 | 0.86 / 0.82 | Class-weighted loss, random-search tuning |
| XGBoost (TimeSeriesSplit CV) | 0.94 | 0.93 / 0.84 | scale_pos_weight for imbalance |
| **LSTM (32-step card sequences)** | **0.99** | **0.98 / 0.95** | Models per-card temporal behaviour |

The LSTM operates on **windows of the last 32 transactions per card** (zero-padded with masking when a card has fewer), letting it learn sequential fraud signatures the tabular models cannot see.

## Cost-Based Threshold Optimisation

Statistical metrics treat false positives and false negatives as equal — in fraud they are not. `Estimating_Business_Cost.ipynb` builds a financial cost model:

- **False negative cost** — mean fraud amount (~$531) × a secondary-cost multiplier (chargebacks, fees, operational handling), sensitivity-tested from 2× to 4×
- **False positive cost** — median lost sale (~$47) + service/support cost ($5–10) + estimated lost future revenue from churned customers (30–45% churn rate, per Riskified research)

The decision threshold is selected on the **validation set** to minimise expected cost, then evaluated on test. Because missed fraud costs an order of magnitude more than a blocked legitimate sale, the cost-optimal threshold sits far below the F1-optimal one (t\* ≈ 0.011 vs 0.28). At the deployed threshold the model catches **97.5% of fraud** (625/641 test cases) with only 94 false positives across 136,737 test transactions.

## Deployment

### Architecture

```
Raw transactions (JSON) → FastAPI → saved preprocessor → 32×17 window → LSTM → probability + decision
```

The service (`service_raw.py`) accepts **raw transactions**, applies the exact fitted preprocessor from training, builds the per-card sequence window server-side, and returns a fraud probability plus a decision at the cost-optimal threshold. All artifacts load once at startup.

### Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow 2.20 / Keras 3.11 |
| API | FastAPI 0.119 + Uvicorn |
| Preprocessing | scikit-learn 1.6 (persisted with joblib) |
| Container | Docker (python-slim base, healthcheck included) |
| Cloud | AWS EC2 — Amazon Linux 2023, t3.micro, Sydney region |

### Run Locally

```bash
cd Model_Deployment
pip install -r requirements.txt
uvicorn service_raw:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

Or with Docker:

```bash
cd Model_Deployment
docker build -t fraud-lstm-api .
docker run -d -p 8000:8000 fraud-lstm-api
curl http://127.0.0.1:8000/health
```

### API

`GET /health`

```json
{
  "status": "ok",
  "timesteps": 32,
  "n_features_encoded": 17,
  "threshold": 0.011,
  "raw_features": ["amt", "trans_hour", "time_since_last", "last_amt", "category"]
}
```

`POST /predict` — send one or more recent transactions for a **single card** (the service builds the sequence window from them):

```json
{
  "transactions": [
    {
      "cc_num": "6011477612335392",
      "trans_date_trans_time": "2025-09-22T21:50:00Z",
      "amt": 3.60,
      "trans_hour": 21,
      "time_since_last": 1626,
      "last_amt": 42.24,
      "category": "home"
    }
  ]
}
```

Response:

```json
{
  "cc_num": "6011477612335392",
  "n_transactions": 1,
  "proba_fraud": 0.0047,
  "decision": 0,
  "threshold": 0.011,
  "window_shape": "(32, 17)"
}
```

> The public EC2 endpoint is only live while the instance is running; use the local instructions above to test.

## Limitations & Future Work

- The simulated dataset is cleaner than production card data; real-world performance would be lower.
- Sequence windows require per-card history — cold-start cards fall back to zero-padded windows, where the model has less signal.
- Cost parameters (churn rate, secondary costs) are research-based estimates; the sensitivity grid bounds but does not eliminate that uncertainty.
- Next steps: model monitoring/drift detection, batch scoring endpoint, CI/CD for retraining, authentication on the API.

## References

- Ali et al. (2022). *Financial Fraud Detection Based on Machine Learning: A Systematic Literature Review.* Applied Sciences, 12(19), 9637.
- LexisNexis Risk Solutions (2024). *True Cost of Fraud Study.*
- J.P. Morgan (2023). *False Positives & Fraud Prevention Tools.*
- Riskified (2025). *How Much Does a False Decline Cost Your Business?*
- Australian Bureau of Statistics (2025). *Personal Fraud, 2023–24 Financial Year.*
