# Credit Card Fraud Detection — LSTM + FastAPI on AWS

End-to-end fraud detection system: from exploratory analysis and model benchmarking on **1.85M credit card transactions**, to a **sequence-based LSTM model**, a **cost-optimised decision threshold**, and a **containerised real-time API deployed on AWS EC2**.

**Final model (LSTM, hold-out test set, per-transaction evaluation):**

| Metric | Score |
|---|---|
| PR-AUC | 0.9659 |
| ROC-AUC | 0.9994 |
| Precision / Recall (F1-optimal threshold, t = 0.28) | 0.81 / 0.96 |
| Recall (deployed cost-optimal threshold, t = 0.011) | 0.977 |

> **Note on evaluation basis.** These numbers score **every transaction in the test slice** (277,858 transactions, 1,447 fraud), using the same zero-padded windows the production API builds for short card histories. An earlier version of this README reported higher figures (precision 0.98 / recall 0.95, PR-AUC 0.9865) that were computed **per window**: the original window builder only scored transactions on cards with ≥32 transactions, at every second position — covering just 641 of the 1,447 fraud cases in the test slice and none of the low-history regime the API actually serves. A self-audit caught this; [`NoteBooks/Honest_Evaluation.ipynb`](NoteBooks/Honest_Evaluation.ipynb) reproduces the old numbers, corrects the basis, and shows exactly where the difference comes from. The model, preprocessing, and thresholds are unchanged — only the measurement was fixed.

---

## Repository Structure

```
├── NoteBooks/
│   ├── Logistic_models.ipynb          # Baseline logistic regression + LASSO feature selection
│   ├── Xg_Boost_models.ipynb          # Tuned XGBoost (TimeSeriesSplit CV) + leakage audit
│   ├── Deep_models.ipynb              # Feed-forward NN + LSTM sequence model
│   ├── Honest_Evaluation.ipynb        # Evaluation audit: per-transaction re-scoring of the deployed model
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

[Kaggle — Simulated Credit Card Transactions](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (`fraudTrain.csv` + `fraudTest.csv`, combined for a custom time-based split): 1,852,394 rows combined, minus 9 records from a single synthetic location with a 100% fraud rate (dropped as a data artefact) — leaving **1,852,385 transactions**, of which ~0.5% are fraudulent.

Because the data is simulated (Sparkov generator), fraud is injected with strongly separable patterns — models routinely score far higher here than on production card data. Absolute numbers should be read with that in mind; the relative comparisons and the methodology are the point.

## Feature Engineering

EDA revealed that fraud commonly appears as **burst behaviour** — a small "test" payment followed shortly by a large transaction on the same card. Two engineered features capture this:

- `time_since_last` — seconds since the card's previous transaction
- `last_amt` — the card's previous transaction amount

Final feature set: `amt`, `trans_hour`, `time_since_last`, `last_amt`, `category` (one-hot encoded → 17 model features). A dedicated leakage audit in `Xg_Boost_models.ipynb` recomputes both engineered features from past-only information and verifies zero look-ahead mismatches.

## Modelling Progression

| Model | Split | PR-AUC | Precision / Recall | Notes |
|---|---|---|---|---|
| Logistic regression (baseline) | chronological | — | 0.02 / 0.76 | Class-weighted; establishes floor |
| + engineered features | chronological | — | 0.06 / 0.86 | Big lift from `time_since_last`, `last_amt` |
| LASSO (L1) feature selection | chronological | — | — | 128/765 features retained; confirms category + engineered features dominate |
| Feed-forward NN | chronological | 0.89 | 0.86 / 0.82 | Class-weighted loss, random-search tuning; threshold from validation |
| XGBoost (TimeSeriesSplit CV) | chronological | 0.94 | — | scale_pos_weight for imbalance. PR-AUC only: the notebook's P/R pair used a test-selected threshold, so it isn't quoted here |
| **LSTM (32-step card sequences)** | **by card** | **0.97** | **0.81 / 0.96** | Per-transaction basis; models per-card temporal behaviour |

Two caveats on comparing across rows, found in the self-audit:

- **The LSTM row uses a different split.** The tabular models train on the past and are tested on the future (chronological split). The LSTM is split **by card** — unseen cards, but spanning the same 2019–2020 period as training — so it answers "does this generalise to new cards?" rather than "does this generalise forward in time?". Part of its lead over XGBoost may come from the easier question. Retraining the LSTM on a chronological split is the top item in Future Work.
- The LSTM operates on **windows of the last 32 transactions per card**, zero-padded when a card has fewer. Training used only full 32-transaction windows, so padded (cold-start) windows are out-of-distribution for the model. The audit shows the cost is precision, not recall: at the deployed threshold the false-positive rate on cold-start windows (≤4 transactions of history) is **~61%**, versus 0.07% on full windows — the model flags most legitimate first purchases on a new card.

## Evaluation Audit & Correction

The original test metrics were computed over sequence *windows* (lookback 32, stride 2, cards with <32 transactions dropped). That basis silently excluded 806 of the 1,447 fraud transactions in the test slice (~56%) — precisely the low-history cases the deployed API still has to score. [`Honest_Evaluation.ipynb`](NoteBooks/Honest_Evaluation.ipynb):

1. rebuilds the data and **reproduces the published per-window confusion matrix exactly** (same model, preprocessor, thresholds), then
2. re-scores **all 277,858 test transactions** with production-identical zero-padding, and
3. breaks results down by card-history length, showing where the previously unmeasured fraud lands.

| Basis | PR-AUC | Precision / Recall @ t = 0.28 | F1 @ t = 0.28 | Recall @ t = 0.011 |
|---|---|---|---|---|
| Per window (as originally published) | 0.9865 | 0.98 / 0.95 | 0.966 | 0.975 |
| **Per transaction (corrected)** | **0.9659** | **0.81 / 0.96** | **0.874** | **0.977** |

The correction barely moves recall — the previously unscored fraud is caught at the same ~0.95–0.98 rate in every history bucket, because the burst/amount signal needs little context on this synthetic data. What the per-window basis hid was **false positives**: 14 → 334 at t = 0.28, concentrated almost entirely in zero-padded cold-start windows the model never saw in training (≤4 transactions of history ⇒ ~48% false-positive rate at t = 0.28, ~61% at the deployed threshold).

## Cost-Based Threshold Optimisation

Statistical metrics treat false positives and false negatives as equal — in fraud they are not. `Estimating_Business_Cost.ipynb` builds a financial cost model:

- **False negative cost** — mean fraud amount (~$531) × a secondary-cost multiplier (chargebacks, fees, operational handling), sensitivity-tested from 2× to 4×
- **False positive cost** — median lost sale (~$47) + service/support cost ($5–10) + estimated lost future revenue from churned customers (30–45% churn rate, per Riskified research)

The decision threshold is selected on the **validation set** to minimise expected cost, then evaluated on test. Because missed fraud costs an order of magnitude more than a blocked legitimate sale, the cost-optimal threshold sits far below the F1-optimal one (t\* ≈ 0.011 vs 0.28). On the corrected per-transaction basis the deployed threshold catches **97.7% of test fraud** (1,413/1,447) at 590 false positives across 277,858 test transactions — 339 of those false positives come from the 620 cold-start windows (see the audit above). The threshold itself was calibrated on per-window validation scores; recalibrating it on a per-transaction basis is listed under Future Work.

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
- The LSTM's by-card split measures generalisation to unseen cards within the same period, not forward in time. **Next: retrain and evaluate the LSTM on the same chronological split as the tabular models.**
- Training used only full 32-transaction windows, while serving zero-pads short histories — cold-start windows are out-of-distribution. **Next: include padded windows in training.**
- The cost-optimal threshold was calibrated on per-window validation scores. **Next: recalibrate on the per-transaction basis.**
- Cost parameters (churn rate, secondary costs) are research-based estimates; the sensitivity grid bounds but does not eliminate that uncertainty.
- Also planned: model monitoring/drift detection, batch scoring endpoint, CI/CD for retraining, authentication on the API.

## References

- Ali et al. (2022). *Financial Fraud Detection Based on Machine Learning: A Systematic Literature Review.* Applied Sciences, 12(19), 9637.
- LexisNexis Risk Solutions (2024). *True Cost of Fraud Study.*
- J.P. Morgan (2023). *False Positives & Fraud Prevention Tools.*
- Riskified (2025). *How Much Does a False Decline Cost Your Business?*
- Australian Bureau of Statistics (2025). *Personal Fraud, 2023–24 Financial Year.*
