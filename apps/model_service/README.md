# FraudGuard 360 - Fraud Detection ML Module (Member 4)

## 1. Overview
The Fraud Detection ML module (Member 4) for **FraudGuard 360** trains machine learning classification models on transaction features, calibrates prediction probabilities, registers pinned prototype models / final demo models to the MLflow Model Registry, and serves real-time fraud predictions, risk scoring, risk categorization, and SHAP-based feature explanations via a RESTful FastAPI service.

---

## 2. Setup Instructions

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

---

## 3. How to Run

```powershell
# 1. Extract feature vectors from real fixture dataset (Primary Workflow - DS_91c85fbe)
python real_data_pipeline.py

# Alternative / Fallback Step: Generate synthetic dummy dataset (Dummy fallback option)
# python dummy_data.py

# 2. Train candidate models, apply probability calibration, and register best model to MLflow Registry
python train.py

# 3. Run unit and API contract test suite
pytest tests/test_model.py -v

# 4. Start production FastAPI inference server (runs on http://localhost:8000)
python main.py

# 5. Launch MLflow UI to inspect experiment runs & Model Registry (opens http://127.0.0.1:5000)
mlflow ui
```

> Interactive API Documentation (Swagger UI) is available at: `http://localhost:8000/docs`

---

## 4. API Documentation

| Method | Endpoint Path | Description / Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/v1/model/predict` | Predicts fraud probability, 0-100 risk score, risk level, and model version tag. |
| `GET` | `/api/v1/model/info` | Returns active model type, version, and `source` (`registry` or `local_fallback`). |
| `GET` | `/api/v1/model/explain/{transaction_id}` | Computes SHAP feature contributions for transaction query parameters. |
| `GET` | `/api/v1/health` | Service health check endpoint returning `{"status": "ok"}`. |

---

## 5. Sample Input / Output

### 5.1 Endpoint: `POST /api/v1/model/predict`

**Request Body (`application/json` - Member 3 Direct Payload):**
```json
{
  "features": {
    "customer_txn_count_60m": 0,
    "customer_amount_mean_prior": 19.0,
    "amount_deviation_ratio": 0.3,
    "is_new_device": 1,
    "is_new_merchant": 1,
    "location_shift": 0,
    "customer_device_degree": 1,
    "customer_merchant_degree": 1,
    "device_customer_degree": 11,
    "merchant_customer_degree": 1,
    "shared_device_customer_count": 10,
    "relationship_risk_score": 0.49
  }
}
```

*Note: Also accepts Member 3 envelope format (`{"data": {...}}`) and flat JSON directly.*

**Response (`200 OK`):**
```json
{
  "fraud_probability": 0.9516,
  "risk_score": 95,
  "prediction": "HIGH_RISK",
  "model_version": "logreg-1.0"
}
```

### 5.2 Endpoint: `GET /api/v1/model/info`

**Response (`200 OK`):**
```json
{
  "model": "LogisticRegression",
  "version": "logreg-1.0-v5",
  "source": "registry",
  "registered_model_name": "fraudguard360-detector",
  "registered_model_version": "5",
  "run_id": "b7b76f72847d409d88edab4288032f6f",
  "feature_schema_version": "1.0",
  "artifact_sha256": "76966a027c034fdc210fac690ad2877fb004c12c99569b10d4b4df11ca3bb688"
}
```

---

## 6. Model Details
- **Primary & Benchmarked Classifiers**:
  - Primary Model (Selected): `XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=auto, eval_metric="aucpr", objective="binary:logistic")` tuned for imbalanced fraud detection.
  - Baseline 1: `LogisticRegression(max_iter=1000)`
  - Baseline 2: `LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)`
- **Probability Calibration**: Top candidate selected by ROC AUC (XGBoost) is wrapped with `CalibratedClassifierCV(estimator=best_model, method="sigmoid", cv=5)` to output well-calibrated posterior probabilities (Platt Scaling).
- **MLflow Model Registry**: Calibrated model registered under `"fraudguard360-detector"` and transitioned to stage `"Staging"` using `MlflowClient`.
- **Pinned Prototype Model Sourcing**: [model_loader.py](model_loader.py) loads from the MLflow Model Registry (`models:/fraudguard360-detector/Staging`) first, gracefully falling back to local artifact [models/best_model.joblib](models/best_model.joblib). Note that all deployed artifacts represent a **pinned prototype model** / **final demo model** for integration and demonstration.

---

## 7. Known Limitations & Model Classification
1. **Model Scope**: The active artifact is designated as a **pinned prototype model** / **final demo model** for demonstration and integration testing, not a production-ready model.
2. **Provisional Decision Thresholds**: Risk category thresholds (`HIGH_RISK` >= 0.7, `MEDIUM_RISK` >= 0.4, `LOW_RISK` < 0.4) are provisional defaults pending joint calibration with Member 5.
3. **SHAP Feature Attribution Scale**: SHAP impacts for linear model baselines are evaluated on log-odds scale relative to reference background data.
4. **Staging Registry Stage**: Model registry stage is initialized to `"Staging"` - promotion to `"Production"` is a manual/deliberate step, not automated.
5. **Local Tracking Store**: MLflow uses a local `sqlite:///mlflow.db` backend store for this prototype rather than a shared remote MLflow server.

---

## 8. Dependencies & Team Handoff
- **Upstream Dependency**: Member 3 (`fraudguard-member3` feature engineering service: supplies canonical 12-feature schema).
- **Downstream Handoffs**:
  - **Member 5** (Risk API Integration & Gateway Orchestration consuming `/api/v1/model/predict` and `/api/v1/model/explain/{transaction_id}`, joint threshold tuning).
  - **Member 6** (Metrics & Operational Monitoring consuming `/api/v1/health` and `/api/v1/model/info`, retraining feedback loop).

---

## 9. Data Scaling & XGBoost Model Training Findings

- **Dataset Scaling to 50,000 Rows**:
  - Scaled dataset generation to 50,000 transaction records maintaining realistic 15% fraud class imbalance (42,500 legitimate / 7,500 fraud transactions).
  - Preserved the full 12 canonical Member 3 feature schema across velocity, amount deviation, graph network degrees, shared device counts, and relationship risk scores.
- **Pipeline Validation & Stratified Split**:
  - Training pipeline ([train.py](train.py)) utilizes an 80/20 stratified split (40,000 train / 10,000 test records, `random_state=42`, `stratify=y`) ensuring consistent fraud representation across train and test sets.
- **Class-Balanced XGBoost Primary Model**:
  - Deployed tuned XGBClassifier with class-imbalance aware `scale_pos_weight` ratio (`n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `eval_metric="aucpr"`).
  - LogisticRegression retained as a baseline benchmark, with XGBoost consistently achieving top ROC AUC and PR AUC performance.
- **Probability Calibration & MLflow Model Registry**:
  - Top XGBoost model is calibrated via Platt Scaling (`CalibratedClassifierCV`) and registered under `"fraudguard360-detector"` in stage `"Staging"`.
- **API Contract Compatibility**:
  - Preserved full 12-feature REST API schema contract for `/api/v1/model/predict` and `/api/v1/model/explain/{transaction_id}`.




