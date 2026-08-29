import os
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import hashlib
from typing import List, Optional, Union

import model_loader
from explain import explain_transaction

# Global state for loaded model and metadata
model = None
model_name = ""
model_version = ""
model_source = ""
registered_model_name = ""
registered_model_version = ""
run_id = ""
artifact_sha256 = ""


def unwrap_model(model_obj):
    """
    Extracts the underlying fitted estimator if wrapped in CalibratedClassifierCV.
    """
    if hasattr(model_obj, "calibrated_classifiers_") and len(model_obj.calibrated_classifiers_) > 0:
        return model_obj.calibrated_classifiers_[0].estimator
    elif hasattr(model_obj, "estimator") and model_obj.estimator is not None:
        if hasattr(model_obj.estimator, "classes_") or hasattr(model_obj.estimator, "coef_") or hasattr(model_obj.estimator, "feature_importances_"):
            return model_obj.estimator
    elif hasattr(model_obj, "base_estimator") and model_obj.base_estimator is not None:
        if hasattr(model_obj.base_estimator, "classes_") or hasattr(model_obj.base_estimator, "coef_") or hasattr(model_obj.base_estimator, "feature_importances_"):
            return model_obj.base_estimator
    return model_obj



def determine_model_version(model_obj) -> str:
    """
    Derives model version tag based on loaded model type.
    """
    base = unwrap_model(model_obj)
    type_name = type(base).__name__
    if "XGB" in type_name:
        return "xgb-1.0"
    elif "LGBM" in type_name:
        return "lgbm-1.0"
    elif "LogisticRegression" in type_name:
        return "logreg-1.0"
    return f"{type_name.lower()}-1.0"


def compute_artifact_sha256(file_path: str = "models/best_model.joblib") -> str:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    return ""


def load_model():
    """
    Loads saved model at startup using model_loader (MLflow Model Registry Staging stage with local joblib fallback).
    """
    global model, model_name, model_version, model_source, registered_model_name, registered_model_version, run_id, artifact_sha256
    model, model_source = model_loader.load_production_model()

    base = unwrap_model(model)
    model_name = type(base).__name__
    reg_name, reg_ver, r_id = model_loader.get_active_model_metadata()
    registered_model_name = reg_name
    registered_model_version = reg_ver
    run_id = r_id
    artifact_sha256 = compute_artifact_sha256("models/best_model.joblib")

    algo_ver = determine_model_version(base)
    if reg_ver != "local_fallback":
        model_version = f"{algo_ver}-v{reg_ver}"
    else:
        model_version = algo_ver

    print(f"[main] Loaded model '{model_name}' ({model_version}) via source '{model_source}' (Run ID: {run_id}, SHA256: {artifact_sha256[:8]}...).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan event context manager to initialize model on service startup.
    """
    load_model()
    yield


app = FastAPI(
    title="FraudGuard 360 - Fraud Detection Model Service",
    description="ML Inference & SHAP Explanation Service for Member 4 module",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware enabling requests from all origins for demo/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Any


# ==============================================================================
# BREAKING CHANGE NOTICE (Member 5 Integration):
# The feature schema for /predict and /explain endpoints has been updated from 4 dummy fields
# to all 12 canonical Member 3 feature service fields.
# Member 5 must update their API request payload to send all 12 real feature fields.
# ==============================================================================

REAL_FEATURE_COLUMNS = [
    "customer_txn_count_60m",
    "customer_amount_mean_prior",
    "amount_deviation_ratio",
    "is_new_device",
    "is_new_merchant",
    "location_shift",
    "customer_device_degree",
    "customer_merchant_degree",
    "device_customer_degree",
    "merchant_customer_degree",
    "shared_device_customer_count",
    "relationship_risk_score",
]


class TransactionFeatures(BaseModel):
    customer_txn_count_60m: int = Field(..., description="Number of transactions in last 1 hour (0-20)")
    customer_amount_mean_prior: float = Field(0.0, description="Customer mean transaction amount prior to this transaction")
    amount_deviation_ratio: float = Field(..., description="Z-score / ratio deviation from normal spending (-1 to 5)")
    is_new_device: int = Field(..., description="Indicator if transaction uses a new device (0 or 1)")
    is_new_merchant: int = Field(0, description="Indicator if transaction uses a new merchant (0 or 1)")
    location_shift: int = Field(0, description="Indicator if location shift detected (0 or 1)")
    customer_device_degree: int = Field(0, description="Count of devices associated with customer")
    customer_merchant_degree: int = Field(0, description="Count of merchants associated with customer")
    device_customer_degree: int = Field(0, description="Count of customers associated with device")
    merchant_customer_degree: int = Field(0, description="Count of customers associated with merchant")
    shared_device_customer_count: int = Field(..., description="Count of other accounts sharing device (0-10)")
    relationship_risk_score: float = Field(0.0, description="Graph-based relationship risk score (0.0 to 1.0)")

    @model_validator(mode="before")
    @classmethod
    def map_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapped = dict(data)
            if "velocity_1h" in mapped and "customer_txn_count_60m" not in mapped:
                mapped["customer_txn_count_60m"] = mapped["velocity_1h"]
            if "amount_deviation" in mapped and "amount_deviation_ratio" not in mapped:
                mapped["amount_deviation_ratio"] = mapped["amount_deviation"]
            if "new_device" in mapped and "is_new_device" not in mapped:
                mapped["is_new_device"] = mapped["new_device"]
            if "shared_device_count" in mapped and "shared_device_customer_count" not in mapped:
                mapped["shared_device_customer_count"] = mapped["shared_device_count"]
            if "shared_device_account_count" in mapped and "shared_device_customer_count" not in mapped:
                mapped["shared_device_customer_count"] = mapped["shared_device_account_count"]
            
            # Cast bools to int if passed
            for bool_col in ["is_new_device", "is_new_merchant", "location_shift"]:
                if bool_col in mapped and isinstance(mapped[bool_col], bool):
                    mapped[bool_col] = int(mapped[bool_col])
            return mapped
        return data

    @property
    def velocity_1h(self) -> int:
        return self.customer_txn_count_60m

    @property
    def amount_deviation(self) -> float:
        return self.amount_deviation_ratio

    @property
    def new_device(self) -> int:
        return self.is_new_device

    @property
    def shared_device_count(self) -> int:
        return self.shared_device_customer_count


class PredictRequest(BaseModel):
    features: TransactionFeatures

    @model_validator(mode="before")
    @classmethod
    def extract_features(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "features" in data:
                return data
            elif "data" in data and isinstance(data["data"], dict):
                return {"features": data["data"]}
            else:
                # Direct flat feature payload
                return {"features": data}
        return data


class PredictResponse(BaseModel):
    fraud_probability: float
    risk_score: int
    prediction: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model: str
    version: str
    source: str
    registered_model_name: Optional[str] = None
    registered_model_version: Optional[Union[str, int]] = None
    run_id: Optional[str] = None
    feature_schema_version: str = "1.0"
    artifact_sha256: Optional[str] = None


class FactorImpact(BaseModel):
    feature: str
    impact: float


class ExplainResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    top_factors: List[FactorImpact]


# --- Endpoints ---

@app.get("/api/v1/health")
def health_check():
    """
    Health check endpoint returning service status.
    """
    return {"status": "ok"}


@app.post("/api/v1/model/predict", response_model=PredictResponse)
def predict_fraud(payload: PredictRequest):
    """
    Predicts transaction fraud probability, risk score, and risk category.
    Accepts Member 3 direct 12-feature schema, envelope payloads ('data'), and legacy aliases.

    BREAKING CHANGE CONTRACT NOTICE:
    Requires 12 real Member 3 feature fields:
    customer_txn_count_60m, customer_amount_mean_prior, amount_deviation_ratio,
    is_new_device, is_new_merchant, location_shift, customer_device_degree,
    customer_merchant_degree, device_customer_degree, merchant_customer_degree,
    shared_device_customer_count, relationship_risk_score.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    feats = payload.features
    feature_dict = {
        "customer_txn_count_60m": feats.customer_txn_count_60m,
        "velocity_1h": feats.customer_txn_count_60m,
        "customer_amount_mean_prior": feats.customer_amount_mean_prior,
        "amount_deviation_ratio": feats.amount_deviation_ratio,
        "amount_deviation": feats.amount_deviation_ratio,
        "is_new_device": int(feats.is_new_device),
        "new_device": int(feats.is_new_device),
        "is_new_merchant": int(feats.is_new_merchant),
        "location_shift": int(feats.location_shift),
        "customer_device_degree": feats.customer_device_degree,
        "customer_merchant_degree": feats.customer_merchant_degree,
        "device_customer_degree": feats.device_customer_degree,
        "merchant_customer_degree": feats.merchant_customer_degree,
        "shared_device_customer_count": feats.shared_device_customer_count,
        "shared_device_account_count": feats.shared_device_customer_count,
        "shared_device_count": feats.shared_device_customer_count,
        "relationship_risk_score": feats.relationship_risk_score,
    }


    df_single = pd.DataFrame([feature_dict])
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        df_single = df_single[expected_cols]
    else:
        df_single = df_single[REAL_FEATURE_COLUMNS]

    fraud_prob = float(model.predict_proba(df_single)[0, 1])
    prob_rounded = round(fraud_prob, 4)
    risk_score = int(round(fraud_prob * 100))

    if fraud_prob >= 0.7:
        prediction = "HIGH_RISK"
    elif fraud_prob >= 0.4:
        prediction = "MEDIUM_RISK"
    else:
        prediction = "LOW_RISK"

    return PredictResponse(
        fraud_probability=prob_rounded,
        risk_score=risk_score,
        prediction=prediction,
        model_version=model_version
    )


import urllib.request
import json

MEMBER3_FEATURE_SERVICE_URL = os.getenv("MEMBER3_FEATURE_SERVICE_URL", "http://localhost:8003/api/v1/features/extract")


class FeatureServicePredictRequest(BaseModel):
    transaction: dict
    feature_service_url: Optional[str] = None


@app.get("/api/v1/model/info", response_model=ModelInfoResponse)
def get_model_info():
    """
    Returns information about the loaded model type, version tag, model source, MLflow registry metadata, feature schema version, and local artifact SHA256.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    return ModelInfoResponse(
        model=model_name,
        version=model_version,
        source=model_source,
        registered_model_name=registered_model_name,
        registered_model_version=registered_model_version,
        run_id=run_id,
        feature_schema_version="1.0",
        artifact_sha256=artifact_sha256 or compute_artifact_sha256("models/best_model.joblib")
    )


@app.post("/api/v1/model/predict_via_feature_service", response_model=PredictResponse)
def predict_via_feature_service(payload: FeatureServicePredictRequest):
    """
    Calls Member 3 Feature Extraction Service at http://localhost:8003/api/v1/features/extract
    to extract features for a raw transaction, then performs real-time ML fraud prediction.
    """
    service_url = payload.feature_service_url or MEMBER3_FEATURE_SERVICE_URL
    
    try:
        req_data = json.dumps(payload.transaction).encode("utf-8")
        req = urllib.request.Request(
            service_url,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            m3_response = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to communicate with Member 3 Feature Service at '{service_url}': {str(e)}"
        )

    # Unwrap Member 3 response envelope and pass to predict_fraud logic
    predict_req = PredictRequest.model_validate(m3_response)
    return predict_fraud(predict_req)


@app.get("/api/v1/model/explain/{transaction_id}", response_model=ExplainResponse)
def explain_tx(
    transaction_id: str,
    customer_txn_count_60m: Optional[int] = Query(None, description="Transactions in last 1 hour"),
    customer_amount_mean_prior: Optional[float] = Query(None, description="Prior mean amount"),
    amount_deviation_ratio: Optional[float] = Query(None, description="Amount deviation ratio"),
    is_new_device: Optional[int] = Query(None, description="New device indicator"),
    is_new_merchant: Optional[int] = Query(None, description="New merchant indicator"),
    location_shift: Optional[int] = Query(None, description="Location shift indicator"),
    customer_device_degree: Optional[int] = Query(None, description="Customer device degree"),
    customer_merchant_degree: Optional[int] = Query(None, description="Customer merchant degree"),
    device_customer_degree: Optional[int] = Query(None, description="Device customer degree"),
    merchant_customer_degree: Optional[int] = Query(None, description="Merchant customer degree"),
    shared_device_customer_count: Optional[int] = Query(None, description="Shared device count"),
    relationship_risk_score: Optional[float] = Query(None, description="Relationship risk score"),
    velocity_1h: Optional[int] = Query(None, description="Legacy velocity_1h"),
    amount_deviation: Optional[float] = Query(None, description="Legacy amount_deviation"),
    new_device: Optional[int] = Query(None, description="Legacy new_device"),
    shared_device_count: Optional[int] = Query(None, description="Legacy shared_device_count")
):
    """
    Computes SHAP feature attribution and explanations for a transaction specified by 12 real features.

    BREAKING CHANGE CONTRACT NOTICE:
    Supports 12 real feature fields for Member 3 integration.
    """
    vel = customer_txn_count_60m if customer_txn_count_60m is not None else velocity_1h
    amt = amount_deviation_ratio if amount_deviation_ratio is not None else amount_deviation
    dev = is_new_device if is_new_device is not None else new_device
    shd = shared_device_customer_count if shared_device_customer_count is not None else shared_device_count

    if vel is None or amt is None or dev is None or shd is None:
        raise HTTPException(status_code=422, detail="Missing required transaction features for explanation.")

    feature_dict = {
        "customer_txn_count_60m": vel,
        "customer_amount_mean_prior": customer_amount_mean_prior or 0.0,
        "amount_deviation_ratio": amt,
        "is_new_device": dev,
        "is_new_merchant": is_new_merchant or 0,
        "location_shift": location_shift or 0,
        "customer_device_degree": customer_device_degree or 0,
        "customer_merchant_degree": customer_merchant_degree or 0,
        "device_customer_degree": device_customer_degree or 0,
        "merchant_customer_degree": merchant_customer_degree or 0,
        "shared_device_customer_count": shd,
        "relationship_risk_score": relationship_risk_score or 0.0,
    }

    try:
        explanation = explain_transaction(feature_dict, model_override=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SHAP explanation: {str(e)}")


    return ExplainResponse(
        transaction_id=transaction_id,
        fraud_probability=explanation["fraud_probability"],
        top_factors=[
            FactorImpact(feature=item["feature"], impact=item["impact"])
            for item in explanation["top_factors"]
        ]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
