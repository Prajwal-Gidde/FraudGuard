import json
import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

MODEL_PATH = "models/best_model.joblib"
DATA_PATH = "data/features_real.csv"

# Global cached variables
_model = None
_explainer = None
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

_feature_names = REAL_FEATURE_COLUMNS

LEGACY_FEATURE_MAPPING = {
    "velocity_1h": "customer_txn_count_60m",
    "amount_deviation": "amount_deviation_ratio",
    "new_device": "is_new_device",
    "shared_device_count": "shared_device_customer_count",
    "shared_device_account_count": "shared_device_customer_count",
}


def normalize_feature_dict(raw_dict: dict) -> dict:
    """
    Normalizes input feature dictionary keys to match Member 3 canonical feature schema.
    """
    normalized = {}
    for key, val in raw_dict.items():
        canonical_key = LEGACY_FEATURE_MAPPING.get(key, key)
        normalized[canonical_key] = val

    # Ensure all 12 feature columns are present
    for col in REAL_FEATURE_COLUMNS:
        if col not in normalized:
            normalized[col] = 0.0 if ("score" in col or "prior" in col or "ratio" in col) else 0

    return normalized



def unwrap_model(model_obj):
    """
    Extracts the underlying estimator from CalibratedClassifierCV if wrapped.
    """
    if hasattr(model_obj, "estimator") and model_obj.estimator is not None:
        return model_obj.estimator
    elif hasattr(model_obj, "base_estimator") and model_obj.base_estimator is not None:
        return model_obj.base_estimator
    elif hasattr(model_obj, "calibrated_classifiers_") and len(model_obj.calibrated_classifiers_) > 0:
        return model_obj.calibrated_classifiers_[0].estimator
    return model_obj


def get_explainer_and_model():
    """
    Loads best_model.joblib and test background dataset, initializing the appropriate SHAP explainer.
    
    Returns
    -------
    tuple
        (model, explainer)
    """
    global _model, _explainer
    if _model is not None and _explainer is not None:
        return _model, _explainer

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please run train.py first.")

    _model = joblib.load(MODEL_PATH)

    # Load dataset to extract background data for SHAP explainers requiring reference samples
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = unwrap_model(_model)
    base_type_name = type(base_model).__name__

    # Detect model type and select appropriate SHAP Explainer
    if isinstance(base_model, (xgb.XGBClassifier, lgb.LGBMClassifier)) or "XGB" in base_type_name or "LGBM" in base_type_name:
        _explainer = shap.TreeExplainer(base_model)
    elif isinstance(base_model, LogisticRegression) or "LogisticRegression" in base_type_name:
        _explainer = shap.LinearExplainer(base_model, X_train)
    else:
        _explainer = shap.Explainer(base_model, X_train)

    return _model, _explainer


def explain_transaction(feature_dict: dict) -> dict:
    """
    Computes fraud probability and feature attribution (SHAP impact) for a single transaction.

    Parameters
    ----------
    feature_dict : dict
        Key-value mapping of input transaction features matching feature schema:
        {"velocity_1h": int, "amount_deviation": float, "new_device": int, "shared_device_count": int}

    Returns
    -------
    dict
        Explanation outcome containing:
        - "fraud_probability": float (0.0 to 1.0)
        - "top_factors": list of dicts [{"feature": str, "impact": float}], sorted by abs(impact) descending
    """
    model, explainer = get_explainer_and_model()

    # Normalize keys if legacy names were passed
    norm_dict = normalize_feature_dict(feature_dict)

    # Create 1-row DataFrame maintaining exact feature column order
    df_single = pd.DataFrame([norm_dict])[_feature_names]

    # Predict calibrated fraud probability (class 1)
    fraud_prob = float(model.predict_proba(df_single)[0, 1])

    # Compute SHAP values using base model explainer
    raw_shap = explainer.shap_values(df_single)

    # Standardize SHAP output into a 1D vector of feature impacts for fraud class (1)
    if isinstance(raw_shap, list):
        # List per output class [class_0, class_1]
        class_1_vals = raw_shap[1]
        impacts = class_1_vals[0] if class_1_vals.ndim == 2 else class_1_vals
    elif isinstance(raw_shap, np.ndarray):
        if raw_shap.ndim == 3:
            impacts = raw_shap[0, :, 1]
        elif raw_shap.ndim == 2:
            impacts = raw_shap[0]
        else:
            impacts = raw_shap
    else:
        exp_obj = explainer(df_single)
        vals = exp_obj.values
        if isinstance(vals, list):
            vals = vals[1]
        if vals.ndim == 3:
            impacts = vals[0, :, 1]
        elif vals.ndim == 2:
            impacts = vals[0]
        else:
            impacts = vals

    top_factors = []
    for feat, imp in zip(_feature_names, impacts):
        top_factors.append({
            "feature": feat,
            "impact": round(float(imp), 4)
        })

    # Sort top factors by absolute impact in descending order
    top_factors.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "fraud_probability": round(fraud_prob, 4),
        "top_factors": top_factors
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SHAP TRANSACTION EXPLAINER TEST RUN")
    print("=" * 70)

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        X = df.drop(columns=["is_fraud"])
        y = df["is_fraud"]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        sample_indices = []
        fraud_idx = y_test[y_test == 1].index
        legit_idx = y_test[y_test == 0].index

        if len(fraud_idx) > 0:
            sample_indices.append(fraud_idx[0])
        if len(legit_idx) > 0:
            sample_indices.append(legit_idx[0])
        if len(X_test.index) > 0 and X_test.index[0] not in sample_indices:
            sample_indices.append(X_test.index[0])

        print(f"\nRunning explain_transaction() on {len(sample_indices)} sample test set transactions:\n")

        for idx in sample_indices:
            sample_features = X_test.loc[idx].to_dict()
            actual_label = y_test.loc[idx]

            explanation = explain_transaction(sample_features)

            print(f"--- Sample Test Index #{idx} (Ground Truth Label: {actual_label}) ---")
            print(f"Input features: {sample_features}")
            print(f"Explanation Output:\n{json.dumps(explanation, indent=2)}\n")

    else:
        print(f"Dataset path '{DATA_PATH}' not found. Testing with custom sample dict...")
        sample_dict = {
            "velocity_1h": 7,
            "amount_deviation": 2.4,
            "new_device": 1,
            "shared_device_count": 4
        }
        explanation = explain_transaction(sample_dict)
        print(f"Input features: {sample_dict}")
        print(f"Explanation Output:\n{json.dumps(explanation, indent=2)}")
