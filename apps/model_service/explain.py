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
    Extracts the underlying fitted base estimator from CalibratedClassifierCV or other wrappers.
    """
    # 1. If wrapped in CalibratedClassifierCV
    if hasattr(model_obj, "calibrated_classifiers_") and len(model_obj.calibrated_classifiers_) > 0:
        cc = model_obj.calibrated_classifiers_[0]
        if hasattr(cc, "estimator") and cc.estimator is not None:
            return unwrap_model(cc.estimator)
        elif hasattr(cc, "base_estimator") and cc.base_estimator is not None:
            return unwrap_model(cc.base_estimator)
        return cc

    # 2. If wrapped in standard Estimator wrapper with estimator/base_estimator
    if hasattr(model_obj, "estimator") and model_obj.estimator is not None:
        est = model_obj.estimator
        if hasattr(est, "classes_") or hasattr(est, "coef_") or hasattr(est, "feature_importances_") or hasattr(est, "predict_proba"):
            return unwrap_model(est)
    if hasattr(model_obj, "base_estimator") and model_obj.base_estimator is not None:
        est = model_obj.base_estimator
        if hasattr(est, "classes_") or hasattr(est, "coef_") or hasattr(est, "feature_importances_") or hasattr(est, "predict_proba"):
            return unwrap_model(est)

    return model_obj


import model_loader


def get_explainer_and_model(model_override=None):
    """
    Loads production model via model_loader and reference background dataset, initializing the SHAP explainer.
    
    Returns
    -------
    tuple
        (model, explainer)
    """
    global _model, _explainer

    if model_override is not None:
        model_to_use = model_override
    elif _model is not None:
        model_to_use = _model
    else:
        try:
            model_to_use, _ = model_loader.load_production_model()
        except Exception:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please run train.py first.")
            model_to_use = joblib.load(MODEL_PATH)
        _model = model_to_use

    if _explainer is not None and model_override is None:
        return _model, _explainer

    base_model = unwrap_model(model_to_use)

    # Determine data path based on available dataset files
    if os.path.exists("data/features_real.csv"):
        data_path = "data/features_real.csv"
    elif os.path.exists("data/features_real_DS_91c85fbe.csv"):
        data_path = "data/features_real_DS_91c85fbe.csv"
    elif os.path.exists(DATA_PATH):
        data_path = DATA_PATH
    else:
        data_path = None

    if hasattr(base_model, "feature_names_in_"):
        feature_cols = list(base_model.feature_names_in_)
    elif hasattr(base_model, "n_features_in_") and base_model.n_features_in_ == 4:
        feature_cols = [
            "customer_txn_count_60m",
            "amount_deviation_ratio",
            "is_new_device",
            "shared_device_account_count",
        ]
    else:
        feature_cols = REAL_FEATURE_COLUMNS

    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        X = df[feature_cols]
        y = df["is_fraud"] if "is_fraud" in df.columns else np.zeros(len(df))
        stratify_target = y if len(np.unique(y)) > 1 else None
        X_train, _, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_target
        )
    else:
        X_train = pd.DataFrame(np.zeros((10, len(feature_cols))), columns=feature_cols)

    base_type_name = type(base_model).__name__

    try:
        if isinstance(base_model, (xgb.XGBClassifier, lgb.LGBMClassifier)) or "XGB" in base_type_name or "LGBM" in base_type_name:
            explainer = shap.TreeExplainer(base_model)
        elif isinstance(base_model, LogisticRegression) or "LogisticRegression" in base_type_name:
            explainer = shap.LinearExplainer(base_model, X_train)
        else:
            explainer = shap.Explainer(base_model, X_train)
    except Exception as e:
        print(f"[explain] Primary explainer creation failed ({e}), trying generic explainer.")
        try:
            explainer = shap.Explainer(base_model, X_train)
        except Exception:
            explainer = None

    if model_override is None:
        _explainer = explainer

    return model_to_use, explainer


def explain_transaction(feature_dict: dict, model_override=None) -> dict:
    """
    Computes fraud probability and feature attribution (SHAP impact) for a single transaction.

    Parameters
    ----------
    feature_dict : dict
        Key-value mapping of input transaction features matching feature schema.
    model_override : optional
        Active model instance passed from service handler.

    Returns
    -------
    dict
        Explanation outcome containing:
        - "fraud_probability": float (0.0 to 1.0)
        - "top_factors": list of dicts [{"feature": str, "impact": float}], sorted by abs(impact) descending
    """
    model, explainer = get_explainer_and_model(model_override=model_override)

    norm_dict = normalize_feature_dict(feature_dict)

    base_model = unwrap_model(model)
    if hasattr(base_model, "feature_names_in_"):
        feature_names = list(base_model.feature_names_in_)
    elif hasattr(base_model, "n_features_in_") and base_model.n_features_in_ == 4:
        feature_names = [
            "customer_txn_count_60m",
            "amount_deviation_ratio",
            "is_new_device",
            "shared_device_account_count",
        ]
    else:
        feature_names = REAL_FEATURE_COLUMNS

    # Ensure all feature columns exist in norm_dict
    for col in feature_names:
        if col not in norm_dict:
            norm_dict[col] = 0.0 if ("score" in col or "prior" in col or "ratio" in col) else 0

    # Create 1-row DataFrame maintaining exact feature column order
    df_single = pd.DataFrame([norm_dict])[feature_names]

    # Predict calibrated fraud probability (class 1)
    fraud_prob = float(model.predict_proba(df_single)[0, 1])

    # Compute feature attribution (SHAP impact with fallback to model feature weights)
    impacts = None

    if explainer is not None:
        try:
            try:
                raw_shap = explainer.shap_values(df_single)
            except Exception:
                raw_shap = explainer(df_single)

            if isinstance(raw_shap, list):
                class_1_vals = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
                if isinstance(class_1_vals, np.ndarray):
                    impacts = class_1_vals[0] if class_1_vals.ndim >= 2 else class_1_vals
                else:
                    impacts = class_1_vals
            elif isinstance(raw_shap, np.ndarray):
                if raw_shap.ndim == 3:
                    impacts = raw_shap[0, :, 1]
                elif raw_shap.ndim == 2:
                    impacts = raw_shap[0]
                else:
                    impacts = raw_shap
            else:
                vals = getattr(raw_shap, "values", raw_shap)
                if isinstance(vals, list) and len(vals) > 1:
                    vals = vals[1]
                if hasattr(vals, "ndim"):
                    if vals.ndim == 3:
                        impacts = vals[0, :, 1]
                    elif vals.ndim == 2:
                        impacts = vals[0]
                    else:
                        impacts = vals
        except Exception as err:
            print(f"[explain] SHAP computation warning: {err}. Using model weight attribution fallback.")
            impacts = None

    # Fallback to model weights if SHAP values could not be computed
    if impacts is None:
        if hasattr(base_model, "coef_"):
            coefs = base_model.coef_[0] if base_model.coef_.ndim == 2 else base_model.coef_
            impacts = df_single.values[0] * coefs
        elif hasattr(base_model, "feature_importances_"):
            impacts = df_single.values[0] * base_model.feature_importances_
        else:
            impacts = df_single.values[0]

    impacts = np.asarray(impacts).flatten()

    top_factors = []
    for feat, imp in zip(feature_names, impacts):
        top_factors.append({
            "feature": str(feat),
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
        explanation = explain_transaction(sample_dict)
        print(f"Input features: {sample_dict}")
        print(f"Explanation Output:\n{json.dumps(explanation, indent=2)}")

