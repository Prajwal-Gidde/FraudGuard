"""
Threshold Sweep Evaluation Script for FraudGuard 360 Model Service (Member 4).

Loads the active production model via model_loader.load_production_model() and evaluates it
against the held-out test split of the feature dataset (reproducing exact train/test split
from train.py: data/features_real.csv, random_state=42, 80/20 stratified split).

Sweeps decision thresholds from 0.20 to 0.90 in increments of 0.05.
For each threshold, computes:
- Precision, Recall, F1 Score
- False Positive Rate (FPR) and False Negative Rate (FNR)
- Confusion Matrix Counts (TP, FP, TN, FN)

Prints a clean evaluation summary table and active MLflow model version info,
and saves the output table to data/threshold_sweep_results.csv.

Read-only: Does not retrain or alter model registry state.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import mlflow
from mlflow.tracking import MlflowClient

import model_loader
from train import load_data, REAL_FEATURE_COLUMNS


def get_active_model_info(source: str):
    """
    Queries MLflow tracking backend to retrieve active registered model name and version.
    """
    model_name = "fraudguard360-detector"
    model_version = "Unknown"

    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = MlflowClient()
        # Query latest Staging versions
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if staging_versions:
            model_version = staging_versions[0].version
        else:
            all_versions = client.get_latest_versions(model_name)
            if all_versions:
                model_version = all_versions[-1].version
    except Exception as err:
        print(f"[Note] Could not retrieve MLflow version details: {err}")

    return model_name, model_version


def run_threshold_sweep(
    data_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    threshold_min: float = 0.20,
    threshold_max: float = 0.90,
    threshold_step: float = 0.05,
    output_csv_path: str = "data/threshold_sweep_results.csv",
) -> pd.DataFrame:
    """
    Executes threshold sweep evaluation on the held-out test split.
    """
    # 1. Load active production model via model_loader
    print("\n" + "=" * 85)
    print("                      FRAUDGUARD 360 - THRESHOLD SWEEP                       ")
    print("=" * 85)

    model, source = model_loader.load_production_model()
    reg_name, reg_version = get_active_model_info(source)

    # 2. Print active MLflow registered model name, version, and source info at the top
    print("-" * 85)
    print(" ACTIVE MODEL METADATA:")
    print(f"   MLflow Registered Model Name : {reg_name}")
    print(f"   MLflow Registered Version     : {reg_version}")
    print(f"   Model Loader Source           : {source}")
    print(f"   Loaded Model Estimator Type   : {type(model).__name__}")
    print("-" * 85)

    # 3. Load dataset and reproduce exact train/test split from train.py
    if data_path is None:
        if os.path.exists("data/features_real.csv"):
            data_path = "data/features_real.csv"
        elif os.path.exists("data/features_real_DS_91c85fbe.csv"):
            data_path = "data/features_real_DS_91c85fbe.csv"
        else:
            data_path = "data/features_real.csv"

    X, y = load_data(data_path)

    # Exact 80/20 stratified split with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    n_total = len(y_test)
    n_fraud = int((y_test == 1).sum())
    n_legit = int((y_test == 0).sum())
    print(
        f" Held-Out Test Set Split : Total={n_total} | Fraud (1)={n_fraud} | Non-Fraud (0)={n_legit} "
        f"(Split={int((1-test_size)*100)}/{int(test_size*100)}, random_state={random_state})"
    )
    print("=" * 85 + "\n")

    # 4. Predict probabilities on held-out test set
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"Loaded model type {type(model).__name__} lacks 'predict_proba' method.")

    y_proba = model.predict_proba(X_test)[:, 1]

    # 5. Sweep decision thresholds from 0.20 to 0.90 in increments of 0.05
    num_steps = int(round((threshold_max - threshold_min) / threshold_step)) + 1
    thresholds = np.round(np.linspace(threshold_min, threshold_max, num_steps), 2)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        total_neg = tn + fp
        total_pos = tp + fn

        fpr = fp / total_neg if total_neg > 0 else 0.0
        fnr = fn / total_pos if total_pos > 0 else 0.0

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        rows.append(
            {
                "Threshold": t,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "FPR": fpr,
                "FNR": fnr,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
            }
        )

    sweep_df = pd.DataFrame(rows)

    # 6. Print clean table with all columns
    print("-" * 85)
    print("                        THRESHOLD SWEEP EVALUATION RESULTS                       ")
    print("-" * 85)
    formatted_table = sweep_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "FPR": "{:.4f}".format,
            "FNR": "{:.4f}".format,
        },
    )
    print(formatted_table)
    print("=" * 85 + "\n")

    # 7. Save table to CSV file
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    sweep_df.to_csv(output_csv_path, index=False)
    print(f"[Success] Threshold sweep table saved to: '{output_csv_path}'\n")

    return sweep_df


if __name__ == "__main__":
    run_threshold_sweep()
