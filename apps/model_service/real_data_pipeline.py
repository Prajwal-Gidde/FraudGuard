"""
Real Data Pipeline for FraudGuard 360 Model Service (Member 4).

Loads real transaction data from parquet fixtures, converts records into Transaction schema
objects, extracts real features using Member 3's feature extraction service, joins the features
with ground-truth fraud labels by transaction_id, and saves the final feature dataset.
"""

import os
import sys
import pandas as pd

# ==============================================================================
# Monorepo Path Configuration
# ==============================================================================
# Adjust sys.path to include the repository root directory.
# This sys.path adjustment is required because apps/feature_service, apps/model_service,
# and shared are sibling packages/directories within the team monorepo structure.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Imports from sibling packages in the monorepo
try:
    from shared.schemas.transaction import Transaction
    from apps.feature_service.features.extractor import extract_batch
except ImportError:
    # Fallback handling for standalone test environments or mock execution
    Transaction = None
    extract_batch = None


# Exact 12-field feature schema provided by Member 3 (apps/feature_service)
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


def load_raw_transactions(parquet_path: str) -> pd.DataFrame:
    """
    Loads raw transaction records from a parquet fixture file.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at expected path: {parquet_path}")

    return pd.read_parquet(parquet_path)


def process_real_data_pipeline(parquet_path: str) -> pd.DataFrame:
    """
    Executes the end-to-end real data pipeline:
    1. Loads raw transactions from parquet file.
    2. Sorts transactions chronologically by timestamp.
    3. Converts rows into Transaction schema objects and extracts features via extract_batch().
    4. Joins extracted feature vectors back with the ground-truth is_fraud label by transaction_id.
    5. Returns a pandas DataFrame containing all 12 real features plus is_fraud.
    """
    # 1. Load raw transaction parquet data
    print(f"Loading raw transaction data from: {parquet_path}")
    df_raw = load_raw_transactions(parquet_path)

    # 2. Sort by timestamp so stateful feature extractors compute in strict chronological sequence
    timestamp_col = None
    for col in ["timestamp", "created_at", "txn_timestamp", "time"]:
        if col in df_raw.columns:
            timestamp_col = col
            break

    if timestamp_col:
        df_raw = df_raw.sort_values(by=timestamp_col).reset_index(drop=True)

    # 3. Convert rows into Transaction objects and execute batch feature extraction
    if Transaction is not None and extract_batch is not None:
        transactions = []
        for _, row in df_raw.iterrows():
            row_dict = row.to_dict()
            txn = Transaction(**row_dict)
            transactions.append(txn)

        print(f"Extracting features for {len(transactions)} transaction records via extract_batch()...")
        extracted_features = extract_batch(transactions)

        # Standardize return type from extract_batch into a DataFrame
        if isinstance(extracted_features, pd.DataFrame):
            features_df = extracted_features.copy()
        elif isinstance(extracted_features, list):
            if len(extracted_features) > 0 and hasattr(extracted_features[0], "dict"):
                features_df = pd.DataFrame([item.dict() for item in extracted_features])
            elif len(extracted_features) > 0 and hasattr(extracted_features[0], "__dict__"):
                features_df = pd.DataFrame([item.__dict__ for item in extracted_features])
            else:
                features_df = pd.DataFrame(extracted_features)
        else:
            features_df = pd.DataFrame(extracted_features)
    else:
        # Fallback if imports are executed in isolation
        print("Warning: Monorepo feature service modules unavailable. Using fallback extraction.")
        features_df = df_raw.copy()

    # 4. Join feature vectors back with the original is_fraud label by transaction_id
    if "transaction_id" in features_df.columns and "transaction_id" in df_raw.columns:
        labels_df = df_raw[["transaction_id", "is_fraud"]].drop_duplicates(subset=["transaction_id"])
        merged_df = features_df.merge(labels_df, on="transaction_id", how="inner")
    else:
        merged_df = features_df.copy()
        if "is_fraud" not in merged_df.columns and "is_fraud" in df_raw.columns:
            merged_df["is_fraud"] = df_raw["is_fraud"].values

    # 5. Format final DataFrame with all 12 real features plus is_fraud
    for col in REAL_FEATURE_COLUMNS:
        if col not in merged_df.columns:
            print(f"Warning: Feature column '{col}' missing from extract_batch output. Initializing default 0.")
            merged_df[col] = 0

    final_columns = REAL_FEATURE_COLUMNS + ["is_fraud"]
    final_df = merged_df[final_columns].copy()

    return final_df


if __name__ == "__main__":
    # 1. Relative path to transactions.parquet from apps/model_service/ up to repo root's data folder
    relative_parquet_path = os.path.join(
        "..", "..", "data", "fixtures", "DS_91c85fbe", "transactions.parquet"
    )
    abs_parquet_path = os.path.abspath(os.path.join(CURRENT_DIR, relative_parquet_path))

    # Output CSV path relative to apps/model_service/
    output_dir = os.path.join(CURRENT_DIR, "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "features_real.csv")

    # Run the real data pipeline
    dataset_df = process_real_data_pipeline(parquet_path=abs_parquet_path)

    # Save output to data/features_real.csv
    dataset_df.to_csv(output_file, index=False)
    print(f"\nSaved feature dataset to: '{output_file}'")

    # Print shape and class balance
    print("\n--- DATASET SUMMARY ---")
    print(f"Dataset Shape: {dataset_df.shape}")
    print("\nClass Balance (is_fraud value counts):")
    print(dataset_df["is_fraud"].value_counts())
