import os
import numpy as np
import pandas as pd


def generate_dummy_dataset(n_rows: int = 2000, fraud_ratio: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic training data for FraudGuard 360 Fraud Detection ML module.
    Produces all 12 Member 3 canonical features plus ground-truth target label `is_fraud`.

    Parameters
    ----------
    n_rows : int, default=2000
        Total number of rows (transaction records) to generate.
    fraud_ratio : float, default=0.15
        Target proportion of fraudulent transactions in the dataset (range 0.0 to 1.0).
    seed : int, default=42
        Random seed for reproducibility across random number generators.

    Returns
    -------
    pd.DataFrame
        DataFrame containing synthetic transaction data with 12 canonical feature columns:
        1. customer_txn_count_60m: integer (0 to 20)
        2. customer_amount_mean_prior: float (>= 0.0)
        3. amount_deviation_ratio: float (-1.0 to 5.0)
        4. is_new_device: int (0 or 1)
        5. is_new_merchant: int (0 or 1)
        6. location_shift: int (0 or 1)
        7. customer_device_degree: int (>= 0)
        8. customer_merchant_degree: int (>= 0)
        9. device_customer_degree: int (>= 0)
        10. merchant_customer_degree: int (>= 0)
        11. shared_device_customer_count: int (0 to 10)
        12. relationship_risk_score: float (0.0 to 1.0)
        13. is_fraud: target label (0 or 1)
    """
    np.random.seed(seed)

    n_fraud = int(n_rows * fraud_ratio)
    n_legit = n_rows - n_fraud

    # Target labels (fraud=1, legit=0)
    is_fraud = np.array([1] * n_fraud + [0] * n_legit)

    # 1. customer_txn_count_60m
    vel_legit = np.random.poisson(lam=1.5, size=n_legit)
    vel_fraud = np.random.poisson(lam=8.0, size=n_fraud)
    customer_txn_count_60m = np.clip(np.concatenate([vel_fraud, vel_legit]), 0, 20).astype(int)

    # 2. customer_amount_mean_prior
    amt_prior_legit = np.random.gamma(shape=2.0, scale=25.0, size=n_legit)
    amt_prior_fraud = np.random.gamma(shape=4.0, scale=35.0, size=n_fraud)
    customer_amount_mean_prior = np.round(np.concatenate([amt_prior_fraud, amt_prior_legit]), 2)

    # 3. amount_deviation_ratio
    amt_dev_legit = np.random.normal(loc=0.0, scale=0.8, size=n_legit)
    amt_dev_fraud = np.random.normal(loc=2.5, scale=1.0, size=n_fraud)
    amount_deviation_ratio = np.round(np.clip(np.concatenate([amt_dev_fraud, amt_dev_legit]), -1.0, 5.0), 4)

    # 4. is_new_device
    new_dev_legit = np.random.binomial(n=1, p=0.08, size=n_legit)
    new_dev_fraud = np.random.binomial(n=1, p=0.65, size=n_fraud)
    is_new_device = np.concatenate([new_dev_fraud, new_dev_legit]).astype(int)

    # 5. is_new_merchant
    new_merch_legit = np.random.binomial(n=1, p=0.15, size=n_legit)
    new_merch_fraud = np.random.binomial(n=1, p=0.55, size=n_fraud)
    is_new_merchant = np.concatenate([new_merch_fraud, new_merch_legit]).astype(int)

    # 6. location_shift
    loc_shift_legit = np.random.binomial(n=1, p=0.05, size=n_legit)
    loc_shift_fraud = np.random.binomial(n=1, p=0.45, size=n_fraud)
    location_shift = np.concatenate([loc_shift_fraud, loc_shift_legit]).astype(int)

    # 7. customer_device_degree
    cust_dev_deg_legit = np.random.poisson(lam=1.2, size=n_legit)
    cust_dev_deg_fraud = np.random.poisson(lam=3.8, size=n_fraud)
    customer_device_degree = np.clip(np.concatenate([cust_dev_deg_fraud, cust_dev_deg_legit]), 0, 15).astype(int)

    # 8. customer_merchant_degree
    cust_merch_deg_legit = np.random.poisson(lam=3.0, size=n_legit)
    cust_merch_deg_fraud = np.random.poisson(lam=8.0, size=n_fraud)
    customer_merchant_degree = np.clip(np.concatenate([cust_merch_deg_fraud, cust_merch_deg_legit]), 0, 30).astype(int)

    # 9. device_customer_degree
    dev_cust_deg_legit = np.random.poisson(lam=1.1, size=n_legit)
    dev_cust_deg_fraud = np.random.poisson(lam=5.0, size=n_fraud)
    device_customer_degree = np.clip(np.concatenate([dev_cust_deg_fraud, dev_cust_deg_legit]), 0, 20).astype(int)

    # 10. merchant_customer_degree
    merch_cust_deg_legit = np.random.poisson(lam=5.0, size=n_legit)
    merch_cust_deg_fraud = np.random.poisson(lam=15.0, size=n_fraud)
    merchant_customer_degree = np.clip(np.concatenate([merch_cust_deg_fraud, merch_cust_deg_legit]), 0, 50).astype(int)

    # 11. shared_device_customer_count
    shared_device_legit = np.random.poisson(lam=0.3, size=n_legit)
    shared_device_fraud = np.random.poisson(lam=3.5, size=n_fraud)
    shared_device_customer_count = np.clip(np.concatenate([shared_device_fraud, shared_device_legit]), 0, 10).astype(int)

    # 12. relationship_risk_score
    rel_risk_legit = np.random.beta(a=0.5, b=5.0, size=n_legit)
    rel_risk_fraud = np.random.beta(a=4.0, b=1.5, size=n_fraud)
    relationship_risk_score = np.round(np.clip(np.concatenate([rel_risk_fraud, rel_risk_legit]), 0.0, 1.0), 4)

    df = pd.DataFrame({
        "customer_txn_count_60m": customer_txn_count_60m,
        "customer_amount_mean_prior": customer_amount_mean_prior,
        "amount_deviation_ratio": amount_deviation_ratio,
        "is_new_device": is_new_device,
        "is_new_merchant": is_new_merchant,
        "location_shift": location_shift,
        "customer_device_degree": customer_device_degree,
        "customer_merchant_degree": customer_merchant_degree,
        "device_customer_degree": device_customer_degree,
        "merchant_customer_degree": merchant_customer_degree,
        "shared_device_customer_count": shared_device_customer_count,
        "relationship_risk_score": relationship_risk_score,
        "is_fraud": is_fraud
    })

    # Shuffle dataset rows with fixed seed
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    output_dir = "dummy_data"
    output_file = os.path.join(output_dir, "train_dummy.csv")

    os.makedirs(output_dir, exist_ok=True)

    dataset = generate_dummy_dataset()
    dataset.to_csv(output_file, index=False)
    print(f"Successfully generated 12-feature synthetic dataset with {len(dataset)} rows at '{output_file}'.")
