import os
import numpy as np
import pandas as pd


def generate_dummy_dataset(n_rows: int = 2000, fraud_ratio: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic training data for FraudGuard 360 Fraud Detection ML module.

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
        DataFrame containing synthetic transaction data with exact contract schema:
        - customer_txn_count_60m: integer, transactions in last hour (0 to 20)
        - amount_deviation_ratio: float, z-score deviation from normal spending (-1.0 to 5.0)
        - is_new_device: int, indicator if device is new (0 or 1)
        - shared_device_account_count: integer, count of shared accounts on device (0 to 10)
        - is_fraud: int, target label (0 or 1)
    """
    np.random.seed(seed)

    n_fraud = int(n_rows * fraud_ratio)
    n_legit = n_rows - n_fraud

    # Target labels
    is_fraud = np.array([1] * n_fraud + [0] * n_legit)

    # Legitimate transaction signal (lower activity, lower deviation, rarely new/shared device)
    velocity_legit = np.random.poisson(lam=1.5, size=n_legit)
    amount_dev_legit = np.random.normal(loc=0.0, scale=0.8, size=n_legit)
    new_device_legit = np.random.binomial(n=1, p=0.08, size=n_legit)
    shared_device_legit = np.random.poisson(lam=0.3, size=n_legit)

    # Fraudulent transaction signal (higher velocity, high amount deviation, frequently new/shared device)
    velocity_fraud = np.random.poisson(lam=8.0, size=n_fraud)
    amount_dev_fraud = np.random.normal(loc=2.5, scale=1.0, size=n_fraud)
    new_device_fraud = np.random.binomial(n=1, p=0.65, size=n_fraud)
    shared_device_fraud = np.random.poisson(lam=3.5, size=n_fraud)

    # Combine distributions
    customer_txn_count_60m = np.concatenate([velocity_fraud, velocity_legit])
    amount_deviation_ratio = np.concatenate([amount_dev_fraud, amount_dev_legit])
    is_new_device = np.concatenate([new_device_fraud, new_device_legit])
    shared_device_account_count = np.concatenate([shared_device_fraud, shared_device_legit])

    # Enforce strict bounds as per team contract
    customer_txn_count_60m = np.clip(customer_txn_count_60m, 0, 20).astype(int)
    amount_deviation_ratio = np.round(np.clip(amount_deviation_ratio, -1.0, 5.0), 4)
    is_new_device = np.clip(is_new_device, 0, 1).astype(int)
    shared_device_account_count = np.clip(shared_device_account_count, 0, 10).astype(int)

    df = pd.DataFrame({
        'customer_txn_count_60m': customer_txn_count_60m,
        'amount_deviation_ratio': amount_deviation_ratio,
        'is_new_device': is_new_device,
        'shared_device_account_count': shared_device_account_count,
        'is_fraud': is_fraud
    })

    # Shuffle dataset rows
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == '__main__':
    output_dir = 'dummy_data'
    output_file = os.path.join(output_dir, 'train_dummy.csv')

    os.makedirs(output_dir, exist_ok=True)

    dataset = generate_dummy_dataset()
    dataset.to_csv(output_file, index=False)
    print(f"Successfully generated synthetic dataset with {len(dataset)} rows at '{output_file}'.")
