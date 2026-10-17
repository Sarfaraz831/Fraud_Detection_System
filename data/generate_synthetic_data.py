"""
Synthetic Credit Card Data Generator

Generates a synthetic dataset modeled after the Kaggle Credit Card Fraud Detection benchmark.
Includes:
- Time: Elapsed seconds across simulated days
- V1-V28: Synthetic PCA components generated with multivariate normal distributions
  (with fraud class featuring realistic distribution shifts on key indicators like V4, V11, V14, V17)
- Amount: Log-normally distributed transaction amounts
- Class: 0 for legitimate transactions, 1 for fraudulent transactions (imbalanced)
"""

import os
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")


def generate_synthetic_transactions(
    n_samples: int = 50000,
    fraud_rate: float = 0.005,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic credit card dataset."""
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    log.info(f"Generating {n_samples:,} synthetic records ({n_legit:,} legit, {n_fraud:,} fraud)...")

    # 1. Generate Timestamps (seconds over 2 days ~ 172,800 seconds)
    time_legit = np.sort(np.random.uniform(0, 172800, n_legit))
    time_fraud = np.sort(np.random.uniform(0, 172800, n_fraud))

    # 2. Generate Amounts (log-normal distribution)
    # Legit: typical spend $5 - $200
    amount_legit = np.random.lognormal(mean=3.0, sigma=1.0, size=n_legit).round(2)
    # Fraud: skewed higher, larger purchases or micro-testing
    amount_fraud = np.random.lognormal(mean=5.2, sigma=1.4, size=n_fraud).round(2)

    # 3. Generate V1 to V28 PCA features
    # Standard Gaussian for baseline
    v_legit = np.random.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    v_fraud = np.random.normal(loc=0.0, scale=1.0, size=(n_fraud, 28))

    # Introduce known fraud signal shifts (0-indexed: V4=idx 3, V11=idx 10, V14=idx 13, V17=idx 16)
    v_fraud[:, 3] = np.random.normal(loc=4.0, scale=1.2, size=n_fraud)    # V4
    v_fraud[:, 10] = np.random.normal(loc=3.8, scale=1.2, size=n_fraud)   # V11
    v_fraud[:, 13] = np.random.normal(loc=-7.5, scale=2.0, size=n_fraud)  # V14
    v_fraud[:, 16] = np.random.normal(loc=-6.8, scale=2.0, size=n_fraud)  # V17
    v_fraud[:, 11] = np.random.normal(loc=-4.5, scale=1.5, size=n_fraud)  # V12

    # Combine into dataframes
    v_cols = [f"V{i}" for i in range(1, 29)]

    df_legit = pd.DataFrame(v_legit, columns=v_cols)
    df_legit["Time"] = time_legit
    df_legit["Amount"] = amount_legit
    df_legit["Class"] = 0

    df_fraud = pd.DataFrame(v_fraud, columns=v_cols)
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    # Reorder columns to match standard schema: Time, V1-V28, Amount, Class
    ordered_cols = ["Time"] + v_cols + ["Amount", "Class"]
    df = df[ordered_cols].sort_values("Time").reset_index(drop=True)

    log.info(f"Synthetic dataset generated successfully. Shape: {df.shape}, Fraud rate: {df['Class'].mean():.4%}")
    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = generate_synthetic_transactions()
    df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"Saved synthetic dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
