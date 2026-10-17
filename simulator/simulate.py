"""
Stage 4 — Transaction Simulator

Continuously sends realistic (and occasionally fraudulent) transactions
to the FastAPI /score endpoint to simulate real-time traffic.

Run: python simulator/simulate.py
"""

import random
import time
import requests
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

API_URL  = "http://localhost:8000/score"
INTERVAL = 1.0   # seconds between transactions

# Pool of simulated users
USER_IDS = [f"user_{i:03d}" for i in range(1, 21)]


def random_v_features(fraud: bool) -> dict:
    """
    Generate V1-V28 PCA features.
    Fraud transactions shift V4, V11, V14 (strongest predictors
    in the Kaggle dataset) into suspicious territory.
    """
    base = {f"V{i}": round(random.gauss(0, 1), 4) for i in range(1, 29)}
    if fraud:
        # These components correlate strongly with fraud in the real dataset
        base["V4"]  = round(random.gauss( 4.0, 1.0), 4)
        base["V11"] = round(random.gauss( 4.0, 1.0), 4)
        base["V14"] = round(random.gauss(-8.0, 2.0), 4)
        base["V17"] = round(random.gauss(-7.0, 2.0), 4)
    return base


def generate_transaction(fraud: bool = False) -> dict:
    """Build one transaction payload."""
    user_id = random.choice(USER_IDS)

    if fraud:
        # Fraud: high amount, odd hours, unusual V features
        amount = round(random.uniform(500, 4000), 2)
        hour   = random.choice([1, 2, 3, 23, 0])    # late night / early morning
    else:
        # Legit: typical everyday purchase
        amount = round(random.uniform(5, 300), 2)
        hour   = random.randint(8, 22)

    txn = {
        "user_id": user_id,
        "amount":  amount,
        "hour":    hour,
        **random_v_features(fraud),
    }
    return txn


def send_transaction(txn: dict):
    """POST the transaction to the API and log the result."""
    try:
        resp = requests.post(API_URL, json=txn, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        tag  = "FRAUD" if data["decision"] == "reject" else data["decision"].upper()
        log.info(
            f"[{tag:7s}]  user={txn['user_id']}  "
            f"amount=${txn['amount']:8.2f}  "
            f"score={data['fraud_score']:.4f}"
        )
    except requests.ConnectionError:
        log.error("Cannot reach API. Is 'uvicorn api.main:app --reload' running?")
    except Exception as e:
        log.error(f"Unexpected error: {e}")


def run():
    log.info(f"Simulator started. Sending to {API_URL} every {INTERVAL}s …")
    log.info("Press Ctrl+C to stop.\n")

    while True:
        # ~10% of transactions are fraud-like
        is_fraud = random.random() < 0.10
        txn = generate_transaction(fraud=is_fraud)
        send_transaction(txn)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    run()
