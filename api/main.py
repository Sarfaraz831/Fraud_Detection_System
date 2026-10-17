"""
Stage 2 + 3 — FastAPI Fraud Scoring Service + Database Logging

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import sys
import logging
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Allow importing db/ from the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from db.database import insert_transaction, fetch_recent, update_actual_label

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

# ── Load model artifacts ──────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
FEAT_PATH  = os.path.join(BASE_DIR, "models", "features.pkl")

try:
    model         = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEAT_PATH)
    log.info(f"Model loaded. Expects {len(feature_names)} features.")
except FileNotFoundError:
    log.error("model.pkl / features.pkl not found. Run training/train.py first.")
    model, feature_names = None, []

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring with XGBoost",
    version="1.0.0",
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class TransactionInput(BaseModel):
    """
    The raw transaction fields sent by the client / simulator.
    V1-V28 are the anonymised PCA features from the Kaggle dataset.
    """
    user_id: Optional[str] = Field(default="unknown", description="User identifier")
    amount:  float         = Field(..., gt=0,          description="Transaction amount in USD")
    hour:    int           = Field(..., ge=0, le=23,   description="Hour of the day (0-23)")

    # PCA components from the Kaggle dataset
    V1: float = 0.0;  V2: float = 0.0;  V3: float = 0.0;  V4: float = 0.0
    V5: float = 0.0;  V6: float = 0.0;  V7: float = 0.0;  V8: float = 0.0
    V9: float = 0.0;  V10: float = 0.0; V11: float = 0.0; V12: float = 0.0
    V13: float = 0.0; V14: float = 0.0; V15: float = 0.0; V16: float = 0.0
    V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0
    V25: float = 0.0; V26: float = 0.0; V27: float = 0.0; V28: float = 0.0


class ScoreResponse(BaseModel):
    fraud_score: float
    decision:    str
    user_id:     str


class FeedbackInput(BaseModel):
    txn_id:       int
    actual_label: int   # 1 = confirmed fraud, 0 = confirmed legit


# ── Feature preparation ───────────────────────────────────────────────────────
def prepare_features(txn: TransactionInput) -> np.ndarray:
    """
    Build the feature vector in the exact order the model was trained on.
    Features engineered here mirror what train.py produces.
    """
    import math

    log_amount = math.log1p(txn.amount)

    # Collect all V features
    v_features = [getattr(txn, f"V{i}") for i in range(1, 29)]

    # Build a dict keyed by feature name
    feat_dict = {
        **{f"V{i}": v_features[i-1] for i in range(1, 29)},
        "hour":                  txn.hour,
        "log_amount":            log_amount,
        "rolling_mean_amount":   txn.amount,   # single txn — best approx is itself
        "rolling_std_amount":    0.0,
        "amount_deviation":      0.0,
    }

    # Order matches training feature list
    vector = [feat_dict.get(f, 0.0) for f in feature_names]
    return np.array(vector).reshape(1, -1), log_amount


# ── Decision logic ────────────────────────────────────────────────────────────
def make_decision(score: float) -> str:
    if score > 0.7:
        return "reject"
    if score > 0.3:
        return "review"
    return "approve"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/score", response_model=ScoreResponse)
def score_transaction(txn: TransactionInput):
    """
    Score a transaction in real time.
    Returns fraud_score (0–1) and decision (approve / review / reject).
    Also logs the full record to SQLite.
    """
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run train.py first.")

    # 1. Build feature vector
    features, log_amount = prepare_features(txn)

    # 2. Score with XGBoost
    fraud_score = float(model.predict_proba(features)[0][1])

    # 3. Decision
    decision = make_decision(fraud_score)

    # 4. Log to database
    insert_transaction({
        "user_id":    txn.user_id,
        "amount":     txn.amount,
        "hour":       txn.hour,
        "log_amount": log_amount,
        "fraud_score": round(fraud_score, 6),
        "decision":   decision,
    })

    log.info(f"user={txn.user_id}  amount={txn.amount:.2f}  "
             f"score={fraud_score:.4f}  decision={decision}")

    return ScoreResponse(
        fraud_score=round(fraud_score, 4),
        decision=decision,
        user_id=txn.user_id or "unknown",
    )


@app.get("/transactions")
def get_transactions(limit: int = 50):
    """Return recent scored transactions (for the dashboard)."""
    return fetch_recent(limit)


@app.post("/feedback")
def submit_feedback(fb: FeedbackInput):
    """
    Stage 6 — Feedback loop.
    Mark a transaction as confirmed fraud or legit.
    These labels can later be used to retrain the model.
    """
    update_actual_label(fb.txn_id, fb.actual_label)
    return {"status": "updated", "txn_id": fb.txn_id, "label": fb.actual_label}
