"""
Stage 1 — Model Training Pipeline
Loads the Kaggle Credit Card Fraud dataset, engineers features,
trains an XGBoost classifier, and saves model + feature list.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Download creditcard.csv and place it in the data/ folder.
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
FEAT_PATH  = os.path.join(BASE_DIR, "models", "features.pkl")


# ── 1. Load data ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        log.warning(f"Dataset not found at {path}.")
        log.info("Generating synthetic transaction dataset for training...")
        import sys
        sys.path.insert(0, BASE_DIR)
        from data.generate_synthetic_data import generate_synthetic_transactions
        df = generate_synthetic_transactions(n_samples=50000, fraud_rate=0.005)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        log.info(f"Synthetic dataset generated and saved to {path}")
        return df

    log.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    log.info(f"Shape: {df.shape}  |  Fraud rate: {df['Class'].mean():.4%}")
    return df


# ── 2. Feature engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering features …")

    # Hour of day extracted from the 'Time' column (seconds since first txn)
    df["hour"] = (df["Time"] // 3600) % 24

    # Normalise Amount (log scale reduces skew)
    df["log_amount"] = np.log1p(df["Amount"])

    # Rolling statistics — approximate spend velocity
    # (In a real system this would be per-user; here it's a global approximation)
    df = df.sort_values("Time").reset_index(drop=True)
    df["rolling_mean_amount"] = (
        df["Amount"].rolling(window=10, min_periods=1).mean()
    )
    df["rolling_std_amount"] = (
        df["Amount"].rolling(window=10, min_periods=1).std().fillna(0)
    )
    # Amount deviation from rolling mean — useful anomaly signal
    df["amount_deviation"] = df["Amount"] - df["rolling_mean_amount"]

    # Drop raw columns we've replaced or that leak information
    df.drop(columns=["Time", "Amount"], inplace=True)

    log.info(f"Features after engineering: {df.columns.tolist()}")
    return df


# ── 3. Split ─────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── 4. Train ─────────────────────────────────────────────────────────────────
def train_model(X_train, y_train):
    # scale_pos_weight balances the heavily imbalanced classes
    # ratio = (number of negatives) / (number of positives)
    neg  = (y_train == 0).sum()
    pos  = (y_train == 1).sum()
    spw  = neg / pos
    log.info(f"scale_pos_weight set to {spw:.1f}  (neg={neg}, pos={pos})")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="aucpr",       # precision-recall AUC — better for imbalanced data
        random_state=42,
        n_jobs=-1,
    )
    log.info("Training XGBoost …")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=50,
    )
    log.info("Training complete.")
    return model


# ── 5. Evaluate ──────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    precision   = precision_score(y_test, y_pred)
    recall      = recall_score(y_test, y_pred)
    roc_auc     = roc_auc_score(y_test, y_prob)

    log.info("─── Evaluation Results ───────────────────────────")
    log.info(f"Precision : {precision:.4f}")
    log.info(f"Recall    : {recall:.4f}")
    log.info(f"ROC-AUC   : {roc_auc:.4f}")
    log.info("\n" + classification_report(y_test, y_pred,
                                          target_names=["legit", "fraud"]))
    log.info("Confusion matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    log.info("──────────────────────────────────────────────────")


# ── 6. Save artifacts ────────────────────────────────────────────────────────
def save_artifacts(model, feature_names: list):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, FEAT_PATH)
    log.info(f"Model saved  → {MODEL_PATH}")
    log.info(f"Features saved → {FEAT_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df                              = load_data(DATA_PATH)
    df                              = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model                           = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_artifacts(model, X_train.columns.tolist())
    log.info("All done! Run the API next: uvicorn api.main:app --reload")
