# 🛡️ Real-Time Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/Database-SQLite3-003B57.svg)](https://www.sqlite.org/)

An end-to-end, real-time transaction fraud scoring and monitoring platform built with Python, XGBoost, FastAPI, SQLite, and Streamlit. The system simulates high-velocity credit card transactions, scores fraud risk via machine learning in sub-100ms latency, logs decisions into a persistent transactional database, and visualizes live activity on an interactive dashboard.

---

## ⚠️ Synthetic Data Notice & Disclosure

> [!IMPORTANT]
> **All training, testing, and simulated transaction data in this repository is strictly SYNTHETIC.**
> 
> Due to strict banking privacy regulations, customer data protection standards (GDPR, PCI-DSS), and ethical considerations, real cardholder transaction records and Personally Identifiable Information (PII) are not stored or tracked in this project.

### About the Synthetic Dataset
* **Distribution-Matched**: The synthetic transaction generation pipeline (`data/generate_synthetic_data.py`) generates multivariate distributions mirroring benchmark transaction profiles (such as the ULB Kaggle Credit Card Fraud benchmark).
* **PCA Transformation ($V_1$ to $V_{28}$)**: Features $V_1$ through $V_{28}$ represent synthetic principal component vectors. In fraudulent scenarios, key statistical shifts are simulated across high-correlation components ($V_4$, $V_{11}$, $V_{14}$, and $V_{17}$).
* **Severe Class Imbalance**: Real-world fraud is extremely rare. The synthetic generator models an authentic low-incidence environment (~0.5% fraud rate), challenging the ML model with heavy class imbalance.
* **Realistic Behavioral Signals**: Timestamps are converted to diurnal transaction cycles (hour of day), and amounts are log-normally distributed to reflect authentic cardholder spending patterns.
* **Zero PII**: Contains no genuine credit card numbers, names, addresses, or authentic bank accounts.

---

## 🏛️ System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │     Transaction Simulator (simulator/simulate.py)      │
       │  Generates realistic & fraudulent transaction payloads │
       └───────────────────────────┬────────────────────────────┘
                                   │ HTTP POST (every 1s)
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │            FastAPI Server (api/main.py)                │
       │             Validates schema via Pydantic              │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │            Feature Engineering Pipeline                │
       │    Calculates hour, log(amount), and rolling stats     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             XGBoost Classification Engine              │
       │        Predicts fraud risk score: [0.0 - 1.0]          │
       └───────────────────────────┬────────────────────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
       ┌──────────────────────┐          ┌──────────────────────┐
       │   Decision Engine    │          │  SQLite Database     │
       │  > 0.70  → Reject    │          │  (db/fraud.db)       │
       │  > 0.30  → Review    │─────────▶│  Logs timestamp,     │
       │  <= 0.30 → Approve   │          │  scores & outcomes   │
       └──────────────────────┘          └──────────┬───────────┘
                                                    │
                                                    ▼
                                  ┌──────────────────────────────────┐
                                  │   Streamlit Live Dashboard       │
                                  │     (dashboard/app.py)           │
                                  │ Real-time metrics & charts (5s)  │
                                  └──────────────────────────────────┘
```

---

## 📁 Project Structure

```text
fraud-detection-system/
├── .gitignore                      # Git exclusion rules (ignores cache, DBs, CSVs, models)
├── README.md                       # Complete documentation & synthetic data disclosure
├── requirements.txt                # Python dependencies
├── data/
│   ├── .gitkeep                    # Tracks folder structure in Git
│   ├── generate_synthetic_data.py  # Generates synthetic benchmark dataset
│   └── creditcard.csv              # Synthetic/benchmark dataset (auto-generated or user-supplied)
├── models/
│   ├── .gitkeep                    # Tracks folder structure in Git
│   ├── model.pkl                   # Trained XGBoost model artifact (generated)
│   └── features.pkl                # Serialized feature column ordering (generated)
├── api/
│   └── main.py                     # FastAPI REST service with /score, /health, /feedback
├── db/
│   ├── database.py                 # SQLite connection and transaction ledger helpers
│   └── fraud.db                    # SQLite database (created on first run)
├── simulator/
│   └── simulate.py                 # Traffic generator simulating live transactions
├── dashboard/
│   └── app.py                      # Streamlit interactive operational dashboard
└── training/
    └── train.py                    # Model training pipeline with evaluation metrics
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup

Ensure you have **Python 3.9+** installed:

```bash
# Clone or navigate to your workspace
cd fraud-detection-system

# Create a virtual environment (optional but recommended)
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Prepare Data & Train the Model

The training script automatically checks for `data/creditcard.csv`. If absent, it **automatically invokes the synthetic data generator** to produce a 50,000-sample synthetic dataset:

```bash
# Train the XGBoost classifier
python training/train.py
```

*Alternatively, generate the synthetic dataset directly anytime:*
```bash
python data/generate_synthetic_data.py
```

Expected output:
```text
INFO  Generating 50,000 synthetic records (49,750 legit, 250 fraud)...
INFO  Synthetic dataset generated successfully. Shape: (50000, 31), Fraud rate: 0.5000%
INFO  Training XGBoost with scale_pos_weight ...
INFO  ─── Evaluation Results ───
INFO  Precision : 0.9412
INFO  Recall    : 0.8889
INFO  ROC-AUC   : 0.9984
INFO  Model saved  → models/model.pkl
INFO  Features saved → models/features.pkl
```

---

### 3. Launch the API Service

Start the FastAPI backend with Uvicorn:

```bash
uvicorn api.main:app --reload --port 8000
```

* API Docs (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Check: [http://localhost:8000/health](http://localhost:8000/health)

#### Test scoring a transaction via cURL:

```bash
curl -X POST http://localhost:8000/score ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\": \"user_007\", \"amount\": 1850.50, \"hour\": 2, \"V4\": 4.2, \"V11\": 3.9, \"V14\": -8.1, \"V17\": -6.9}"
```

Sample response:
```json
{
  "fraud_score": 0.9412,
  "decision": "reject",
  "user_id": "user_007"
}
```

---

### 4. Run the Transaction Simulator

In a separate terminal, run the simulator to mimic incoming real-time transactions:

```bash
python simulator/simulate.py
```

Console output will stream transactions in real time:
```text
INFO  [APPROVE]  user=user_014  amount=$   24.50  score=0.0084
INFO  [REVIEW ]  user=user_003  amount=$  320.00  score=0.4215
INFO  [FRAUD  ]  user=user_007  amount=$ 2450.00  score=0.9632
```

---

### 5. Open the Streamlit Dashboard

In another terminal, launch the dashboard:

```bash
streamlit run dashboard/app.py
```

The browser will open at [http://localhost:8501](http://localhost:8501).
* **Live KPI cards**: Total volume, Approved, Flagged for Review, Rejected, and overall Fraud Rate.
* **Score Distribution**: Histogram showing probability density of incoming scores.
* **Score Timeline**: Dynamic line chart tracking score trends.
* **Transaction Table**: Real-time color-coded transaction stream.
* **User Breakdown**: High-risk user rankings.

---

## 🔄 Human-in-the-Loop Feedback Loop

To support continuous model monitoring and future retraining, the API exposes a feedback endpoint:

```bash
curl -X POST http://localhost:8000/feedback ^
  -H "Content-Type: application/json" ^
  -d "{\"txn_id\": 1, \"actual_label\": 1}"
```

Investigated transactions can be marked as confirmed fraud (`1`) or confirmed legitimate (`0`).

---

## 🛠️ API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Verifies service status and confirms model weights are loaded |
| `/score` | `POST` | Scores a transaction, applies decision threshold, and logs to SQLite |
| `/transactions` | `GET` | Retrieves recent transactions with scores and decisions |
| `/feedback` | `POST` | Updates actual label for a transaction (feedback loop) |

---

## 📤 Pushing to Your GitHub Repository

To push this project to your GitHub repository:

```bash
# 1. Initialize git (if not already done)
git init -b main

# 2. Stage all important files (respects .gitignore)
git add .

# 3. Create your initial commit
git commit -m "feat: initial commit for real-time fraud detection system"

# 4. Add your GitHub remote repository URL
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git

# 5. Push to GitHub
git push -u origin main
```

---

## 📄 License

This project is licensed under the MIT License.
