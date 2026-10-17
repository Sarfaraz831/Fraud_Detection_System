"""
Stage 5 — Streamlit Dashboard

Connects to fraud.db and visualises live results.

Run: streamlit run dashboard/app.py
"""

import sys
import os
import time
import pandas as pd
import streamlit as st

# Allow importing db/ from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from db.database import fetch_recent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Real-Time Fraud Detection Dashboard")
st.caption("Auto-refreshes every 5 seconds. Data from SQLite fraud.db.")

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data(limit: int) -> pd.DataFrame:
    rows = fetch_recent(limit)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")
refresh_rate = st.sidebar.slider("Refresh interval (s)", 2, 30, 5)
row_limit    = st.sidebar.slider("Rows to load", 50, 500, 200)

df = load_data(row_limit)

user_options = ["All users"]
if not df.empty and "user_id" in df.columns:
    user_options += sorted(df["user_id"].dropna().unique().tolist())
user_filter = st.sidebar.selectbox("Filter by user", user_options)

if df.empty:
    st.warning("No transactions yet. Start the simulator: `python simulator/simulate.py`")
    time.sleep(refresh_rate)
    st.rerun()

# Apply user filter
if user_filter != "All users":
    df = df[df["user_id"] == user_filter]

# ── Rebuild sidebar user list dynamically ─────────────────────────────────────
# (Streamlit re-runs top to bottom, so the selectbox will update on next refresh)

# ── Top-level metrics ─────────────────────────────────────────────────────────
total      = len(df)
n_reject   = (df["decision"] == "reject").sum()
n_review   = (df["decision"] == "review").sum()
n_approve  = (df["decision"] == "approve").sum()
fraud_rate = n_reject / total * 100 if total else 0
avg_score  = df["fraud_score"].mean() if total else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total transactions", f"{total:,}")
col2.metric("Approved",           f"{n_approve:,}")
col3.metric("Flagged for review", f"{n_review:,}")
col4.metric("Rejected (fraud)",   f"{n_reject:,}")
col5.metric("Fraud rate",         f"{fraud_rate:.1f}%")

st.divider()

# ── Row 1: Score distribution + Decision breakdown ────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Fraud score distribution")
    # Histogram using Streamlit's built-in chart (no extra libraries needed)
    hist_df = df["fraud_score"].value_counts(bins=20, sort=False).reset_index()
    hist_df.columns = ["score_range", "count"]
    st.bar_chart(hist_df.set_index("score_range")["count"])

with right:
    st.subheader("Decision breakdown")
    decision_counts = df["decision"].value_counts().rename_axis("decision").reset_index(name="count")
    st.bar_chart(decision_counts.set_index("decision"))

# ── Row 2: Score over time ────────────────────────────────────────────────────
st.subheader("Fraud score over time (last 200 transactions)")
time_df = df.sort_values("timestamp").tail(200)[["timestamp", "fraud_score"]].set_index("timestamp")
st.line_chart(time_df)

# ── Row 3: Recent transactions table ─────────────────────────────────────────
st.subheader("Recent transactions (last 20)")

display_cols = ["id", "timestamp", "user_id", "amount", "hour",
                "fraud_score", "decision", "actual_label"]
recent = df.head(20)[display_cols].copy()

# Colour-code decisions
def colour_decision(val):
    colours = {"reject": "background-color:#ffd6d6",
               "review": "background-color:#fff3cd",
               "approve": "background-color:#d6f0d6"}
    return colours.get(val, "")

if hasattr(recent.style, "map"):
    styled = recent.style.map(colour_decision, subset=["decision"])
else:
    styled = recent.style.applymap(colour_decision, subset=["decision"])
st.dataframe(styled, use_container_width=True)

# ── Row 4: Per-user stats ─────────────────────────────────────────────────────
st.subheader("Per-user fraud summary")
user_stats = (
    df.groupby("user_id")
    .agg(
        transactions=("id", "count"),
        avg_score=("fraud_score", "mean"),
        rejections=("decision", lambda x: (x == "reject").sum()),
    )
    .sort_values("rejections", ascending=False)
    .reset_index()
)
st.dataframe(user_stats, use_container_width=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.caption(f"Next refresh in {refresh_rate}s …")
time.sleep(refresh_rate)
st.rerun()
