"""
Database helper — creates and manages the SQLite fraud.db.
Imported by both the API (writes) and the dashboard (reads).
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "db", "fraud.db")


def get_connection() -> sqlite3.Connection:
    """Return a connection with row_factory so rows behave like dicts."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the transactions table if it doesn't exist yet."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            user_id         TEXT,
            amount          REAL,
            hour            INTEGER,
            log_amount      REAL,
            fraud_score     REAL,
            decision        TEXT,
            actual_label    INTEGER DEFAULT NULL   -- filled in later for feedback loop
        )
    """)
    conn.commit()
    conn.close()


def insert_transaction(record: dict):
    """
    Insert one scored transaction.
    record must contain: user_id, amount, hour, log_amount,
                         fraud_score, decision
    """
    conn = get_connection()
    conn.execute("""
        INSERT INTO transactions
            (timestamp, user_id, amount, hour, log_amount, fraud_score, decision)
        VALUES
            (:timestamp, :user_id, :amount, :hour, :log_amount,
             :fraud_score, :decision)
    """, {**record, "timestamp": datetime.utcnow().isoformat()})
    conn.commit()
    conn.close()


def fetch_recent(limit: int = 200) -> list:
    """Return the most recent transactions as a list of dicts."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM transactions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_actual_label(txn_id: int, label: int):
    """
    Feedback loop: mark a transaction as confirmed fraud (1) or legit (0).
    This lets you later retrain on real outcomes.
    """
    conn = get_connection()
    conn.execute(
        "UPDATE transactions SET actual_label = ? WHERE id = ?",
        (label, txn_id)
    )
    conn.commit()
    conn.close()


# Initialise the DB when this module is first imported
init_db()
