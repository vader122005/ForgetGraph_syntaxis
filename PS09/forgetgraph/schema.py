"""FORGETGRAPH SQLite schema + synthetic data generation.

Build order step 1: create tables and populate 200 synthetic users
(fixed random seed = 42) into ``forgetgraph.db``.

Tables:
    users        — id, age, account_balance, num_transactions, tenure_days, risk_score
    features     — id, user_id, <derived feature columns>
    predictions  — id, user_id, predicted_value, cached_at
    tombstones   — user_id, deleted_at
    deletion_log — id, user_id, step_name, status, timestamp
"""

from __future__ import annotations

import sqlite3

import numpy as np

from forgetgraph import DB_PATH, N_USERS, SEED


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return a new sqlite3 connection with row_factory=sqlite3.Row."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """Create users / features / predictions / tombstones / deletion_log."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            age REAL NOT NULL,
            account_balance REAL NOT NULL,
            num_transactions REAL NOT NULL,
            tenure_days REAL NOT NULL,
            risk_score REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL UNIQUE,
            age_norm REAL NOT NULL,
            balance_log REAL NOT NULL,
            tx_per_tenure REAL NOT NULL,
            balance_per_tx REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL UNIQUE,
            predicted_value REAL NOT NULL,
            cached_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tombstones (
            user_id INTEGER PRIMARY KEY,
            deleted_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS deletion_log (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            step_name TEXT NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
    )
    conn.commit()


def generate_synthetic_users(n: int = N_USERS, seed: int = SEED) -> list[dict]:
    """Generate n synthetic user dicts with numpy (fixed seed).

    Features: age, account_balance, num_transactions, tenure_days.
    Target:   risk_score (continuous), a noisy linear function of the features.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 81, size=n).astype(float)
    account_balance = rng.lognormal(mean=8.0, sigma=1.0, size=n)
    num_transactions = rng.integers(1, 501, size=n).astype(float)
    tenure_days = rng.integers(30, 3651, size=n).astype(float)

    risk_score = (
        0.02 * age
        + 0.0005 * account_balance
        + 0.1 * num_transactions
        + 0.002 * tenure_days
        + rng.normal(0.0, 2.0, size=n)
    )

    users = []
    for i in range(n):
        users.append(
            {
                "id": i + 1,
                "age": float(age[i]),
                "account_balance": float(account_balance[i]),
                "num_transactions": float(num_transactions[i]),
                "tenure_days": float(tenure_days[i]),
                "risk_score": float(risk_score[i]),
            }
        )
    return users


def seed_database(db_path: str = DB_PATH, n: int = N_USERS, seed: int = SEED) -> None:
    """Create tables and insert the synthetic users (idempotent reset)."""
    import os

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = get_connection(db_path)
    try:
        create_tables(conn)
        users = generate_synthetic_users(n=n, seed=seed)
        conn.executemany(
            "INSERT INTO users (id, age, account_balance, num_transactions, "
            "tenure_days, risk_score) VALUES (:id, :age, :account_balance, "
            ":num_transactions, :tenure_days, :risk_score)",
            users,
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    seed_database()
    print(f"Seeded {N_USERS} users into {DB_PATH}")
