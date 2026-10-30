"""FORGETGRAPH core: models, features, deletion pipeline, re-ingestion guard.

Build order steps 2-3, 5:
    - train_model        — Ridge(alpha=1.0, random_state=42), used everywhere
    - compute_features   — derived numeric columns for one user row
    - rebuild_features   — recompute the features table from current users
    - retrain_model      — replacement model on remaining features
    - ingest_user        — re-ingestion guard (checks tombstones first)
    - delete_user        — the deletion pipeline (steps 1-6, ordered)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from forgetgraph import DB_PATH, RIDGE_ALPHA, SEED

FEATURE_NAMES = [
    "age_norm",
    "balance_log",
    "tx_per_tenure",
    "balance_per_tx",
]

PIPELINE_STEPS = [
    "source_removed",
    "features_recomputed",
    "cache_invalidated",
    "model_retrained",
    "verification_run",
    "tombstone_recorded",
]


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return a new sqlite3 connection with row_factory=sqlite3.Row."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def train_model(X, y) -> Ridge:
    """Fit and return Ridge(alpha=1.0, random_state=42) on X, y."""
    model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
    model.fit(X, y)
    return model


def compute_features(user_row: dict[str, Any]) -> dict[str, float]:
    """Derive feature columns from a single raw user row.

    Pure function — shared verbatim by the pipeline and the verifier.
    """
    age = float(user_row["age"])
    account_balance = float(user_row["account_balance"])
    num_transactions = float(user_row["num_transactions"])
    tenure_days = float(user_row["tenure_days"])

    return {
        "age_norm": age / 100.0,
        "balance_log": np.log1p(max(account_balance, 0.0)),
        "tx_per_tenure": num_transactions / max(tenure_days, 1.0),
        "balance_per_tx": account_balance / max(num_transactions, 1.0),
    }


def rebuild_features(conn: sqlite3.Connection) -> None:
    """Recompute the features table from all rows currently in users."""
    rows = conn.execute(
        "SELECT id, age, account_balance, num_transactions, tenure_days FROM users"
    ).fetchall()

    conn.execute("DELETE FROM features")
    for row in rows:
        feats = compute_features(dict(row))
        conn.execute(
            "INSERT INTO features (user_id, age_norm, balance_log, "
            "tx_per_tenure, balance_per_tx) VALUES (?, ?, ?, ?, ?)",
            (
                row["id"],
                feats["age_norm"],
                feats["balance_log"],
                feats["tx_per_tenure"],
                feats["balance_per_tx"],
            ),
        )
    conn.commit()


def load_training_data(conn: sqlite3.Connection):
    """Return (X, y, user_ids) built from the features + users tables."""
    rows = conn.execute(
        """
        SELECT u.id, u.risk_score, f.age_norm, f.balance_log,
               f.tx_per_tenure, f.balance_per_tx
        FROM features f JOIN users u ON u.id = f.user_id
        ORDER BY u.id
        """
    ).fetchall()

    X = np.array(
        [[r["age_norm"], r["balance_log"], r["tx_per_tenure"], r["balance_per_tx"]] for r in rows]
    )
    y = np.array([r["risk_score"] for r in rows])
    user_ids = [r["id"] for r in rows]
    return X, y, user_ids


def populate_predictions(conn: sqlite3.Connection, model: Ridge) -> None:
    """Cache model predictions for every user into the predictions table."""
    X, _, user_ids = load_training_data(conn)
    if len(user_ids) == 0:
        conn.execute("DELETE FROM predictions")
        conn.commit()
        return

    preds = model.predict(X)
    now = utc_now()
    conn.execute("DELETE FROM predictions")
    conn.executemany(
        "INSERT INTO predictions (user_id, predicted_value, cached_at) VALUES (?, ?, ?)",
        [(uid, float(p), now) for uid, p in zip(user_ids, preds)],
    )
    conn.commit()


def utc_now() -> str:
    """ISO-8601 UTC timestamp used across the tables."""
    return datetime.now(timezone.utc).isoformat()


def ingest_user(conn: sqlite3.Connection, user: dict[str, Any]) -> str:
    """Insert a new user row, guarded by the tombstones table.

    Returns "accepted" or "blocked"; blocks (and logs) any tombstoned user_id.
    """
    user_id = int(user["id"])

    tombstoned = conn.execute(
        "SELECT 1 FROM tombstones WHERE user_id = ?", (user_id,)
    ).fetchone()
    if tombstoned is not None:
        conn.execute(
            "INSERT INTO deletion_log (user_id, step_name, status, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (user_id, "reingestion_blocked", "rejected", utc_now()),
        )
        conn.commit()
        return "blocked"

    existing = conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
    if existing is not None:
        conn.execute(
            "INSERT INTO deletion_log (user_id, step_name, status, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (user_id, "reingestion_blocked", "rejected_duplicate_id", utc_now()),
        )
        conn.commit()
        return "blocked"

    conn.execute(
        "INSERT INTO users (id, age, account_balance, num_transactions, "
        "tenure_days, risk_score) VALUES (?, ?, ?, ?, ?, ?)",
        (
            user_id,
            float(user["age"]),
            float(user["account_balance"]),
            float(user["num_transactions"]),
            float(user["tenure_days"]),
            float(user["risk_score"]),
        ),
    )
    conn.execute(
        "INSERT INTO deletion_log (user_id, step_name, status, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (user_id, "reingestion_accepted", "accepted", utc_now()),
    )
    conn.commit()
    return "accepted"


def log_step(
    conn: sqlite3.Connection,
    user_id: int,
    step_name: str,
    status: str,
) -> None:
    """Append one row to deletion_log (step_name per spec)."""
    conn.execute(
        "INSERT INTO deletion_log (user_id, step_name, status, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (user_id, step_name, status, utc_now()),
    )
    conn.commit()


def retrain_model(conn: sqlite3.Connection) -> Ridge:
    """Retrain Ridge (alpha=1.0, random_state=42) on current features."""
    X, y, _ = load_training_data(conn)
    return train_model(X, y)


def delete_user(user_id: int, db_path: str = DB_PATH) -> dict[str, Any]:
    """Run the full deletion pipeline for user_id (idempotent).

    Steps in order:
        1. tombstone check (early return "already deleted")
        2. source_removed       — remove from users
        3. features_recomputed  — rebuild features excluding deleted user
        4. cache_invalidated    — delete predictions rows
        5. model_retrained      — replacement model (Ridge alpha=1.0, seed=42)
        6. log every step in deletion_log
        7. run independent verification (verify.run_verification, separate module)
        8. tombstone_recorded   — write user_id + timestamp to tombstones
    """
    from forgetgraph import verify

    conn = get_connection(db_path)
    try:
        tombstoned = conn.execute(
            "SELECT 1 FROM tombstones WHERE user_id = ?", (user_id,)
        ).fetchone()
        if tombstoned is not None:
            return {"status": "already_deleted", "replacement_model": None, "verification": None}

        existing = conn.execute(
            "SELECT 1 FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if existing is None:
            return {"status": "user_not_found", "replacement_model": None, "verification": None}

        conn.execute("DELETE FROM features WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        log_step(conn, user_id, "source_removed", "ok")

        rebuild_features(conn)
        log_step(conn, user_id, "features_recomputed", "ok")

        conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
        conn.commit()
        log_step(conn, user_id, "cache_invalidated", "ok")

        replacement_model = retrain_model(conn)
        log_step(conn, user_id, "model_retrained", "ok")

        verification = verify.run_verification(replacement_model, user_id, db_path=db_path)
        log_step(conn, user_id, "verification_run", "pass" if verification["passed"] else "fail")

        conn.execute(
            "INSERT INTO tombstones (user_id, deleted_at) VALUES (?, ?)",
            (user_id, utc_now()),
        )
        conn.commit()
        log_step(conn, user_id, "tombstone_recorded", "ok")

        return {
            "status": "deleted",
            "replacement_model": replacement_model,
            "verification": verification,
        }
    finally:
        conn.close()
