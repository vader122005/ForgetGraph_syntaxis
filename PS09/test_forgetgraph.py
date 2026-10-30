"""FORGETGRAPH pytest suite (build order step 7).

Tests:
    1. Deleting a user removes them from users, features, and predictions
    2. Replacement model coefficients differ from the original model's
    3. Reference model matches replacement model within tolerance
    4. Deleting an already-tombstoned user is idempotent
    5. Re-ingesting a tombstoned user_id is blocked
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from forgetgraph import ATOL, RTOL
from forgetgraph.pipeline_core import (
    delete_user,
    get_connection,
    ingest_user,
    populate_predictions,
    rebuild_features,
    train_model,
)
from forgetgraph.schema import seed_database


@pytest.fixture
def fresh_db(tmp_path):
    """Yield a fresh seeded DB path; original model trained on all 200 users."""
    db_path = str(tmp_path / "test_forgetgraph.db")
    seed_database(db_path)

    conn = get_connection(db_path)
    try:
        rebuild_features(conn)
        X = []
        y = []
        rows = conn.execute(
            "SELECT u.risk_score, f.age_norm, f.balance_log, f.tx_per_tenure, "
            "f.balance_per_tx FROM features f JOIN users u ON u.id = f.user_id "
            "ORDER BY u.id"
        ).fetchall()
        for r in rows:
            X.append([r["age_norm"], r["balance_log"], r["tx_per_tenure"], r["balance_per_tx"]])
            y.append(r["risk_score"])
        original_model = train_model(np.array(X), np.array(y))
        populate_predictions(conn, original_model)
    finally:
        conn.close()

    yield {"db_path": db_path, "original_model": original_model}

    if os.path.exists(db_path):
        os.remove(db_path)


def test_deletion_removes_user_from_all_tables(fresh_db):
    db_path = fresh_db["db_path"]
    user_id = 10

    result = delete_user(user_id, db_path=db_path)
    assert result["status"] == "deleted"

    conn = get_connection(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM users WHERE id = ?", (user_id,)).fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM features WHERE user_id = ?", (user_id,)).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM tombstones WHERE user_id = ?", (user_id,)).fetchone()[
                0
            ]
            == 1
        )
    finally:
        conn.close()


def test_replacement_model_differs_from_original(fresh_db):
    db_path = fresh_db["db_path"]
    original_model = fresh_db["original_model"]

    result = delete_user(25, db_path=db_path)
    assert result["status"] == "deleted"
    replacement_model = result["replacement_model"]

    assert not np.allclose(
        original_model.coef_,
        replacement_model.coef_,
        atol=ATOL,
        rtol=RTOL,
    )


def test_reference_matches_replacement_within_tolerance(fresh_db):
    db_path = fresh_db["db_path"]

    result = delete_user(50, db_path=db_path)
    assert result["status"] == "deleted"

    replacement_model = result["replacement_model"]
    verification = result["verification"]
    assert verification is not None
    assert verification["deleted_user_absent"] is True
    assert verification["coefficients_match"] is True
    assert verification["intercept_match"] is True
    assert verification["passed"] is True

    assert np.allclose(
        replacement_model.coef_,
        verification["reference_coef"],
        atol=ATOL,
        rtol=RTOL,
    )
    assert np.allclose(
        [replacement_model.intercept_],
        [verification["reference_intercept"]],
        atol=ATOL,
        rtol=RTOL,
    )


def test_delete_is_idempotent(fresh_db):
    db_path = fresh_db["db_path"]
    user_id = 75

    first = delete_user(user_id, db_path=db_path)
    assert first["status"] == "deleted"

    conn = get_connection(db_path)
    try:
        log_count_after_first = conn.execute(
            "SELECT COUNT(*) FROM deletion_log WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
    finally:
        conn.close()

    second = delete_user(user_id, db_path=db_path)
    assert second["status"] == "already_deleted"
    assert second["replacement_model"] is None
    assert second["verification"] is None

    conn = get_connection(db_path)
    try:
        log_count_after_second = conn.execute(
            "SELECT COUNT(*) FROM deletion_log WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        tombstone_count = conn.execute(
            "SELECT COUNT(*) FROM tombstones WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
    finally:
        conn.close()

    assert log_count_after_second == log_count_after_first
    assert tombstone_count == 1


def test_reingestion_of_tombstoned_user_is_blocked(fresh_db):
    db_path = fresh_db["db_path"]
    user_id = 100

    assert delete_user(user_id, db_path=db_path)["status"] == "deleted"

    user = {
        "id": user_id,
        "age": 40.0,
        "account_balance": 500.0,
        "num_transactions": 10.0,
        "tenure_days": 200.0,
        "risk_score": 5.0,
    }
    conn = get_connection(db_path)
    try:
        outcome = ingest_user(conn, user)
        assert outcome == "blocked"

        still_gone = conn.execute(
            "SELECT COUNT(*) FROM users WHERE id = ?", (user_id,)
        ).fetchone()[0]
        assert still_gone == 0

        rejection_logged = conn.execute(
            "SELECT COUNT(*) FROM deletion_log WHERE user_id = ? AND step_name = ?",
            (user_id, "reingestion_blocked"),
        ).fetchone()[0]
        assert rejection_logged >= 1

        fresh_user = dict(user, id=9999)
        assert ingest_user(conn, fresh_user) == "accepted"
    finally:
        conn.close()
