"""FORGETGRAPH independent verification (separate code path).

Build order step 4. No shared in-memory state with the pipeline:

    - opens a FRESH sqlite3 connection (never receives one from the pipeline)
    - queries retained user_ids directly from users
    - recomputes features by calling pipeline_core.compute_features directly
      (the pure function only — no pipeline object or cached variable)
    - trains its own Ridge(alpha=1.0, random_state=42) reference model
    - compares against the replacement model with numpy.allclose
"""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from forgetgraph import ATOL, DB_PATH, RIDGE_ALPHA, RTOL, SEED
from forgetgraph.pipeline_core import FEATURE_NAMES, compute_features


def _fresh_connection(db_path: str) -> sqlite3.Connection:
    """Open a brand-new DB connection; never reuses a pipeline connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def run_verification(
    replacement_model,
    user_id: int,
    db_path: str = DB_PATH,
) -> dict[str, Any]:
    """Verify the deletion of user_id against a replacement model.

    Must NOT be called from inside delete_user with shared state; opens its own
    sqlite3 connection and trains its own reference model.

    Returns {
        "user_id": ...,
        "deleted_user_absent": bool,
        "coefficients_match": bool,
        "intercept_match": bool,
        "passed": bool,
        "reference_coef": np.ndarray,
        "reference_intercept": float,
        "retained_user_ids": list[int],
    }
    """
    conn = _fresh_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, age, account_balance, num_transactions, tenure_days, "
            "risk_score FROM users ORDER BY id"
        ).fetchall()

        retained_user_ids = [r["id"] for r in rows]
        deleted_user_absent = user_id not in retained_user_ids

        X = np.array(
            [list(compute_features(dict(r)).values()) for r in rows]
        )
        y = np.array([r["risk_score"] for r in rows])

        reference_model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
        reference_model.fit(X, y)

        coefficients_match = bool(
            np.allclose(
                replacement_model.coef_,
                reference_model.coef_,
                atol=ATOL,
                rtol=RTOL,
            )
        )
        intercept_match = bool(
            np.allclose(
                [replacement_model.intercept_],
                [reference_model.intercept_],
                atol=ATOL,
                rtol=RTOL,
            )
        )
    finally:
        conn.close()

    return {
        "user_id": user_id,
        "deleted_user_absent": deleted_user_absent,
        "coefficients_match": coefficients_match,
        "intercept_match": intercept_match,
        "passed": bool(deleted_user_absent and coefficients_match and intercept_match),
        "reference_coef": reference_model.coef_,
        "reference_intercept": float(reference_model.intercept_),
        "retained_user_ids": retained_user_ids,
    }
