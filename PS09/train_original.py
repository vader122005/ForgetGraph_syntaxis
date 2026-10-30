"""One-off: train and save the original model (build order step 2).

Run after schema.py. Fits Ridge(alpha=1.0, random_state=42) on all 200 users
and saves it to original_model.pkl; also populates cached predictions and
prints the original coefficients for later before/after comparison.
"""

from __future__ import annotations

import pickle

from forgetgraph import DB_PATH
from forgetgraph.pipeline_core import (
    get_connection,
    load_training_data,
    populate_predictions,
    rebuild_features,
    train_model,
)
from forgetgraph.schema import seed_database

MODEL_PATH = "original_model.pkl"


def main(db_path: str = DB_PATH, reset: bool = True) -> None:
    if reset:
        seed_database(db_path)

    conn = get_connection(db_path)
    try:
        rebuild_features(conn)
        X, y, user_ids = load_training_data(conn)

        original_model = train_model(X, y)
        populate_predictions(conn, original_model)
    finally:
        conn.close()

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(original_model, f)

    print(f"Original model trained on {len(user_ids)} users")
    print(f"coef_     = {original_model.coef_}")
    print(f"intercept_ = {original_model.intercept_}")
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
