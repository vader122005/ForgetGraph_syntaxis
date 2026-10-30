"""FORGETGRAPH — dependency-aware data deletion verification.

Modules:
    schema          — SQLite schema + synthetic data generation (seed=42)
    pipeline_core   — models, features, deletion pipeline, re-ingestion guard
    verify          — independent verification (separate code path)
    app             — Streamlit dashboard
    train_original  — one-off script to train/persist the original model
"""

__version__ = "0.1.0"

DB_PATH = "forgetgraph.db"
SEED = 42
N_USERS = 200
FEATURE_COLUMNS = [
    "age",
    "account_balance",
    "num_transactions",
    "tenure_days",
]
TARGET_COLUMN = "risk_score"
RIDGE_ALPHA = 1.0
ATOL = 1e-6
RTOL = 1e-5
