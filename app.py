"""FORGETGRAPH Streamlit dashboard (build order step 6).

Sections:
    1. Dependency graph visual (users -> features -> predictions -> model)
       via NetworkX + matplotlib
    2. "Delete a user" — user_id input + delete button
    3. Live step-by-step deletion progress from deletion_log
    4. Before/after model comparison — replacement vs reference coefficients
       side by side + pass/fail verification badge
    5. Tombstone list viewer (deleted user_ids + timestamps)
    6. "Simulate re-ingestion" — attempt insert, show accepted/blocked
"""

from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from forgetgraph import DB_PATH
from forgetgraph.pipeline_core import (
    FEATURE_NAMES,
    delete_user,
    get_connection,
    ingest_user,
)
from forgetgraph.schema import create_tables

MODEL_PATH = "original_model.pkl"

PIPELINE_STEPS = [
    "source_removed",
    "features_recomputed",
    "cache_invalidated",
    "model_retrained",
    "verification_run",
    "tombstone_recorded",
]


def ensure_db() -> None:
    conn = get_connection(DB_PATH)
    try:
        create_tables(conn)
    finally:
        conn.close()


def load_original_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def section_dependency_graph() -> None:
    st.subheader("1. Dependency Graph")
    st.markdown("Deleting a user must propagate through every layer:")

    G = nx.DiGraph()
    G.add_edge("users\n(raw rows)", "features\n(derived)")
    G.add_edge("features", "predictions\n(cached)")
    G.add_edge("features", "model\n(Ridge)")

    pos = {
        "users\n(raw rows)": (0, 0),
        "features\n(derived)": (1, 0),
        "predictions\n(cached)": (2, 0.6),
        "model\n(Ridge)": (2, -0.6),
    }
    fig, ax = plt.subplots(figsize=(8, 3))
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color="#7fc8a9",
        node_size=9000,
        font_size=9,
        arrows=True,
        arrowsize=25,
    )
    ax.margins(0.2)
    st.pyplot(fig)


def section_delete_user() -> None:
    st.subheader("2. Delete a User")
    user_id = st.number_input("user_id", min_value=1, value=1, step=1)
    if st.button("Delete user", type="primary"):
        with st.spinner(f"Running deletion pipeline for user {user_id}..."):
            result = delete_user(int(user_id), db_path=DB_PATH)
        st.session_state["last_result"] = {
            "user_id": int(user_id),
            "result": {
                k: v for k, v in result.items() if k != "replacement_model"
            },
        }
        st.session_state["replacement_model"] = result["replacement_model"]
        if result["status"] == "already_deleted":
            st.info(f"User {user_id} is already tombstoned — nothing to do (idempotent).")
        elif result["status"] == "user_not_found":
            st.warning(f"User {user_id} does not exist in users.")
        else:
            st.success(f"User {user_id} deleted and pipeline completed.")


def section_deletion_progress() -> None:
    st.subheader("3. Deletion Progress (deletion_log)")
    last = st.session_state.get("last_result")
    if not last:
        st.caption("Delete a user above to see step-by-step progress.")
        return
    user_id = last["user_id"]
    conn = get_connection(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT step_name, status, timestamp FROM deletion_log "
            "WHERE user_id = ? ORDER BY id",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        st.caption(f"No deletion_log rows for user {user_id}.")
        return
    df = pd.DataFrame([dict(r) for r in rows])
    st.table(df)
    completed = {r["step_name"] for r in rows}
    missing = [s for s in PIPELINE_STEPS if s not in completed]
    if missing:
        st.warning(f"Missing steps: {', '.join(missing)}")
    else:
        st.caption("All 6 pipeline steps logged for this user.")


def section_model_comparison() -> None:
    st.subheader("4. Before / After Model Comparison")
    original_model = load_original_model()
    last = st.session_state.get("last_result")
    replacement_model = st.session_state.get("replacement_model")

    if original_model is None:
        st.caption("Run `python train_original.py` to produce original_model.pkl.")
        return
    if not last or replacement_model is None:
        st.caption("Delete a user above to compare models.")
        return

    verification = last["result"].get("verification") or {}
    ref_coef = verification.get("reference_coef")
    ref_intercept = verification.get("reference_intercept")

    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        rows.append(
            {
                "feature": name,
                "original_coef": original_model.coef_[i],
                "replacement_coef": replacement_model.coef_[i],
                "reference_coef": ref_coef[i] if ref_coef is not None else None,
            }
        )
    rows.append(
        {
            "feature": "intercept",
            "original_coef": original_model.intercept_,
            "replacement_coef": replacement_model.intercept_,
            "reference_coef": ref_intercept,
        }
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if verification.get("passed"):
        st.success(
            "Verification: PASS — reference model matches replacement model "
            "within tolerance (atol=1e-6, rtol=1e-5)."
        )
    elif verification:
        st.error(
            "Verification: FAIL — reference model does not match replacement model."
        )
    else:
        st.caption("No verification result recorded for this deletion.")


def section_tombstones() -> None:
    st.subheader("5. Tombstones")
    conn = get_connection(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT user_id, deleted_at FROM tombstones ORDER BY user_id"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        st.caption("No users have been deleted yet.")
        return
    st.dataframe(pd.DataFrame([dict(r) for r in rows]), use_container_width=True)


def section_reingestion() -> None:
    st.subheader("6. Simulate Re-ingestion")
    st.markdown(
        "Attempt to insert a user_id back into `users`. Tombstoned ids are "
        "rejected; a genuinely new id is accepted."
    )
    user_id = st.number_input("user_id to ingest", min_value=1, value=1, step=1, key="reingest")
    if st.button("Attempt insert"):
        user = {
            "id": int(user_id),
            "age": 33.0,
            "account_balance": 1234.0,
            "num_transactions": 42.0,
            "tenure_days": 500.0,
            "risk_score": 10.0,
        }
        conn = get_connection(DB_PATH)
        try:
            outcome = ingest_user(conn, user)
        finally:
            conn.close()
        if outcome == "accepted":
            st.success(f"User {user_id} accepted (not tombstoned, fresh id).")
        else:
            st.error(f"User {user_id} blocked — tombstoned or duplicate id. Rejection logged.")


def main() -> None:
    st.set_page_config(page_title="FORGETGRAPH", layout="wide")
    st.title("FORGETGRAPH")
    st.caption("Dependency-aware data deletion verification (GDPR-style right to be forgotten)")

    ensure_db()
    section_dependency_graph()
    st.divider()
    section_delete_user()
    section_deletion_progress()
    st.divider()
    section_model_comparison()
    st.divider()
    section_tombstones()
    st.divider()
    section_reingestion()


if __name__ == "__main__":
    main()
