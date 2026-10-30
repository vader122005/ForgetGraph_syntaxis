"""FORGETGRAPH Streamlit dashboard — reference-style dark admin layout.

Sidebar nav controls which section renders. All numbers come from the live
database, the model file, or package constants — nothing is fabricated.

Rendering layer only: delete_user / ingest_user / verification / deletion_log
queries are unchanged from the original implementation.
"""

from __future__ import annotations

import pickle

import streamlit as st

from forgetgraph import DB_PATH, N_USERS, RIDGE_ALPHA, SEED
from forgetgraph.pipeline_core import (
    FEATURE_NAMES,
    PIPELINE_STEPS,
    delete_user,
    get_connection,
    ingest_user,
)
from forgetgraph.schema import create_tables

MODEL_PATH = "original_model.pkl"

BG = "#0a0e1a"
CARD = "#131824"
BORDER = "#232b3e"
TEXT = "#e6e9f0"
MUTED = "#8b93a7"
GREEN = "#4ade80"
RED = "#f87171"
AMBER = "#fbbf24"
ACCENT = "#6c8cff"

STEP_LABELS = {
    "source_removed": "Source Deleted",
    "features_recomputed": "Features Rebuilt",
    "cache_invalidated": "Cache Invalidated",
    "model_retrained": "Model Retrained",
    "verification_run": "Verification",
    "tombstone_recorded": "Tombstone Recorded",
}

CSS = f"""
<style>
  html, body, .stApp {{ background: {BG} !important; color: {TEXT}; }}
  .stApp, section.stApp {{ background: {BG} !important; }}
  html, body, [class*="css"] {{
      font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
  }}
  h1, h2, h3 {{ color: {TEXT}; font-weight: 700; }}
  .fg-sub {{ color: {MUTED}; font-size: 0.88rem; }}
  .stButton > button {{
      background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
      border-radius: 10px; font-weight: 600;
  }}
  .stButton > button[kind="primary"], .stButton > button:hover {{
      background: {ACCENT}; color: #fff; border-color: {ACCENT};
  }}
  section[data-testid="stSidebar"] {{
      background: #0d1220 !important; border-right: 1px solid {BORDER};
  }}
  .fg-card {{
      background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px;
      padding: 1.2rem 1.4rem; margin-bottom: 1rem;
  }}
  .fg-kpi {{
      background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px;
      padding: 1rem 1.1rem; height: 100%;
  }}
  .fg-kpi-label {{ color: {MUTED}; font-size: 0.75rem; text-transform: uppercase;
                   letter-spacing: 0.08em; }}
  .fg-kpi-value {{ color: {TEXT}; font-size: 1.9rem; font-weight: 800; line-height: 1.2; }}
  .fg-kpi-sub   {{ color: {MUTED}; font-size: 0.75rem; margin-top: 0.2rem; }}
  .fg-badge {{
      display: inline-block; padding: 0.25rem 0.9rem; border-radius: 999px;
      font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em;
  }}
  .fg-badge.pass  {{ background: rgba(74,222,128,0.15); color: {GREEN}; border: 1px solid {GREEN}; }}
  .fg-badge.fail  {{ background: rgba(248,113,113,0.15); color: {RED}; border: 1px solid {RED}; }}
  .fg-badge.block {{ background: rgba(248,113,113,0.15); color: {RED}; border: 1px solid {RED}; }}
  .fg-badge.ok    {{ background: rgba(74,222,128,0.15); color: {GREEN}; border: 1px solid {GREEN}; }}
  .fg-badge.info  {{ background: rgba(251,191,36,0.15); color: {AMBER}; border: 1px solid {AMBER}; }}
  .fg-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  .fg-table th {{
      text-align: left; color: {MUTED}; font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: 0.08em; padding: 0.55rem 0.7rem; border-bottom: 1px solid {BORDER};
  }}
  .fg-table td {{ padding: 0.55rem 0.7rem; border-bottom: 1px solid rgba(35,43,62,0.5); }}
  .fg-node {{
      background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
      padding: 0.9rem 1rem; text-align: center; flex: 1;
  }}
  .fg-node-title {{ font-weight: 700; font-size: 0.92rem; color: {TEXT}; }}
  .fg-node-count {{ font-size: 1.5rem; font-weight: 800; margin: 0.15rem 0; }}
  .fg-node-sub   {{ color: {MUTED}; font-size: 0.72rem; }}
  .fg-arrow {{ display: flex; align-items: center; justify-content: center;
               color: {MUTED}; font-size: 1.3rem; padding: 0 0.35rem; }}
  .fg-step {{ display: flex; flex-direction: column; align-items: center;
              text-align: center; flex: 1; min-width: 0; }}
  .fg-dot {{
      width: 2.6rem; height: 2.6rem; border-radius: 50%; display: flex;
      align-items: center; justify-content: center; font-size: 1.1rem;
      margin-bottom: 0.45rem; border: 2px solid {BORDER}; background: {CARD};
  }}
  .fg-dot.done {{ border-color: {GREEN}; background: rgba(74,222,128,0.12); }}
  .fg-dot.run  {{ border-color: {AMBER}; background: rgba(251,191,36,0.12); }}
  .fg-step-label {{ font-size: 0.78rem; font-weight: 600; color: {TEXT}; }}
  .fg-step-sub   {{ font-size: 0.68rem; color: {MUTED}; }}
  .fg-track {{ height: 2px; background: {BORDER}; margin: 1.3rem -1rem 0.5rem -1rem;
               border-radius: 2px; position: relative; top: -2.6rem; z-index: 0; }}
</style>
"""


def css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers

def db_count(sql: str, params: tuple = ()) -> int:
    conn = get_connection(DB_PATH)
    try:
        return conn.execute(sql, params).fetchone()[0]
    finally:
        conn.close()


def db_rows(sql: str, params: tuple = ()) -> list[dict]:
    conn = get_connection(DB_PATH)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


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


def badge(kind: str, text: str) -> str:
    return f'<span class="fg-badge {kind}">{text}</span>'


def kpi(label: str, value: str, sub: str, icon: str) -> str:
    return (
        f'<div class="fg-kpi"><div class="fg-kpi-label">{icon}&nbsp; {label}</div>'
        f'<div class="fg-kpi-value">{value}</div>'
        f'<div class="fg-kpi-sub">{sub}</div></div>'
    )


# ---------------------------------------------------------------- sections

def header() -> None:
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.15rem;">Data Lifecycle Orchestrator</h1>
        <div class="fg-sub">Dependency-aware deletion verification —
        {N_USERS} synthetic users · Ridge α={RIDGE_ALPHA} (seed {SEED})</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border:none;border-top:1px solid #232b3e;margin:1rem 0;">', unsafe_allow_html=True)


def kpi_row() -> None:
    total = db_count("SELECT COUNT(*) FROM users")
    deleted = db_count("SELECT COUNT(*) FROM tombstones")
    blocked = db_count("SELECT COUNT(*) FROM deletion_log WHERE step_name = 'reingestion_blocked'")
    features = db_count("SELECT COUNT(*) FROM features")
    verif_pass = db_count(
        "SELECT COUNT(*) FROM deletion_log WHERE step_name = 'verification_run' AND status = 'pass'"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("Total Users", f"{total}", f"of {N_USERS} originally seeded", "👥"), unsafe_allow_html=True)
    c2.markdown(kpi("Deleted Users", f"{deleted}", "tombstoned, idempotent re-delete", "🗑️"), unsafe_allow_html=True)
    c3.markdown(kpi("Blocked Re-ingestions", f"{blocked}", "tombstone guard rejections", "⛔"), unsafe_allow_html=True)
    c4.markdown(kpi("Features In Store", f"{features}", "recomputed on each deletion", "⚙️"), unsafe_allow_html=True)
    c5.markdown(kpi("Verifications Passed", f"{verif_pass}", "reference ≈ replacement", "✅"), unsafe_allow_html=True)


def dependency_graph() -> None:
    st.markdown("#### Dependency Graph")
    users_n = db_count("SELECT COUNT(*) FROM users")
    feats_n = db_count("SELECT COUNT(*) FROM features")
    preds_n = db_count("SELECT COUNT(*) FROM predictions")
    model = load_original_model()
    model_state = "trained" if model is not None else "not trained — run train_original.py"

    row = (
        f'<div style="display:flex;align-items:stretch;gap:0;">'
        f'<div class="fg-node"><div class="fg-node-title">Source Dataset</div>'
        f'<div class="fg-node-count" style="color:{ACCENT}">{users_n}</div>'
        f'<div class="fg-node-sub">users rows</div></div>'
        f'<div class="fg-arrow">→</div>'
        f'<div class="fg-node"><div class="fg-node-title">Feature Dataset</div>'
        f'<div class="fg-node-count" style="color:{GREEN}">{feats_n}</div>'
        f'<div class="fg-node-sub">derived features</div></div>'
        f'<div class="fg-arrow">→</div>'
        f'<div class="fg-node"><div class="fg-node-title">Ridge Model</div>'
        f'<div class="fg-node-count" style="color:{AMBER}">α={RIDGE_ALPHA}</div>'
        f'<div class="fg-node-sub">{model_state}</div></div>'
        f'</div>'
        f'<div style="display:flex;align-items:stretch;margin-top:0.6rem;">'
        f'<div style="flex:1;"></div>'
        f'<div class="fg-node" style="flex:0 0 30%;margin-left:auto;">'
        f'<div class="fg-node-title">Prediction Cache</div>'
        f'<div class="fg-node-count" style="color:{RED}">{preds_n}</div>'
        f'<div class="fg-node-sub">cached predictions</div></div>'
        f'</div>'
    )
    st.markdown(f'<div class="fg-card">{row}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="fg-sub">A delete request propagates left → right: source row removed, '
        f'features rebuilt without the user, cache invalidated, model retrained.</div>',
        unsafe_allow_html=True,
    )


def deletion_center() -> None:
    st.markdown("#### Deletion Center")
    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="fg-card" style="margin-bottom:0;">', unsafe_allow_html=True)
        st.markdown("**Delete a user**")
        st.markdown('<div class="fg-sub">Runs the 6-step pipeline; tombstoned ids are skipped.</div>', unsafe_allow_html=True)
        user_id = st.number_input("user_id", min_value=1, value=1, step=1, key="del_uid")
        if st.button("🗑 Delete user", type="primary", use_container_width=True):
            with st.spinner("Running deletion pipeline…"):
                result = delete_user(int(user_id), db_path=DB_PATH)
            st.session_state["last_result"] = {
                "user_id": int(user_id),
                "result": {k: v for k, v in result.items() if k != "replacement_model"},
            }
            st.session_state["replacement_model"] = result["replacement_model"]
            if result["status"] == "already_deleted":
                st.session_state["del_banner"] = ("info", f"User {user_id} already tombstoned — nothing to do (idempotent).")
            elif result["status"] == "user_not_found":
                st.session_state["del_banner"] = ("fail", f"User {user_id} not found in users table.")
            else:
                st.session_state["del_banner"] = ("ok", f"User {user_id} deleted — pipeline completed.")
        if "del_banner" in st.session_state:
            kind, text = st.session_state["del_banner"]
            st.markdown(badge(kind, text.upper()), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="fg-card" style="margin-bottom:0;">', unsafe_allow_html=True)
        st.markdown("**Deletion Timeline**")
        last = st.session_state.get("last_result")
        if not last:
            st.markdown(
                '<div class="fg-sub">No deletion run yet in this session. Delete a user to see the 6-step timeline.</div>',
                unsafe_allow_html=True,
            )
        else:
            rows = db_rows(
                "SELECT step_name, status, timestamp FROM deletion_log "
                "WHERE user_id = ? AND step_name IN (?, ?, ?, ?, ?, ?) ORDER BY id",
                (last["user_id"], *PIPELINE_STEPS),
            )
            by_step = {r["step_name"]: r for r in rows}
            cols = st.columns(6)
            for i, step in enumerate(PIPELINE_STEPS):
                r = by_step.get(step)
                if r is None:
                    dot_cls, icon, sub = "", "•", "pending"
                elif r["status"] in ("ok", "pass"):
                    dot_cls, icon, sub = "done", "✓", r["status"]
                else:
                    dot_cls, icon, sub = "run", "✗", r["status"]
                cols[i].markdown(
                    f'<div class="fg-step"><div class="fg-dot {dot_cls}">{icon}</div>'
                    f'<div class="fg-step-label">{STEP_LABELS[step]}</div>'
                    f'<div class="fg-step-sub">{sub}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


def model_comparison() -> None:
    st.markdown("#### Model Comparison")
    original_model = load_original_model()
    if original_model is None:
        st.markdown(
            f'<div class="fg-card"><div class="fg-sub">Run <code>python train_original.py</code> to produce {MODEL_PATH}.</div></div>',
            unsafe_allow_html=True,
        )
        return

    last = st.session_state.get("last_result")
    replacement_model = st.session_state.get("replacement_model")
    if not last or replacement_model is None:
        st.markdown(
            '<div class="fg-card"><div class="fg-sub">Delete a user to compare original / replacement / reference coefficients.</div></div>',
            unsafe_allow_html=True,
        )
        return

    verification = last["result"].get("verification") or {}
    ref_coef = verification.get("reference_coef")
    ref_intercept = verification.get("reference_intercept")

    if verification.get("passed"):
        verdict = badge("pass", "VERIFICATION PASS")
    elif verification:
        verdict = badge("fail", "VERIFICATION FAIL")
    else:
        verdict = ""

    header_cells = "<tr><th>feature</th><th>original</th><th>replacement</th><th>reference</th></tr>"
    body = ""
    for i, name in enumerate(FEATURE_NAMES):
        ref = f"{ref_coef[i]:.6f}" if ref_coef is not None else "—"
        body += (
            f"<tr><td>{name}</td><td>{original_model.coef_[i]:.6f}</td>"
            f"<td>{replacement_model.coef_[i]:.6f}</td><td>{ref}</td></tr>"
        )
    ref_i = f"{ref_intercept:.6f}" if ref_intercept is not None else "—"
    body += (
        f"<tr><td><b>intercept</b></td><td>{original_model.intercept_:.6f}</td>"
        f"<td>{replacement_model.intercept_:.6f}</td><td>{ref_i}</td></tr>"
    )
    st.markdown(
        f'<div class="fg-card"><div style="margin-bottom:0.7rem;">{verdict}</div>'
        f'<table class="fg-table">{header_cells}{body}</table>'
        f'<div class="fg-sub" style="margin-top:0.5rem;">Tolerance: numpy.allclose(atol=1e-6, rtol=1e-5) · '
        f"reference model trained independently on retained rows.</div></div>",
        unsafe_allow_html=True,
    )


def tombstones() -> None:
    st.markdown("#### Deleted Users (Tombstones)")
    rows = db_rows(
        "SELECT user_id, deleted_at FROM tombstones ORDER BY user_id"
    )
    if not rows:
        st.markdown(
            '<div class="fg-card"><div class="fg-sub">No users deleted yet.</div></div>',
            unsafe_allow_html=True,
        )
        return
    body = "".join(
        f"<tr><td>{r['user_id']}</td><td>{r['deleted_at']}</td>{badge('block', 'TOMBSTONED')}</tr>"
        for r in rows
    )
    st.markdown(
        f'<div class="fg-card"><table class="fg-table">'
        f"<tr><th>User ID</th><th>Deletion Date &amp; Time</th><th>Status</th></tr>"
        f"{body}</table></div>",
        unsafe_allow_html=True,
    )


def reingestion() -> None:
    st.markdown("#### Re-ingestion Guard")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="fg-card" style="margin-bottom:0;">', unsafe_allow_html=True)
        st.markdown("**Simulate re-ingestion**")
        st.markdown(
            '<div class="fg-sub">Attempt to insert a user_id back into <code>users</code>. '
            "Tombstoned ids are rejected; fresh ids pass.</div>",
            unsafe_allow_html=True,
        )
        uid = st.number_input("user_id to ingest", min_value=1, value=1, step=1, key="reing_uid")
        if st.button("Attempt insert", type="primary", use_container_width=True):
            user = {
                "id": int(uid),
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
                st.session_state["reing_result"] = ("ok", f"ACCEPTED — user {uid} inserted (fresh id).")
            else:
                st.session_state["reing_result"] = ("block", f"BLOCKED — user {uid} is tombstoned or duplicate; rejection logged.")
    with right:
        st.markdown('<div class="fg-card" style="margin-bottom:0;">', unsafe_allow_html=True)
        st.markdown("**Result**")
        res = st.session_state.get("reing_result")
        if res:
            st.markdown(badge(res[0], res[1]), unsafe_allow_html=True)
        else:
            st.markdown('<div class="fg-sub">No re-ingestion attempt yet in this session.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


NAV = ["Overview", "Dependency Graph", "Deletion Center", "Tombstones", "Re-ingestion Guard"]


def main() -> None:
    st.set_page_config(page_title="FORGETGRAPH", layout="wide")
    css()
    ensure_db()

    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:0.2rem 0 0.9rem 0.4rem;">
              <div style="font-size:1.3rem;font-weight:800;color:{TEXT};letter-spacing:0.02em;">FORGETGRAPH</div>
              <div style="color:{MUTED};font-size:0.75rem;">Data Lifecycle Orchestrator</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", NAV, label_visibility="collapsed")

    header()
    kpi_row()

    if page == "Overview":
        dependency_graph()
        deletion_center()
        model_comparison()
    elif page == "Dependency Graph":
        dependency_graph()
    elif page == "Deletion Center":
        deletion_center()
        model_comparison()
    elif page == "Tombstones":
        tombstones()
    elif page == "Re-ingestion Guard":
        reingestion()


if __name__ == "__main__":
    main()
