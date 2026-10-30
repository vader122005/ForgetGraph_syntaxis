"""FORGETGRAPH Streamlit dashboard — dark command-center theme.

Pages (all backed by real data — DB counts, model file, or package constants):
    Overview          — KPI cards + activity chart + operation donut + recent log
    Dependency Graph  — NetworkX graph, dark-styled, with propagation legend
    Deletion Center   — delete-user input + 6-step vertical stepper with pills
    Model Comparison  — before/after/reference coefficients + PASS/FAIL banner
    Tombstones        — deleted users card
    Re-ingestion      — simulator with dramatic blocked/accepted treatment
    Activity Log      — full deletion_log table
    Settings          — read-only info card (paths, constants, table counts)

Backend modules (pipeline_core, verify, schema, constants) are unchanged;
this is a UI-layer redesign only.
"""

from __future__ import annotations

import pickle
from collections import Counter
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from forgetgraph import (
    ATOL,
    DB_PATH,
    N_USERS,
    RIDGE_ALPHA,
    RTOL,
    SEED,
)
from forgetgraph.pipeline_core import (
    FEATURE_NAMES,
    PIPELINE_STEPS,
    delete_user,
    get_connection,
    ingest_user,
)
from forgetgraph.schema import create_tables

MODEL_PATH = "original_model.pkl"

NODE_USERS = "users\n(raw rows)"
NODE_FEATURES = "features\n(derived)"
NODE_PREDICTIONS = "predictions\n(cached)"
NODE_MODEL = "model\n(Ridge)"

BG = "#0F172A"
CARD = "#1E293B"
BORDER = "#334155"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
ACCENT = "#38BDF8"
GREEN = "#4ADE80"
RED = "#F87171"
AMBER = "#FBBF24"
VIOLET = "#A78BFA"

STATUS_COLORS = {
    "ok": GREEN,
    "pass": GREEN,
    "accepted": GREEN,
    "fail": RED,
    "rejected": RED,
    "rejected_duplicate_id": RED,
}

ACCENT_BY_STEP = {
    "source_removed": ACCENT,
    "features_recomputed": VIOLET,
    "cache_invalidated": AMBER,
    "model_retrained": GREEN,
    "verification_run": ACCENT,
    "tombstone_recorded": RED,
    "reingestion_blocked": RED,
    "reingestion_accepted": GREEN,
}

CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
  html, body, [class*="css"], .stApp {{
      font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
      background: {BG};
      color: {TEXT};
  }}
  .stApp {{ background: {BG}; }}
  section[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {BORDER}; }}
  h1, h2, h3 {{ color: {TEXT}; font-weight: 800; letter-spacing: -0.02em; }}
  .fg-sub {{ color: {MUTED}; font-size: 0.9rem; margin-top: -0.4rem; }}
  .fg-card {{
      background: {CARD};
      border: 1px solid {BORDER};
      border-radius: 14px;
      padding: 1.1rem 1.2rem;
      margin-bottom: 1rem;
  }}
  .fg-kpi {{
      background: {CARD};
      border: 1px solid {BORDER};
      border-top: 3px solid var(--kpi-accent, {ACCENT});
      border-radius: 14px;
      padding: 0.9rem 1rem 0.7rem 1rem;
      height: 100%;
  }}
  .fg-kpi-label {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }}
  .fg-kpi-value {{ color: {TEXT}; font-size: 1.7rem; font-weight: 800; }}
  .fg-kpi-delta {{ font-size: 0.75rem; font-weight: 600; }}
  .fg-chip {{
      display: inline-block; padding: 0.2rem 0.8rem; border-radius: 999px;
      font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
      background: rgba(74, 222, 128, 0.14); color: {GREEN};
      border: 1px solid {GREEN};
  }}
  .fg-chip.down {{ background: rgba(248, 113, 113, 0.14); color: {RED}; border-color: {RED}; }}
  .fg-banner-pass {{
      background: rgba(74, 222, 128, 0.12); border: 1px solid {GREEN};
      border-radius: 14px; padding: 1rem 1.2rem; text-align: center;
      font-size: 1.3rem; font-weight: 800; color: {GREEN}; margin-bottom: 1rem;
  }}
  .fg-banner-fail {{
      background: rgba(248, 113, 113, 0.12); border: 1px solid {RED};
      border-radius: 14px; padding: 1rem 1.2rem; text-align: center;
      font-size: 1.3rem; font-weight: 800; color: {RED}; margin-bottom: 1rem;
  }}
  .fg-banner-info {{
      background: rgba(56, 189, 248, 0.10); border: 1px solid {ACCENT};
      border-radius: 14px; padding: 1rem 1.2rem; text-align: center;
      font-size: 1.1rem; font-weight: 700; color: {ACCENT}; margin-bottom: 1rem;
  }}
  .fg-step {{ display: flex; align-items: center; gap: 0.6rem; padding: 0.35rem 0; }}
  .fg-step-index {{
      width: 1.6rem; height: 1.6rem; border-radius: 50%; flex: 0 0 auto;
      display: flex; align-items: center; justify-content: center;
      font-size: 0.75rem; font-weight: 800; color: {BG};
  }}
  .fg-step-name {{ font-weight: 600; font-size: 0.92rem; }}
  .fg-pill {{
      margin-left: auto; padding: 0.12rem 0.7rem; border-radius: 999px;
      font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
  }}
  .fg-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  .fg-table th {{
      text-align: left; color: {MUTED}; font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: 0.08em; padding: 0.5rem 0.6rem; border-bottom: 1px solid {BORDER};
  }}
  .fg-table td {{ padding: 0.5rem 0.6rem; border-bottom: 1px solid rgba(51,65,85,0.4); }}
  .fg-legend-item {{ display: flex; gap: 0.5rem; align-items: center; margin: 0.25rem 0; }}
  .fg-legend-swatch {{ width: 0.8rem; height: 0.8rem; border-radius: 4px; flex: 0 0 auto; }}
  div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; }}
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


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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


def system_health() -> str:
    try:
        db_count("SELECT COUNT(*) FROM users")
        model_ok = load_original_model() is not None
        return "OPERATIONAL" if model_ok else "DEGRADED (run train_original.py)"
    except Exception:
        return "DB UNREACHABLE"


def kpi_card(label: str, value: str, accent: str, delta: str | None = None) -> str:
    delta_html = (
        f'<div class="fg-kpi-delta" style="color:{MUTED}">{delta}</div>' if delta else ""
    )
    return (
        f'<div class="fg-kpi" style="--kpi-accent:{accent}">'
        f'<div class="fg-kpi-label">{label}</div>'
        f'<div class="fg-kpi-value">{value}</div>{delta_html}</div>'
    )


def status_pill(status: str) -> str:
    color = STATUS_COLORS.get(status, MUTED)
    return (
        f'<span class="fg-pill" style="background:rgba({_rgb(color)},0.14);'
        f'color:{color};border:1px solid {color}">{status.upper()}</span>'
    )


def _rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def step_color(step: str) -> str:
    return ACCENT_BY_STEP.get(step, ACCENT)


# ---------------------------------------------------------------- header

def render_header() -> None:
    health = system_health()
    chip = (
        f'<span class="fg-chip">● {health}</span>'
        if health == "OPERATIONAL"
        else f'<span class="fg-chip down">● {health}</span>'
    )
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.4rem;">
          <div>
            <h1 style="margin:0; font-size:1.9rem;">FORGETGRAPH</h1>
            <div class="fg-sub">Dependency-aware deletion verification · GDPR-style right-to-be-forgotten</div>
          </div>
          <div style="margin-left:auto; text-align:right;">
            <div class="fg-sub">{now_utc()}</div>
            <div style="margin-top:0.3rem;">{chip}</div>
          </div>
        </div>
        <hr style="border:none; border-top:1px solid {BORDER}; margin:0.6rem 0 1.2rem 0;">
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- pages

def page_overview() -> None:
    st.subheader("Overview")
    st.markdown('<div class="fg-sub">Live system state — every number sourced from the database or model file.</div>', unsafe_allow_html=True)

    total = db_count("SELECT COUNT(*) FROM users")
    deleted = db_count("SELECT COUNT(*) FROM tombstones")
    blocked = db_count("SELECT COUNT(*) FROM deletion_log WHERE step_name = 'reingestion_blocked'")
    verif_pass = db_count("SELECT COUNT(*) FROM deletion_log WHERE step_name = 'verification_run' AND status = 'pass'")
    verif_total = db_count("SELECT COUNT(*) FROM deletion_log WHERE step_name = 'verification_run'")
    active_model = f"Ridge α={RIDGE_ALPHA} (seed {SEED})"

    pct_deleted = (deleted / N_USERS * 100) if N_USERS else 0
    pct_pass = (verif_pass / verif_total * 100) if verif_total else 100.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi_card("Total Records", str(total), ACCENT, f"{N_USERS} originally seeded"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Deleted Users", str(deleted), RED, f"{pct_deleted:.1f}% of original"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Blocked Re-ingestions", str(blocked), AMBER, "tombstone guard rejections"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Active Model", active_model, VIOLET, "replacement in production"), unsafe_allow_html=True)
    c5.markdown(kpi_card("Verifications Passed", str(verif_pass), GREEN, f"{pct_pass:.0f}% pass rate"), unsafe_allow_html=True)

    st.markdown("### Data Lifecycle Activity")
    rows = db_rows("SELECT timestamp, step_name FROM deletion_log ORDER BY id")
    if rows:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["event"] = 1
        life = df.set_index("timestamp").resample("min")["event"].sum().fillna(0)
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.set_facecolor(CARD)
        fig.patch.set_facecolor(BG)
        ax.plot(life.index, life.values.cumsum(), color=ACCENT, linewidth=2)
        ax.fill_between(life.index, life.values.cumsum(), color=ACCENT, alpha=0.15)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.set_xlabel("")
        st.pyplot(fig)
    else:
        st.markdown('<div class="fg-card fg-sub">No pipeline activity yet — delete a user from the Deletion Center.</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Operation Status")
        counts = Counter(r["step_name"] for r in rows)
        if counts:
            fig, ax = plt.subplots(figsize=(4.6, 4.0))
            fig.patch.set_facecolor(BG)
            labels = list(counts.keys())
            values = list(counts.values())
            colors = [step_color(l) for l in labels]
            wedges, _ = ax.pie(values, colors=colors, startangle=90,
                              wedgeprops=dict(width=0.42, edgecolor=BG))
            ax.text(0, 0, str(sum(values)), ha="center", va="center",
                    fontsize=22, fontweight="bold", color=TEXT)
            ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                      frameon=False, labelcolor=MUTED, fontsize=8)
            st.pyplot(fig)
        else:
            st.markdown('<div class="fg-card fg-sub">No operations logged yet.</div>', unsafe_allow_html=True)
    with right:
        st.markdown("### Recent Activity")
        recent = db_rows("SELECT user_id, step_name, status, timestamp FROM deletion_log ORDER BY id DESC LIMIT 10")
        if recent:
            items = "".join(
                f'<div class="fg-legend-item">'
                f'<div class="fg-legend-swatch" style="background:{step_color(r["step_name"])}"></div>'
                f'<div><b>user {r["user_id"]}</b> · {r["step_name"]} {status_pill(r["status"])}'
                f'<div class="fg-sub" style="font-size:0.7rem">{r["timestamp"]}</div></div></div>'
                for r in reversed(recent)
            )
            st.markdown(f'<div class="fg-card">{items}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="fg-card fg-sub">Nothing logged yet.</div>', unsafe_allow_html=True)

    st.markdown("### Quick Actions")
    qa1, qa2, qa3 = st.columns(3)
    if qa1.button("⚡ Go to Deletion Center", use_container_width=True):
        st.session_state["_nav"] = "Deletion Center"
        st.rerun()
    if qa2.button("⚡ Go to Model Comparison", use_container_width=True):
        st.session_state["_nav"] = "Model Comparison"
        st.rerun()
    if qa3.button("⚡ Go to Re-ingestion Simulator", use_container_width=True):
        st.session_state["_nav"] = "Re-ingestion"
        st.rerun()


def page_dependency_graph() -> None:
    st.subheader("Dependency Graph")
    st.markdown('<div class="fg-sub">Deleting a user must propagate through every layer of the chain.</div>', unsafe_allow_html=True)

    G = nx.DiGraph()
    G.add_edge(NODE_USERS, NODE_FEATURES)
    G.add_edge(NODE_FEATURES, NODE_PREDICTIONS)
    G.add_edge(NODE_FEATURES, NODE_MODEL)

    pos = {
        NODE_USERS: (0, 0),
        NODE_FEATURES: (1, 0),
        NODE_PREDICTIONS: (2, 0.6),
        NODE_MODEL: (2, -0.6),
    }
    node_colors = {NODE_USERS: ACCENT, NODE_FEATURES: VIOLET, NODE_PREDICTIONS: AMBER, NODE_MODEL: GREEN}
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color=[node_colors[n] for n in G.nodes],
        edge_color=MUTED,
        node_size=9000,
        font_size=9,
        font_color=TEXT,
        font_weight="bold",
        arrows=True,
        arrowsize=25,
        node_shape="o",
        linewidths=2.0,
        edgecolors=TEXT,
    )
    ax.margins(0.2)
    st.pyplot(fig)

    legend = (
        f'<div class="fg-card">'
        f'<b>Propagation chain:</b> <code>users → features → predictions / model</code><br>'
        f'<span class="fg-sub">A delete request removes the raw row, recomputes derived features '
        f'without the user, invalidates cached predictions, and retrains the Ridge model — '
        f'leaving no residual influence at any layer.</span><br>'
        f'<div style="margin-top:0.5rem">'
        f'<div class="fg-legend-item"><div class="fg-legend-swatch" style="background:{ACCENT}"></div>raw rows (source of truth)</div>'
        f'<div class="fg-legend-item"><div class="fg-legend-swatch" style="background:{VIOLET}"></div>derived numeric columns</div>'
        f'<div class="fg-legend-item"><div class="fg-legend-swatch" style="background:{AMBER}"></div>cached model outputs</div>'
        f'<div class="fg-legend-item"><div class="fg-legend-swatch" style="background:{GREEN}"></div>trained Ridge regression</div>'
        f'</div></div>'
    )
    st.markdown(legend, unsafe_allow_html=True)


def page_deletion_center() -> None:
    st.subheader("Deletion Center")
    st.markdown('<div class="fg-sub">Run the 6-step deletion pipeline for a user. Tombstoned users are skipped (idempotent).</div>', unsafe_allow_html=True)

    user_id = st.number_input("user_id to delete", min_value=1, value=1, step=1)
    if st.button("🗑 Delete user", type="primary"):
        with st.spinner(f"Running deletion pipeline for user {user_id}..."):
            result = delete_user(int(user_id), db_path=DB_PATH)
        st.session_state["last_result"] = {
            "user_id": int(user_id),
            "result": {k: v for k, v in result.items() if k != "replacement_model"},
        }
        st.session_state["replacement_model"] = result["replacement_model"]
        if result["status"] == "already_deleted":
            st.markdown(f'<div class="fg-banner-info">USER {user_id} ALREADY TOMBSTONED — nothing to do (idempotent).</div>', unsafe_allow_html=True)
        elif result["status"] == "user_not_found":
            st.markdown(f'<div class="fg-banner-fail">USER {user_id} NOT FOUND in users table.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="fg-banner-pass">USER {user_id} DELETED — pipeline completed, 6 steps logged.</div>', unsafe_allow_html=True)

    last = st.session_state.get("last_result")
    if not last:
        st.markdown('<div class="fg-card fg-sub">No deletion run yet in this session.</div>', unsafe_allow_html=True)
        return

    log_rows = db_rows(
        "SELECT step_name, status FROM deletion_log WHERE user_id = ? ORDER BY id",
        (last["user_id"],),
    )
    pipeline_rows = [r for r in log_rows if r["step_name"] in PIPELINE_STEPS]
    steps_html = ""
    for i, r in enumerate(pipeline_rows, 1):
        color = STATUS_COLORS.get(r["status"], MUTED)
        steps_html += (
            f'<div class="fg-step">'
            f'<div class="fg-step-index" style="background:{color}">{i}</div>'
            f'<div class="fg-step-name">{r["step_name"]}</div>'
            f'{status_pill(r["status"])}'
            f'</div>'
        )
    st.markdown(
        f'<div class="fg-card"><b>Pipeline progress — user {last["user_id"]}</b>'
        f'<div style="margin-top:0.5rem">{steps_html}</div></div>',
        unsafe_allow_html=True,
    )


def page_model_comparison() -> None:
    st.subheader("Model Comparison")
    st.markdown('<div class="fg-sub">Original vs replacement vs independent reference coefficients.</div>', unsafe_allow_html=True)

    original_model = load_original_model()
    if original_model is None:
        st.markdown('<div class="fg-card fg-sub">Run <code>python train_original.py</code> to produce original_model.pkl.</div>', unsafe_allow_html=True)
        return

    last = st.session_state.get("last_result")
    replacement_model = st.session_state.get("replacement_model")
    if not last or replacement_model is None:
        st.markdown('<div class="fg-card fg-sub">Delete a user in the Deletion Center to compare models.</div>', unsafe_allow_html=True)
        return

    verification = last["result"].get("verification") or {}
    ref_coef = verification.get("reference_coef")
    ref_intercept = verification.get("reference_intercept")

    if verification.get("passed"):
        st.markdown('<div class="fg-banner-pass">✓ VERIFICATION PASS — reference model matches replacement model (atol=1e-6, rtol=1e-5)</div>', unsafe_allow_html=True)
    elif verification:
        st.markdown('<div class="fg-banner-fail">✗ VERIFICATION FAIL — reference model does not match replacement model.</div>', unsafe_allow_html=True)

    header = (
        '<tr><th>feature</th><th>original</th><th>replacement</th><th>reference</th></tr>'
    )
    body = ""
    for i, name in enumerate(FEATURE_NAMES):
        ref_val = f"{ref_coef[i]:.6f}" if ref_coef is not None else "—"
        body += (
            f'<tr><td>{name}</td><td>{original_model.coef_[i]:.6f}</td>'
            f'<td>{replacement_model.coef_[i]:.6f}</td><td>{ref_val}</td></tr>'
        )
    ref_int = f"{ref_intercept:.6f}" if ref_intercept is not None else "—"
    body += (
        f'<tr><td><b>intercept</b></td><td>{original_model.intercept_:.6f}</td>'
        f'<td>{replacement_model.intercept_:.6f}</td><td>{ref_int}</td></tr>'
    )
    st.markdown(f'<div class="fg-card"><table class="fg-table">{header}{body}</table></div>', unsafe_allow_html=True)


def page_tombstones() -> None:
    st.subheader("Tombstones")
    st.markdown('<div class="fg-sub">Every deleted user_id is permanently recorded — re-ingestion of these ids is blocked.</div>', unsafe_allow_html=True)
    rows = db_rows("SELECT user_id, deleted_at FROM tombstones ORDER BY user_id")
    if not rows:
        st.markdown('<div class="fg-card fg-sub">No users deleted yet.</div>', unsafe_allow_html=True)
        return
    body = "".join(f'<tr><td>{r["user_id"]}</td><td>{r["deleted_at"]}</td></tr>' for r in rows)
    st.markdown(
        f'<div class="fg-card"><table class="fg-table">'
        f'<tr><th>user_id</th><th>deleted_at</th></tr>{body}</table></div>',
        unsafe_allow_html=True,
    )


def page_reingestion() -> None:
    st.subheader("Re-ingestion Simulator")
    st.markdown('<div class="fg-sub">Attempt to insert a user_id back into <code>users</code>. Tombstoned ids are rejected by the guard; fresh ids are accepted.</div>', unsafe_allow_html=True)

    user_id = st.number_input("user_id to ingest", min_value=1, value=1, step=1, key="reingest")
    if st.button("Attempt insert", type="primary"):
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
            st.markdown(
                f'<div class="fg-banner-pass">✓ ACCEPTED — user {user_id} inserted (fresh id, no tombstone).</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="fg-banner-fail">⛔ BLOCKED — user {user_id} is tombstoned or a duplicate. '
                f"The insert was rejected and the rejection logged in deletion_log.</div>",
                unsafe_allow_html=True,
            )


def page_activity_log() -> None:
    st.subheader("Activity Log")
    st.markdown('<div class="fg-sub">Full deletion_log — every pipeline step, verification, and re-ingestion attempt.</div>', unsafe_allow_html=True)
    rows = db_rows("SELECT id, user_id, step_name, status, timestamp FROM deletion_log ORDER BY id")
    if not rows:
        st.markdown('<div class="fg-card fg-sub">No activity logged yet.</div>', unsafe_allow_html=True)
        return
    body = ""
    for r in rows:
        body += (
            f'<tr><td>{r["id"]}</td><td>{r["user_id"]}</td>'
            f'<td><span style="color:{step_color(r["step_name"])}">●</span> {r["step_name"]}</td>'
            f'<td>{status_pill(r["status"])}</td><td class="fg-sub">{r["timestamp"]}</td></tr>'
        )
    st.markdown(
        f'<div class="fg-card"><table class="fg-table">'
        f'<tr><th>#</th><th>user</th><th>step</th><th>status</th><th>timestamp</th></tr>'
        f'{body}</table></div>',
        unsafe_allow_html=True,
    )


def page_settings() -> None:
    st.subheader("Settings (read-only)")
    st.markdown('<div class="fg-sub">System configuration — sourced from package constants and live table counts.</div>', unsafe_allow_html=True)
    counts = {
        table: db_count(f"SELECT COUNT(*) FROM {table}")
        for table in ("users", "features", "predictions", "tombstones", "deletion_log")
    }
    rows = [
        ("Database path", DB_PATH),
        ("Model file", MODEL_PATH),
        ("Random seed", str(SEED)),
        ("Ridge alpha", str(RIDGE_ALPHA)),
        ("Verification tolerances", f"atol={ATOL}, rtol={RTOL}"),
        ("Table counts", ", ".join(f"{t}={c}" for t, c in counts.items())),
    ]
    body = "".join(f'<tr><td class="fg-sub">{k}</td><td><code>{v}</code></td></tr>' for k, v in rows)
    st.markdown(f'<div class="fg-card"><table class="fg-table">{body}</table></div>', unsafe_allow_html=True)


PAGES = {
    "Overview": page_overview,
    "Dependency Graph": page_dependency_graph,
    "Deletion Center": page_deletion_center,
    "Model Comparison": page_model_comparison,
    "Tombstones": page_tombstones,
    "Re-ingestion": page_reingestion,
    "Activity Log": page_activity_log,
    "Settings": page_settings,
}


def main() -> None:
    st.set_page_config(page_title="FORGETGRAPH", layout="wide")
    css()
    ensure_db()

    with st.sidebar:
        st.markdown(
            f'<div style="padding:0.4rem 0 0.8rem 0">'
            f'<div style="font-size:1.25rem; font-weight:800; color:{TEXT}">◈ FORGETGRAPH</div>'
            f'<div class="fg-sub" style="font-size:0.72rem">Deletion Verification Command Center</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        selected = option_menu(
            menu_title=None,
            options=list(PAGES.keys()),
            icons=["gauge", "diagram-3", "trash", "graph-up", "checkbox-x", "shield-x", "clock-history", "gear"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": CARD, "border-radius": "12px", "border": f"1px solid {BORDER}"},
                "icon": {"color": MUTED, "font-size": "0.9rem"},
                "nav-link": {"font-size": "0.88rem", "padding": "0.5rem 0.8rem", "--hover-color": "#fff"},
                "nav-link-selected": {"background-color": f"rgba(56,189,248,0.18)", "color": ACCENT, "font-weight": "600"},
            },
        )

    if "_nav" in st.session_state and st.session_state["_nav"] != selected:
        st.session_state["_nav"] = selected

    render_header()
    PAGES[selected]()


if __name__ == "__main__":
    main()
