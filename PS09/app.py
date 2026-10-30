"""FORGETGRAPH Streamlit dashboard — premium glass + aurora dark theme.

Styling/layout layer only: delete_user / ingest_user / verification /
deletion_log queries are unchanged. Every displayed number comes from the
live database, the model file, or package constants — zero fabrication.

Fonts are embedded locally (PS09/assets/fonts/) so the demo works offline.
"""

from __future__ import annotations

import base64
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from forgetgraph import ATOL, DB_PATH, N_USERS, RIDGE_ALPHA, RTOL, SEED
from forgetgraph.pipeline_core import (
    FEATURE_NAMES,
    PIPELINE_STEPS,
    delete_user,
    get_connection,
    ingest_user,
)
from forgetgraph.schema import create_tables

MODEL_PATH = "original_model.pkl"
FONT_DIR = Path(__file__).parent / "assets" / "fonts"

BG = "#05070f"
SURFACE = "#0b0f1d"
TEXT = "#e8ebf4"
MUTED = "#8b93a7"
GREEN = "#4ade80"
RED = "#f87171"
AMBER = "#fbbf24"
ACCENT = "#6c8cff"
VIOLET = "#a78bfa"

STEP_LABELS = {
    "source_removed": "Source Deleted",
    "features_recomputed": "Features Rebuilt",
    "cache_invalidated": "Cache Invalidated",
    "model_retrained": "Model Retrained",
    "verification_run": "Verification",
    "tombstone_recorded": "Tombstone Recorded",
}


def _font_b64(name: str) -> str:
    p = FONT_DIR / name
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return ""


def _font_face(family: str, weight: int, filename: str) -> str:
    b64 = _font_b64(filename)
    if not b64:
        return ""
    return (
        f"@font-face {{ font-family: '{family}'; font-weight: {weight}; "
        f"font-style: normal; font-display: swap; "
        f"src: url(data:font/woff2;base64,{b64}) format('woff2'); }}"
    )


FACES = (
    _font_face("Inter", 400, "inter-regular.woff2")
    + _font_face("Inter", 600, "inter-semibold.woff2")
    + _font_face("Inter", 700, "inter-bold.woff2")
    + _font_face("Space Grotesk", 700, "space-grotesk-bold.woff2")
)

CSS = f"""
<style>
{FACES}
  @keyframes fgFadeUp {{
      from {{ opacity: 0; transform: translateY(14px); }}
      to   {{ opacity: 1; transform: none; }}
  }}
  @keyframes fgPulse {{
      0%, 100% {{ box-shadow: 0 0 0 0 rgba(74,222,128,0.45); }}
      50%      {{ box-shadow: 0 0 0 12px rgba(74,222,128,0); }}
  }}
  @keyframes fgAurora {{
      0%, 100% {{ background-position: 0% 0%, 100% 100%; }}
      50%      {{ background-position: 20% 30%, 80% 70%; }}
  }}
  @keyframes fgSheen {{
      0%   {{ transform: translateX(-120%) skewX(-18deg); }}
      100% {{ transform: translateX(240%) skewX(-18deg); }}
  }}
  @keyframes fgSpin {{ to {{ transform: rotate(360deg); }} }}
  @keyframes fgGradient {{
      0%, 100% {{ background-position: 0% 50%; }}
      50%      {{ background-position: 100% 50%; }}
  }}

  /* ---------- base ---------- */
  html, body, .stApp {{ background: {BG} !important; color: {TEXT}; }}
  .stApp, section.stApp {{ background: {BG} !important; }}
  header {{ visibility: hidden; height: 0 !important; }}
  .block-container, section.main .block-container {{
      padding-top: 1.2rem !important; max-width: 1200px !important;
      margin: 0 auto;
  }}
  [class*="css"], html, body {{ font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; }}
  h1, h2, h3, .fg-h4 {{ font-family: 'Space Grotesk', 'Inter', sans-serif; }}

  /* ---------- aurora background ---------- */
  .stApp::before, .stApp::after {{
      content: ""; position: fixed; z-index: 0; pointer-events: none;
      width: 70vw; height: 70vw; border-radius: 50%;
      filter: blur(90px); opacity: 1;
  }}
  .stApp::before {{
      background: radial-gradient(circle, rgba(108,140,255,0.15), transparent 65%);
      top: -25vw; left: -15vw;
  }}
  .stApp::after {{
      background: radial-gradient(circle, rgba(74,222,128,0.10), transparent 65%);
      bottom: -30vw; right: -15vw;
  }}
  .stApp > div {{ position: relative; z-index: 1; }}

  /* ---------- sidebar glass ---------- */
  section[data-testid="stSidebar"] {{
      background: rgba(11,15,29,0.8) !important;
      backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      border-right: 1px solid rgba(255,255,255,0.06);
  }}
  section[data-testid="stSidebar"] label {{
      display: flex; align-items: center; padding: 0.5rem 0.75rem;
      border-radius: 10px; color: {MUTED}; font-weight: 600;
      cursor: pointer; transition: all 0.2s ease;
  }}
  section[data-testid="stSidebar"] label:hover {{
      background: rgba(108,140,255,0.10); transform: scale(1.02); color: {TEXT};
  }}
  section[data-testid="stSidebar"] label:has(input:checked) {{
      background: rgba(108,140,255,0.18); border: 1px solid {ACCENT}; color: #ffffff;
  }}

  /* ---------- glass primitives ---------- */
  .fg-card, .fg-kpi, .fg-node {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
      box-shadow: 0 4px 30px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
      animation: fgFadeUp 0.6s cubic-bezier(0.21,0.68,0.36,1) both;
      transition: all 0.25s ease;
  }}
  .fg-card {{ padding: 1.3rem 1.5rem; margin-bottom: 1.2rem; }}
  .fg-card:hover {{
      transform: translateY(-3px); border-color: rgba(108,140,255,0.45);
      box-shadow: 0 8px 30px rgba(0,0,0,0.4), 0 0 24px rgba(108,140,255,0.12),
                  inset 0 1px 0 rgba(255,255,255,0.06);
  }}
  .fg-kpi {{
      position: relative; overflow: hidden; padding: 1.1rem 1.2rem; height: 100%;
  }}
  .fg-kpi:hover {{
      transform: translateY(-3px); border-color: rgba(108,140,255,0.45);
      box-shadow: 0 8px 30px rgba(0,0,0,0.4), 0 0 24px rgba(108,140,255,0.12),
                  inset 0 1px 0 rgba(255,255,255,0.06);
  }}
  .fg-kpi::after {{
      content: ""; position: absolute; top: -20%; bottom: -20%; width: 40%;
      left: 0; background: linear-gradient(90deg, transparent,
      rgba(255,255,255,0.05), transparent);
      animation: fgSheen 1.2s ease-out both; pointer-events: none;
  }}
  .fg-kpi:nth-of-type(2)::after {{ animation-delay: 0.12s; }}
  .fg-kpi:nth-of-type(3)::after {{ animation-delay: 0.24s; }}
  .fg-kpi:nth-of-type(4)::after {{ animation-delay: 0.36s; }}
  .fg-kpi:nth-of-type(5)::after {{ animation-delay: 0.48s; }}
  .fg-kpi:nth-of-type(1) {{ animation-delay: 0s; }}
  .fg-kpi:nth-of-type(2) {{ animation-delay: 0.06s; }}
  .fg-kpi:nth-of-type(3) {{ animation-delay: 0.12s; }}
  .fg-kpi:nth-of-type(4) {{ animation-delay: 0.18s; }}
  .fg-kpi:nth-of-type(5) {{ animation-delay: 0.24s; }}
  .fg-kpi-label {{ color: {MUTED}; font-size: 0.72rem; text-transform: uppercase;
                   letter-spacing: 0.1em; }}
  .fg-kpi-value {{ color: {TEXT}; font-size: 2.2rem; font-weight: 700;
                   font-family: 'Space Grotesk', 'Inter', sans-serif;
                   line-height: 1.15; margin: 0.15rem 0; }}
  .fg-kpi-sub   {{ color: {MUTED}; font-size: 0.72rem; }}
  .fg-kpi-bar   {{ height: 3px; border-radius: 2px; margin-top: 0.7rem;
                   background: rgba(255,255,255,0.06); overflow: hidden; }}
  .fg-kpi-bar > div {{ height: 100%; border-radius: 2px; }}

  /* ---------- hero ---------- */
  .fg-hero {{
      position: relative; overflow: hidden;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08); border-radius: 20px;
      backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
      box-shadow: 0 4px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
      padding: 2rem 2.2rem; margin-bottom: 1.6rem;
      display: flex; align-items: center; gap: 1.5rem;
      animation: fgFadeUp 0.6s cubic-bezier(0.21,0.68,0.36,1) both;
  }}
  .fg-hero-word {{
      font-family: 'Space Grotesk', 'Inter', sans-serif;
      font-size: 2.6rem; font-weight: 700; letter-spacing: 0.01em;
      background: linear-gradient(90deg, {ACCENT}, {GREEN}, {VIOLET}, {ACCENT});
      background-size: 300% 100%;
      -webkit-background-clip: text; background-clip: text;
      color: transparent; line-height: 1.05;
      animation: fgGradient 8s ease infinite;
  }}
  .fg-hero-sub {{ color: {MUTED}; font-size: 0.95rem; margin-top: 0.35rem; }}
  .fg-hero-right {{ margin-left: auto; text-align: right; }}
  .fg-health {{
      display: inline-flex; align-items: center; gap: 0.5rem;
      background: rgba(74,222,128,0.10); border: 1px solid {GREEN};
      color: {GREEN}; border-radius: 999px; padding: 0.35rem 1rem;
      font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
  }}
  .fg-health-dot {{
      width: 0.55rem; height: 0.55rem; border-radius: 50%; background: {GREEN};
      animation: fgPulse 2s ease-in-out infinite;
  }}
  .fg-hero-stats {{ color: {MUTED}; font-size: 0.8rem; margin-top: 0.55rem; }}
  .fg-hero-motif {{
      position: absolute; right: 30%; top: 50%; transform: translateY(-50%);
      opacity: 0.05; pointer-events: none;
  }}

  /* ---------- overline + headings ---------- */
  .fg-overline {{
      color: {ACCENT}; font-size: 0.7rem; font-weight: 700;
      letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.25rem;
  }}
  .fg-h4 {{
      font-size: 1.15rem; font-weight: 700; color: {TEXT};
      margin: 0 0 0.8rem 0; letter-spacing: 0.02em;
      font-family: 'Space Grotesk', 'Inter', sans-serif;
  }}
  .fg-section {{ margin-top: 2.2rem; }}
  .fg-sub {{ color: {MUTED}; font-size: 0.88rem; }}

  /* ---------- badges ---------- */
  .fg-badge {{
      display: inline-block; padding: 0.28rem 0.95rem; border-radius: 999px;
      font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em;
      border: 1px solid; transition: all 0.2s ease;
  }}
  .fg-badge.pass  {{ background: rgba(74,222,128,0.22); color: #b9f6d4; border-color: {GREEN};
                     animation: fgPulse 2s ease-in-out infinite; }}
  .fg-badge.fail  {{ background: rgba(248,113,113,0.22); color: #ffd4d4; border-color: {RED}; }}
  .fg-badge.block {{ background: rgba(248,113,113,0.22); color: #ffd4d4; border-color: {RED}; }}
  .fg-badge.ok    {{ background: rgba(74,222,128,0.22); color: #b9f6d4; border-color: {GREEN}; }}
  .fg-badge.info  {{ background: rgba(251,191,36,0.22); color: #ffe9b8; border-color: {AMBER}; }}

  /* ---------- table ---------- */
  .fg-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  .fg-table th {{
      text-align: left; color: {MUTED}; font-size: 0.7rem; text-transform: uppercase;
      letter-spacing: 0.1em; padding: 0.55rem 0.7rem; border-bottom: 1px solid rgba(255,255,255,0.08);
  }}
  .fg-table td {{ padding: 0.6rem 0.7rem; border-bottom: 1px solid rgba(255,255,255,0.04);
                  transition: background 0.2s ease; }}
  .fg-table tr:hover td {{ background: rgba(108,140,255,0.06); }}

  /* ---------- dependency nodes ---------- */
  .fg-node {{ padding: 0.95rem 1.05rem; text-align: center; flex: 1; }}
  .fg-node-title {{ font-weight: 700; font-size: 0.9rem; color: {TEXT}; }}
  .fg-node-count {{ font-size: 1.6rem; font-weight: 700; margin: 0.15rem 0;
                    font-family: 'Space Grotesk', 'Inter', sans-serif; }}
  .fg-node-sub   {{ color: {MUTED}; font-size: 0.7rem; }}
  .fg-arrow {{ display: flex; align-items: center; justify-content: center;
               color: {ACCENT}; font-size: 1.3rem; padding: 0 0.35rem; }}

  /* ---------- timeline ---------- */
  .fg-step {{ display: flex; flex-direction: column; align-items: center;
              text-align: center; flex: 1; min-width: 0;
              animation: fgFadeUp 0.6s cubic-bezier(0.21,0.68,0.36,1) both; }}
  .fg-step:nth-of-type(2) {{ animation-delay: 0.07s; }}
  .fg-step:nth-of-type(3) {{ animation-delay: 0.14s; }}
  .fg-step:nth-of-type(4) {{ animation-delay: 0.21s; }}
  .fg-step:nth-of-type(5) {{ animation-delay: 0.28s; }}
  .fg-step:nth-of-type(6) {{ animation-delay: 0.35s; }}
  .fg-dot {{
      width: 2.7rem; height: 2.7rem; border-radius: 50%; display: flex;
      align-items: center; justify-content: center; font-size: 1.1rem;
      margin-bottom: 0.45rem; border: 2px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.04); z-index: 2;
      backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
  }}
  .fg-dot.done {{ border-color: {GREEN}; background: rgba(74,222,128,0.14); }}
  .fg-dot.run  {{ border-color: {AMBER}; background: rgba(251,191,36,0.14);
                  border-style: dashed; animation: fgSpin 2.5s linear infinite; }}
  .fg-dot.pass {{ border-color: {GREEN}; background: rgba(74,222,128,0.18);
                  animation: fgPulse 2s ease-in-out infinite; }}
  .fg-step-label {{ font-size: 0.76rem; font-weight: 600; color: {TEXT}; }}
  .fg-step-sub   {{ font-size: 0.66rem; color: {MUTED}; }}

  /* ---------- buttons ---------- */
  .stButton > button {{
      background: rgba(255,255,255,0.04); color: {TEXT};
      border: 1px solid rgba(255,255,255,0.10); border-radius: 12px;
      font-weight: 600; transition: all 0.2s ease;
  }}
  .stButton > button:hover, .stButton > button[kind="primary"] {{
      background: {ACCENT}; color: #fff; border-color: {ACCENT};
      box-shadow: 0 0 18px rgba(108,140,255,0.25);
  }}

  @media (prefers-reduced-motion: reduce) {{
      *, *::before, *::after {{
          animation: none !important; transition: none !important;
      }}
  }}
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


def overline(num: str, label: str) -> str:
    return f'<div class="fg-overline">{num} — {label}</div>'


KPI_ACCENTS = [ACCENT, RED, AMBER, VIOLET, GREEN]


def kpi(label: str, value: str, sub: str, icon: str, accent: str, ratio: float | None) -> str:
    bar = ""
    if ratio is not None:
        pct = max(0.0, min(1.0, ratio)) * 100
        bar = f'<div class="fg-kpi-bar"><div style="width:{pct:.1f}%;background:{accent};"></div></div>'
    return (
        f'<div class="fg-kpi"><div class="fg-kpi-label">{icon}&nbsp; {label}</div>'
        f'<div class="fg-kpi-value" style="color:{accent};">{value}</div>'
        f'<div class="fg-kpi-sub">{sub}</div>{bar}</div>'
    )


# ---------------------------------------------------------------- sections

def hero() -> None:
    total = db_count("SELECT COUNT(*) FROM users")
    deleted = db_count("SELECT COUNT(*) FROM tombstones")
    verif_pass = db_count(
        "SELECT COUNT(*) FROM deletion_log WHERE step_name = 'verification_run' AND status = 'pass'"
    )
    verif_total = db_count("SELECT COUNT(*) FROM deletion_log WHERE step_name = 'verification_run'")
    pct = (verif_pass / verif_total * 100) if verif_total else 100.0
    motif = (
        '<svg class="fg-hero-motif" width="260" height="150" viewBox="0 0 260 150">'
        '<g fill="none" stroke="#ffffff" stroke-width="6">'
        '<rect x="10" y="55" width="55" height="40" rx="8"/>'
        '<rect x="100" y="55" width="55" height="40" rx="8"/>'
        '<rect x="190" y="15" width="55" height="40" rx="8"/>'
        '<rect x="190" y="95" width="55" height="40" rx="8"/>'
        '<path d="M65 75 H100"/><path d="M155 75 L190 35"/>'
        '<path d="M155 75 L190 115"/>'
        '</g></svg>'
    )
    st.markdown(
        f"""
        <div class="fg-hero">
          {motif}
          <div>
            <div class="fg-hero-word">FORGETGRAPH</div>
            <div class="fg-hero-sub">Dependency-aware deletion verification ·
            GDPR-style right to be forgotten</div>
          </div>
          <div class="fg-hero-right">
            <span class="fg-health"><span class="fg-health-dot"></span>OPERATIONAL</span>
            <div class="fg-hero-stats">{total} users · 1 Ridge model · {pct:.0f}% verified deletions</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row() -> None:
    total = db_count("SELECT COUNT(*) FROM users")
    deleted = db_count("SELECT COUNT(*) FROM tombstones")
    blocked = db_count("SELECT COUNT(*) FROM deletion_log WHERE step_name = 'reingestion_blocked'")
    features = db_count("SELECT COUNT(*) FROM features")
    verif_pass = db_count(
        "SELECT COUNT(*) FROM deletion_log WHERE step_name = 'verification_run' AND status = 'pass'"
    )
    verif_total = db_count("SELECT COUNT(*) FROM deletion_log WHERE step_name = 'verification_run'")
    log_total = db_count("SELECT COUNT(*) FROM deletion_log")
    pct_verif = (verif_pass / verif_total) if verif_total else 1.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("Total Users", f"{total}", f"of {N_USERS} originally seeded", "👥",
                    KPI_ACCENTS[0], total / N_USERS if N_USERS else None), unsafe_allow_html=True)
    c2.markdown(kpi("Deleted Users", f"{deleted}", "tombstoned, idempotent re-delete", "🗑️",
                    KPI_ACCENTS[1], deleted / N_USERS if N_USERS else None), unsafe_allow_html=True)
    c3.markdown(kpi("Blocked Re-ingestions", f"{blocked}", "tombstone guard rejections", "⛔",
                    KPI_ACCENTS[2], blocked / log_total if log_total else 0.0), unsafe_allow_html=True)
    c4.markdown(kpi("Features In Store", f"{features}", "recomputed on each deletion", "⚙️",
                    KPI_ACCENTS[3], features / N_USERS if N_USERS else None), unsafe_allow_html=True)
    c5.markdown(kpi("Verifications Passed", f"{verif_pass}", "reference ≈ replacement", "✅",
                    KPI_ACCENTS[4], pct_verif), unsafe_allow_html=True)


def dependency_graph() -> None:
    st.markdown(overline("02", "LINEAGE"), unsafe_allow_html=True)
    st.markdown('<div class="fg-h4">Dependency Graph</div>', unsafe_allow_html=True)
    users_n = db_count("SELECT COUNT(*) FROM users")
    feats_n = db_count("SELECT COUNT(*) FROM features")
    preds_n = db_count("SELECT COUNT(*) FROM predictions")
    model = load_original_model()
    model_state = "trained" if model is not None else "not trained — run train_original.py"

    row = (
        f'<div style="display:flex;align-items:stretch;">'
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
        '<div class="fg-sub">A delete request propagates left → right: source row removed, '
        "features rebuilt without the user, cache invalidated, model retrained.</div>",
        unsafe_allow_html=True,
    )


def deletion_center() -> None:
    st.markdown(overline("03", "ORCHESTRATION"), unsafe_allow_html=True)
    st.markdown('<div class="fg-h4">Deletion Center</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Delete a user**")
        st.markdown(
            '<div class="fg-sub">Runs the 6-step pipeline; tombstoned ids are skipped.</div>',
            unsafe_allow_html=True,
        )
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

    with right:
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
                elif step == "verification_run" and r["status"] == "pass":
                    dot_cls, icon, sub = "pass", "✓", r["status"]
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


def coefficient_chart(original_model, replacement_model, ref_coef, ref_intercept) -> None:
    """Two-panel chart: features (left, auto-scaled) and intercept (right, own scale)."""
    feat_labels = list(FEATURE_NAMES)
    feat_orig = list(original_model.coef_)
    feat_repl = list(replacement_model.coef_)
    int_orig = float(original_model.intercept_)
    int_repl = float(replacement_model.intercept_)
    if ref_coef is not None and ref_intercept is not None:
        feat_ref = list(ref_coef)
        int_ref = float(ref_intercept)
    else:
        feat_ref = [None] * len(feat_labels)
        int_ref = None

    fig, (ax1, ax2) = plt.subplots(
        1, 2, width_ratios=[4, 1], figsize=(11, 4.2),
        gridspec_kw={"wspace": 0.35},
    )
    fig.patch.set_alpha(0.0)

    def style_axis(ax):
        ax.set_facecolor("none")
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color=(1, 1, 1), alpha=0.06, linewidth=0.6)
        ax.set_axisbelow(True)

    def label_bars(ax, pairs):
        """Plain text labels; skipped for near-zero bars to avoid collisions."""
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        pad = 0.02 * span
        threshold = 0.05 * span
        for b, v in pairs:
            if abs(v) < threshold:
                continue
            if v >= 0:
                ax.text(b.get_x() + b.get_width() / 2, v + pad, f"{v:.2g}",
                        ha="center", va="bottom", fontsize=8, color=MUTED)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v - pad, f"{v:.2g}",
                        ha="center", va="top", fontsize=8, color=MUTED)

    def draw_group(ax, x, triples, width=0.26):
        pairs = []
        for j, (name, vals, color) in enumerate(triples):
            if vals is None or any(v is None for v in vals):
                continue
            bars = ax.bar(x + (j - 1) * width, vals, width, label=name,
                          color=color, alpha=0.85, edgecolor="none", zorder=3)
            pairs.extend(zip(bars, vals))
        return pairs

    # left panel: features only
    xf = np.arange(len(feat_labels))
    pairs1 = draw_group(ax1, xf, [
        ("original", feat_orig, MUTED),
        ("replacement", feat_repl, ACCENT),
        ("reference", feat_ref, GREEN),
    ])
    ax1.set_xticks(xf)
    ax1.set_xticklabels(feat_labels, color=MUTED, fontsize=9)
    ax1.axhline(0, color=(1, 1, 1), alpha=0.12, linewidth=0.8, zorder=2)
    style_axis(ax1)
    ax1.set_title("feature coefficients", color=ACCENT, fontsize=10, pad=10)
    label_bars(ax1, pairs1)

    # right panel: intercept only
    triples_i = [
        ("original", [int_orig], MUTED),
        ("replacement", [int_repl], ACCENT),
    ]
    if int_ref is not None:
        triples_i.append(("reference", [int_ref], GREEN))
    pairs2 = draw_group(ax2, np.arange(1), triples_i)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["intercept"], color=MUTED, fontsize=9)
    style_axis(ax2)
    ax2.set_title("intercept", color=ACCENT, fontsize=10, pad=10)
    label_bars(ax2, pairs2)

    fig.suptitle("Ridge coefficients — original vs replacement vs reference",
                 color=ACCENT, fontsize=11, y=0.99)
    handles, labels_ = ax1.get_legend_handles_labels()
    legend = fig.legend(handles, labels_, loc="lower center", ncol=3,
                        facecolor=(0.043, 0.059, 0.114, 0.85),
                        edgecolor=(1, 1, 1, 0.12), labelcolor=TEXT, fontsize=9,
                        bbox_to_anchor=(0.5, -0.02), framealpha=0.85)
    legend.get_frame().set_boxstyle("round", rounding_size=8)
    legend.set_zorder(12)
    fig.subplots_adjust(top=0.85, bottom=0.16)
    st.pyplot(fig)
    plt.close(fig)


def model_comparison() -> None:
    st.markdown(overline("04", "EVIDENCE"), unsafe_allow_html=True)
    st.markdown('<div class="fg-h4">Model Comparison</div>', unsafe_allow_html=True)
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
        f'<div class="fg-sub" style="margin-top:0.5rem;">Tolerance: numpy.allclose(atol={ATOL}, rtol={RTOL}) · '
        f"reference model trained independently on retained rows.</div></div>",
        unsafe_allow_html=True,
    )
    coefficient_chart(original_model, replacement_model, ref_coef, ref_intercept)


def tombstones() -> None:
    st.markdown(overline("05", "REGISTRY"), unsafe_allow_html=True)
    st.markdown('<div class="fg-h4">Deleted Users (Tombstones)</div>', unsafe_allow_html=True)
    rows = db_rows("SELECT user_id, deleted_at FROM tombstones ORDER BY user_id")
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
    st.markdown(overline("06", "GUARD"), unsafe_allow_html=True)
    st.markdown('<div class="fg-h4">Re-ingestion Guard</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
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
        st.markdown("**Result**")
        res = st.session_state.get("reing_result")
        if res:
            st.markdown(badge(res[0], res[1]), unsafe_allow_html=True)
        else:
            st.markdown('<div class="fg-sub">No re-ingestion attempt yet in this session.</div>', unsafe_allow_html=True)


NAV = ["Overview", "Deletion Center", "Tombstones", "Re-ingestion Guard"]


def main() -> None:
    st.set_page_config(page_title="FORGETGRAPH", layout="wide")
    css()
    ensure_db()

    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:0.2rem 0 0.9rem 0.4rem;">
              <div style="font-family:'Space Grotesk','Inter',sans-serif;font-size:1.3rem;
                          font-weight:700;color:{TEXT};letter-spacing:0.02em;">FORGETGRAPH</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", NAV, label_visibility="collapsed")

    if page == "Overview":
        hero()
        kpi_row()
        dependency_graph()
        deletion_center()
        model_comparison()
    elif page == "Deletion Center":
        hero()
        deletion_center()
        model_comparison()
    elif page == "Tombstones":
        hero()
        tombstones()
    elif page == "Re-ingestion Guard":
        hero()
        reingestion()


if __name__ == "__main__":
    main()
