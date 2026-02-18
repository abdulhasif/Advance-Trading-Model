"""
src/ui/dashboard.py — Phase 5: Streamlit Mission Control
==========================================================
Dark-mode Bloomberg Terminal aesthetic with auto-refresh.

Run:  streamlit run src/ui/dashboard.py
"""

import json
import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

import config

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Institutional Fortress — Mission Control",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark Theme CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; color: #e0e0e0; }
    h1,h2,h3,h4,h5,h6 { color: #00ff88 !important; font-family: 'Courier New', monospace; }
    [data-testid="stMetricValue"] { color: #00ff88; font-family: 'Courier New', monospace; font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"] { color: #888; font-family: 'Courier New', monospace; }
    .yellow-alert { background: #ff8800; color: #000; padding: 8px 16px; border-radius: 4px;
                    font-weight: bold; font-size: 1.2rem; text-align: center;
                    animation: pulse 1s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
    .green-status { background: #00ff88; color: #000; padding: 8px 16px; border-radius: 4px;
                    font-weight: bold; font-size: 1.2rem; text-align: center; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


def load_state() -> dict | None:
    if not config.LIVE_STATE_FILE.exists():
        return None
    try:
        with open(config.LIVE_STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


st.markdown("# 🏰 INSTITUTIONAL FORTRESS — Mission Control")
st.markdown("---")

state = load_state()

if state is None:
    st.warning("⏳ Waiting for live_state.json — make sure live engine is running.")
    time.sleep(2); st.rerun()
else:
    top_signals  = state.get("top_signals", [])
    chart_bricks = state.get("chart_bricks", [])
    chart_symbol = state.get("chart_symbol", "—")
    health       = state.get("health", {})
    state_ts     = state.get("timestamp", "")

    # ── Leaderboard + Health ───────────────────────────────────────────────
    col_lead, col_hp = st.columns([3, 1])

    with col_lead:
        st.markdown("## 📊 LEADERBOARD — Top 3 Signals")
        if top_signals:
            for i, sig in enumerate(top_signals):
                d = sig.get("direction","—")
                c = "#00ff88" if d == "BUY" else "#ff4444"
                a = "▲" if d == "BUY" else "▼"
                v = " ⚠️VETOED" if sig.get("is_vetoed") else ""
                st.markdown(f"""
                <div style="background:#111;border-left:4px solid {c};padding:12px 16px;margin:6px 0;
                            border-radius:4px;font-family:'Courier New',monospace;">
                  <span style="color:{c};font-size:1.3rem;font-weight:bold;">
                    #{i+1} {sig.get('symbol','—')} {a} {d}</span>
                  <span style="color:#666;margin-left:20px;">
                    Score: <b style="color:#fff;">{sig.get('score',0):.1f}</b>
                    &nbsp;|&nbsp; P(up): {sig.get('brain1_prob',0):.2%}
                    &nbsp;|&nbsp; Conv: {sig.get('brain2_conviction',0):.1f}%
                    &nbsp;|&nbsp; Vel: {sig.get('velocity',0):.3f}
                    &nbsp;|&nbsp; RS: {sig.get('rs',0):.3f}
                    &nbsp;|&nbsp; ₹{sig.get('price',0):,.2f}{v}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No active signals. Waiting for brick formation ...")

    with col_hp:
        st.markdown("## 🩺 HEALTH")
        st.metric("Latency", f"{health.get('loop_latency_ms',0):.0f} ms")
        st.metric("Active", str(health.get("active_symbols",0)))
        drift = health.get("drift_accuracy")
        if drift is not None:
            st.metric("Accuracy", f"{drift*100:.1f}%", delta="⚠️ LOW" if drift < .5 else "✅ OK")
        else:
            st.metric("Accuracy", "N/A")
        if health.get("yellow_alert"):
            st.markdown('<div class="yellow-alert">🟡 YELLOW ALERT — STOP TRADING</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="green-status">🟢 SYSTEM NOMINAL</div>', unsafe_allow_html=True)
        st.caption(f"Updated: {state_ts}")

    # ── Renko Chart ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"## 📈 LIVE RENKO — {chart_symbol}")
    if chart_bricks:
        bdf = pd.DataFrame(chart_bricks)
        bdf["brick_timestamp"] = pd.to_datetime(bdf["brick_timestamp"])
        fig = go.Figure()
        for _, row in bdf.iterrows():
            c = "#00ff88" if row["direction"] > 0 else "#ff4444"
            fig.add_trace(go.Candlestick(
                x=[row["brick_timestamp"]],
                open=[row["brick_open"]], high=[row["brick_high"]],
                low=[row["brick_low"]], close=[row["brick_close"]],
                increasing=dict(line=dict(color=c), fillcolor=c),
                decreasing=dict(line=dict(color="#ff4444"), fillcolor="#ff4444"),
                showlegend=False))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            xaxis=dict(showgrid=False, zeroline=False, rangeslider=dict(visible=False)),
            yaxis=dict(showgrid=False, zeroline=False, title="Price (₹)", titlefont=dict(color="#00ff88")),
            margin=dict(l=50,r=20,t=30,b=40), height=500, font=dict(family="Courier New", color="#888"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No brick data for chart yet.")

    # ── Signal Table ───────────────────────────────────────────────────────
    if top_signals:
        st.markdown("---")
        st.markdown("## 📋 SIGNAL DETAILS")
        cols = ["symbol","direction","score","brain1_prob","brain2_conviction","velocity","wick_pressure","rs","price","is_vetoed"]
        sdf = pd.DataFrame(top_signals)
        st.dataframe(sdf[[c for c in cols if c in sdf.columns]], use_container_width=True, hide_index=True)

time.sleep(1)
st.rerun()
