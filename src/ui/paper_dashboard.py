"""
src/ui/paper_dashboard.py -- Paper Trading Performance Dashboard
==================================================================
Dark-mode visualization of paper trading results: equity curve, trade
history, stock breakdown, P&L analysis, and signal statistics.

Run:  python -m streamlit run src/ui/paper_dashboard.py --server.port 8502
"""

import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import config

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paper Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 30 seconds
st.markdown(
    '<meta http-equiv="refresh" content="30">',
    unsafe_allow_html=True,
)

# ── Dark Theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; color: #e0e0e0; }
    h1,h2,h3 { color: #00ff88 !important; font-family: 'Courier New', monospace; }
    [data-testid="stMetricValue"] {
        color: #00ff88; font-family: 'Courier New', monospace; font-size: 1.4rem !important;
    }
    [data-testid="stMetricLabel"] { color: #999; font-family: 'Courier New', monospace; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #111; color: #888; border-radius: 4px;
        padding: 8px 20px; font-family: 'Courier New', monospace;
    }
    .stTabs [aria-selected="true"] { background: #00ff88 !important; color: #000 !important; }
    .big-pnl-positive { color: #00ff88; font-size: 2.5rem; font-weight: bold;
                         font-family: 'Courier New', monospace; }
    .big-pnl-negative { color: #ff4444; font-size: 2.5rem; font-weight: bold;
                         font-family: 'Courier New', monospace; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0a0a",
    plot_bgcolor="#0a0a0a",
    font=dict(family="Courier New", color="#888"),
    margin=dict(l=50, r=20, t=40, b=40),
)


# ── Data Loading ────────────────────────────────────────────────────────────
TRADE_LOG = config.LOGS_DIR / "paper_trades.csv"
DAILY_LOG = config.LOGS_DIR / "paper_daily.csv"
SIGNAL_LOG = config.LOGS_DIR / "paper_signals.csv"
PNL_FILE = config.PROJECT_ROOT / "paper_pnl.json"


@st.cache_data(ttl=5)
def load_trades():
    if not TRADE_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(TRADE_LOG)
    if df.empty:
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    return df


@st.cache_data(ttl=5)
def load_daily():
    if not DAILY_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(DAILY_LOG)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(ttl=5)
def load_signals():
    if not SIGNAL_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(SIGNAL_LOG)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_live_pnl():
    if not PNL_FILE.exists():
        return None
    try:
        with open(PNL_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


# ── SIDEBAR CONTROLS ────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛡️ ENGINE CONTROLS")

def get_trading_status():
    if not config.TRADE_CONTROL_FILE.exists():
        # Ensure directory exists before writing
        config.TRADE_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(config.TRADE_CONTROL_FILE, "w") as f:
            json.dump({"active": True}, f)
        return True
    try:
        with open(config.TRADE_CONTROL_FILE, "r") as f:
            return json.load(f).get("active", True)
    except:
        return True

def set_trading_status(active: bool):
    config.TRADE_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.TRADE_CONTROL_FILE, "w") as f:
        json.dump({"active": active}, f)

current_status = get_trading_status()
if st.sidebar.button("🔴 STOP TRADING" if current_status else "🟢 RESUME TRADING", use_container_width=True):
    set_trading_status(not current_status)
    st.rerun()

status_label = "ACTIVE" if current_status else "PAUSED"
status_color = "#00ff88" if current_status else "#ff4444"
st.sidebar.markdown(f"Status: **<span style='color:{status_color}'>{status_label}</span>**", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🛰️ MARKET SENTINEL")
st.sidebar.info("Sentiment: **CAUTIOUS RALLY**")
st.sidebar.markdown(f"""
**Feb 23 Close:**
- Nifty: 25,713 (+0.55%)
- Sensex: 83,294 (+0.58%)

**Key Insight:**
- Rally on US Tariff blocks neutralized by new 15% tariff plan.
- **IT Sector (-1.42%)** is the primary laggard.
- Whipsaw risk: **HIGH**.
""")
st.sidebar.markdown("---")


# ── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("# 📊 PAPER TRADING DASHBOARD")

trades = load_trades()
daily = load_daily()
signals = load_signals()
pnl_state = load_live_pnl()

if trades.empty and pnl_state is None:
    st.warning("No paper trading data found. Run `python main.py paper` first.")
    st.info(f"Looking for: `{TRADE_LOG}`, `{DAILY_LOG}`, `{SIGNAL_LOG}`")
    st.stop()


# ── LIVE STATUS BAR ─────────────────────────────────────────────────────────
if pnl_state:
    mode = pnl_state.get("mode", "UNKNOWN")
    equity = pnl_state.get("total_equity", 0)
    starting = pnl_state.get("starting_capital", 100000)
    ret = (equity / starting - 1) * 100 if starting > 0 else 0
    pnl_class = "big-pnl-positive" if ret >= 0 else "big-pnl-negative"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mode", mode)
    c2.markdown(f'<div class="{pnl_class}">Rs {equity:,.2f}</div>',
                unsafe_allow_html=True)
    c3.metric("Total Trades", pnl_state.get("total_trades", 0))
    c4.metric("Win Rate", f"{pnl_state.get('win_rate', 0):.1f}%")
    c5.metric("Return", f"{ret:+.2f}%")

    # Live open positions
    open_pos = pnl_state.get("open_positions", [])
    if open_pos:
        pos_df = pd.DataFrame(open_pos)
        # Force static table to avoid rendering crashes
        st.table(pos_df)

st.markdown("---")
st.caption("DEBUG: Reached Tabs Section")


# ── TABS ────────────────────────────────────────────────────────────────────
tab_overview, tab_trades, tab_stocks, tab_signals = st.tabs([
    "Overview", "Trade History", "Stock Analysis", "Signal Log"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    if not trades.empty:
        wins = (trades["net_pnl"] > 0).sum()
        losses = (trades["net_pnl"] <= 0).sum()
        total_pnl = trades["net_pnl"].sum()
        avg_win = trades[trades["net_pnl"] > 0]["net_pnl"].mean() if wins > 0 else 0
        avg_loss = trades[trades["net_pnl"] <= 0]["net_pnl"].mean() if losses > 0 else 0
        profit_factor = abs(trades[trades["net_pnl"] > 0]["net_pnl"].sum() /
                           trades[trades["net_pnl"] <= 0]["net_pnl"].sum()) \
            if losses > 0 and trades[trades["net_pnl"] <= 0]["net_pnl"].sum() != 0 else float("inf")
        best = trades["net_pnl"].max()
        worst = trades["net_pnl"].min()

        # Headline metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Trades", len(trades))
        m2.metric("Win Rate", f"{wins/len(trades)*100:.1f}%")
        m3.metric("Total P&L", f"Rs {total_pnl:+,.2f}")
        m4.metric("Profit Factor", f"{profit_factor:.2f}")
        m5.metric("Best Trade", f"Rs {best:+,.2f}")
        m6.metric("Worst Trade", f"Rs {worst:+,.2f}")

        st.markdown("---")

        # Equity Curve
        col_eq, col_dist = st.columns([2, 1])

        with col_eq:
            st.markdown("### Equity Curve")
            starting_cap = pnl_state.get("starting_capital", 100000) if pnl_state else 100000
            equity_series = [starting_cap]
            for pnl in trades["net_pnl"]:
                equity_series.append(equity_series[-1] + pnl)

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                y=equity_series, mode="lines",
                line=dict(color="#00ff88", width=2),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.05)",
                name="Equity"
            ))
            fig_eq.add_hline(y=starting_cap, line_dash="dot",
                            line_color="#444", annotation_text="Starting Capital")
            fig_eq.update_layout(**PLOTLY_LAYOUT, height=400,
                                yaxis_title="Equity (Rs)",
                                xaxis_title="Trade #")
            st.plotly_chart(fig_eq, use_container_width=True)

        with col_dist:
            st.markdown("### P&L Distribution")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=trades["net_pnl"], nbinsx=30,
                marker_color="#00ff88", opacity=0.7,
                name="P&L"
            ))
            fig_hist.add_vline(x=0, line_dash="solid", line_color="#ff4444")
            fig_hist.update_layout(**PLOTLY_LAYOUT, height=400,
                                  xaxis_title="P&L (Rs)",
                                  yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Win/Loss and Exit Reasons
        col_wl, col_exit = st.columns(2)

        with col_wl:
            st.markdown("### Win / Loss Breakdown")
            fig_wl = go.Figure(data=[go.Pie(
                labels=["Wins", "Losses"],
                values=[int(wins), int(losses)],
                marker=dict(colors=["#00ff88", "#ff4444"]),
                hole=0.5,
                textinfo="label+percent",
                textfont=dict(color="#fff", family="Courier New"),
            )])
            fig_wl.update_layout(**PLOTLY_LAYOUT, height=350,
                                showlegend=False)
            st.plotly_chart(fig_wl, use_container_width=True)

        with col_exit:
            st.markdown("### Exit Reasons")
            if "exit_reason" in trades.columns:
                exit_counts = trades["exit_reason"].value_counts()
                fig_exit = go.Figure(data=[go.Bar(
                    x=exit_counts.index, y=exit_counts.values,
                    marker_color=["#00ff88", "#ff8800", "#ff4444",
                                  "#4488ff", "#aa44ff", "#ffff44"][:len(exit_counts)],
                    text=exit_counts.values, textposition="auto",
                    textfont=dict(color="#fff"),
                )])
                fig_exit.update_layout(**PLOTLY_LAYOUT, height=350,
                                      xaxis_title="Exit Reason",
                                      yaxis_title="Count")
                st.plotly_chart(fig_exit, use_container_width=True)

    # Daily P&L
    if not daily.empty:
        st.markdown("---")
        st.markdown("### Daily P&L")
        colors = ["#00ff88" if x >= 0 else "#ff4444" for x in daily["realized_pnl"]]
        fig_daily = go.Figure(data=[go.Bar(
            x=daily["date"].dt.strftime("%d %b"),
            y=daily["realized_pnl"],
            marker_color=colors,
            text=[f"Rs {v:+.0f}" for v in daily["realized_pnl"]],
            textposition="auto",
            textfont=dict(color="#fff", size=10),
        )])
        fig_daily.update_layout(**PLOTLY_LAYOUT, height=350,
                               xaxis_title="Date", yaxis_title="Daily P&L (Rs)")
        st.plotly_chart(fig_daily, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: TRADE HISTORY
# ═══════════════════════════════════════════════════════════════════════════
with tab_trades:
    if not trades.empty:
        st.markdown("### All Trades")

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            side_filter = st.multiselect("Side", ["LONG", "SHORT"],
                                          default=["LONG", "SHORT"])
        with fc2:
            exit_filter = st.multiselect("Exit Reason",
                                          trades["exit_reason"].unique().tolist(),
                                          default=trades["exit_reason"].unique().tolist())
        with fc3:
            pnl_filter = st.radio("P&L", ["All", "Winners", "Losers"],
                                   horizontal=True)

        filtered = trades[
            trades["side"].isin(side_filter) &
            trades["exit_reason"].isin(exit_filter)
        ]
        if pnl_filter == "Winners":
            filtered = filtered[filtered["net_pnl"] > 0]
        elif pnl_filter == "Losers":
            filtered = filtered[filtered["net_pnl"] <= 0]

        # Add Date column
        filtered = filtered.copy()
        filtered["Date"] = filtered["entry_time"].dt.strftime("%Y-%m-%d")

        display_cols = [
            "Date", "symbol", "side", "qty", "entry_price", "exit_price",
            "gross_pnl", "cost", "net_pnl", "exit_reason"
        ]
        
        # Ensure columns exist before filtering (safety)
        available_cols = [c for c in display_cols if c in filtered.columns]
        display_df = filtered[available_cols].copy()
        
        # Rename for cleaner UI
        rename_map = {
            "symbol": "Stock", "side": "Side", "qty": "Qty",
            "entry_price": "Entry Price", "exit_price": "Exit Price",
            "gross_pnl": "Gross P&L", "cost": "Brokerage & Taxes",
            "net_pnl": "Net P&L", "exit_reason": "Reason"
        }
        display_df = display_df.rename(columns=rename_map)

        # Color and Format
        st.dataframe(
            display_df.style.map(
                lambda v: "color: #00ff88" if isinstance(v, (int, float)) and v > 0 else "color: #ff4444" if isinstance(v, (int, float)) and v < 0 else "",
                subset=[c for c in ["Gross P&L", "Net P&L"] if c in display_df.columns]
            ).format({
                "Entry Price": "₹{:,.2f}",
                "Exit Price": "₹{:,.2f}",
                "Gross P&L": "₹{:,.2f}",
                "Brokerage & Taxes": "₹{:,.2f}",
                "Net P&L": "₹{:,.2f}",
            }),
            use_container_width=True, hide_index=True, height=500
        )

        # Summary stats for filtered
        st.markdown(f"**Showing {len(filtered)} / {len(trades)} trades** | "
                    f"Total P&L: Rs {filtered['net_pnl'].sum():+,.2f} | "
                    f"Avg: Rs {filtered['net_pnl'].mean():+,.2f} | "
                    f"Median: Rs {filtered['net_pnl'].median():+,.2f}")
    else:
        st.info("No trades recorded yet.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: STOCK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab_stocks:
    if not trades.empty:
        # Per-stock P&L
        stock_pnl = trades.groupby("symbol").agg(
            trades_count=("net_pnl", "count"),
            total_pnl=("net_pnl", "sum"),
            avg_pnl=("net_pnl", "mean"),
            win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
            best_trade=("net_pnl", "max"),
            worst_trade=("net_pnl", "min"),
        ).sort_values("total_pnl", ascending=False).reset_index()

        col_tbl, col_chart = st.columns([1, 1])

        with col_tbl:
            st.markdown("### Stock Performance Ranking")
            st.dataframe(
                stock_pnl.style.map(
                    lambda v: "color: #00ff88" if isinstance(v, (int, float)) and v > 0
                    else "color: #ff4444" if isinstance(v, (int, float)) and v < 0
                    else "",
                    subset=["total_pnl", "avg_pnl", "best_trade", "worst_trade"]
                ).format({
                    "total_pnl": "Rs {:+,.2f}",
                    "avg_pnl": "Rs {:+,.2f}",
                    "win_rate": "{:.1f}%",
                    "best_trade": "Rs {:+,.2f}",
                    "worst_trade": "Rs {:+,.2f}",
                }),
                use_container_width=True, hide_index=True, height=500
            )

        with col_chart:
            st.markdown("### P&L by Stock")
            top_n = stock_pnl.head(15)
            colors = ["#00ff88" if v >= 0 else "#ff4444" for v in top_n["total_pnl"]]
            fig_stock = go.Figure(data=[go.Bar(
                x=top_n["symbol"], y=top_n["total_pnl"],
                marker_color=colors,
                text=[f"Rs {v:+,.0f}" for v in top_n["total_pnl"]],
                textposition="auto",
                textfont=dict(color="#fff", size=10),
            )])
            fig_stock.update_layout(**PLOTLY_LAYOUT, height=500,
                                   xaxis_title="Stock", yaxis_title="Total P&L (Rs)")
            st.plotly_chart(fig_stock, use_container_width=True)

        # Sector breakdown
        if "sector" in trades.columns:
            st.markdown("---")
            st.markdown("### Sector Performance")
            sector_pnl = trades.groupby("sector").agg(
                trades_count=("net_pnl", "count"),
                total_pnl=("net_pnl", "sum"),
                win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
            ).sort_values("total_pnl", ascending=False).reset_index()

            col_sec_chart, col_sec_tbl = st.columns([1, 1])
            with col_sec_chart:
                fig_sec = go.Figure(data=[go.Pie(
                    labels=sector_pnl["sector"],
                    values=sector_pnl["trades_count"],
                    marker=dict(colors=px.colors.qualitative.Set3),
                    hole=0.4,
                    textinfo="label+percent",
                    textfont=dict(color="#fff", family="Courier New", size=11),
                )])
                fig_sec.update_layout(**PLOTLY_LAYOUT, height=400,
                                     title="Trades by Sector")
                st.plotly_chart(fig_sec, use_container_width=True)

            with col_sec_tbl:
                st.dataframe(
                    sector_pnl.style.format({
                        "total_pnl": "Rs {:+,.2f}",
                        "win_rate": "{:.1f}%",
                    }),
                    use_container_width=True, hide_index=True
                )
    else:
        st.info("No trades recorded yet.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: SIGNAL LOG
# ═══════════════════════════════════════════════════════════════════════════
with tab_signals:
    if not signals.empty:
        # Signal action distribution
        col_act, col_dir = st.columns(2)

        with col_act:
            st.markdown("### Signal Actions")
            act_counts = signals["action"].value_counts()
            fig_act = go.Figure(data=[go.Bar(
                x=act_counts.index, y=act_counts.values,
                marker_color=["#00ff88" if a == "ENTRY" else
                              "#ff4444" if a == "EXIT" else
                              "#ff8800" if a == "VETOED" else "#666"
                              for a in act_counts.index],
                text=act_counts.values, textposition="auto",
                textfont=dict(color="#fff"),
            )])
            fig_act.update_layout(**PLOTLY_LAYOUT, height=350,
                                 xaxis_title="Action", yaxis_title="Count")
            st.plotly_chart(fig_act, use_container_width=True)

        with col_dir:
            st.markdown("### Signal Direction")
            dir_counts = signals["direction"].value_counts()
            fig_dir = go.Figure(data=[go.Pie(
                labels=dir_counts.index,
                values=dir_counts.values,
                marker=dict(colors=["#00ff88", "#ff4444"]),
                hole=0.5,
                textinfo="label+percent",
                textfont=dict(color="#fff", family="Courier New"),
            )])
            fig_dir.update_layout(**PLOTLY_LAYOUT, height=350)
            st.plotly_chart(fig_dir, use_container_width=True)

        # Full signal log table
        st.markdown("### Recent Signals")
        recent = signals.tail(100).sort_values("timestamp", ascending=False)
        st.dataframe(recent, use_container_width=True, hide_index=True, height=400)

        st.caption(f"Total signals: {len(signals):,} | "
                   f"Entries: {len(signals[signals['action']=='ENTRY']):,} | "
                   f"Exits: {len(signals[signals['action']=='EXIT']):,} | "
                   f"Vetoed: {len(signals[signals['action']=='VETOED']):,} | "
                   f"Skipped: {len(signals[signals['action']=='SKIP']):,}")
    else:
        st.info("No signals logged yet.")


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')} | "
           f"Data: {TRADE_LOG.name}, {DAILY_LOG.name}, {SIGNAL_LOG.name}")
