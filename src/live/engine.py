"""
src/live/engine.py — Phase 4: Real-Time Trading Engine
========================================================
Daily lifecycle: 08:50 wake → 09:08 warmup → 09:15–15:30 trade → 15:35 shutdown.
Writes live_state.json every 1s for the dashboard.

Run:  python -m src.live.engine
"""

import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

import xgboost as xgb

import config
from src.core.renko import LiveRenkoState
from src.core.features import compute_features_live
from src.core.risk import RiskFortress
from src.live.tick_provider import TickProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LIVE_LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── Model Loader ────────────────────────────────────────────────────────────

def load_models():
    b1 = xgb.XGBClassifier();  b1.load_model(str(config.BRAIN1_MODEL_PATH))
    b2 = xgb.XGBRegressor();   b2.load_model(str(config.BRAIN2_MODEL_PATH))
    logger.info("Models loaded")
    return b1, b2


# ── Execute Trade Placeholder ──────────────────────────────────────────────

def execute_trade(signal: dict):
    """[FUTURE] Auto-Pilot — currently no-op (Human-in-the-Loop)."""
    pass


# ── Warmup ──────────────────────────────────────────────────────────────────

def warmup_brick_sizes(universe: pd.DataFrame) -> dict[str, float]:
    """Compute today's brick size from previous-day close for each symbol."""
    logger.info("WARMUP — Calculating brick sizes")
    sizes = {}
    for _, row in universe.iterrows():
        sym, sec = row["symbol"], row["sector"]
        stock_dir = config.DATA_DIR / sec / sym
        if stock_dir.exists():
            pqs = sorted(stock_dir.glob("*.parquet"))
            if pqs:
                try:
                    df = pd.read_parquet(pqs[-1])
                    if not df.empty:
                        sizes[sym] = df["brick_close"].iloc[-1] * config.NATR_BRICK_PERCENT
                        continue
                except Exception:
                    pass
        sizes[sym] = 500 * config.NATR_BRICK_PERCENT  # fallback
    logger.info(f"Brick sizes ready for {len(sizes)} symbols")
    return sizes


# ── State Writer ────────────────────────────────────────────────────────────

def write_live_state(top_signals, renko_states, risk: RiskFortress, latency_ms: float):
    chart_bricks = []
    if top_signals:
        sym = top_signals[0]["symbol"]
        if sym in renko_states:
            bdf = renko_states[sym].to_dataframe()
            if not bdf.empty:
                chart_bricks = bdf.tail(200).to_dict(orient="records")
                for b in chart_bricks:
                    for k in ["brick_timestamp", "brick_start_time", "brick_end_time"]:
                        if k in b and hasattr(b[k], "isoformat"):
                            b[k] = b[k].isoformat()

    state = {
        "timestamp": datetime.now().isoformat(),
        "top_signals": top_signals,
        "chart_symbol": top_signals[0]["symbol"] if top_signals else None,
        "chart_bricks": chart_bricks,
        "health": {
            "loop_latency_ms": round(latency_ms, 1),
            "drift_accuracy": risk.drift_accuracy,
            "yellow_alert": risk.yellow_alert,
            "active_symbols": len(renko_states),
        },
    }
    try:
        with open(config.LIVE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"State write failed: {e}")


# ── Sector Direction Helper ────────────────────────────────────────────────

def get_sector_directions(sector_renko: dict[str, LiveRenkoState]) -> dict[str, int]:
    return {
        st.sector: (st.bricks[-1]["direction"] if st.bricks else 0)
        for st in sector_renko.values()
    }


# ── Main Live Loop ──────────────────────────────────────────────────────────

def run_live_engine():
    logger.info("=" * 70)
    logger.info("LIVE ENGINE — Starting")
    logger.info("=" * 70)

    FEAT_COLS = ["velocity", "wick_pressure", "relative_strength",
                 "brick_size", "duration_seconds", "direction"]

    # ── Sleep until 09:00 ──────────────────────────────────────────────────
    now = datetime.now()
    target = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now < target:
        logger.info(f"Sleeping {(target-now).total_seconds():.0f}s until 09:00")
        time.sleep((target - now).total_seconds())

    # ── Load universe & models ─────────────────────────────────────────────
    universe = pd.read_csv(config.UNIVERSE_CSV)
    universe["is_index"] = universe["is_index"].astype(str).str.lower().isin(["true","1","yes"])
    stocks  = universe[~universe["is_index"]].reset_index(drop=True)
    indices = universe[ universe["is_index"]].reset_index(drop=True)

    brain1, brain2 = load_models()
    sector_index_map = {r["sector"]: r["symbol"] for _, r in indices.iterrows()}

    # ── Warmup at 09:08 ────────────────────────────────────────────────────
    wt = datetime.now().replace(hour=9, minute=8, second=0, microsecond=0)
    if datetime.now() < wt:
        time.sleep((wt - datetime.now()).total_seconds())
    brick_sizes = warmup_brick_sizes(universe)

    renko_states = {
        r["symbol"]: LiveRenkoState(r["symbol"], r["sector"], brick_sizes.get(r["symbol"], 0.75))
        for _, r in stocks.iterrows()
    }
    sector_renko = {
        r["symbol"]: LiveRenkoState(r["symbol"], r["sector"], brick_sizes.get(r["symbol"], 0.75))
        for _, r in indices.iterrows()
    }

    risk = RiskFortress()

    # ── Wait for 09:15 ─────────────────────────────────────────────────────
    ot = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    if datetime.now() < ot:
        time.sleep((ot - datetime.now()).total_seconds())
    logger.info("09:15 — TRADING LOOP STARTED")

    tick_provider = TickProvider(list(renko_states) + list(sector_renko))
    tick_provider.connect()

    last_write = 0.0
    try:
        while True:
            t0 = time.time()
            now = datetime.now()

            # Shutdown check
            if now.hour > config.SYSTEM_SHUTDOWN_HOUR or \
               (now.hour == config.SYSTEM_SHUTDOWN_HOUR and now.minute >= config.SYSTEM_SHUTDOWN_MINUTE):
                logger.info("15:35 — MARKET CLOSED. Bye.")
                tick_provider.disconnect(); sys.exit(0)

            ticks = tick_provider.get_latest_ticks()

            # Process sector ticks
            for sym, st in sector_renko.items():
                if sym in ticks:
                    t = ticks[sym]
                    st.process_tick(t["ltp"], t["high"], t["low"], t["timestamp"])

            sector_dirs = get_sector_directions(sector_renko)

            # Process stock ticks
            all_signals = []
            for sym, st in renko_states.items():
                if sym not in ticks:
                    continue
                t = ticks[sym]
                prev_cnt = len(st.bricks)
                st.process_tick(t["ltp"], t["high"], t["low"], t["timestamp"])
                if len(st.bricks) <= prev_cnt or len(st.bricks) < 2:
                    continue

                sec_sym = sector_index_map.get(st.sector, "")
                sec_bdf = sector_renko[sec_sym].to_dataframe() if sec_sym in sector_renko else pd.DataFrame()
                bdf = compute_features_live(st.to_dataframe(), sec_bdf)
                latest = bdf.iloc[-1]

                X = pd.DataFrame([latest[FEAT_COLS].fillna(0).to_dict()])
                b1p = float(brain1.predict_proba(X)[0, 1])
                b1d = 1 if b1p > 0.5 else -1

                X_m = pd.DataFrame([{
                    "brain1_prob": b1p,
                    "velocity": float(latest.get("velocity", 0)),
                    "wick_pressure": float(latest.get("wick_pressure", 0)),
                    "relative_strength": float(latest.get("relative_strength", 0)),
                }])
                b2c = float(np.clip(brain2.predict(X_m)[0], 0, 100))

                sec_dir = sector_dirs.get(st.sector, 0)
                score = risk.score_signal(b1p, b2c, b1d, sec_dir)

                sig = {
                    "symbol": sym, "sector": st.sector,
                    "direction": "BUY" if b1d > 0 else "SELL",
                    "brain1_prob": round(b1p, 4),
                    "brain2_conviction": round(b2c, 2),
                    "score": round(score, 2),
                    "velocity": round(float(latest.get("velocity",0)), 4),
                    "wick_pressure": round(float(latest.get("wick_pressure",0)), 4),
                    "rs": round(float(latest.get("relative_strength",0)), 4),
                    "price": round(float(t["ltp"]), 2),
                    "brick_count": len(st.bricks),
                    "is_vetoed": b1d != sec_dir,
                    "timestamp": now.isoformat(),
                }
                all_signals.append(sig)
                execute_trade(sig)

            top = risk.rank_signals(all_signals)
            latency = (time.time() - t0) * 1000

            if (time.time() - last_write) >= config.STATE_WRITE_INTERVAL:
                write_live_state(top, renko_states, risk, latency)
                last_write = time.time()

            elapsed = time.time() - t0
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        tick_provider.disconnect()
        logger.info("Engine shut down.")


if __name__ == "__main__":
    run_live_engine()
