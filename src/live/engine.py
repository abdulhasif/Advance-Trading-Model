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
from src.live.execution_guard import LiveExecutionGuard, SyncPendingOrderGuard
from src.live.paper_trader import PaperPortfolio


# =============================================================================
# ENGINE CONSTANTS
# =============================================================================
ENTRY_PROB_THRESH = 0.70         # Raised to Elite
ENTRY_CONV_THRESH = 45.0         # Adjusted (Strict but Realistic)
ENTRY_RS_THRESHOLD = 1.0          # Only trade leaders/laggards
MAX_ENTRY_WICK     = 0.40         # Avoid absorption traps

# ── Whipsaw Protection ──────────────────────────────────────────────────────
MIN_CONSECUTIVE_BRICKS = 3       # Require N same-direction bricks before entry
MIN_BRICKS_TODAY       = 2       # Out of the N bricks, at least M must be from today

# ── Trading Control ────────────────────────────────────────────────────────

def is_trading_active() -> bool:
    """Check if trading is paused by the user via the control file."""
    if not config.TRADE_CONTROL_FILE.exists():
        return True
    try:
        with open(config.TRADE_CONTROL_FILE, "r") as f:
            data = json.load(f)
            return data.get("active", True)
    except Exception:
        return True

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


# ── Module-Level Singletons (initialised inside run_live_engine) ───────────
# These are set once at startup so execute_trade() can access them without
# passing them through every function call in the hot loop.
_paper_portfolio: PaperPortfolio | None  = None
_order_guard:     SyncPendingOrderGuard  = SyncPendingOrderGuard(lock_timeout_seconds=5)


# ── Execute Trade ──────────────────────────────────────────────────────────

def execute_trade(signal: dict) -> bool:
    """
    FIX 5 (COMPLETE): Pending-Order-Guarded paper order placement.

    Flow:
      1. try_acquire() — non-blocking mutex check. If symbol already has a
         pending order in this 0.1s loop window, drop the signal immediately.
      2. open_position() — delegate to PaperPortfolio. Returns False if already
         in the stock or max positions reached (double-entry prevention).
      3. release() — ALWAYS in finally to guarantee the mutex is freed even if
         open_position() raises an unexpected exception.

    Returns:
        True  if order was accepted and position opened.
        False if blocked by mutex or portfolio rules.
    """
    global _paper_portfolio, _order_guard

    if _paper_portfolio is None:
        logger.error("execute_trade called before _paper_portfolio initialised!")
        return False

    symbol    = signal["symbol"]
    sector    = signal["sector"]
    side      = signal["direction"]   # "BUY" or "SELL"
    price     = signal["price"]
    ts        = datetime.now()

    # ── Fix 5: Non-blocking mutex — drop signal if already pending ────────
    if not _order_guard.try_acquire(symbol, side):
        return False

    try:
        # Delegate to PaperPortfolio (has its own duplicate-symbol guard)
        opened = _paper_portfolio.open_position(
            symbol = symbol,
            sector = sector,
            side   = "LONG" if side == "BUY" else "SHORT",
            price  = price,
            ts     = ts,
        )
        if opened:
            logger.info(f"[Engine→Paper] ORDER SENT: {side} {symbol} @ Rs {price:.2f} "
                        f"| prob={signal.get('brain1_prob',0):.3f} "
                        f"| conv={signal.get('brain2_conviction',0):.1f}")
        return opened

    except Exception as e:
        logger.error(f"[Engine→Paper] execute_trade EXCEPTION {symbol}: {e}")
        return False

    finally:
        # ALWAYS release — even on exception — to prevent permanent lockout
        _order_guard.release(symbol)


# ── Soft Veto ──────────────────────────────────────────────────────────────

def passes_soft_veto(signal: str, rel_strength: float) -> bool:
    if signal == "LONG" and rel_strength < -0.5:
        return False
    if signal == "SHORT" and rel_strength > 0.5:
        return False
    return True


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
                 "brick_size", "duration_seconds", "direction",
                 "consecutive_same_dir", "brick_oscillation_rate"]
    # After retraining, add: "consecutive_same_dir", "brick_oscillation_rate"

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

    # Initialise the module-level PaperPortfolio singleton used by execute_trade()
    global _paper_portfolio
    _paper_portfolio = PaperPortfolio()
    logger.info("PaperPortfolio initialised.")

    # ── Warmup at 09:08 ────────────────────────────────────────────────────
    wt = datetime.now().replace(hour=9, minute=8, second=0, microsecond=0)
    if datetime.now() < wt:
        sleep_sec = (wt - datetime.now()).total_seconds()
        logger.info(f"Sleeping {sleep_sec:.0f}s until 09:08 AM Warmup...")
        time.sleep(sleep_sec)
    brick_sizes = warmup_brick_sizes(universe)

    renko_states = {}
    for _, r in stocks.iterrows():
        st = LiveRenkoState(r["symbol"], r["sector"], brick_sizes.get(r["symbol"], 0.75))
        st.load_history(100)
        renko_states[r["symbol"]] = st

    sector_renko = {}
    for _, r in indices.iterrows():
        st = LiveRenkoState(r["symbol"], r["sector"], brick_sizes.get(r["symbol"], 0.75))
        st.load_history(100)
        sector_renko[r["symbol"]] = st

    risk = RiskFortress()

    # FIX 1 + 3 + 4 + 5: Build the execution guard at warm-up time
    sym_sector_map = {r["symbol"]: r["sector"] for _, r in stocks.iterrows()}
    exec_guard = LiveExecutionGuard(
        symbols   = list(renko_states.keys()),
        sectors   = sym_sector_map,
        silence_threshold  = 60,    # FIX 3: 60s silence → heartbeat candle
        order_lock_timeout = 30,    # FIX 5: 30s pending order mutex timeout
    )
    # FIX 1: Load historical bricks to warm up all indicators before 09:15
    exec_guard.warm_up_all()


    # ── Wait for 09:15 ─────────────────────────────────────────────────────
    ot = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    if datetime.now() < ot:
        sleep_sec = (ot - datetime.now()).total_seconds()
        logger.info(f"Brick sizes calculating complete. Sleeping {sleep_sec:.0f}s until 09:15 AM Market Open...")
        time.sleep(sleep_sec)
    logger.info("09:15 — TRADING LOOP STARTED")

    tick_provider = TickProvider(list(renko_states) + list(sector_renko))
    tick_provider.connect()

    last_write = 0.0
    last_entry_minutes = {}  # Hyper-trading protection
    
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
            executable_signals = []  # Collecting signals for priority execution

            for sym, st in renko_states.items():
                # FIX 3: Inject heartbeat if WebSocket has gone silent for 60s
                exec_guard.heartbeat.check_and_inject(sym, st, now)

                if sym not in ticks:
                    continue
                t = ticks[sym]

                # FIX 3: Register this live tick so heartbeat knows it's alive
                exec_guard.heartbeat.register_tick(sym, t["ltp"])

                prev_cnt = len(st.bricks)
                st.process_tick(t["ltp"], t["high"], t["low"], t["timestamp"])
                if len(st.bricks) <= prev_cnt or len(st.bricks) < 2:
                    continue

                # FIX 4: Append new brick to rolling buffer (O(1), not DataFrame)
                if st.bricks:
                    exec_guard.buffers[sym].append(st.bricks[-1])

                sec_sym = sector_index_map.get(st.sector, "")
                sec_bdf = sector_renko[sec_sym].to_dataframe() if sec_sym in sector_renko else pd.DataFrame()
                bdf = compute_features_live(st.to_dataframe(), sec_bdf)
                latest = bdf.iloc[-1]

                X = pd.DataFrame([latest[FEAT_COLS].infer_objects(copy=False).fillna(0).to_dict()])
                b1p = float(brain1.predict_proba(X)[0, 1])
                b1d = 1 if b1p > 0.5 else -1
                signal_str = "LONG" if b1d > 0 else "SHORT"

                X_m = pd.DataFrame([{
                    "brain1_prob": b1p,
                    "velocity": float(latest.get("velocity", 0)),
                    "wick_pressure": float(latest.get("wick_pressure", 0)),
                    "relative_strength": float(latest.get("relative_strength", 0)),
                }])
                b2c = float(np.clip(brain2.predict(X_m)[0], 0, 100))

                sec_dir = sector_dirs.get(st.sector, 0)
                score = risk.score_signal(b1p, b2c, b1d, sec_dir)
                rel_str_val = float(latest.get("relative_strength", 0))

                sig = {
                    "symbol": sym, "sector": st.sector,
                    "direction": "BUY" if b1d > 0 else "SELL",
                    "brain1_prob": round(b1p, 4),
                    "brain2_conviction": round(b2c, 2),
                    "score": round(score, 2),
                    "velocity": round(float(latest.get("velocity",0)), 4),
                    "wick_pressure": round(float(latest.get("wick_pressure",0)), 4),
                    "rs": round(rel_str_val, 4),
                    "price": round(float(t["ltp"]), 2),
                    "brick_count": len(st.bricks),
                    "is_vetoed": not passes_soft_veto(signal_str, rel_str_val),
                    "timestamp": now.isoformat(),
                }
                all_signals.append(sig)

                current_minute = now.replace(second=0, microsecond=0)
                if sym not in last_entry_minutes:
                    last_entry_minutes[sym] = None

                # ── Time Boundary ──────────────────────────────────────────
                no_entry = (now.hour > 15) or (now.hour == 15 and now.minute >= 0)
                if no_entry:
                    continue

                # Entry Gates
                if b1d > 0:
                    entry_prob_ok = (b1p >= ENTRY_PROB_THRESH)
                else:
                    entry_prob_ok = ((1 - b1p) >= ENTRY_PROB_THRESH)

                if entry_prob_ok and b2c >= ENTRY_CONV_THRESH and not sig["is_vetoed"]:
                    # Gate 2: RS Anchor
                    if signal_str == "LONG" and rel_str_val < ENTRY_RS_THRESHOLD:
                        continue
                    if signal_str == "SHORT" and rel_str_val > -ENTRY_RS_THRESHOLD:
                        continue
                        
                    # Gate 3: Wick Trap
                    wick_p = float(latest.get("wick_pressure", 0))
                    if wick_p > MAX_ENTRY_WICK:
                        continue

                    # Whipsaw Guard: Consecutive brick filter + Session Check
                    # Whipsaw Guard: Consecutive brick filter + Session Check
                    if len(st.bricks) >= MIN_CONSECUTIVE_BRICKS:
                        recent_bricks = st.bricks[-MIN_CONSECUTIVE_BRICKS:]
                        recent_dirs = [b["direction"] for b in recent_bricks]
                        expected_dir = (1 if signal_str == "LONG" else -1)
                        
                        # Same direction check
                        if not all(d == expected_dir for d in recent_dirs):
                            continue
                            
                        # Fresh session check: ensure today's momentum is real
                        today_date = now.date()
                        bricks_today = sum(1 for b in recent_bricks if b["brick_timestamp"].date() == today_date)
                        if bricks_today < MIN_BRICKS_TODAY:
                            continue

                    if last_entry_minutes[sym] != current_minute:
                        # Passed all gates! Add to priority queue instead of executing instantly.
                        executable_signals.append(sig)
                        last_entry_minutes[sym] = current_minute

            # ── PRIORITY EXECUTION QUEUE ─────────────────────────────────
            # Sort all collected signals by their mathematical score (Highest first)
            # This ensures limited margin goes to the best setups, not just the fastest ticks.
            if executable_signals:
                executable_signals.sort(key=lambda x: x["score"], reverse=True)
                logger.info(f"Priority Queue: {len(executable_signals)} signals firing. Executing highest score first.")
                
                for sig in executable_signals:
                    if is_trading_active():
                        execute_trade(sig)
                    else:
                        logger.info(f"TRADE SUPPRESSED: Engine Paused by User | {sig['symbol']}")

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
