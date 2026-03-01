"""
src/live/paper_trader.py -- Paper Trading Engine
=============================================================================
Runs the FULL live trading pipeline but executes VIRTUAL trades only.
Tracks positions, P&L, and logs every signal and trade to CSV files.

Purpose: Validate backtest results against live market data before
deploying real capital. Run for 2-4 weeks with paper money to confirm
the model's edge is real.

Run:  python main.py paper
"""

import sys
import json
import time
import csv
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import xgboost as xgb

import config
from src.core.renko import LiveRenkoState
import xgboost as xgb

import config
from src.core.renko import LiveRenkoState
from src.core.features import compute_features_live
from src.core.risk import RiskFortress
from src.live.tick_provider import TickProvider
from src.live.upstox_simulator import UpstoxSimulator
from src.live.control_state import CONTROL_STATE, _thread_lock
from src.live.daily_logger import log_brick_event
from src.live.execution_guard import HeartbeatCandle
from src.api.server import register_brick_signal


# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "paper_trader.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# PAPER TRADING CONSTANTS
# =============================================================================
PAPER_CAPITAL      = 100_000     # Rs 1 Lakh paper money
POSITION_SIZE_PCT  = 0.02        # 2% capital risk per trade
INTRADAY_LEVERAGE  = 5           # 5x MIS margin (standard intraday)
ENTRY_PROB_THRESH  = 0.65        # Balanced Mode (was 0.70)
ENTRY_CONV_THRESH  = 5.0         # Brain2 gate — 5 bps min expected move (avg ~20 bps after bps re-calibration)
ENTRY_RS_THRESHOLD = 1.0         # Must be a leader/laggard (|RS| > 1.0)
MAX_ENTRY_WICK     = 0.40        # Block if wick > 40% (absorption trap)
EXIT_CONV_THRESH   = 0.0         # Brain2 conviction threshold for exit
MAX_ADVERSE_BRICKS = 5           # Stop-loss: consecutive adverse bricks
MAX_HOLD_BRICKS    = 60          # Max hold time per trade
MAX_OPEN_POSITIONS = 10          # Max simultaneous positions
EOD_EXIT_HOUR      = 15
EOD_EXIT_MINUTE    = 14

# Anti-Myopia: Hysteresis Dead-Zone (Probability State Machine)
# A held position will NOT be exited unless the model is STRONGLY against it.
# This prevents fake reversals from micro-pauses in the trend.
HYST_LONG_SELL_FLOOR   = 0.40   # LONG: only exit Trend Reversal if prob < 0.40
HYST_SHORT_SELL_CEIL   = 0.60   # SHORT: only exit Trend Reversal if prob > 0.60
# Everything between 0.40 and 0.60 is the Dead-Zone — hold and ignore noise.

# Anti-Myopia: 2-Brick Structural Stop (chart-based safety override)
# If XGBoost is confused but 2 consecutive adverse bricks form, exit regardless.
STRUCTURAL_REVERSAL_BRICKS = 2  # Consecutive adverse bricks before hard struct exit

# ── Whipsaw Protection ──────────────────────────────────────────────────────
MIN_CONSECUTIVE_BRICKS = 2       # Require N same-direction bricks before entry
MIN_BRICKS_TODAY       = 2       # Out of the N bricks, at least M must be from today
MAX_LOSSES_PER_STOCK   = 1       # Max losing trades per stock per day
NO_ENTRY_HOUR      = 15
NO_ENTRY_MINUTE    = 0

# ── Upstox Intraday Equity Charges (official rates) ─────────────────────────
BROKERAGE_PER_ORDER  = 20.0       # Rs 20 flat per executed order
BROKERAGE_PCT        = 0.0005     # or 0.05% of turnover, whichever is lower
STT_SELL_PCT         = 0.00025    # 0.025% on sell-side only
STAMP_DUTY_BUY_PCT   = 0.00003    # 0.003% on buy-side only
EXCHANGE_TXN_PCT     = 0.0000297  # NSE exchange transaction charge (both sides)
SEBI_TURNOVER_FEE    = 10.0       # Rs 10 per crore (both sides)
GST_PCT              = 0.18       # 18% on (brokerage + exchange charges)


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

# (calculate_charges removed, UpstoxSimulator handles exact Indian taxes now)


FEAT_COLS = [
    "velocity", "wick_pressure", "relative_strength",
    "brick_size", "duration_seconds",
    # "direction" removed — was 70% of model gain, causing momentum-echo bias
    "consecutive_same_dir", "brick_oscillation_rate",
    # Must match brain_trainer.py FEATURE_COLS exactly
    "fracdiff_price",      # Fractional Differentiation
    "hurst",               # Hurst Regime Feature
    "is_trending_regime",  # Boolean regime gate
]

# Output files
SIGNAL_LOG    = config.LOGS_DIR / "paper_signals.csv"
TRADE_LOG     = config.LOGS_DIR / "paper_trades.csv"
DAILY_LOG     = config.LOGS_DIR / "paper_daily.csv"
LIVE_PNL_FILE = config.PROJECT_ROOT / "paper_pnl.json"


# (PaperPosition removed, using dictionary for states and UpstoxSimulator for exact math)


# =============================================================================
# PAPER PORTFOLIO
# =============================================================================
class PaperPortfolio:
    """Tracks all virtual positions, P&L, and generates reports."""

    def __init__(self, starting_capital: float = PAPER_CAPITAL):
        self.starting_capital = starting_capital
        # Instantiate the exact Upstox tax and margin simulator
        self.simulator = UpstoxSimulator(starting_capital=starting_capital)
        
        self.positions: dict[str, dict] = {}  # Keep minimal dict for CSV logging compatibility
        self.closed_trades: list[dict] = []
        
        self.trade_counter = 0
        self.historical_trades_count = 0
        self.historical_wins = 0
        self.historical_realized_pnl = 0.0

        if TRADE_LOG.exists():
            try:
                df = pd.read_csv(TRADE_LOG)
                if not df.empty and "trade_id" in df.columns:
                    self.trade_counter = int(df["trade_id"].max())
                if not df.empty and "net_pnl" in df.columns:
                    self.historical_trades_count = len(df)
                    self.historical_wins = int((df["net_pnl"] > 0).sum())
                    self.historical_realized_pnl = float(df["net_pnl"].sum())
                    self.simulator.total_capital += self.historical_realized_pnl
                    self.simulator.available_margin = self.simulator.total_capital
            except Exception as e:
                logger.warning(f"Could not load trade history for resumption: {e}")
        self.daily_pnl: list[dict] = []
        self._today_realized = 0.0
        self._today_trades = 0
        self._today_wins = 0
        self._signals_seen = 0
        self._signals_vetoed = 0
        self._daily_stock_losses: dict[str, int] = {}  # symbol -> loss count today

        # Init CSV files with headers
        self._init_csv(SIGNAL_LOG, [
            "timestamp", "symbol", "sector", "direction", "brain1_prob",
            "brain2_conviction", "rel_strength", "score", "price",
            "action", "reason"
        ])
        self._init_csv(TRADE_LOG, [
            "trade_id", "symbol", "sector", "side", "entry_time",
            "entry_price", "exit_time", "exit_price", "qty",
            "bricks_held", "favorable", "adverse", "gross_pnl",
            "cost", "net_pnl", "exit_reason"
        ])
        self._init_csv(DAILY_LOG, [
            "date", "trades", "wins", "losses", "realized_pnl",
            "unrealized_pnl", "total_equity", "win_rate",
            "open_positions", "signals_seen", "signals_vetoed"
        ])

    def _init_csv(self, path: Path, headers: list[str]):
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(headers)

    def open_position(self, symbol: str, sector: str, side: str,
                      price: float, ts: datetime) -> bool:
        """Try to open a virtual position. Returns True if opened."""
        if symbol in self.positions:
            return False  # Already in this stock
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return False  # Max positions reached

        self.trade_counter += 1
        
        # Calculate exactly how much to allocate (2% of current total capital * 5x Leverage)
        alloc = self.simulator.total_capital * POSITION_SIZE_PCT * self.simulator.LEVERAGE
        qty = max(1, int(alloc / price))

        # Delegate logic to the exact Upstox Simulator
        order = self.simulator.place_order(symbol, side, qty, price, ts)
        
        if order.state == "REJECTED":
            # Simulator blocked it (e.g. Insufficient Margin)
            return False
            
        # Simulate instant fill for Paper Trading
        self.simulator.fill_pending_order(symbol, ts)

        # Keep a proxy dict for CSV logging compatibility
        pos = {
            "trade_id": self.trade_counter,
            "symbol": symbol,
            "sector": sector,
            "side": side,
            "entry_time": ts,
            "entry_price": price,
            "qty": qty,
            "last_price": price,
            "bricks_held": 0,
            "favorable_bricks": 0,
            "adverse_bricks": 0
        }
        self.positions[symbol] = pos
        self._today_trades += 1

        logger.info(f"PAPER ENTRY #{pos['trade_id']} | {side} {symbol} @ Rs {price:.2f} "
                    f"x {qty} | Sector: {sector} | Locked: Rs {order.locked_margin:.2f}")
        return True
    def close_position(self, symbol: str, price: float, ts: datetime,
                       reason: str) -> Optional[dict]:
        """Close a virtual position and record the trade."""
        if symbol not in self.positions:
            return None

        pos = self.positions.pop(symbol)
        
        # Exact Upstox Math via Simulator
        self.simulator.close_position(symbol, price, ts, reason)
        sim_order = self.simulator.trade_history[-1]

        pnl = sim_order.net_pnl
        self._today_realized += pnl
        if pnl > 0:
            self._today_wins += 1
        else:
            self._daily_stock_losses[symbol] = self._daily_stock_losses.get(symbol, 0) + 1
            
        self.closed_trades.append(pos)

        # Log to CSV with exact simulator taxes
        with open(TRADE_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                pos["trade_id"], symbol, pos["sector"], pos["side"],
                pos["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                round(pos["entry_price"], 2),
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                round(price, 2), pos["qty"],
                pos["bricks_held"], pos["favorable_bricks"], pos["adverse_bricks"],
                round(sim_order.gross_pnl, 2), round(sim_order.total_friction, 2), round(pnl, 2),
                reason,
            ])

        status = "WIN" if pnl > 0 else "LOSS"
        logger.info(f"PAPER EXIT  #{pos['trade_id']} | {pos['side']} {symbol} @ Rs {price:.2f} "
                    f"| {status} Rs {pnl:+.2f} | Reason: {reason} | Taxes: Rs {sim_order.total_friction:.2f}")
        return pos

    def update_position(self, symbol: str, price: float, brick_dir: int,
                        conviction: float, signal: str, prob: float):
        """Update an open position and check exit rules."""
        if symbol not in self.positions:
            return
            
        self.simulator.update_active_price(symbol, price)
            
        pos = self.positions[symbol]
        pos["last_price"] = price
        pos["bricks_held"] += 1

        # Track brick direction
        if pos["side"] == "LONG":
            if brick_dir > 0:
                pos["favorable_bricks"] += 1
                pos["adverse_bricks"] = 0
            else:
                pos["adverse_bricks"] += 1
        else:
            if brick_dir < 0:
                pos["favorable_bricks"] += 1
                pos["adverse_bricks"] = 0
            else:
                pos["adverse_bricks"] += 1

    def check_exit(self, symbol: str, price: float, ts: datetime,
                   conviction: float, signal: str, prob: float,
                   brick_dir: int = 0) -> Optional[str]:
        """Check if any exit rule triggers. Returns exit reason or None."""
        if symbol not in self.positions:
            return None
        pos = self.positions[symbol]

        # Exit Rule 1: Low conviction
        if conviction < EXIT_CONV_THRESH:
            return "LOW_CONVICTION"

        # Exit Rule 2: Hysteresis Dead-Zone Trend Reversal
        # Anti-Myopia: We do NOT exit on any probability drop.
        # The model must STRONGLY confirm reversal before we flee.
        # Dead-Zone: [HYST_LONG_SELL_FLOOR, HYST_SHORT_SELL_CEIL] → HOLD (ignore noise)
        if pos["side"] == "LONG":
            # Only exit if prob STRONGLY confirms bearish (< sell floor)
            if signal == "SHORT" and prob < HYST_LONG_SELL_FLOOR:
                return "TREND_REVERSAL"
        elif pos["side"] == "SHORT":
            # Only exit if prob STRONGLY confirms bullish (> sell ceiling)
            if signal == "LONG" and prob > HYST_SHORT_SELL_CEIL:
                return "TREND_REVERSAL"

        # Exit Rule 3: 2-Brick Structural Trailing Stop (chart override)
        # Hard override: if 2 consecutive adverse bricks form, the chart structure
        # is unambiguous regardless of XGBoost confusion — protect capital NOW.
        if pos["adverse_bricks"] >= STRUCTURAL_REVERSAL_BRICKS:
            if pos["side"] == "LONG" and brick_dir < 0:
                return "STRUCTURAL_2BRICK_REVERSAL"
            if pos["side"] == "SHORT" and brick_dir > 0:
                return "STRUCTURAL_2BRICK_REVERSAL"

        # Exit Rule 4: Stop loss (full adverse bricks limit)
        if pos["adverse_bricks"] >= MAX_ADVERSE_BRICKS:
            return "MAX_ADVERSE_BRICKS_HIT"

        # Exit Rule 5: Max hold time
        if pos["bricks_held"] >= MAX_HOLD_BRICKS:
            return "MAX_HOLD_TIME_REACHED"

        return None

    # ── Log a signal ───────────────────────────────────────────────────────
    def log_signal(self, ts: datetime, symbol: str, sector: str,
                   direction: str, prob: float, conviction: float,
                   rel_str: float, score: float, price: float,
                   action: str, reason: str = ""):
        """Append a signal row to the signal log CSV."""
        self._signals_seen += 1
        if action == "VETOED":
            self._signals_vetoed += 1
        with open(SIGNAL_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                symbol, sector, direction,
                round(prob, 4), round(conviction, 2),
                round(rel_str, 4), round(score, 2),
                round(price, 2), action, reason,
            ])

    # ── EOD close all ──────────────────────────────────────────────────────
    def close_all_eod(self, ts: datetime):
        """Force close all open positions at end of day."""
        for symbol in list(self.positions.keys()):
            if symbol in self.positions:
                pos = self.positions[symbol]
                self.close_position(symbol, pos["last_price"], ts, "EOD_SQUARE_OFF")

        # FIX: Also force cancel any abandoned pending orders so margin isn't locked overnight
        for symbol in list(self.simulator.pending_orders.keys()):
            self.simulator.cancel_pending_order(symbol, ts, "EOD_CANCEL_PENDING")

    # ── Daily summary ──────────────────────────────────────────────────────
    def record_daily_summary(self, date: str):
        """Record end-of-day summary."""
        unrealized = sum(p.gross_pnl for p in self.simulator.active_trades.values())
        total_equity = self.simulator.total_capital + unrealized
        losses = self._today_trades - self._today_wins
        wr = (self._today_wins / self._today_trades * 100) if self._today_trades > 0 else 0

        summary = {
            "date": date,
            "trades": self._today_trades,
            "wins": self._today_wins,
            "losses": losses,
            "realized_pnl": round(self._today_realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_equity": round(total_equity, 2),
            "win_rate": round(wr, 1),
            "open_positions": len(self.positions),
            "signals_seen": self._signals_seen,
            "signals_vetoed": self._signals_vetoed,
        }
        self.daily_pnl.append(summary)

        with open(DAILY_LOG, "a", newline="") as f:
            csv.writer(f).writerow(list(summary.values()))

        # Reset daily counters
        self._today_realized = 0.0
        self._today_trades = 0
        self._today_wins = 0
        self._signals_seen = 0
        self._signals_vetoed = 0

        logger.info(f"DAILY SUMMARY | {date} | Trades: {summary['trades']} "
                    f"| Win Rate: {summary['win_rate']}% "
                    f"| Day PnL: Rs {summary['realized_pnl']:+.2f} "
                    f"| Equity: Rs {summary['total_equity']:,.2f}")
        return summary

    # ── Live PnL state (for dashboard) ─────────────────────────────────────
    def write_pnl_state(self):
        """Write current portfolio state to JSON for dashboard."""
        unrealized = sum(p.unrealized_pnl for p in self.simulator.active_trades.values())
        open_pos = []
        for sym, pos in self.positions.items():
            # Get the UpstoxSimulator's active trade object for this symbol
            sim_trade = self.simulator.active_trades.get(sym)
            if sim_trade:
                open_pos.append({
                    "symbol": sym, "side": pos["side"],
                    "entry_price": round(pos["entry_price"], 2),
                    "current_price": round(pos["last_price"], 2),
                    "qty": pos["qty"],
                    "unrealized_pnl": round(sim_trade.unrealized_pnl, 2),
                    "bricks_held": pos["bricks_held"],
                })

        total_closed = len(self.closed_trades) + self.historical_trades_count
        # The `closed_trades` list now contains the original `pos` dicts, not `PaperPosition` objects.
        # We need to sum the `net_pnl` from the simulator's trade history for accurate realized PnL.
        # This is a bit tricky as `closed_trades` is just a proxy for logging.
        # For dashboard, we should rely on `simulator.trade_history` for realized PnL.
        
        # Recalculate total wins and realized PnL from simulator's history
        total_realized_pnl_from_sim = sum(t.net_pnl for t in self.simulator.trade_history)
        total_wins_from_sim = sum(1 for t in self.simulator.trade_history if t.net_pnl > 0)

        state = {
            "timestamp": datetime.now().isoformat(),
            "mode": "PAPER_TRADING",
            "starting_capital": self.starting_capital,
            "cash": round(self.simulator.available_margin, 2), # Use simulator's available margin as cash
            "unrealized_pnl": round(unrealized, 2),
            "total_equity": round(self.simulator.total_capital + unrealized, 2),
            "total_trades": len(self.simulator.trade_history), # Total trades from simulator
            "total_wins": total_wins_from_sim,
            "win_rate": round(total_wins_from_sim / len(self.simulator.trade_history) * 100, 1) if len(self.simulator.trade_history) > 0 else 0,
            "total_realized_pnl": round(total_realized_pnl_from_sim, 2),
            "open_positions": open_pos,
            "max_positions": MAX_OPEN_POSITIONS,
        }
        try:
            with open(LIVE_PNL_FILE, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write PnL state: {e}")


# =============================================================================
# SOFT VETO
# =============================================================================
def passes_soft_veto(signal: str, rel_strength: float) -> bool:
    if signal == "LONG" and rel_strength < -0.5:
        return False
    if signal == "SHORT" and rel_strength > 0.5:
        return False
    return True


# =============================================================================
# MAIN PAPER TRADING LOOP
# =============================================================================
def run_paper_trader():
    logger.info("=" * 72)
    logger.info("PAPER TRADING ENGINE -- Virtual Execution Mode")
    logger.info(f"Capital: Rs {PAPER_CAPITAL:,} | Max Positions: {MAX_OPEN_POSITIONS}")
    logger.info(f"Entry: Prob>{ENTRY_PROB_THRESH} Conv>{ENTRY_CONV_THRESH} | "
                f"StopLoss: {MAX_ADVERSE_BRICKS} bricks")
    logger.info("=" * 72)

    # ── Load models ─────────────────────────────────────────────────────────
    b1 = xgb.XGBClassifier();  b1.load_model(str(config.BRAIN1_MODEL_PATH))
    b2 = xgb.XGBRegressor();   b2.load_model(str(config.BRAIN2_MODEL_PATH))
    logger.info("Models loaded: Brain1 (Direction) + Brain2 (Conviction)")

    # ── Load universe ───────────────────────────────────────────────────────
    universe = pd.read_csv(config.UNIVERSE_CSV)
    universe["is_index"] = universe["is_index"].astype(str).str.lower().isin(["true","1","yes"])
    stocks  = universe[~universe["is_index"]].reset_index(drop=True)
    indices = universe[ universe["is_index"]].reset_index(drop=True)
    sector_index_map = {r["sector"]: r["symbol"] for _, r in indices.iterrows()}

    # ── Sleep until 09:00 ───────────────────────────────────────────────────
    now = datetime.now()
    target = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now < target:
        wait_s = (target - now).total_seconds()
        logger.info(f"Sleeping {wait_s:.0f}s until 09:00")
        time.sleep(wait_s)

    # ── Warmup brick sizes at 09:08 ─────────────────────────────────────────
    logger.info("WARMUP -- Calculating brick sizes")
    brick_sizes = {}
    for _, row in universe.iterrows():
        sym, sec = row["symbol"], row["sector"]
        stock_dir = config.DATA_DIR / sec / sym
        if stock_dir.exists():
            pqs = sorted(stock_dir.glob("*.parquet"))
            if pqs:
                try:
                    df = pd.read_parquet(pqs[-1])
                    if not df.empty:
                        brick_sizes[sym] = df["brick_close"].iloc[-1] * config.NATR_BRICK_PERCENT
                        continue
                except Exception:
                    pass
        brick_sizes[sym] = 500 * config.NATR_BRICK_PERCENT

    renko_states = {}
    for _, r in stocks.iterrows():
        st = LiveRenkoState(r["symbol"], r["sector"], brick_sizes.get(r["symbol"], 0.75))
        renko_states[r["symbol"]] = st

    sector_renko = {}
    for _, r in indices.iterrows():
        st = LiveRenkoState(r["symbol"], r["sector"], brick_sizes.get(r["symbol"], 0.75))
        sector_renko[r["symbol"]] = st

    from src.live.execution_guard import LiveExecutionGuard
    stock_sectors = {r["symbol"]: r["sector"] for _, r in stocks.iterrows()}
    all_syms = list(renko_states.keys()) + list(sector_renko.keys())
    guard = LiveExecutionGuard(symbols=all_syms, sectors=stock_sectors)
    guard.warm_up_all()

    for sym, st in list(renko_states.items()) + list(sector_renko.items()):
        bdf = guard.buffers[sym].to_dataframe()
        if not bdf.empty:
            st.renko_level = bdf["brick_close"].iloc[-1]
            st.brick_start_time = bdf["brick_timestamp"].iloc[-1]

    # HeartbeatCandle: inject flat ticks when WebSocket goes silent > 60s
    heartbeat = guard.heartbeat
    last_preds = {}

    risk = RiskFortress()
    portfolio = PaperPortfolio(PAPER_CAPITAL)
    
    # Track the last minute we entered a trade per symbol to prevent Execution Illusion
    last_entry_minutes = {}

    # ── Wait for 09:15 ──────────────────────────────────────────────────────
    ot = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    if datetime.now() < ot:
        sleep_sec = (ot - datetime.now()).total_seconds()
        logger.info(f"Brick sizes calculating complete. Sleeping {sleep_sec:.0f}s until 09:15 AM Market Open...")
        time.sleep(sleep_sec)
    logger.info("09:15 -- PAPER TRADING LOOP STARTED")

    tick_provider = TickProvider(list(renko_states) + list(sector_renko))
    tick_provider.connect()

    last_write = 0.0
    current_date = datetime.now().strftime("%Y-%m-%d")

    try:
        while True:
            t0 = time.time()
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")

            # ── DELIVERABLE 4A: Global Kill Switch ─────────────────────────
            # Android biometric trigger -> square off everything and exit loop.
            with _thread_lock:
                _kill = CONTROL_STATE["GLOBAL_KILL"]
            if _kill:
                logger.critical("GLOBAL KILL SWITCH: squaring off all positions and shutting down.")
                portfolio.simulator.square_off_all(now)
                portfolio.record_daily_summary(today_str)
                break

            # Day change detection
            if today_str != current_date:
                portfolio.record_daily_summary(current_date)
                portfolio._daily_stock_losses.clear()  # Reset whipsaw tracker
                current_date = today_str

            # Shutdown check (15:35)
            if now.hour > config.SYSTEM_SHUTDOWN_HOUR or \
               (now.hour == config.SYSTEM_SHUTDOWN_HOUR and
                now.minute >= config.SYSTEM_SHUTDOWN_MINUTE):
                # Final cleanup
                for sym in list(portfolio.positions.keys()):
                    pos = portfolio.positions[sym]
                    portfolio.close_position(sym, pos["last_price"], now, "SYSTEM_SHUTDOWN")

                portfolio.record_daily_summary(today_str)
                logger.info(f"Summary — Total Trades: {portfolio.historical_trades_count + portfolio._today_trades}")
                logger.info(f"Summary — Realized PnL: Rs {portfolio.historical_realized_pnl + portfolio._today_realized:.2f}")
                logger.info(f"Summary — Total Equity: Rs {portfolio.simulator.total_capital:.2f}")
                logger.info("=== Paper Trader Offline ===")
                sys.exit(0)

            # EOD exit window
            is_eod = (now.hour > EOD_EXIT_HOUR) or (now.hour == EOD_EXIT_HOUR and now.minute >= EOD_EXIT_MINUTE)
            no_entry = (now.hour > NO_ENTRY_HOUR) or (now.hour == NO_ENTRY_HOUR and now.minute >= NO_ENTRY_MINUTE)

            if is_eod:
                portfolio.close_all_eod(now)

            ticks = tick_provider.get_latest_ticks()

            # Process sector ticks
            for sym, st in sector_renko.items():
                if sym in ticks:
                    t = ticks[sym]
                    new_b = st.process_tick(t["ltp"], t["high"], t["low"], t["timestamp"])
                    for b in new_b:
                        guard.buffers[sym].append(b)

            sector_dirs = {
                st.sector: (guard.buffers[sym]._buffer[-1]["direction"] if guard.buffers[sym].size > 0 else 0)
                for sym, st in sector_renko.items()
            }

            # Process stock ticks
            for sym, st in renko_states.items():
                if sym in ticks:
                    t = ticks[sym]
                    price = t["ltp"]
                    heartbeat.register_tick(sym, price)
                    new_bricks = st.process_tick(price, t["high"], t["low"], t["timestamp"])
                else:
                    # Heartbeat injection ensures silent ticks are registered
                    heartbeat.check_and_inject(sym, st, now)
                    price = heartbeat._last_ltp.get(sym, st.renko_level or 0.0)
                    new_bricks = []

                for b in new_bricks:
                    guard.buffers[sym].append(b)

                # Update live position state unconditionally (Red Team Fix)
                if sym in portfolio.positions:
                    portfolio.positions[sym]["last_price"] = price

                    if sym in last_preds:
                        lp = last_preds[sym]
                        portfolio.update_position(sym, price, lp["brick_dir"], lp["b2c"], lp["signal"], lp["b1p"])
                        exit_reason = portfolio.check_exit(sym, price, now, lp["b2c"], lp["signal"], lp["b1p"], brick_dir=lp["brick_dir"])
                        if exit_reason:
                            portfolio.close_position(sym, price, now, exit_reason)
                            portfolio.log_signal(now, sym, st.sector, lp["signal"],
                                               lp["b1p"], lp["b2c"], lp["rel_str"], lp["score"], price,
                                               "EXIT", exit_reason)
                            # Let it re-evaluate entry naturally if desired
                            
                # Skip ML inference if no new bricks formed (O(1) DataFrame conversion)
                if not new_bricks or guard.buffers[sym].size < 2:
                    continue

                # Compute features via O(1) buffer
                sec_sym = sector_index_map.get(st.sector, "")
                sec_bdf = guard.buffers[sec_sym].to_dataframe() if sec_sym in guard.buffers else pd.DataFrame()
                bdf = compute_features_live(guard.buffers[sym].to_dataframe(), sec_bdf)
                latest = bdf.iloc[-1]

                # Brain predictions
                X = pd.DataFrame([latest[FEAT_COLS].infer_objects(copy=False).fillna(0).to_dict()])
                b1p = float(b1.predict_proba(X)[0, 1])
                b1d = 1 if b1p > 0.5 else -1
                signal = "LONG" if b1d > 0 else "SHORT"

                X_m = pd.DataFrame([{
                    "brain1_prob": b1p,
                    "velocity": float(latest.get("velocity", 0)),
                    "wick_pressure": float(latest.get("wick_pressure", 0)),
                    "relative_strength": float(latest.get("relative_strength", 0)),
                }])
                b2c = float(np.clip(b2.predict(X_m)[0], 0, 100))

                sec_dir = sector_dirs.get(st.sector, 0)
                score = risk.score_signal(b1p, b2c, b1d, sec_dir)
                price = t["ltp"]
                rel_str = float(latest.get("relative_strength", 0))
                brick_dir = int(latest.get("direction", 0))
                wick_p_raw = float(latest.get("wick_pressure", 0))

                # Update Market Regime Telemetry
                register_brick_signal(brick_dir, b2c)

                # ── Audit snapshot for daily_logger ────────────────────
                _dbg = dict(
                    ts=now, symbol=sym, sector=st.sector,
                    price=price, brick_dir=brick_dir, sec_dir=sec_dir,
                    new_bricks=len(st.bricks),
                    velocity=float(latest.get("velocity", 0)),
                    wick_pressure=wick_p_raw,
                    relative_strength=rel_str,
                    brick_size=float(latest.get("brick_size", 0)),
                    duration_seconds=float(latest.get("duration_seconds", 0)),
                    consecutive_same=int(latest.get("consecutive_same_dir", 0)),
                    oscillation_rate=float(latest.get("brick_oscillation_rate", 0)),
                    brain1_prob=b1p, brain2_conv=b2c, signal=signal, score=score,
                    # control state (read before acquiring any lock below)
                    global_kill=CONTROL_STATE["GLOBAL_KILL"],
                    global_pause=CONTROL_STATE["GLOBAL_PAUSE"],
                    ticker_paused=sym in CONTROL_STATE["PAUSED_TICKERS"],
                    bias=CONTROL_STATE["BIAS"].get(sym, ""),
                    eff_prob_thresh=0.75,   # default; updated by bias logic
                    # gate verdicts default to SKIP (updated as they run)
                    gate_prob="SKIP", gate_conv="SKIP", gate_rs="SKIP",
                    gate_wick="SKIP", gate_whipsaw="SKIP",
                    gate_losses="SKIP", gate_positions="SKIP",
                    action="", reason="",
                    open_positions=len(portfolio.positions),
                    live_pnl=portfolio.simulator.get_live_pnl(),
                )

                last_preds[sym] = {
                    "b1p": b1p, "b2c": b2c, "signal": signal, 
                    "score": score, "rel_str": rel_str, "brick_dir": brick_dir
                }

                # ── Note: Exits already evaluated above so we can jump straight to entry validation ───────

                # ── Check entry for new position ──────────────────────
                if no_entry:
                    continue

                # ── DELIVERABLE 4B: 3-Tier Pause Check ────────────────────────
                # Tier 1: GLOBAL_PAUSE suspends entries for ALL tickers engine-wide.
                # Tier 2: PAUSED_TICKERS suppresses entries for specific symbols.
                # Exits are always monitored — that logic runs in the block above.
                with _thread_lock:
                    _global_pause = CONTROL_STATE["GLOBAL_PAUSE"]
                    _ticker_paused = sym in CONTROL_STATE["PAUSED_TICKERS"]
                if _global_pause or _ticker_paused:
                    reason = "GLOBAL_PAUSE" if _global_pause else "TICKER_PAUSED_BY_ANDROID"
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", reason)
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": reason})
                    continue

                # ── DELIVERABLE 4C: Soft Bias / Hunter Mode ────────────────
                # Human-in-the-Loop: Android sends a news-based directional bias.
                # Rules:
                #   1. If a bias is set, OPPOSING signals are STRICTLY ignored.
                #   2. If signal matches bias direction, threshold drops from
                #      base_threshold (0.75) -> bias_threshold (0.65).
                #   3. No bias set -> base_threshold (0.75) applies to all signals.
                #
                # IMPORTANT: All downstream gates (Gate 2 RS, Gate 3 wick,
                # whipsaw guard, conviction) still run — the AI retains final
                # execution authority. Bias is a nudge, not an override.
                with _thread_lock:
                    _bias = CONTROL_STATE["BIAS"].get(sym, None)

                base_threshold = ENTRY_PROB_THRESH   # Synced — never hard-code here
                bias_threshold = 0.65

                if _bias == "LONG":
                    if signal == "SHORT":
                        # Strict: ignore all opposing signals when bias is active
                        portfolio.log_signal(now, sym, st.sector, signal,
                                           b1p, b2c, rel_str, score, price,
                                           "SKIP", "SOFT_BIAS_LONG_BLOCKS_SHORT")
                        log_brick_event(**{**_dbg, "action": "SKIP", "reason": "SOFT_BIAS_LONG_BLOCKS_SHORT"})
                        continue
                    eff_prob_thresh = bias_threshold   # 0.75 -> 0.65 for LONG
                elif _bias == "SHORT":
                    if signal == "LONG":
                        portfolio.log_signal(now, sym, st.sector, signal,
                                           b1p, b2c, rel_str, score, price,
                                           "SKIP", "SOFT_BIAS_SHORT_BLOCKS_LONG")
                        log_brick_event(**{**_dbg, "action": "SKIP", "reason": "SOFT_BIAS_SHORT_BLOCKS_LONG"})
                        continue
                    eff_prob_thresh = bias_threshold   # 0.75 -> 0.65 for SHORT
                else:
                    eff_prob_thresh = ENTRY_PROB_THRESH   # No bias: use configured threshold

                _dbg["eff_prob_thresh"] = eff_prob_thresh  # update snapshot

                # Gate 1: Elite Stats — uses eff_prob_thresh (AI retains authority)
                if signal == "LONG":
                    entry_prob_ok = b1p >= eff_prob_thresh
                else:
                    entry_prob_ok = (1 - b1p) >= eff_prob_thresh

                _dbg["gate_prob"] = "PASS" if entry_prob_ok else "FAIL"
                _dbg["gate_conv"] = "PASS" if b2c >= ENTRY_CONV_THRESH else "FAIL"

                if not entry_prob_ok or b2c < ENTRY_CONV_THRESH:
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "ELITE_STATS_FILTER")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": "ELITE_STATS_FILTER"})
                    continue

                # Gate 2: RS Anchor (Only trade the strongest/weakest)
                _rs_ok_long  = signal == "LONG"  and rel_str >= ENTRY_RS_THRESHOLD
                _rs_ok_short = signal == "SHORT" and rel_str <= -ENTRY_RS_THRESHOLD
                _dbg["gate_rs"] = "PASS" if (_rs_ok_long or _rs_ok_short) else "FAIL"

                if signal == "LONG" and rel_str < ENTRY_RS_THRESHOLD:
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "WEAK_RS_FILTER")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": "WEAK_RS_FILTER"})
                    continue
                if signal == "SHORT" and rel_str > 0.0:
                    # SHORT: only trade stocks underperforming their sector (RS < 0)
                    # NOT -ENTRY_RS_THRESHOLD (-1.0) — too strict, blocks all shorts in uptrend
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "WEAK_RS_FILTER")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": "WEAK_RS_FILTER"})
                    continue

                # Gate 3: Wick Trap (Absorption check)
                wick_p = float(latest.get("wick_pressure", 0))
                _dbg["gate_wick"] = "FAIL" if wick_p > MAX_ENTRY_WICK else "PASS"
                if wick_p > MAX_ENTRY_WICK:
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", f"WICK_TRAP_FILTER_{round(wick_p,2)}")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": f"WICK_TRAP_FILTER_{round(wick_p,2)}"})
                    continue

                # Whipsaw Guard 1: Consecutive brick filter + Session Check

                # Whipsaw Guard 1: Consecutive brick filter
                # Require N same-direction bricks before entry + Session Check
                if len(st.bricks) >= MIN_CONSECUTIVE_BRICKS:
                    recent_bricks = st.bricks[-MIN_CONSECUTIVE_BRICKS:]
                    recent_dirs = [b["direction"] for b in recent_bricks]
                    expected_dir = 1 if signal == "LONG" else -1

                    # Same direction check
                    _whip_dirs_ok = all(d == expected_dir for d in recent_dirs)
                    if not _whip_dirs_ok:
                        _dbg["gate_whipsaw"] = "FAIL"
                        portfolio.log_signal(now, sym, st.sector, signal,
                                           b1p, b2c, rel_str, score, price,
                                           "SKIP", "WHIPSAW_BRICK_FILTER")
                        log_brick_event(**{**_dbg, "action": "SKIP", "reason": "WHIPSAW_BRICK_FILTER"})
                        continue

                    # Fresh session check: ensure today's momentum is real
                    today_date = now.date()
                    bricks_today = sum(1 for b in recent_bricks if b["brick_timestamp"].date() == today_date)
                    if bricks_today < MIN_BRICKS_TODAY:
                        _dbg["gate_whipsaw"] = "FAIL"
                        portfolio.log_signal(now, sym, st.sector, signal,
                                           b1p, b2c, rel_str, score, price,
                                           "SKIP", "WHIPSAW_STALE_TREND")
                        log_brick_event(**{**_dbg, "action": "SKIP", "reason": "WHIPSAW_STALE_TREND"})
                        continue

                    _dbg["gate_whipsaw"] = "PASS"

                # Whipsaw Guard 2: Daily stock loss limit
                stock_losses = portfolio._daily_stock_losses.get(sym, 0)
                _dbg["gate_losses"] = "FAIL" if stock_losses >= MAX_LOSSES_PER_STOCK else "PASS"
                if stock_losses >= MAX_LOSSES_PER_STOCK:
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "WHIPSAW_DAILY_LOSS_LIMIT")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": "WHIPSAW_DAILY_LOSS_LIMIT"})
                    continue

                # Soft veto
                if not passes_soft_veto(signal, rel_str):
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "VETOED", "SECTOR_MISALIGN")
                    log_brick_event(**{**_dbg, "action": "VETOED", "reason": "SECTOR_MISALIGN"})
                    continue

                # Execution Illusion Fix
                now_minute = now.replace(second=0, microsecond=0)
                if sym in last_entry_minutes and last_entry_minutes[sym] == now_minute:
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "SAME_MINUTE_ENTRY")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": "SAME_MINUTE_ENTRY"})
                    continue

                # Kill Switch check
                if not is_trading_active():
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "PAUSED_BY_USER")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": "PAUSED_BY_USER"})
                    continue

                # Position cap check
                _dbg["gate_positions"] = "FAIL" if len(portfolio.positions) >= MAX_OPEN_POSITIONS else "PASS"

                # OPEN position
                opened = portfolio.open_position(sym, st.sector, signal, price, now)
                action = "ENTRY" if opened else "SKIP"
                reason = "" if opened else "MAX_POSITIONS"
                if opened:
                    last_entry_minutes[sym] = now_minute

                portfolio.log_signal(now, sym, st.sector, signal,
                                   b1p, b2c, rel_str, score, price,
                                   action, reason)
                log_brick_event(**{**_dbg, "action": action, "reason": reason,
                                   "open_positions": len(portfolio.positions),
                                   "live_pnl": portfolio.simulator.get_live_pnl()})

            # Write PnL state periodically
            if (time.time() - last_write) >= config.STATE_WRITE_INTERVAL:
                portfolio.write_pnl_state()
                last_write = time.time()

            # Throttle
            elapsed = time.time() - t0
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

    except KeyboardInterrupt:
        logger.info("Stopped by user")
        portfolio.close_all_eod(datetime.now())
        portfolio.record_daily_summary(current_date)
        _print_session_summary(portfolio)
    finally:
        tick_provider.disconnect()
        portfolio.write_pnl_state()
        logger.info("Paper trading engine shut down.")


def _print_session_summary(portfolio: PaperPortfolio):
    """Print end-of-session performance summary."""
    trades = portfolio.simulator.trade_history
    if not trades:
        logger.info("No trades executed this session.")
        return

    wins = sum(1 for t in trades if t.net_pnl > 0)
    losses = len(trades) - wins
    total_pnl = sum(t.net_pnl for t in trades)
    win_rate = wins / len(trades) * 100

    print("\n" + "=" * 60)
    print("   PAPER TRADING SESSION SUMMARY")
    print("=" * 60)
    print(f"   {'Total Trades:':<25} {len(trades)}")
    print(f"   {'Wins:':<25} {wins}")
    print(f"   {'Losses:':<25} {losses}")
    print(f"   {'Win Rate:':<25} {win_rate:.1f}%")
    print(f"   {'Total P&L:':<25} Rs {total_pnl:+,.2f}")
    print(f"   {'Starting Capital:':<25} Rs {portfolio.starting_capital:,.2f}")
    print(f"   {'Final Equity:':<25} Rs {portfolio.simulator.total_capital:,.2f}")
    print(f"   {'Return:':<25} {(portfolio.simulator.total_capital/portfolio.starting_capital-1)*100:+.2f}%")
    print("=" * 60)
    print(f"   Logs: {SIGNAL_LOG.name}, {TRADE_LOG.name}, {DAILY_LOG.name}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_paper_trader()
