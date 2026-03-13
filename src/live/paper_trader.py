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
from src.core.features import compute_features_live, FeatureSanityCheck
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
# PAPER TRADING CONSTANTS (Synchronized with config.py)
# =============================================================================
PAPER_CAPITAL      = config.STARTING_CAPITAL
POSITION_SIZE_PCT  = config.POSITION_SIZE_PCT
INTRADAY_LEVERAGE  = config.INTRADAY_LEVERAGE

# High-Conviction "Sniper" Thresholds
LONG_ENTRY_PROB_THRESH  = config.LONG_ENTRY_PROB_THRESH
SHORT_ENTRY_PROB_THRESH = config.SHORT_ENTRY_PROB_THRESH
ENTRY_CONV_THRESH       = config.ENTRY_CONV_THRESH

# Technical Alpha Filters
ENTRY_RS_THRESHOLD = config.ENTRY_RS_THRESHOLD
MAX_ENTRY_WICK     = config.MAX_ENTRY_WICK
EXIT_CONV_THRESH   = config.EXIT_CONV_THRESH
MAX_ADVERSE_BRICKS = config.STRUCTURAL_REVERSAL_BRICKS # STOP: Matches structural reversal
MAX_HOLD_BRICKS    = config.MAX_HOLD_BRICKS
MAX_OPEN_POSITIONS = config.MAX_OPEN_POSITIONS
MIN_PRICE_FILTER   = config.MIN_PRICE_FILTER

# Anti-Myopia: Hysteresis Dead-Zone (Probability State Machine)
# A held position will NOT be exited unless the model is STRONGLY against it.
HYST_LONG_SELL_FLOOR   = 0.40   # LONG: only exit Trend Reversal if prob < 0.40
HYST_SHORT_SELL_CEIL   = 0.60   # SHORT: only exit Trend Reversal if prob > 0.60

# Anti-Myopia: Structural Stop (chart-based safety override)
STRUCTURAL_REVERSAL_BRICKS = config.STRUCTURAL_REVERSAL_BRICKS

# ── Whipsaw Protection ──────────────────────────────────────────────────────
MIN_CONSECUTIVE_BRICKS = config.MIN_CONSECUTIVE_BRICKS
MIN_BRICKS_TODAY       = config.MIN_BRICKS_TODAY
MAX_LOSSES_PER_STOCK   = config.MAX_LOSSES_PER_STOCK

# ── Dynamic Realism & Charges ──────────────────────────────────────────────
TRANSACTION_COST_PCT = config.TRANSACTION_COST_PCT
T1_SLIPPAGE_PCT      = config.T1_SLIPPAGE_PCT
JITTER_SECONDS       = config.JITTER_SECONDS
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


FEAT_COLS = config.FEATURE_COLS

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
                      price: float, sl_price: float, ts: datetime) -> bool:
        # Safety Gate: Penny Stock Filter
        if price < MIN_PRICE_FILTER:
            logger.warning(f"[Paper] Rejected {symbol} @ Rs {price:.2f} (Below MIN_PRICE_FILTER)")
            return False
        """Try to open a virtual position. Returns True if opened."""
        if symbol in self.positions:
            return False  # Already in this stock
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return False  # Max positions reached

        self.trade_counter += 1
        
        # FIX #10: T+1 slippage — entry price penalty for API latency
        # Align Paper Trading with Backtester's pessimistic fill assumptions
        slippage_mult = (1.0 + T1_SLIPPAGE_PCT) if side == "LONG" else (1.0 - T1_SLIPPAGE_PCT)
        effective_price = price * slippage_mult

        # Calculate exactly how much to allocate (2% of current total capital * 5x Leverage)
        alloc = self.simulator.total_capital * POSITION_SIZE_PCT * self.simulator.LEVERAGE
        qty = max(1, int(alloc / effective_price))

        # Delegate logic to the exact Upstox Simulator
        order = self.simulator.place_order(symbol, side, qty, effective_price, sl_price, ts)
        
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
            "sl_price": sl_price,
            "qty": qty,
            "last_price": price,
            "bricks_held": 0,
            "favorable_bricks": 0,     # Highest count of favorable bricks reached
            "adverse_bricks": 0,       # Current contiguous adverse bricks from peak
            "max_run_seen": 0          # Peak favorable bricks (High-water mark)
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
        elif pnl <= 0:
            self._daily_stock_losses[symbol] = self._daily_stock_losses.get(symbol, 0) + 1
            
        self.closed_trades.append(pos)

        # Log to CSV with exact simulator taxes and timings
        entry_ts = sim_order.filled_at if sim_order.filled_at else pos["entry_time"]
        with open(TRADE_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                pos["trade_id"], symbol, pos["sector"], pos["side"],
                entry_ts.strftime("%Y-%m-%d %H:%M:%S"),
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
                        conviction: float, signal: str, prob: float,
                        new_bricks_formed: bool = False):
        """Update an open position and check exit rules."""
        if symbol not in self.positions:
            return
            
        self.simulator.update_active_price(symbol, price)
            
        pos = self.positions[symbol]
        pos["last_price"] = price
        
        # Only increment brick-based counters if a physical brick was formed.
        # This prevents "tick-drift" where sub-second ticks trigger 300-brick exits.
        if new_bricks_formed:
            pos["bricks_held"] += 1
    
            # Track peak brick run and adverse pullbacks from that peak
            if pos["side"] == "LONG":
                if brick_dir > 0:
                    pos["favorable_bricks"] += 1
                    pos["max_run_seen"] = max(pos["max_run_seen"], pos["favorable_bricks"])
                    pos["adverse_bricks"] -= 1 # Recover one adverse brick
                    pos["adverse_bricks"] = max(0, pos["adverse_bricks"])
                else:
                    pos["favorable_bricks"] -= 1 # Erode the run
                    pos["adverse_bricks"] += 1
            else:
                if brick_dir < 0:
                    pos["favorable_bricks"] += 1
                    pos["max_run_seen"] = max(pos["max_run_seen"], pos["favorable_bricks"])
                    pos["adverse_bricks"] -= 1
                    pos["adverse_bricks"] = max(0, pos["adverse_bricks"])
                else:
                    pos["favorable_bricks"] -= 1
                    pos["adverse_bricks"] += 1

    def check_exit(self, symbol: str, price: float, ts: datetime,
                   conviction: float, signal: str, prob: float,
                   brick_dir: int = 0) -> Optional[str]:
        """Check if any exit rule triggers. Returns exit reason or None."""
        if symbol not in self.positions:
            return None
        pos = self.positions[symbol]

        # Exit Rule 1: Low conviction
        if conviction < config.EXIT_CONV_THRESH:
            return "LOW_CONVICTION"

        # Exit Rule 1a: Activation Trailing Stop (Chop Protection)
        # Instead of taking profit instantly and capping runners, we use +3 as an ACTIVATION ZONE.
        # 1. At +3 bricks, we lock in Break-Even (+0 buffer).
        # 2. Beyond +3 bricks, we trail the price dynamically by TRAIL_DISTANCE_BRICKS.
        if conviction < config.STRONG_CONVICTION_THRESH:
            # Use max_run_seen to see if the trade EVER reached the activation threshold
            if pos["max_run_seen"] >= config.TRAIL_ACTIVATION_BRICKS:
                # The maximum adverse bricks allowed from the PEAK (favorable_bricks)
                dynamic_trail_allowance = config.TRAIL_DISTANCE_BRICKS
                
                # Once activated, if we fall back by the trail distance, exit immediately to lock profit
                if pos["adverse_bricks"] >= dynamic_trail_allowance:
                    return "TRAIL_PROFIT_ACTIVATED"

        # Exit Rule 2: Hysteresis Dead-Zone Trend Reversal
        # Anti-Myopia: We do NOT exit on any probability drop.
        # The model must STRONGLY confirm reversal before we flee.
        # Dead-Zone: [HYST_LONG_SELL_FLOOR, HYST_SHORT_SELL_CEIL] → HOLD (ignore noise)
        if pos["side"] == "LONG":
            # Exit LONG if model STRONGLY leans SHORT (> 0.60)
            if signal == "SHORT" and prob > (1.0 - config.HYST_LONG_SELL_FLOOR):
                return "TREND_REVERSAL"
        elif pos["side"] == "SHORT":
            # Exit SHORT if model STRONGLY leans LONG (> 0.60)
            if signal == "LONG" and prob > config.HYST_SHORT_SELL_CEIL:
                return "TREND_REVERSAL"

        # Exit Rule 3: 2-Brick Structural Trailing Stop (chart override)
        # Hard override: if 2 consecutive adverse bricks form, the chart structure
        # is unambiguous regardless of XGBoost confusion — protect capital NOW.
        if pos["adverse_bricks"] >= STRUCTURAL_REVERSAL_BRICKS:
            if pos["side"] == "LONG" and brick_dir < 0:
                return "STRUCTURAL_REVERSAL"
            if pos["side"] == "SHORT" and brick_dir > 0:
                return "STRUCTURAL_REVERSAL"

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
def passes_soft_veto(signal: str, rel_strength: float, conviction: float = 0.0) -> bool:
    # FIX #6: Use config.SOFT_VETO_THRESHOLD instead of hardcoded 0.5 to align with backtester and engine.
    """
    [NEW] Override: If conviction is exceptionally high (>VETO_BYPASS_CONV),
    we trust the model's reversal/trend prediction over the sector alignment.
    """
    if conviction >= config.VETO_BYPASS_CONV:
        return True

    if signal == "LONG" and rel_strength < -config.SOFT_VETO_THRESHOLD:
        return False
    if signal == "SHORT" and rel_strength > config.SOFT_VETO_THRESHOLD:
        return False
    return True


# =============================================================================
# MAIN PAPER TRADING LOOP
# =============================================================================
def run_paper_trader():
    logger.info("=" * 72)
    logger.info("PAPER TRADING ENGINE -- Virtual Execution Mode")
    logger.info(f"Capital: Rs {PAPER_CAPITAL:,} | Max Positions: {MAX_OPEN_POSITIONS}")
    logger.info(f"Entry: Long>{LONG_ENTRY_PROB_THRESH} Short>{SHORT_ENTRY_PROB_THRESH} Conv>{ENTRY_CONV_THRESH} | "
                f"StopLoss: {config.STRUCTURAL_REVERSAL_BRICKS} bricks")
    logger.info("=" * 72)

    # ── Load models (Dual-Brain Calibrated) ─────────────────────────────────
    # FIX #3: Load dual calibrated LONG/SHORT models instead of the legacy single brain1_direction.json model.
    if config.USE_CALIBRATED_MODELS:
        b1_long = joblib.load(str(config.BRAIN1_CALIBRATED_LONG_PATH))
        b1_short = joblib.load(str(config.BRAIN1_CALIBRATED_SHORT_PATH))
        mode_str = "Calibrated .pkl"
    else:
        b1_long = xgb.XGBClassifier(); b1_long.load_model(str(config.BRAIN1_MODEL_LONG_PATH))
        b1_short = xgb.XGBClassifier(); b1_short.load_model(str(config.BRAIN1_MODEL_SHORT_PATH))
        mode_str = "Raw .json"
    
    b2 = xgb.XGBRegressor(); b2.load_model(str(config.BRAIN2_MODEL_PATH))
    logger.info(f"Models loaded (Brain1: {mode_str}, Brain2: JSON)")

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

    # Patch 3: Feature Sanity Check — detects live vs historical feature drift
    # Fits on a representative stock's historical parquet. Disable after diagnosis.
    sanity = FeatureSanityCheck(enabled=True)
    sanity.fit_from_parquet("Finance", "NIACL")

    # Track the last minute we entered a trade per symbol to prevent Execution Illusion
    last_entry_minutes = {}

    # ── Wait for Market Open ────────────────────────────────────────────────
    ot = datetime.now().replace(hour=config.MARKET_OPEN_HOUR, minute=config.MARKET_OPEN_MINUTE, second=0, microsecond=0)
    if datetime.now() < ot:
        sleep_sec = (ot - datetime.now()).total_seconds()
        logger.info(f"Brick sizes calculating complete. Sleeping {sleep_sec:.0f}s until {config.MARKET_OPEN_HOUR:02d}:{config.MARKET_OPEN_MINUTE:02d} AM Market Open...")
        time.sleep(sleep_sec)
    logger.info(f"{config.MARKET_OPEN_HOUR:02d}:{config.MARKET_OPEN_MINUTE:02d} -- PAPER TRADING LOOP STARTED")

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

            # morning Entry Lock (Respect config.ENTRY_LOCK_MINUTES)
            morning_lock_min = config.MARKET_OPEN_MINUTE + config.ENTRY_LOCK_MINUTES
            morning_lock_hour = config.MARKET_OPEN_HOUR + (morning_lock_min // 60)
            morning_lock_min %= 60
            
            # Use current wall clock for EOD checks, but tick timestamp for entry logic where possible
            is_eod = (now.hour > config.EOD_SQUARE_OFF_HOUR) or (now.hour == config.EOD_SQUARE_OFF_HOUR and now.minute >= config.EOD_SQUARE_OFF_MIN)
            is_too_early = (now.hour < morning_lock_hour) or (now.hour == morning_lock_hour and now.minute < morning_lock_min)
            no_entry = (now.hour > config.NO_NEW_ENTRY_HOUR) or (now.hour == config.NO_NEW_ENTRY_HOUR and now.minute >= config.NO_NEW_ENTRY_MIN) or is_too_early

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

                    # Gate 0: Penny Stock Filter
                    if price < MIN_PRICE_FILTER:
                        continue
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
                        has_new_bricks = len(new_bricks) > 0
                        # Use the actual brick direction seen (or current tick momentum) instead of predicted
                        current_dir = new_bricks[-1]["direction"] if new_bricks else lp.get("brick_dir", 0)
                        portfolio.update_position(sym, price, current_dir, lp["b2c"], lp["signal"], lp["b1p"], new_bricks_formed=has_new_bricks)
                        exit_reason = portfolio.check_exit(sym, price, now, lp["b2c"], lp["signal"], lp["b1p"], brick_dir=current_dir)
                        if exit_reason:
                            portfolio.close_position(sym, price, now, exit_reason)
                            portfolio.log_signal(now, sym, st.sector, lp["signal"],
                                               lp["b1p"], lp["b2c"], lp["rel_str"], lp["score"], price,
                                               "EXIT", exit_reason)
                            # Patch 1 + 2: Release position lock & record exit for cooldown
                            guard.entry_lock.confirm_exit(sym)
                            guard.cooldown.record_exit(sym, guard.buffers[sym]._total_bricks_seen)
                            
                # Skip ML inference if no new bricks formed (O(1) DataFrame conversion)
                if not new_bricks or guard.buffers[sym].size < 2:
                    continue

                # Compute features via O(1) buffer
                sec_sym = sector_index_map.get(st.sector, "")
                sec_bdf = guard.buffers[sec_sym].to_dataframe() if sec_sym in guard.buffers else pd.DataFrame()
                bdf = compute_features_live(guard.buffers[sym].to_dataframe(), sec_bdf)
                latest = bdf.iloc[-1]

                # Dual Brain predictions
                # FIX #4: Dual Brain predictions to match backtester architecture instead of predicting on a single model.
                X = pd.DataFrame([latest[FEAT_COLS].infer_objects(copy=False).fillna(0).to_dict()])
                
                # Fast numpy extraction
                X_arr = X.values.astype(np.float32)
                p_long  = float(b1_long.predict_proba(X_arr)[0, 1])
                p_short = float(b1_short.predict_proba(X_arr)[0, 1])

                signal = "FLAT"
                b1p = 0.0
                
                # Dynamic Threshold Selection
                thresh_long  = config.LONG_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_LONG_ENTRY_PROB_THRESH
                thresh_short = config.SHORT_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_SHORT_ENTRY_PROB_THRESH

                long_ok  = p_long  >= thresh_long
                short_ok = p_short >= thresh_short

                if long_ok and short_ok:
                    if p_long >= p_short:
                        signal, b1p = "LONG", p_long
                    else:
                        signal, b1p = "SHORT", p_short
                elif long_ok:
                    signal, b1p = "LONG", p_long
                elif short_ok:
                    signal, b1p = "SHORT", p_short

                # Patch 3: Feature Sanity Check
                sanity.check(X.iloc[0].to_dict(), sym, now, prob=p_long)

                # meta-regressor now sees the full context (Trend, Alpha, etc.)
                X_m = pd.DataFrame([{c: float(latest.get(c, 0)) for c in FEAT_COLS}])
                X_m["brain1_prob"] = b1p
                b2c = float(np.clip(b2.predict(X_m)[0], 0, config.TARGET_CLIPPING_BPS))

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
                if sym in portfolio.positions:
                    continue  # Duplicate Trade Prevention
                    
                if no_entry:
                    continue

                # ── Option 2: No Gate Rush at 09:30 ──
                # Use config.ENTRY_LOCK_MINUTES for consistency
                morning_lock_min = config.MARKET_OPEN_MINUTE + config.ENTRY_LOCK_MINUTES
                morning_lock_hour = config.MARKET_OPEN_HOUR + (morning_lock_min // 60)
                morning_lock_min %= 60

                # Use the tick timestamp for the time gate check for perfect spoofer alignment
                tick_ts = t.get("timestamp", now) if sym in ticks else now
                if tick_ts.hour < morning_lock_hour or (tick_ts.hour == morning_lock_hour and tick_ts.minute < morning_lock_min):
                    portfolio.log_signal(tick_ts, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "TIME_GATE_LOCK")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": "TIME_GATE_LOCK", "ts": tick_ts})
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

                base_threshold = config.LONG_ENTRY_PROB_THRESH   # Synced — never hard-code here
                bias_threshold = config.BIAS_ENTRY_THRESHOLD

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
                    eff_prob_thresh = config.LONG_ENTRY_PROB_THRESH   # No bias: use configured threshold

                _dbg["eff_prob_thresh"] = eff_prob_thresh  # update snapshot

                # Gate 1: Elite Stats — per-direction threshold
                if signal == "LONG":
                    _thresh = LONG_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_LONG_ENTRY_PROB_THRESH
                    entry_prob_ok = b1p >= eff_prob_thresh if _bias else b1p >= _thresh
                else:
                    _thresh = SHORT_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_SHORT_ENTRY_PROB_THRESH
                    # FIX #5: Compare b1p (which is exactly p_short from the short model) directly, rather than (1-b1p).
                    entry_prob_ok = b1p >= eff_prob_thresh if _bias else b1p >= _thresh

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

                # Bypass RS Anchor if conviction is exceptionally high
                if b2c < config.VETO_BYPASS_CONV:
                    if signal == "LONG" and rel_str < ENTRY_RS_THRESHOLD:
                        portfolio.log_signal(now, sym, st.sector, signal,
                                           b1p, b2c, rel_str, score, price,
                                           "SKIP", "WEAK_RS_FILTER")
                        log_brick_event(**{**_dbg, "action": "SKIP", "reason": "WEAK_RS_FILTER"})
                        continue
                    if signal == "SHORT" and rel_str > 0.0:
                        # SHORT: only trade stocks underperforming their sector (RS < 0)
                        portfolio.log_signal(now, sym, st.sector, signal,
                                           b1p, b2c, rel_str, score, price,
                                           "SKIP", "WEAK_RS_FILTER")
                        log_brick_event(**{**_dbg, "action": "SKIP", "reason": "WEAK_RS_FILTER"})
                        continue

                # Gate 2.5: VWAP Z-Score Exhaustion
                z_vwap = float(latest.get("vwap_zscore", 0))
                if signal == "LONG" and z_vwap > getattr(config, "MAX_VWAP_ZSCORE", 3.0):
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", f"VWAP_ZSCORE_EXHAUSTION_({round(z_vwap,2)})")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": f"VWAP_ZSCORE_EXHAUSTION"})
                    continue
                if signal == "SHORT" and z_vwap < -getattr(config, "MAX_VWAP_ZSCORE", 3.0):
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", f"VWAP_ZSCORE_EXHAUSTION_({round(z_vwap,2)})")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": f"VWAP_ZSCORE_EXHAUSTION"})
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

                # Gate 4: FOMO Protection (Ghost Momentum)
                if int(latest.get("consecutive_same_dir", 0)) >= config.STREAK_LIMIT:
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "FOMO_GHOST_MOMENTUM_FILTER")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": "FOMO_GHOST_MOMENTUM_FILTER"})
                    continue

                # Whipsaw Guard 1: Consecutive brick filter + Session Check
                if len(st.bricks) < MIN_CONSECUTIVE_BRICKS:
                    _dbg["gate_whipsaw"] = "FAIL"
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "WHIPSAW_INSUFFICIENT_BRICKS")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": "WHIPSAW_INSUFFICIENT_BRICKS"})
                    continue

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
                # USER REQUEST: Commented out MIN_BRICKS_TODAY check to match spoofer behavior.
                # live_bricks_today = guard.splicers[sym].live_brick_count
                # if live_bricks_today < MIN_BRICKS_TODAY:
                #     _dbg["gate_whipsaw"] = "FAIL"
                #     portfolio.log_signal(now, sym, st.sector, signal,
                #                        b1p, b2c, rel_str, score, price,
                #                        "SKIP", "WHIPSAW_STALE_TREND")
                #     log_brick_event(**{**_dbg, "action": "SKIP", "reason": "WHIPSAW_STALE_TREND"})
                #     continue

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
                if not passes_soft_veto(signal, rel_str, b2c):
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

                # Patch 1: Authoritative entry state lock (prevents duplicate entry)
                if not guard.entry_lock.try_enter(sym):
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "ENTRY_LOCK_ALREADY_OPEN")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": "ENTRY_LOCK_ALREADY_OPEN"})
                    continue

                # Patch 2: Brick cooldown — must see N bricks after any exit before re-entry
                _bricks_now = guard.buffers[sym]._total_bricks_seen
                if not guard.cooldown.is_cooled_down(sym, _bricks_now):
                    # Release the entry lock we just acquired since we won't enter
                    guard.entry_lock.confirm_exit(sym)
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "BRICK_COOLDOWN_ACTIVE")
                    log_brick_event(**{**_dbg, "action": "SKIP", "reason": "BRICK_COOLDOWN_ACTIVE"})
                    continue

                # OPEN position
                opened = portfolio.open_position(sym, st.sector, signal, price, now)
                action = "ENTRY" if opened else "SKIP"
                reason = "" if opened else "MAX_POSITIONS"
                if opened:
                    last_entry_minutes[sym] = now_minute
                else:
                    # open_position() rejected (e.g. margin) — release the lock
                    guard.entry_lock.confirm_exit(sym)

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
