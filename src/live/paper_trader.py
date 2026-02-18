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
from src.core.features import compute_features_live
from src.core.risk import RiskFortress
from src.live.tick_provider import TickProvider


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
POSITION_SIZE_PCT  = 0.02        # 2% per trade
REALITY_TAX        = 0.001       # 10 bps per trade
ENTRY_PROB_THRESH  = 0.55        # Brain1 probability threshold
ENTRY_CONV_THRESH  = 65.0        # Brain2 conviction threshold for entry
EXIT_CONV_THRESH   = 40.0        # Brain2 conviction threshold for exit
MAX_ADVERSE_BRICKS = 5           # Stop-loss: consecutive adverse bricks
MAX_HOLD_BRICKS    = 60          # Max hold time per trade
MAX_OPEN_POSITIONS = 3           # Max simultaneous positions
EOD_EXIT_HOUR      = 15
EOD_EXIT_MINUTE    = 14
NO_ENTRY_HOUR      = 15
NO_ENTRY_MINUTE    = 0

FEAT_COLS = ["velocity", "wick_pressure", "relative_strength",
             "brick_size", "duration_seconds", "direction"]

# Output files
SIGNAL_LOG    = config.LOGS_DIR / "paper_signals.csv"
TRADE_LOG     = config.LOGS_DIR / "paper_trades.csv"
DAILY_LOG     = config.LOGS_DIR / "paper_daily.csv"
LIVE_PNL_FILE = config.PROJECT_ROOT / "paper_pnl.json"


# =============================================================================
# VIRTUAL POSITION
# =============================================================================
@dataclass
class PaperPosition:
    trade_id: int
    symbol: str
    sector: str
    side: str                      # LONG or SHORT
    entry_time: datetime
    entry_price: float
    qty: int                       # shares (based on position sizing)
    bricks_held: int = 0
    favorable_bricks: int = 0
    adverse_bricks: int = 0
    last_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""

    @property
    def unrealized_pnl(self) -> float:
        price = self.last_price if self.last_price > 0 else self.entry_price
        if self.side == "LONG":
            return (price - self.entry_price) * self.qty
        else:
            return (self.entry_price - price) * self.qty

    @property
    def realized_pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        if self.side == "LONG":
            gross = (self.exit_price - self.entry_price) * self.qty
        else:
            gross = (self.entry_price - self.exit_price) * self.qty
        cost = self.entry_price * self.qty * REALITY_TAX
        return gross - cost


# =============================================================================
# PAPER PORTFOLIO
# =============================================================================
class PaperPortfolio:
    """Tracks all virtual positions, P&L, and generates reports."""

    def __init__(self, starting_capital: float = PAPER_CAPITAL):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions: dict[str, PaperPosition] = {}  # symbol -> position
        self.closed_trades: list[PaperPosition] = []
        self.trade_counter = 0
        self.daily_pnl: list[dict] = []
        self._today_realized = 0.0
        self._today_trades = 0
        self._today_wins = 0
        self._signals_seen = 0
        self._signals_vetoed = 0

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

    # ── Entry ──────────────────────────────────────────────────────────────
    def open_position(self, symbol: str, sector: str, side: str,
                      price: float, ts: datetime) -> bool:
        """Try to open a virtual position. Returns True if opened."""
        if symbol in self.positions:
            return False  # Already in this stock
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return False  # Max positions reached

        self.trade_counter += 1
        alloc = self.starting_capital * POSITION_SIZE_PCT
        qty = max(1, int(alloc / price))

        pos = PaperPosition(
            trade_id=self.trade_counter,
            symbol=symbol,
            sector=sector,
            side=side,
            entry_time=ts,
            entry_price=price,
            qty=qty,
            last_price=price,
        )
        self.positions[symbol] = pos
        self._today_trades += 1

        logger.info(f"PAPER ENTRY #{pos.trade_id} | {side} {symbol} @ Rs {price:.2f} "
                    f"x {qty} | Sector: {sector}")
        return True

    # ── Exit ───────────────────────────────────────────────────────────────
    def close_position(self, symbol: str, price: float, ts: datetime,
                       reason: str) -> Optional[PaperPosition]:
        """Close a virtual position and record the trade."""
        if symbol not in self.positions:
            return None

        pos = self.positions.pop(symbol)
        pos.exit_time = ts
        pos.exit_price = price
        pos.exit_reason = reason

        pnl = pos.realized_pnl
        self._today_realized += pnl
        if pnl > 0:
            self._today_wins += 1
        self.cash += pnl
        self.closed_trades.append(pos)

        # Log to CSV
        gross = (price - pos.entry_price) * pos.qty if pos.side == "LONG" \
            else (pos.entry_price - price) * pos.qty
        cost = pos.entry_price * pos.qty * REALITY_TAX

        with open(TRADE_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                pos.trade_id, symbol, pos.sector, pos.side,
                pos.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                round(pos.entry_price, 2),
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                round(price, 2), pos.qty,
                pos.bricks_held, pos.favorable_bricks, pos.adverse_bricks,
                round(gross, 2), round(cost, 2), round(pnl, 2),
                reason,
            ])

        status = "WIN" if pnl > 0 else "LOSS"
        logger.info(f"PAPER EXIT  #{pos.trade_id} | {pos.side} {symbol} @ Rs {price:.2f} "
                    f"| {status} Rs {pnl:+.2f} | Reason: {reason}")
        return pos

    # ── Update prices / check exits ────────────────────────────────────────
    def update_position(self, symbol: str, price: float, brick_dir: int,
                        conviction: float, signal: str, prob: float):
        """Update an open position and check exit rules."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        pos.last_price = price
        pos.bricks_held += 1

        # Track brick direction
        if pos.side == "LONG":
            if brick_dir > 0:
                pos.favorable_bricks += 1
                pos.adverse_bricks = 0
            else:
                pos.adverse_bricks += 1
        else:
            if brick_dir < 0:
                pos.favorable_bricks += 1
                pos.adverse_bricks = 0
            else:
                pos.adverse_bricks += 1

    def check_exit(self, symbol: str, price: float, ts: datetime,
                   conviction: float, signal: str, prob: float) -> Optional[str]:
        """Check if any exit rule triggers. Returns exit reason or None."""
        if symbol not in self.positions:
            return None
        pos = self.positions[symbol]

        # Exit Rule 1: Low conviction
        if conviction < EXIT_CONV_THRESH:
            return "LOW_CONVICTION"

        # Exit Rule 2: Trend reversal
        if pos.side == "LONG" and signal == "SHORT" and prob < 0.40:
            return "TREND_REVERSAL"
        if pos.side == "SHORT" and signal == "LONG" and prob > 0.60:
            return "TREND_REVERSAL"

        # Exit Rule 3: Stop-loss
        if pos.adverse_bricks >= MAX_ADVERSE_BRICKS:
            return "STOP_LOSS"

        # Exit Rule 4: Max hold
        if pos.bricks_held >= MAX_HOLD_BRICKS:
            return "MAX_HOLD"

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
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            self.close_position(sym, pos.last_price, ts, "EOD_EXIT")

    # ── Daily summary ──────────────────────────────────────────────────────
    def record_daily_summary(self, date: str):
        """Record end-of-day summary."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_equity = self.cash + unrealized
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
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        open_pos = []
        for sym, pos in self.positions.items():
            open_pos.append({
                "symbol": sym, "side": pos.side,
                "entry_price": round(pos.entry_price, 2),
                "current_price": round(pos.last_price, 2),
                "qty": pos.qty,
                "unrealized_pnl": round(pos.unrealized_pnl, 2),
                "bricks_held": pos.bricks_held,
            })

        total_closed = len(self.closed_trades)
        total_wins = sum(1 for t in self.closed_trades if t.realized_pnl > 0)
        total_realized = sum(t.realized_pnl for t in self.closed_trades)

        state = {
            "timestamp": datetime.now().isoformat(),
            "mode": "PAPER_TRADING",
            "starting_capital": self.starting_capital,
            "cash": round(self.cash, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_equity": round(self.cash + unrealized, 2),
            "total_trades": total_closed,
            "total_wins": total_wins,
            "win_rate": round(total_wins / total_closed * 100, 1) if total_closed > 0 else 0,
            "total_realized_pnl": round(total_realized, 2),
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

    renko_states = {
        r["symbol"]: LiveRenkoState(r["symbol"], r["sector"],
                                     brick_sizes.get(r["symbol"], 0.75))
        for _, r in stocks.iterrows()
    }
    sector_renko = {
        r["symbol"]: LiveRenkoState(r["symbol"], r["sector"],
                                     brick_sizes.get(r["symbol"], 0.75))
        for _, r in indices.iterrows()
    }

    risk = RiskFortress()
    portfolio = PaperPortfolio(PAPER_CAPITAL)

    # ── Wait for 09:15 ──────────────────────────────────────────────────────
    ot = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    if datetime.now() < ot:
        time.sleep((ot - datetime.now()).total_seconds())
    logger.info("09:15 -- PAPER TRADING LOOP STARTED")

    tick_provider = TickProvider(list(renko_states) + list(sector_renko))
    tick_provider.connect()

    last_write = 0.0
    current_date = datetime.now().strftime("%Y-%m-%d")

    try:
        while True:
            t0 = time.time()
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            # Day change detection
            if today != current_date:
                portfolio.record_daily_summary(current_date)
                current_date = today

            # Shutdown check (15:35)
            if now.hour > config.SYSTEM_SHUTDOWN_HOUR or \
               (now.hour == config.SYSTEM_SHUTDOWN_HOUR and
                now.minute >= config.SYSTEM_SHUTDOWN_MINUTE):
                # EOD: close all positions
                portfolio.close_all_eod(now)
                portfolio.record_daily_summary(current_date)
                portfolio.write_pnl_state()

                # Print session summary
                _print_session_summary(portfolio)

                logger.info("15:35 -- MARKET CLOSED. Paper trading session ended.")
                tick_provider.disconnect()
                sys.exit(0)

            # EOD exit window
            is_eod = (now.hour >= EOD_EXIT_HOUR and now.minute >= EOD_EXIT_MINUTE)
            no_entry = (now.hour >= NO_ENTRY_HOUR and now.minute >= NO_ENTRY_MINUTE)

            if is_eod:
                portfolio.close_all_eod(now)

            ticks = tick_provider.get_latest_ticks()

            # Process sector ticks
            for sym, st in sector_renko.items():
                if sym in ticks:
                    t = ticks[sym]
                    st.process_tick(t["ltp"], t["high"], t["low"], t["timestamp"])

            sector_dirs = {
                st.sector: (st.bricks[-1]["direction"] if st.bricks else 0)
                for st in sector_renko.values()
            }

            # Process stock ticks
            for sym, st in renko_states.items():
                if sym not in ticks:
                    continue

                t = ticks[sym]
                prev_cnt = len(st.bricks)
                st.process_tick(t["ltp"], t["high"], t["low"], t["timestamp"])

                # Update open position price
                if sym in portfolio.positions:
                    portfolio.positions[sym].last_price = t["ltp"]

                # Only process on NEW brick
                if len(st.bricks) <= prev_cnt or len(st.bricks) < 2:
                    continue

                # Compute features
                sec_sym = sector_index_map.get(st.sector, "")
                sec_bdf = sector_renko[sec_sym].to_dataframe() if sec_sym in sector_renko else pd.DataFrame()
                bdf = compute_features_live(st.to_dataframe(), sec_bdf)
                latest = bdf.iloc[-1]

                # Brain predictions
                X = pd.DataFrame([latest[FEAT_COLS].fillna(0).to_dict()])
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

                # ── Check exits for open position ──────────────────────
                if sym in portfolio.positions:
                    portfolio.update_position(sym, price, brick_dir, b2c, signal, b1p)
                    exit_reason = portfolio.check_exit(sym, price, now, b2c, signal, b1p)
                    if exit_reason:
                        portfolio.close_position(sym, price, now, exit_reason)
                        portfolio.log_signal(now, sym, st.sector, signal,
                                           b1p, b2c, rel_str, score, price,
                                           "EXIT", exit_reason)
                    continue

                # ── Check entry for new position ──────────────────────
                if no_entry:
                    continue

                # Entry gates
                if signal == "LONG":
                    entry_prob_ok = b1p > ENTRY_PROB_THRESH
                else:
                    entry_prob_ok = (1 - b1p) > ENTRY_PROB_THRESH

                if not entry_prob_ok or b2c <= ENTRY_CONV_THRESH:
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "SKIP", "BELOW_THRESHOLD")
                    continue

                # Soft veto
                if not passes_soft_veto(signal, rel_str):
                    portfolio.log_signal(now, sym, st.sector, signal,
                                       b1p, b2c, rel_str, score, price,
                                       "VETOED", "SECTOR_MISALIGN")
                    continue

                # OPEN position
                opened = portfolio.open_position(sym, st.sector, signal, price, now)
                action = "ENTRY" if opened else "SKIP"
                reason = "" if opened else "MAX_POSITIONS"
                portfolio.log_signal(now, sym, st.sector, signal,
                                   b1p, b2c, rel_str, score, price,
                                   action, reason)

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
    trades = portfolio.closed_trades
    if not trades:
        logger.info("No trades executed this session.")
        return

    wins = sum(1 for t in trades if t.realized_pnl > 0)
    losses = len(trades) - wins
    total_pnl = sum(t.realized_pnl for t in trades)
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
    print(f"   {'Final Equity:':<25} Rs {portfolio.cash:,.2f}")
    print(f"   {'Return:':<25} {(portfolio.cash/portfolio.starting_capital-1)*100:+.2f}%")
    print("=" * 60)
    print(f"   Logs: {SIGNAL_LOG.name}, {TRADE_LOG.name}, {DAILY_LOG.name}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_paper_trader()
