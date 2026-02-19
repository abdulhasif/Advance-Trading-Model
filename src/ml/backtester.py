"""
src/ml/backtester.py -- The "Truth Teller" Backtest Engine v2
===========================================================================
Strict, event-driven INTRADAY simulation of the dual-brain trading system
on UNSEEN data (2025-2026). Applies Reality Tax, Soft Veto, proper
day-boundary exits, and produces a boardroom-quality performance report.

KEY FIX in v2:
  - Enforces INTRADAY-only: all positions close by 15:25 each day
  - Re-evaluates conviction on every brick for dynamic exit
  - Adds stop-loss (max adverse bricks) and take-profit logic
  - Properly handles both LONG and SHORT trades

Run:  python main.py backtest
"""

import sys
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import config

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "backtester.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================
REALITY_TAX        = 0.001       # 10 bps per trade (slippage + brokerage + STT)
ENTRY_PROB_THRESH  = 0.55        # Brain1 probability threshold (raised for quality)
ENTRY_CONV_THRESH  = 65.0        # Brain2 conviction threshold for entry
EXIT_CONV_THRESH   = 40.0        # Brain2 conviction threshold for exit
STARTING_CAPITAL   = 1_000_000   # Rs 10 Lakh notional capital

# Intraday constraints
EOD_EXIT_HOUR      = 15           # Force exit at 15:14
EOD_EXIT_MINUTE    = 14
MAX_ADVERSE_BRICKS = 5            # Stop-loss: exit after 5 adverse bricks
MAX_HOLD_BRICKS    = 60           # Max hold time in bricks (prevents stuck trades)

FEATURE_COLS = [
    "velocity", "wick_pressure", "relative_strength",
    "brick_size", "duration_seconds", "direction",
]

META_COLS = [
    "brain1_prob", "velocity", "wick_pressure", "relative_strength",
]

# Default test window (can be overridden via CLI)
DEFAULT_START_YEAR = 2025
DEFAULT_END_YEAR   = 2026   # inclusive


# =============================================================================
# TRADE DATA CLASS
# =============================================================================
@dataclass
class Trade:
    trade_id: int
    symbol: str
    sector: str
    side: str                     # LONG or SHORT
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    bricks_held: int = 0
    favorable_bricks: int = 0
    adverse_bricks: int = 0
    gross_pnl_pct: float = 0.0
    cost_pct: float = REALITY_TAX
    net_pnl_pct: float = 0.0
    exit_reason: str = ""


# =============================================================================
# DATA LOADING
# =============================================================================
def load_test_data(start_year: int = DEFAULT_START_YEAR,
                   end_year: int = DEFAULT_END_YEAR) -> pd.DataFrame:
    """Load feature-enriched data for the given test window."""
    if not config.FEATURES_DIR.exists():
        logger.error("Features dir missing. Run: python main.py features")
        sys.exit(1)

    frames = []
    for sector_dir in config.FEATURES_DIR.iterdir():
        if not sector_dir.is_dir():
            continue
        for pf in sorted(sector_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(pf)
                df["_sector"] = sector_dir.name
                df["_symbol"] = pf.stem
                frames.append(df)
            except Exception as e:
                logger.warning(f"Skip {pf}: {e}")

    if not frames:
        logger.error("No feature files found.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("brick_timestamp").reset_index(drop=True)

    # Filter to test window
    test_start = pd.Timestamp(f"{start_year}-01-01", tz="Asia/Kolkata")
    test_end   = pd.Timestamp(f"{end_year + 1}-01-01", tz="Asia/Kolkata")
    mask = (combined["brick_timestamp"] >= test_start) & (combined["brick_timestamp"] < test_end)
    test = combined[mask].reset_index(drop=True)

    # Add trading date column for day-boundary logic
    test["_trade_date"] = test["brick_timestamp"].dt.date

    logger.info(f"Test data loaded: {len(test):,} bricks from "
                f"{len(test['_symbol'].unique())} stocks across "
                f"{test['_trade_date'].nunique()} trading days  "
                f"[{start_year}-{end_year}]")
    return test


def load_models():
    """Load trained Brain 1 and Brain 2 models."""
    b1_path = config.BRAIN1_MODEL_PATH
    b2_path = config.BRAIN2_MODEL_PATH

    if not b1_path.exists() or not b2_path.exists():
        logger.error("Trained models not found. Run: python main.py train")
        sys.exit(1)

    b1 = xgb.XGBClassifier()
    b1.load_model(str(b1_path))

    b2 = xgb.XGBRegressor()
    b2.load_model(str(b2_path))

    logger.info("Models loaded: Brain1 (Direction) + Brain2 (Conviction)")
    return b1, b2


# =============================================================================
# PREDICTION ENGINE
# =============================================================================
def generate_signals(df: pd.DataFrame, brain1, brain2) -> pd.DataFrame:
    """Run both models on the test data and attach probabilities + conviction."""
    X = df[FEATURE_COLS].fillna(0)

    # Brain 1: Direction probability (probability of UP)
    prob_up = brain1.predict_proba(X)[:, 1]
    df = df.copy()
    df["brain1_prob"] = prob_up
    df["brain1_signal"] = np.where(prob_up > 0.5, "LONG", "SHORT")

    # Brain 2: Conviction score (0-100)
    X_meta = df[META_COLS].fillna(0)
    df["brain2_conviction"] = brain2.predict(X_meta).clip(0, 100)

    long_count = (df["brain1_signal"] == "LONG").sum()
    short_count = (df["brain1_signal"] == "SHORT").sum()
    logger.info(f"Signals generated: LONG={long_count:,}  SHORT={short_count:,}  "
                f"Avg Conviction={df['brain2_conviction'].mean():.1f}")
    return df


# =============================================================================
# SOFT VETO CHECK
# =============================================================================
def passes_soft_veto(signal: str, rel_strength: float) -> bool:
    """
    Soft Veto Rule (sector alignment):
      - LONG  + rel_strength < -0.5 (strongly weak sector)  -> VETOED
      - SHORT + rel_strength > +0.5 (strongly strong sector) -> VETOED
    Loosened threshold to allow more trades while still filtering garbage.
    """
    if signal == "LONG" and rel_strength < -0.5:
        return False
    if signal == "SHORT" and rel_strength > 0.5:
        return False
    return True


# =============================================================================
# CLOSE POSITION HELPER
# =============================================================================
def close_position(position: Trade, price: float, ts: pd.Timestamp,
                   reason: str) -> Trade:
    """Calculate PnL and close a position."""
    position.exit_time = ts
    position.exit_price = price
    position.exit_reason = reason

    if position.side == "LONG":
        position.gross_pnl_pct = (price - position.entry_price) / position.entry_price
    else:  # SHORT
        position.gross_pnl_pct = (position.entry_price - price) / position.entry_price

    # Reality Tax: deduct 10 bps
    position.net_pnl_pct = position.gross_pnl_pct - REALITY_TAX
    return position


# =============================================================================
# INTRADAY SIMULATION ENGINE
# =============================================================================
def run_simulation(df: pd.DataFrame) -> List[Trade]:
    """
    Event-driven INTRADAY backtesting loop.
    - Processes each stock per day independently
    - Forces exit at 15:25 (no overnight positions)
    - Re-evaluates conviction on every brick
    - Implements stop-loss and max-hold limits
    """
    all_trades: List[Trade] = []
    trade_counter = 0
    vetoed_count = 0
    eod_exits = 0

    symbols = df["_symbol"].unique()
    trading_days = sorted(df["_trade_date"].unique())

    logger.info(f"Simulating {len(symbols)} stocks across {len(trading_days)} days...")

    for sym_idx, symbol in enumerate(symbols):
        sym_df = df[df["_symbol"] == symbol]
        if len(sym_df) < 10:
            continue

        sector = sym_df["_sector"].iloc[0]
        sym_days = sorted(sym_df["_trade_date"].unique())

        for day in sym_days:
            day_df = sym_df[sym_df["_trade_date"] == day].reset_index(drop=True)
            if len(day_df) < 3:
                continue

            position: Optional[Trade] = None

            for i in range(len(day_df)):
                row = day_df.iloc[i]
                prob = row["brain1_prob"]
                signal = row["brain1_signal"]
                conviction = row["brain2_conviction"]
                rel_str = row["relative_strength"] if pd.notna(row["relative_strength"]) else 0.0
                price = row["brick_close"]
                ts = row["brick_timestamp"]
                brick_dir = row["direction"]

                # ── Force EOD exit at 15:25 ──────────────────────────
                if ts.hour >= EOD_EXIT_HOUR and ts.minute >= EOD_EXIT_MINUTE:
                    if position is not None:
                        position = close_position(position, price, ts, "EOD_EXIT")
                        all_trades.append(position)
                        eod_exits += 1
                        position = None
                    continue  # No new entries after 15:25

                # ── If we have an open position, check EXIT rules ────
                if position is not None:
                    exit_reason = None
                    position.bricks_held += 1

                    # Track favorable vs adverse bricks
                    if position.side == "LONG":
                        if brick_dir > 0:
                            position.favorable_bricks += 1
                            position.adverse_bricks = 0  # reset consecutive counter
                        else:
                            position.adverse_bricks += 1
                    else:  # SHORT
                        if brick_dir < 0:
                            position.favorable_bricks += 1
                            position.adverse_bricks = 0
                        else:
                            position.adverse_bricks += 1

                    # Exit Rule 1: Conviction drops below threshold
                    if conviction < EXIT_CONV_THRESH:
                        exit_reason = "LOW_CONVICTION"

                    # Exit Rule 2: Trend reversal (strong opposite signal)
                    if position.side == "LONG" and signal == "SHORT" and prob < 0.40:
                        exit_reason = "TREND_REVERSAL"
                    elif position.side == "SHORT" and signal == "LONG" and prob > 0.60:
                        exit_reason = "TREND_REVERSAL"

                    # Exit Rule 3: Stop-loss (consecutive adverse bricks)
                    if position.adverse_bricks >= MAX_ADVERSE_BRICKS:
                        exit_reason = "STOP_LOSS"

                    # Exit Rule 4: Max hold time exceeded
                    if position.bricks_held >= MAX_HOLD_BRICKS:
                        exit_reason = "MAX_HOLD"

                    if exit_reason:
                        position = close_position(position, price, ts, exit_reason)
                        all_trades.append(position)
                        position = None
                    continue  # Don't open new position on same brick as exit

                # ── No open position, check ENTRY rules ──────────────
                # Don't enter in last 30 bricks worth of time near EOD
                if ts.hour >= 15 and ts.minute >= 0:
                    continue

                # For LONG: prob > threshold; For SHORT: (1-prob) > threshold
                if signal == "LONG":
                    entry_prob_ok = prob > ENTRY_PROB_THRESH
                elif signal == "SHORT":
                    entry_prob_ok = (1 - prob) > ENTRY_PROB_THRESH  # prob < 0.45
                else:
                    continue

                # Entry Gate 1: High confidence only
                if not entry_prob_ok or conviction <= ENTRY_CONV_THRESH:
                    continue

                # Entry Gate 2: Soft Veto (sector alignment)
                if not passes_soft_veto(signal, rel_str):
                    vetoed_count += 1
                    continue

                # OPEN new position
                trade_counter += 1
                position = Trade(
                    trade_id=trade_counter,
                    symbol=symbol,
                    sector=sector,
                    side=signal,
                    entry_time=ts,
                    entry_price=price,
                    bricks_held=0,
                )

            # Safety: close any position still open at end of day data
            if position is not None:
                last_row = day_df.iloc[-1]
                position = close_position(
                    position, last_row["brick_close"],
                    last_row["brick_timestamp"], "EOD_EXIT"
                )
                all_trades.append(position)
                eod_exits += 1
                position = None

        # Progress log
        if (sym_idx + 1) % 20 == 0:
            logger.info(f"  Progress: {sym_idx + 1}/{len(symbols)} stocks | "
                        f"{len(all_trades)} trades so far")

    logger.info(f"Simulation complete: {len(all_trades)} trades | "
                f"{vetoed_count} vetoed | {eod_exits} EOD exits")
    return all_trades


# =============================================================================
# PERFORMANCE REPORT
# =============================================================================
def generate_report(trades: List[Trade]) -> dict:
    """Generate the boardroom-quality performance report."""
    if not trades:
        logger.warning("No trades executed. Cannot generate report.")
        return {}

    net_pnls = [t.net_pnl_pct for t in trades]

    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]

    # Core metrics
    total_trades = len(trades)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = np.mean(wins) * 100 if wins else 0
    avg_loss = np.mean(losses) * 100 if losses else 0
    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else float("inf")

    # Net ROI — simple sum (not compounded, to avoid overflow with 28K+ trades)
    simple_roi = sum(net_pnls) * 100
    avg_pnl_per_trade = np.mean(net_pnls) * 100

    # Equity curve using additive PnL on fixed capital (realistic for intraday)
    # Each trade risks a fixed fraction of starting capital
    pnl_rs = np.array(net_pnls) * STARTING_CAPITAL * 0.02  # 2% position size
    cumulative_pnl = np.cumsum(pnl_rs)
    equity = STARTING_CAPITAL + cumulative_pnl

    # Max Drawdown on equity curve
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_drawdown = abs(drawdowns.min()) * 100

    # Trade duration stats
    durations = [t.bricks_held for t in trades]
    avg_duration = np.mean(durations) if durations else 0

    # Side breakdown
    long_trades = [t for t in trades if t.side == "LONG"]
    short_trades = [t for t in trades if t.side == "SHORT"]
    long_wins = sum(1 for t in long_trades if t.net_pnl_pct > 0)
    short_wins = sum(1 for t in short_trades if t.net_pnl_pct > 0)

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Sector breakdown
    sector_pnl = {}
    for t in trades:
        if t.sector not in sector_pnl:
            sector_pnl[t.sector] = []
        sector_pnl[t.sector].append(t.net_pnl_pct)

    # Daily P&L
    daily_pnl = {}
    for t in trades:
        day = str(t.entry_time.date()) if t.entry_time else "unknown"
        daily_pnl.setdefault(day, []).append(t.net_pnl_pct)
    profitable_days = sum(1 for pnls in daily_pnl.values() if sum(pnls) > 0)
    total_days = len(daily_pnl)

    report = {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_roi": simple_roi,
        "max_drawdown": max_drawdown,
        "period": "configured",
    }

    # ── Print Boardroom Report ──────────────────────────────────────────
    print("\n")
    print("=" * 72)
    print("   THE TRUTH TELLER v2 -- Backtest Performance Report")
    print(f"   Institutional Fortress | Test Period: {report.get('period', 'N/A')}")
    print("=" * 72)

    print(f"\n   CONFIGURATION")
    print(f"   {'Reality Tax:':<30} {REALITY_TAX*100:.1f}% per trade (10 bps)")
    print(f"   {'Entry Threshold:':<30} Prob > {ENTRY_PROB_THRESH} AND Conv > {ENTRY_CONV_THRESH}")
    print(f"   {'Exit Threshold:':<30} Conv < {EXIT_CONV_THRESH} OR Reversal OR StopLoss")
    print(f"   {'Stop-Loss:':<30} {MAX_ADVERSE_BRICKS} consecutive adverse bricks")
    print(f"   {'Max Hold:':<30} {MAX_HOLD_BRICKS} bricks")
    print(f"   {'Intraday Only:':<30} Force exit at {EOD_EXIT_HOUR}:{EOD_EXIT_MINUTE:02d}")
    print(f"   {'Soft Veto:':<30} Sector Alignment (threshold +-0.5)")

    print(f"\n   {'='*68}")
    print(f"   HEADLINE NUMBERS")
    print(f"   {'='*68}")
    print(f"   {'Total Trades:':<30} {total_trades:,}")
    print(f"   {'Win Rate:':<30} {win_rate:.2f}%")
    print(f"   {'Profit Factor:':<30} {profit_factor:.2f}")
    print(f"   {'Net ROI (Simple Sum):':<30} {simple_roi:+.2f}%")
    print(f"   {'Avg PnL/Trade:':<30} {avg_pnl_per_trade:+.4f}%")
    print(f"   {'Max Drawdown:':<30} {max_drawdown:.2f}%")
    print(f"   {'Profitable Days:':<30} {profitable_days}/{total_days} ({profitable_days/total_days*100:.1f}%)" if total_days > 0 else "")

    print(f"\n   {'='*68}")
    print(f"   DETAILED METRICS")
    print(f"   {'='*68}")
    print(f"   {'Avg Win:':<30} {avg_win:+.4f}%")
    print(f"   {'Avg Loss:':<30} {avg_loss:+.4f}%")
    print(f"   {'Avg Duration:':<30} {avg_duration:.1f} bricks")
    print(f"   {'Starting Capital:':<30} Rs {STARTING_CAPITAL:,.0f}")
    print(f"   {'Final Capital:':<30} Rs {equity[-1]:,.0f}")
    print(f"   {'Peak Capital:':<30} Rs {equity.max():,.0f}")
    print(f"   {'Lowest Capital:':<30} Rs {equity.min():,.0f}")

    print(f"\n   {'='*68}")
    print(f"   SIDE BREAKDOWN")
    print(f"   {'='*68}")
    long_wr = (long_wins / len(long_trades) * 100) if long_trades else 0
    short_wr = (short_wins / len(short_trades) * 100) if short_trades else 0
    print(f"   {'LONG Trades:':<30} {len(long_trades):,} (Win Rate: {long_wr:.1f}%)")
    print(f"   {'SHORT Trades:':<30} {len(short_trades):,} (Win Rate: {short_wr:.1f}%)")

    print(f"\n   {'='*68}")
    print(f"   EXIT REASONS")
    print(f"   {'='*68}")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        pct = count / total_trades * 100
        print(f"   {reason + ':':<30} {count:,} ({pct:.1f}%)")

    print(f"\n   {'='*68}")
    print(f"   SECTOR PERFORMANCE")
    print(f"   {'='*68}")
    print(f"   {'Sector':<20} {'Trades':>8} {'Win%':>8} {'Avg PnL':>10} {'Total PnL':>10}")
    print(f"   {'-'*58}")
    for sector in sorted(sector_pnl.keys()):
        pnls = sector_pnl[sector]
        s_wins = sum(1 for p in pnls if p > 0)
        s_wr = s_wins / len(pnls) * 100 if pnls else 0
        s_avg = np.mean(pnls) * 100
        s_total = sum(pnls) * 100
        print(f"   {sector:<20} {len(pnls):>8} {s_wr:>7.1f}% {s_avg:>+9.3f}% {s_total:>+9.2f}%")

    print(f"\n{'=' * 72}")
    print(f"   Trade log: storage/logs/backtest_trades.csv")
    print(f"{'=' * 72}\n")

    return report


def save_trade_log(trades: List[Trade]):
    """Save detailed trade log to CSV."""
    records = []
    for t in trades:
        records.append({
            "trade_id": t.trade_id,
            "symbol": t.symbol,
            "sector": t.sector,
            "side": t.side,
            "entry_time": t.entry_time,
            "entry_price": round(t.entry_price, 2),
            "exit_time": t.exit_time,
            "exit_price": round(t.exit_price, 2) if t.exit_price else None,
            "bricks_held": t.bricks_held,
            "favorable_bricks": t.favorable_bricks,
            "adverse_bricks": t.adverse_bricks,
            "gross_pnl_pct": round(t.gross_pnl_pct * 100, 4),
            "cost_pct": round(t.cost_pct * 100, 4),
            "net_pnl_pct": round(t.net_pnl_pct * 100, 4),
            "exit_reason": t.exit_reason,
        })

    out = Path(config.LOGS_DIR / "backtest_trades.csv")
    pd.DataFrame(records).to_csv(out, index=False)
    logger.info(f"Trade log saved: {out} ({len(records)} trades)")


# =============================================================================
# ORCHESTRATOR
# =============================================================================
def run_backtester():
    """Entry point — parses --start / --end from sys.argv."""
    import argparse
    parser = argparse.ArgumentParser(description="Truth Teller v2 Backtest")
    parser.add_argument("--start", type=int, default=DEFAULT_START_YEAR,
                        help=f"Start year of test window (default: {DEFAULT_START_YEAR})")
    parser.add_argument("--end", type=int, default=DEFAULT_END_YEAR,
                        help=f"End year of test window inclusive (default: {DEFAULT_END_YEAR})")
    # Only parse the args after 'backtest'
    args, _ = parser.parse_known_args(sys.argv[2:])

    start_year = args.start
    end_year   = args.end
    period_label = f"{start_year}-{end_year}"

    logger.info("=" * 72)
    logger.info("THE TRUTH TELLER v2 -- Intraday Backtest Engine")
    logger.info(f"Test Period: {period_label}")
    logger.info(f"Reality Tax: {REALITY_TAX*100:.1f}% | Entry: Prob>{ENTRY_PROB_THRESH} Conv>{ENTRY_CONV_THRESH}")
    logger.info(f"Intraday Only | StopLoss: {MAX_ADVERSE_BRICKS} bricks | MaxHold: {MAX_HOLD_BRICKS} bricks")
    logger.info("=" * 72)

    # Phase 1: Load data & models
    test_data = load_test_data(start_year, end_year)
    brain1, brain2 = load_models()

    # Phase 2: Generate signals
    test_data = generate_signals(test_data, brain1, brain2)

    # Phase 3: Run simulation
    trades = run_simulation(test_data)

    if not trades:
        logger.warning("No trades generated. Check entry thresholds or data.")
        return

    # Phase 4: Save trade log
    save_trade_log(trades)

    # Phase 5: Generate report
    report = generate_report(trades)
    report["period"] = period_label

    logger.info("TRUTH TELLER v2 COMPLETE")
    return report


if __name__ == "__main__":
    run_backtester()
