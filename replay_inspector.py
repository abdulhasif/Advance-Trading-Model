"""
replay_inspector.py - Brick-by-Brick State Machine Replay Tool
==============================================================
Loads a specific stock on a specific date, pre-warms the rolling features
using the previous 5 trading days, runs the frozen XGBoost model, and
prints a chronological terminal log of every Renko brick with the full
state machine internals exposed.

Usage:
    python replay_inspector.py --symbol RELIANCE --date 2026-02-26
    python replay_inspector.py --symbol INDUSINDBK --date 2026-02-27 --verbose

Requirements:
    - Run from the project root: C:\\Trading Platform\\Advance Trading Model
    - python main.py features  (feature parquet files must exist)
    - python main.py train     (brain1_direction.json must exist)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xgboost as xgb

# ── Project root path setup ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config

# ── Constants — must match backtester.py + paper_trader.py exactly ───────────
ENTRY_PROB_THRESH      = 0.65
ENTRY_CONV_THRESH      = 5.0
ENTRY_RS_THRESHOLD     = 1.0
MAX_ENTRY_WICK         = 0.40
HYST_LONG_SELL_FLOOR   = 0.40    # LONG: hold unless prob < 0.40
HYST_SHORT_SELL_CEIL   = 0.60    # SHORT: hold unless prob > 0.60
STRUCTURAL_REVERSAL_BRICKS = 2
MAX_ADVERSE_BRICKS     = 5
MAX_HOLD_BRICKS        = 60
EOD_EXIT_HOUR          = 15
EOD_EXIT_MINUTE        = 14
NO_NEW_ENTRY_HOUR      = 15
NO_NEW_ENTRY_MIN       = 0
MIN_CONSECUTIVE_BRICKS = 2
MAX_LOSSES_PER_DAY     = 1

FEATURE_COLS = [
    "velocity", "wick_pressure", "relative_strength",
    "brick_size", "duration_seconds",
    "consecutive_same_dir", "brick_oscillation_rate",
    "fracdiff_price", "hurst", "is_trending_regime",
    "velocity_long", "trend_slope", "rolling_range_pct", "momentum_acceleration",
]
META_COLS = ["brain1_prob", "velocity", "wick_pressure", "relative_strength"]

# Force UTF-8 output on Windows so ANSI + Unicode chars don't crash cp1252
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def find_feature_file(symbol: str) -> Path:
    """Search all sector subdirs for the symbol's feature parquet."""
    for sector_dir in config.FEATURES_DIR.iterdir():
        if not sector_dir.is_dir():
            continue
        pf = sector_dir / f"{symbol}.parquet"
        if pf.exists():
            return pf, sector_dir.name
    raise FileNotFoundError(
        f"No feature file found for '{symbol}' in {config.FEATURES_DIR}\n"
        f"Run: python main.py features"
    )


def load_symbol_data(symbol: str, target_date: str, warmup_days: int = 5) -> pd.DataFrame:
    """
    Load feature data for the symbol.
    Returns warmup_days prior trading days + the target_date, sorted chronologically.
    """
    pf, sector = find_feature_file(symbol)
    df = pd.read_parquet(pf)
    df["_symbol"] = symbol
    df["_sector"] = sector

    # Standardize to Naive IST
    ts = pd.to_datetime(df["brick_timestamp"])
    if ts.dt.tz is not None:
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        else:
            ts = ts.dt.tz_localize(None) if hasattr(ts.dt, 'tz_localize') else ts
    else:
        # If already naive, assume it's IST and keep it naive
        ts = ts.dt.tz_localize(None)
    df["brick_timestamp"] = ts
    df = df.sort_values("brick_timestamp").reset_index(drop=True)
    df["_trade_date"] = df["brick_timestamp"].dt.date

    # Get all unique trading days, find the target date
    all_days = sorted(df["_trade_date"].unique())
    target = datetime.strptime(target_date, "%Y-%m-%d").date()

    if target not in all_days:
        available = sorted(d for d in all_days if d.year >= target.year - 1)[-10:]
        raise ValueError(
            f"Date {target_date} not found in feature data for {symbol}.\n"
            f"  Latest available dates: {[str(d) for d in available]}"
        )

    target_idx = all_days.index(target)
    warmup_start_idx = max(0, target_idx - warmup_days)
    days_to_load = all_days[warmup_start_idx : target_idx + 1]

    df_loaded = df[df["_trade_date"].isin(days_to_load)].reset_index(drop=True)
    warmup_bricks = len(df_loaded[df_loaded["_trade_date"] != target])
    target_bricks = len(df_loaded[df_loaded["_trade_date"] == target])
    print(f"{DIM}  Warmup: {len(days_to_load)-1} trading days ({warmup_bricks} bricks){RESET}")
    print(f"{DIM}  Target: {target_date} ({target_bricks} bricks){RESET}")
    return df_loaded, sector


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_models():
    b1_path = config.BRAIN1_MODEL_PATH
    b2_path = config.BRAIN2_MODEL_PATH
    if not b1_path.exists():
        raise FileNotFoundError(f"Brain1 model not found: {b1_path}\nRun: python main.py train")
    if not b2_path.exists():
        raise FileNotFoundError(f"Brain2 model not found: {b2_path}\nRun: python main.py train")
    b1 = xgb.XGBClassifier(); b1.load_model(str(b1_path))
    b2 = xgb.XGBRegressor();  b2.load_model(str(b2_path))
    return b1, b2


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class Position:
    def __init__(self, entry_time, entry_price: float, side: str):
        self.entry_time   = entry_time
        self.entry_price  = entry_price
        self.side         = side           # LONG or SHORT
        self.bricks_held  = 0
        self.adverse_bricks = 0
        self.favorable_bricks = 0

    def pnl_pct(self, price: float) -> float:
        if self.side == "LONG":
            return (price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - price) / self.entry_price * 100

    def update(self, brick_dir: int):
        self.bricks_held += 1
        if self.side == "LONG":
            if brick_dir > 0:
                self.favorable_bricks += 1
                self.adverse_bricks = 0
            else:
                self.adverse_bricks += 1
        else:
            if brick_dir < 0:
                self.favorable_bricks += 1
                self.adverse_bricks = 0
            else:
                self.adverse_bricks += 1


# ═══════════════════════════════════════════════════════════════════════════════
# REPLAY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_replay(df_all: pd.DataFrame, symbol: str, target_date: str,
               b1, b2, verbose: bool = False):
    """
    Step through every brick on the target date, applying the full
    entry/exit state machine and printing the result.
    """
    target = datetime.strptime(target_date, "%Y-%m-%d").date()

    # ── Batch-score the FULL dataframe (warmup + target) at once ─────────────
    X_all = df_all[FEATURE_COLS].fillna(0)
    prob_all = b1.predict_proba(X_all)[:, 1]
    df_all = df_all.copy()
    df_all["brain1_prob"] = prob_all

    X_meta = df_all[META_COLS].fillna(0)
    df_all["brain2_conv"] = b2.predict(X_meta).clip(0, 100)

    # ── Slice to target date only ──────────────────────────────────────────────
    target_df = df_all[df_all["_trade_date"] == target].reset_index(drop=True)

    if target_df.empty:
        print(f"{RED}No bricks found for {symbol} on {target_date}{RESET}")
        return

    # ── Header ────────────────────────────────────────────────────────────────
    price_range = f"Rs {target_df['brick_close'].min():.2f} → Rs {target_df['brick_close'].max():.2f}"
    print()
    print(f"{BOLD}{'='*112}{RESET}")
    print(f"{BOLD}  REPLAY INSPECTOR  |  {symbol}  |  {target_date}  |  {len(target_df)} bricks  |  {price_range}{RESET}")
    print(f"{BOLD}{'='*112}{RESET}")
    header = (
        f"{'TIME':<8}  {'PRICE':>9}  {'DIR':>4}  {'PROB':>6}  {'CONV':>6}  "
        f"{'RS':>6}  {'ADV':>4}  {'HELD':>5}  {'STATE / EVENT':<40}"
    )
    print(f"{DIM}{header}{RESET}")
    print(f"{DIM}{'-'*112}{RESET}")

    # ── State variables ────────────────────────────────────────────────────────
    position: Position | None = None
    daily_losses = 0
    trades = []

    for i, row in target_df.iterrows():
        ts         = row["brick_timestamp"]
        price      = row["brick_close"]
        brick_dir  = int(row["direction"])
        prob       = float(row["brain1_prob"])
        conv       = float(row["brain2_conv"])
        rel_str    = float(row.get("relative_strength", 0))
        wick_p     = float(row.get("wick_pressure", 0))
        consec     = int(row.get("consecutive_same_dir", 0))

        signal     = "LONG" if prob > 0.5 else "SHORT"
        dir_arrow  = f"{GREEN}UP{RESET}" if brick_dir > 0 else f"{RED}DN{RESET}"
        time_str   = ts.strftime("%H:%M")

        is_eod     = (ts.hour > EOD_EXIT_HOUR) or (ts.hour == EOD_EXIT_HOUR and ts.minute >= EOD_EXIT_MINUTE)
        no_entry   = (ts.hour > NO_NEW_ENTRY_HOUR) or (ts.hour == NO_NEW_ENTRY_HOUR and ts.minute >= NO_NEW_ENTRY_MIN)

        state      = ""
        color      = RESET

        # ── If in a position, check exits first ────────────────────────────────
        if position:
            position.update(brick_dir)
            adv = position.adverse_bricks
            held = position.bricks_held
            current_pnl = position.pnl_pct(price)

            exit_reason = None

            # EOD forced exit
            if is_eod:
                exit_reason = "EOD_EXIT"

            # Hysteresis dead-zone trend reversal
            elif position.side == "LONG" and signal == "SHORT" and prob < HYST_LONG_SELL_FLOOR:
                exit_reason = "TREND_REVERSAL"
            elif position.side == "SHORT" and signal == "LONG" and prob > HYST_SHORT_SELL_CEIL:
                exit_reason = "TREND_REVERSAL"

            # 2-brick structural hard stop
            elif adv >= STRUCTURAL_REVERSAL_BRICKS:
                if position.side == "LONG" and brick_dir < 0:
                    exit_reason = "STRUCTURAL_2BRICK_STOP"
                elif position.side == "SHORT" and brick_dir > 0:
                    exit_reason = "STRUCTURAL_2BRICK_STOP"

            # Max adverse bricks stop
            elif adv >= MAX_ADVERSE_BRICKS:
                exit_reason = "MAX_ADVERSE_STOP"

            # Max hold time
            elif held >= MAX_HOLD_BRICKS:
                exit_reason = "MAX_HOLD_TIME"

            if exit_reason:
                pnl_str = f"{current_pnl:+.3f}%"
                pnl_color = GREEN if current_pnl > 0 else RED
                state = f"{pnl_color}<< EXIT [{exit_reason}]  PnL={pnl_str}  Entry=Rs {position.entry_price:.2f}{RESET}"
                color = RED if current_pnl < 0 else GREEN
                if current_pnl < 0:
                    daily_losses += 1
                trades.append({
                    "entry_time": position.entry_time.strftime("%H:%M"),
                    "exit_time": time_str,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "exit_price": price,
                    "pnl_pct": current_pnl,
                    "bricks_held": held,
                    "exit_reason": exit_reason,
                })
                position = None
            else:
                # Holding in dead-zone or favorable trend
                dead_zone = (HYST_LONG_SELL_FLOOR <= prob <= HYST_SHORT_SELL_CEIL)
                if dead_zone and signal != position.side:
                    state = f"{YELLOW}⋯ HOLDING [DEAD-ZONE prob={prob:.3f}]  adv={adv}  pnl={current_pnl:+.2f}%{RESET}"
                else:
                    state = f"{CYAN}>> HOLDING [{position.side}]  adv={adv}  held={held}  pnl={current_pnl:+.2f}%{RESET}"

        # ── No position — check for entry ──────────────────────────────────────
        if position is None:
            if is_eod or no_entry:
                state = state or f"{DIM}  flat [EOD — no entry]{RESET}"

            elif daily_losses >= MAX_LOSSES_PER_DAY:
                state = f"{DIM}  flat [daily loss limit reached]{RESET}"

            else:
                # Gate 1: Prob threshold
                prob_ok = (signal == "LONG"  and prob >= ENTRY_PROB_THRESH) or \
                          (signal == "SHORT" and (1 - prob) >= ENTRY_PROB_THRESH)

                # Gate 2: Conviction
                conv_ok = conv >= ENTRY_CONV_THRESH

                # Gate 3: RS
                rs_ok = (signal == "LONG"  and rel_str >= ENTRY_RS_THRESHOLD) or \
                        (signal == "SHORT" and rel_str <= -ENTRY_RS_THRESHOLD)

                # Gate 4: Wick trap
                wick_ok = wick_p <= MAX_ENTRY_WICK

                # Gate 5: Consecutive bricks
                consec_ok = consec >= MIN_CONSECUTIVE_BRICKS

                if prob_ok and conv_ok and rs_ok and wick_ok and consec_ok and not state:
                    # ENTRY
                    position = Position(ts, price, signal)
                    state = (f"{GREEN if signal == 'LONG' else RED}"
                             f">>> BUY [{signal}] @ Rs {price:.2f}  "
                             f"prob={prob:.3f}  conv={conv:.1f}bps  RS={rel_str:.2f}"
                             f"{RESET}")
                elif not state:
                    # Show which gate blocked
                    fails = []
                    if not prob_ok:  fails.append(f"prob={prob:.3f}<{ENTRY_PROB_THRESH}")
                    if not conv_ok:  fails.append(f"conv={conv:.1f}<{ENTRY_CONV_THRESH}")
                    if not rs_ok:    fails.append(f"RS={rel_str:.2f}")
                    if not wick_ok:  fails.append(f"wick={wick_p:.2f}>{MAX_ENTRY_WICK}")
                    if not consec_ok:fails.append(f"consec={consec}<{MIN_CONSECUTIVE_BRICKS}")
                    state = f"{DIM}  skip [{', '.join(fails)}]{RESET}" if verbose else ""

        # ── Print the row ──────────────────────────────────────────────────────
        if state or verbose:
            adv_str  = f"{position.adverse_bricks}" if position else "-"
            held_str = f"{position.bricks_held}"    if position else "-"
            prob_color = GREEN if prob >= ENTRY_PROB_THRESH else (YELLOW if prob >= 0.50 else RED)
            row_str = (
                f"{time_str:<8}  {price:>9.2f}  {dir_arrow:>5}  "
                f"{prob_color}{prob:>6.3f}{RESET}  {conv:>6.1f}  "
                f"{rel_str:>6.2f}  {adv_str:>4}  {held_str:>5}  {state}"
            )
            print(row_str)

    # ── End-of-day summary ──────────────────────────────────────────────────────
    print(f"{DIM}{'-'*112}{RESET}")
    print(f"\n{BOLD}  TRADE SUMMARY — {symbol} on {target_date}{RESET}")
    if not trades:
        print(f"  {YELLOW}No trades executed.{RESET}")
        print(f"  Check: prob threshold ({ENTRY_PROB_THRESH}), RS gate ({ENTRY_RS_THRESHOLD}), conviction ({ENTRY_CONV_THRESH} bps)")
    else:
        total_pnl = sum(t["pnl_pct"] for t in trades)
        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        print(f"  Total Trades : {len(trades)}")
        print(f"  Win Rate     : {wins}/{len(trades)} ({wins/len(trades)*100:.1f}%)")
        print(f"  Net PnL      : {GREEN if total_pnl > 0 else RED}{total_pnl:+.3f}%{RESET}")
        print()
        print(f"  {'ENTRY':>5}  {'EXIT':>5}  {'SIDE':>5}  {'ENTRY Px':>10}  {'EXIT Px':>10}  {'PnL%':>8}  {'BRICKS':>7}  REASON")
        print(f"  {'-'*80}")
        for t in trades:
            pnl_color = GREEN if t["pnl_pct"] > 0 else RED
            print(
                f"  {t['entry_time']:>5}  {t['exit_time']:>5}  "
                f"{'L' if t['side']=='LONG' else 'S':>5}  "
                f"{t['entry_price']:>10.2f}  {t['exit_price']:>10.2f}  "
                f"{pnl_color}{t['pnl_pct']:>+8.3f}%{RESET}  "
                f"{t['bricks_held']:>7}  {t['exit_reason']}"
            )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Brick-by-Brick Replay Inspector for the XGBoost Trading System"
    )
    parser.add_argument("--symbol", required=True, type=str,
                        help="NSE symbol, e.g. RELIANCE")
    parser.add_argument("--date", required=True, type=str,
                        help="Trading date in YYYY-MM-DD format, e.g. 2026-02-26")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of prior trading days to load for feature warmup (default: 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every brick including skipped ones")
    args = parser.parse_args()

    symbol      = args.symbol.upper().strip()
    target_date = args.date.strip()

    print(f"\n{BOLD}REPLAY INSPECTOR{RESET} — Loading {symbol} | {target_date}")
    print(f"{DIM}  ENTRY_PROB_THRESH={ENTRY_PROB_THRESH}  "
          f"ENTRY_CONV_THRESH={ENTRY_CONV_THRESH}bps  "
          f"RS_THRESHOLD={ENTRY_RS_THRESHOLD}  "
          f"HYSTERESIS=[{HYST_LONG_SELL_FLOOR}–{HYST_SHORT_SELL_CEIL}]  "
          f"STRUCTURAL_STOP={STRUCTURAL_REVERSAL_BRICKS}bricks{RESET}")

    # Load data
    df_all, sector = load_symbol_data(symbol, target_date, warmup_days=args.warmup)
    print(f"{DIM}  Sector: {sector}{RESET}")

    # Load models
    print(f"{DIM}  Loading models from {config.BRAIN1_MODEL_PATH.name} + {config.BRAIN2_MODEL_PATH.name}...{RESET}")
    b1, b2 = load_models()

    # Run replay
    run_replay(df_all, symbol, target_date, b1, b2, verbose=args.verbose)


if __name__ == "__main__":
    main()
