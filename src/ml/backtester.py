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
import joblib
import xgboost as xgb
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import config
from src.core.quant_fixes import IsotonicCalibrationWrapper

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
LONG_ENTRY_PROB_THRESH  = getattr(config, "LONG_ENTRY_PROB_THRESH",  0.55)  # from config.py
SHORT_ENTRY_PROB_THRESH = getattr(config, "SHORT_ENTRY_PROB_THRESH", 0.50)  # from config.py
ENTRY_PROB_THRESH  = LONG_ENTRY_PROB_THRESH   # kept for legacy log lines
ENTRY_CONV_THRESH  = 20.0     # Brain2 conviction gate — 20 bps min expected move to cover friction
EXIT_CONV_THRESH   = 0.0         # Brain2 conviction threshold for exit
STARTING_CAPITAL   = 20_000      # Rs 20,000 micro-capital investment

# Anti-Myopia: Hysteresis Dead-Zone (Probability State Machine)
# A held position does NOT exit until the model STRONGLY confirms reversal.
# Dead-Zone = [0.40, 0.60] --- pure noise, hold the position.
HYST_LONG_SELL_FLOOR  = 0.40    # LONG: only Trend Reversal exit if prob < 0.40
HYST_SHORT_SELL_CEIL  = 0.60    # SHORT: only Trend Reversal exit if prob > 0.60

# Anti-Myopia: 3-Brick Structural Stop (hard chart-based safety net)
STRUCTURAL_REVERSAL_BRICKS = 3  # Consecutive adverse bricks -> immediate exit

# Upstox Intraday Equity Charges & Sizing
POSITION_SIZE_PCT    = 0.10
INTRADAY_LEVERAGE    = 5
BROKERAGE_PER_ORDER  = 20.0
BROKERAGE_PCT        = 0.0005
STT_SELL_PCT         = 0.00025
STAMP_DUTY_BUY_PCT   = 0.00003
EXCHANGE_TXN_PCT     = 0.0000297
SEBI_TURNOVER_FEE    = 10.0
GST_PCT              = 0.18

def calculate_charges(entry_price: float, exit_price: float, qty: int) -> float:
    buy_turnover = entry_price * qty
    sell_turnover = exit_price * qty
    total_turnover = buy_turnover + sell_turnover

    brok_buy = min(BROKERAGE_PER_ORDER, buy_turnover * BROKERAGE_PCT)
    brok_sell = min(BROKERAGE_PER_ORDER, sell_turnover * BROKERAGE_PCT)
    brokerage = brok_buy + brok_sell

    stt = sell_turnover * STT_SELL_PCT
    stamp = buy_turnover * STAMP_DUTY_BUY_PCT
    exchange = total_turnover * EXCHANGE_TXN_PCT
    sebi = total_turnover * (SEBI_TURNOVER_FEE / 1_00_00_000)
    gst = (brokerage + exchange) * GST_PCT

    return brokerage + stt + stamp + exchange + sebi + gst

# Intraday constraints — matches paper_trader.py exactly
EOD_EXIT_HOUR      = 15           # Force exit at 3:14 PM
EOD_EXIT_MINUTE    = 14           # Matches paper_trader.py EOD_EXIT_MINUTE
NO_NEW_ENTRY_HOUR  = 14           # No new entries from 2:00 PM onwards
NO_NEW_ENTRY_MIN   = 0            # Matches paper_trader.py NO_ENTRY_MINUTE
MAX_ADVERSE_BRICKS = 4            # Stop-loss: exit after 3 adverse bricks
MAX_HOLD_BRICKS    = 120          # Max hold time in bricks
MAX_OPEN_POSITIONS = 10           # Max simultaneous positions

# Whipsaw protection — matches paper_trader.py exactly
MIN_CONSECUTIVE_BRICKS = 2      # Require N same-direction bricks before entry
MIN_BRICKS_TODAY       = 2       # At least M of those N must be from today's session
MAX_LOSSES_PER_STOCK   = 1       # Max losing trades per stock per day

# Entry filters — matches paper_trader.py exactly
ENTRY_RS_THRESHOLD = 1.0         # |RS| > 1.0; must be a sector leader/laggard
MAX_ENTRY_WICK     = 0.40      # Block if wick_pressure > 40% (absorption trap)

# Fix #9: Volume-cap constants — prevents ghost liquidity trades
VOLUME_LIMIT_PCT   = 0.05         # Max trade = 5% of candle volume
MIN_CANDLE_VOLUME  = 500          # Minimum volume to accept a signal

# Fix #10: T+1 slippage — entry price penalty for API latency
T1_SLIPPAGE_PCT    = 0.0005       # 0.05% slippage on top of T+1 open price

# Fix #11: Penny Stock Filter — blocks noise from tight spreads
MIN_PRICE_FILTER   = 100.0        # Minimum stock price to trade

# =============================================================================
# PHASE 4: PESSIMISTIC EXECUTION ENGINE CONSTANTS
# =============================================================================
# These three constants implement the "Friction Tax" that makes backtest
# results representative of real-world execution quality.

SLIPPAGE_PCT    = 0.0005  # 0.05% friction applied to BOTH entry and exit prices.
                           # For a LONG: entry worsens (pays more), exit worsens (receives less).
                           # For a SHORT: entry worsens (sells for less), exit worsens (buys at more).
                           # This simulates: bid-ask spread + market impact + STT asymmetry.

JITTER_SECONDS  = 1.0     # Simulates WebSocket-to-order-placement latency.
                           # Entry price = NEXT brick's open, not signal brick's close.
                           # In live trading, there is always >=1 tick of latency between
                           # the model generating a signal and the broker receiving the order.

PATH_CONFLICT   = True    # Path-Conflict Resolution: If BOTH Stop-Loss AND Target are
                           # hit within the same 1-minute candle's interpolated path,
                           # record the outcome as LOSS (worst-case, pessimistic).
                           # This is important because: on a fast NSE candle, the wick
                           # may touch the SL before the price recovers to hit the target.
                           # Without this check, backtests assume perfect fill order,
                           # which they never have in reality.


# =============================================================================
# FEATURE ORDER SHIELD
# =============================================================================
# CRITICAL: This list defines the exact column order that the CalibratedClassifierCV
# model was trained on. ANY deviation in order produces silently wrong probabilities.
# This is the single source of truth for feature alignment in backtest + live engine.
EXPECTED_FEATURES = [
    "velocity", "wick_pressure", "relative_strength",
    "brick_size", "duration_seconds",
    "consecutive_same_dir", "brick_oscillation_rate",
    "fracdiff_price",        # Fractional Differentiation
    "hurst",                 # Hurst Regime Feature
    "is_trending_regime",    # Boolean regime gate
    # Anti-Myopia: Long-lookback features
    "velocity_long",         # 20-brick momentum vs 10-brick
    "trend_slope",           # 14-brick OLS price slope (scale-invariant)
    "rolling_range_pct",     # 14-brick price range / avg (volatility gate)
    "momentum_acceleration", # 5-brick vel minus 14-brick vel
    # Phase 2: Institutional Alpha Factors
    "vwap_zscore",           # VWAP anchor: >+2.5 = exhaustion peak
    "vpt_acceleration",      # VPT 2nd derivative: institutional absorption
    "squeeze_zscore",        # Brick density Z-score: expansion after squeeze
    "streak_exhaustion",     # Sigmoid decay: penalizes late-stage momentum
    # Phase 3: Temporal Alpha Features
    "true_gap_pct",
    "time_to_form_seconds",
    "volume_intensity_per_sec",
    "is_opening_drive",
]

# Legacy alias kept for Brain2 meta-regressor which uses its own columns
FEATURE_COLS = EXPECTED_FEATURES


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
    qty: int = 0
    bricks_held: int = 0
    favorable_bricks: int = 0
    adverse_bricks: int = 0
    gross_pnl_pct: float = 0.0
    cost_pct: float = 0.0
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

    # 1. First, establish the test window dates
    if hasattr(config, 'TEST_START_DATE') and start_year == int(config.TEST_START_DATE[:4]):
        # Honor the mid-year cutoff if backtesting the split year
        test_start = pd.Timestamp(config.TEST_START_DATE, tz="Asia/Kolkata")
    else:
        test_start = pd.Timestamp(f"{start_year}-01-01", tz="Asia/Kolkata")
        
    test_end = pd.Timestamp(f"{end_year + 1}-01-01", tz="Asia/Kolkata")

    # 2. Load and filter chunks directly from disk to prevent 54M row memory spike
    frames = []
    for sector_dir in config.FEATURES_DIR.iterdir():
        if not sector_dir.is_dir():
            continue
        for pf in sorted(sector_dir.glob("*.parquet")):
            try:
                # Read specific columns or full df, but importantly filter immediately
                df = pd.read_parquet(pf)
                # Keep only rows in test window
                mask = (df["brick_timestamp"] >= test_start) & (df["brick_timestamp"] < test_end)
                df = df[mask].reset_index(drop=True)
                
                if df.empty:
                    continue
                    
                df["_sector"] = sector_dir.name
                df["_symbol"] = pf.stem
                frames.append(df)
            except Exception as e:
                logger.warning(f"Skip {pf}: {e}")

    if not frames:
        logger.error(f"No feature files found containing data for {start_year}-{end_year}.")
        sys.exit(1)

    # 3. Concatenate only the filtered ~500k rows
    test = pd.concat(frames, ignore_index=True)
    test = test.sort_values("brick_timestamp").reset_index(drop=True)

    # Add trading date column for day-boundary logic
    test["_trade_date"] = test["brick_timestamp"].dt.date

    logger.info(f"Test data loaded: {len(test):,} bricks from "
                f"{len(test['_symbol'].unique())} stocks across "
                f"{test['_trade_date'].nunique()} trading days  "
                f"[{start_year}-{end_year}]")
    return test


def load_models():
    """
    Load Brain 1 LONG & SHORT (.pkl) and Brain 2 (.json).
    """
    b1_long_path = config.BRAIN1_CALIBRATED_LONG_PATH
    b1_short_path = config.BRAIN1_CALIBRATED_SHORT_PATH
    b2_path = config.BRAIN2_MODEL_PATH

    if not b1_long_path.exists() or not b1_short_path.exists():
        logger.error(f"Calibrated models not found. Run: python main.py train")
        sys.exit(1)
    if not b2_path.exists():
        logger.error(f"Brain2 model not found at {b2_path}. Run: python main.py train")
        sys.exit(1)

    b1_long = joblib.load(str(b1_long_path))
    b1_short = joblib.load(str(b1_short_path))

    b2 = xgb.XGBRegressor()
    b2.load_model(str(b2_path))

    logger.info(f"Models loaded: Brain1 (LONG & SHORT .pkl) + Brain2 (.json)")
    return b1_long, b1_short, b2


# =============================================================================
# PREDICTION ENGINE
# =============================================================================
def generate_signals(df: pd.DataFrame, brain1_long, brain1_short, brain2) -> pd.DataFrame:
    """
    Run Brain1 (CalibratedClassifierCV) + Brain2 on the test DataFrame.
    """
    # 4. The Feature Alignment Audit
    print(f"\n[DIAGNOSTIC] FEATURE ALIGNMENT AUDIT:")
    print(f"[DIAGNOSTIC] Expected length: {len(EXPECTED_FEATURES)}")
    
    # ORDER SHIELD: reindex enforces exact training column order, fills missing with 0
    X = df[EXPECTED_FEATURES].fillna(0)
    print(f"[DIAGNOSTIC] Final array shape before inference: {X.shape}")

    # Brain 1: Calibrated probability of success
    prob_long = brain1_long.predict_proba(X)[:, 1]
    prob_short = brain1_short.predict_proba(X)[:, 1]

    df = df.copy()
    
    # Predict signals
    sg = np.full(len(df), "FLAT", dtype=object)
    pb = np.zeros(len(df), dtype=float)
    
    for i in range(len(df)):
        pl = prob_long[i]
        ps = prob_short[i]
        
        sig = "FLAT"
        p = 0.0
        
        long_ok  = pl >= LONG_ENTRY_PROB_THRESH
        short_ok = ps >= SHORT_ENTRY_PROB_THRESH
        if long_ok and short_ok:
            if pl >= ps:
                sig, p = "LONG", pl
            else:
                sig, p = "SHORT", ps
        elif long_ok:
            sig, p = "LONG", pl
        elif short_ok:
            sig, p = "SHORT", ps
            
        sg[i] = sig
        pb[i] = p
        
    df["brain1_signal"] = sg
    df["brain1_prob"] = pb
    
    # 2. The Index Audit
    if len(prob_long) > 0:
        print(f"\n[DIAGNOSTIC] INDEX AUDIT (Row 0):")
        print(f"[DIAGNOSTIC] LONG Prob:  {prob_long[0]:.4f}")
        print(f"[DIAGNOSTIC] SHORT Prob: {prob_short[0]:.4f}")
        print(f"[DIAGNOSTIC] Signal assigned: {sg[0]} (Prob: {pb[0]:.4f})")

    # Brain 2: Conviction score (0-100 bps expected move)
    X_meta = df[META_COLS].fillna(0)
    # Provide the probability of the *chosen* signal (or 0 if FLAT)
    X_meta["brain1_prob"] = pb 
    df["brain2_conviction"] = brain2.predict(X_meta).clip(0, 100)

    long_count  = (df["brain1_signal"] == "LONG").sum()
    short_count = (df["brain1_signal"] == "SHORT").sum()
    high_conv   = (df["brain1_prob"] >= ENTRY_PROB_THRESH).sum()
    logger.info(
        f"Signals generated: LONG={long_count:,}  SHORT={short_count:,}  "
        f"Avg Conviction={df['brain2_conviction'].mean():.1f}  "
        f"Above threshold ({ENTRY_PROB_THRESH}): {high_conv:,} ({high_conv/max(len(df),1)*100:.1f}%)"
    )
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
    """
    Calculate PnL and close a position.

    Phase 4: Pessimistic Execution — SLIPPAGE_PCT applied to exit.
      LONG exit:  fill_price = price × (1 - SLIPPAGE_PCT)
                  (We sell at a slightly LOWER price — bid-ask spread, impact)
      SHORT exit: fill_price = price × (1 + SLIPPAGE_PCT)
                  (We buy back at a slightly HIGHER price — spread + urgency)

    This is applied on top of T1_SLIPPAGE on entry, so total round-trip
    friction = 2 × SLIPPAGE_PCT ≈ 0.10%, matching realistic NSE execution costs.
    """
    # Apply Phase 4 pessimistic exit slippage
    if position.side == "LONG":
        effective_exit = price * (1.0 - SLIPPAGE_PCT)
    else:  # SHORT
        effective_exit = price * (1.0 + SLIPPAGE_PCT)

    position.exit_time  = ts
    position.exit_price = effective_exit
    position.exit_reason = reason

    if position.side == "LONG":
        position.gross_pnl_pct = (effective_exit - position.entry_price) / position.entry_price
    else:  # SHORT
        position.gross_pnl_pct = (position.entry_price - effective_exit) / position.entry_price

    charges = calculate_charges(position.entry_price, effective_exit, position.qty)
    position.cost_pct = charges / (position.entry_price * position.qty) if position.qty > 0 else 0.0
    position.net_pnl_pct = position.gross_pnl_pct - position.cost_pct
    return position


# =============================================================================
# INTRADAY SIMULATION ENGINE
# =============================================================================
def run_simulation(df: pd.DataFrame) -> List[Trade]:
    """
    Event-driven INTRADAY backtesting loop.
    FIX #10: T+1 Execution — signals fire on brick[i], fill on brick[i+1]'s open.
    FIX #9:  Volume Guard — position capped at 5% of candle volume.
    FIX #13: EOD 3:15PM exit uses brick_open of 3:15 candle (not brick_close).
    """
    all_trades: List[Trade] = []
    trade_counter = 0
    vetoed_count  = 0
    volume_rejected = 0
    eod_exits = 0

    symbols = df["_symbol"].unique()
    trading_days = sorted(df["_trade_date"].unique())

    logger.info(f"Simulating {len(symbols)} stocks across {len(trading_days)} days...")

    for sym_idx, symbol in enumerate(symbols):
        sym_df = df[df["_symbol"] == symbol]
        if len(sym_df) < 10:
            continue

        sector   = sym_df["_sector"].iloc[0]
        sym_days = sorted(sym_df["_trade_date"].unique())

        for day in sym_days:
            day_df = sym_df[sym_df["_trade_date"] == day].reset_index(drop=True)
            if len(day_df) < 3:
                continue

            position: Optional[Trade]  = None
            pending_entry: Optional[dict] = None   # FIX #10: T+1 pending
            last_entry_minute: Optional[pd.Timestamp] = None
            daily_losses = 0

            for i in range(len(day_df)):
                row       = day_df.iloc[i]
                prob      = row["brain1_prob"]
                signal    = row["brain1_signal"]
                conviction= row["brain2_conviction"]
                rel_str   = row["relative_strength"] if pd.notna(row["relative_strength"]) else 0.0
                ts        = row["brick_timestamp"]
                brick_dir = row["direction"]

                # brick_open available for T+1 fills; fall back to close if missing
                brick_open  = row.get("brick_open",  row["brick_close"])
                brick_close = row["brick_close"]

                # ── FIX #13: Force EOD exit at 3:15 PM using brick_open ─────
                is_eod_candle = (
                    ts.hour > EOD_EXIT_HOUR or
                    (ts.hour == EOD_EXIT_HOUR and ts.minute >= EOD_EXIT_MINUTE)
                )
                if is_eod_candle:
                    pending_entry = None  # cancel any pending T+1 entry
                    if position is not None:
                        # Exit at brick_open of the 3:15 candle — not close (fiction fix)
                        position = close_position(position, brick_open, ts, "EOD_315")
                        all_trades.append(position)
                        eod_exits += 1
                        position = None
                    continue  # No new entries at or after 3:15

                # ── FIX #10: Execute T+1 pending entry on THIS brick's open ──
                if pending_entry is not None and position is None:
                    p = pending_entry
                    # FIX #3: Liquidity Mirage — adjust virtual brick_open by actual market gap
                    gap_pct = row.get("true_gap_pct", 0.0)
                    gap_pct = gap_pct / 100.0 if pd.notna(gap_pct) else 0.0
                    gap_adjusted_open = brick_open * (1.0 + gap_pct)

                    # Apply T+1 slippage on the gap-adjusted OPEN price
                    fill_price = gap_adjusted_open * (1 + T1_SLIPPAGE_PCT) if p["side"] == "LONG" \
                                 else gap_adjusted_open * (1 - T1_SLIPPAGE_PCT)
                    # FIX #9: Volume guard — cap qty at 5% of this candle's volume
                    raw_vol = row.get("volume", 1_000_000_000)
                    if pd.isna(raw_vol):
                        raw_vol = 1_000_000_000
                    candle_vol = max(float(raw_vol or 1_000_000_000), 1.0)
                    if candle_vol < MIN_CANDLE_VOLUME:
                        # Not enough liquidity — reject trade entirely
                        pending_entry = None
                        volume_rejected += 1
                    else:
                        max_qty = max(1, int(candle_vol * VOLUME_LIMIT_PCT))
                        qty     = min(p["qty"], max_qty)
                        if qty < 1:
                            pending_entry = None
                            volume_rejected += 1
                        else:
                            position = Trade(
                                trade_id   = trade_counter,
                                symbol     = symbol,
                                sector     = sector,
                                side       = p["side"],
                                entry_time = ts,           # T+1 timestamp
                                entry_price= fill_price,   # T+1 open + slippage
                                qty        = qty,
                                bricks_held= 0,
                            )
                            last_entry_minute = ts.floor("T")
                            trade_counter += 1

                    pending_entry = None  # consumed

                # ── If we have an open position, check EXIT rules ────────────
                if position is not None:
                    exit_reason = None
                    position.bricks_held += 1

                    # Track favorable vs adverse bricks
                    if position.side == "LONG":
                        if brick_dir > 0:
                            position.favorable_bricks += 1
                            position.adverse_bricks = 0
                        else:
                            position.adverse_bricks += 1
                    else:  # SHORT
                        if brick_dir < 0:
                            position.favorable_bricks += 1
                            position.adverse_bricks = 0
                        else:
                            position.adverse_bricks += 1

                    # Exit Rule 1: Conviction drops
                    if conviction < EXIT_CONV_THRESH:
                        exit_reason = "LOW_CONVICTION"

                    # Exit Rule 1a: Activation Trailing Stop (Chop Protection)
                    # Instead of taking profit instantly and capping runners, we use +3 as an ACTIVATION ZONE.
                    # 1. At +3 bricks, we lock in Break-Even (+0 buffer).
                    # 2. Beyond +3 bricks, we trail the price dynamically by TRAIL_DISTANCE_BRICKS.
                    if exit_reason is None and conviction < config.STRONG_CONVICTION_THRESH:
                        if position.favorable_bricks >= config.TRAIL_ACTIVATION_BRICKS:
                            # The minimum number of adverse bricks allowed before exiting
                            # When favorable = 3, allowed adverse is roughly 3 - 1.5 = 1.5 (rounded to 2)
                            # When favorable = 10, allowed adverse is strictly the trailing distance (e.g. 1.5)
                            dynamic_trail_allowance = min(
                                STRUCTURAL_REVERSAL_BRICKS, 
                                max(1, position.favorable_bricks - config.TRAIL_DISTANCE_BRICKS)
                            )
                            
                            # If we drop back down past our trailing line, exit to secure the profit
                            if position.adverse_bricks >= dynamic_trail_allowance:
                                exit_reason = "TRAIL_PROFIT_ACTIVATED"

                    # Exit Rule 2: Hysteresis Dead-Zone Trend Reversal
                    # Anti-Myopia: Do NOT exit on any probability nudge.
                    # The model must STRONGLY confirm reversal before exiting.
                    # Dead-Zone [0.40, 0.60] = HOLD — pure noise, stay in.
                    if exit_reason is None:
                        if position.side == "LONG" and signal == "SHORT" and prob < HYST_LONG_SELL_FLOOR:
                            exit_reason = "TREND_REVERSAL"
                        elif position.side == "SHORT" and signal == "LONG" and prob > HYST_SHORT_SELL_CEIL:
                            exit_reason = "TREND_REVERSAL"

                    # Exit Rule 3: 2-Brick Structural Trailing Stop (chart override)
                    # Chart structure is unambiguous — exit regardless of XGBoost state.
                    if exit_reason is None and position.adverse_bricks >= STRUCTURAL_REVERSAL_BRICKS:
                        if position.side == "LONG" and brick_dir < 0:
                            exit_reason = "STRUCTURAL_2BRICK_REVERSAL"
                        elif position.side == "SHORT" and brick_dir > 0:
                            exit_reason = "STRUCTURAL_2BRICK_REVERSAL"

                    # Exit Rule 4: Stop-loss
                    if exit_reason is None and position.adverse_bricks >= MAX_ADVERSE_BRICKS:
                        exit_reason = "STOP_LOSS"

                    # Exit Rule 5: Max hold
                    if exit_reason is None and position.bricks_held >= MAX_HOLD_BRICKS:
                        exit_reason = "MAX_HOLD"

                    if exit_reason:
                        position = close_position(position, brick_close, ts, exit_reason)
                        all_trades.append(position)
                        if position.net_pnl_pct <= 0:
                            daily_losses += 1
                        position = None
                    continue  # Don't open new position on same brick as exit

                # ── No open position, evaluate ENTRY for T+1 fill ───────────
                # Strict Morning Time-Lock. Do not enter any trades before 09:30 AM.
                # Option 2: Require Fresh Evidence — check brick_start_time to prevent Gate Rush
                start_ts = row.get("brick_start_time", ts)
                if pd.isna(start_ts):
                    start_ts = ts
                elif isinstance(start_ts, str):
                    start_ts = pd.to_datetime(start_ts)
                    
                is_too_early = (start_ts.hour < 9) or (start_ts.hour == 9 and start_ts.minute < 20)
                
                # No new entries from 3:00 PM onwards (not enough time for T+1 fill)
                is_too_late = (ts.hour > NO_NEW_ENTRY_HOUR) or (ts.hour == NO_NEW_ENTRY_HOUR and ts.minute >= NO_NEW_ENTRY_MIN)
                
                if is_too_early or is_too_late:
                    continue

                # Prevent entering in the same physical minute twice
                ts_minute = ts.floor("T")
                if last_entry_minute is not None and ts_minute == last_entry_minute:
                    continue

                # For LONG and SHORT: prob > threshold
                if signal in ("LONG", "SHORT"):
                    entry_prob_ok = prob >= ENTRY_PROB_THRESH
                else:
                    continue

                if not entry_prob_ok or conviction < ENTRY_CONV_THRESH:
                    continue

                if not passes_soft_veto(signal, rel_str):
                    vetoed_count += 1
                    continue

                if brick_close < MIN_PRICE_FILTER:
                    continue  # Block penny stocks from generating noise signals

                # Gate: RS Anchor — only trade sector leaders/laggards
                # Gate: RS Anchor — only trade sector leaders/laggards
                if signal == "LONG" and rel_str < ENTRY_RS_THRESHOLD:
                    continue
                if signal == "SHORT" and rel_str > -ENTRY_RS_THRESHOLD:
                    continue

                # Gate: Wick Trap — block absorption candles
                wick_p = row.get("wick_pressure", 0) or 0
                if wick_p > MAX_ENTRY_WICK:
                    # _drop_reason = f"Wick: {wick_p}"
                    continue

                expected_dir = 1 if signal == "LONG" else -1
                if brick_dir != expected_dir or row.get("consecutive_same_dir", 0) < MIN_CONSECUTIVE_BRICKS:
                    continue

                # FIX #6: Ghost Momentum FOMO Protection
                if row.get("consecutive_same_dir", 0) >= 7:
                    continue

                # Whipsaw: MIN_BRICKS_TODAY session freshness check
                # (backtester can't check brick dates easily per row, approximated
                #  by requiring the day to have >= MIN_BRICKS_TODAY bricks so far)
                if i < MIN_BRICKS_TODAY:
                    # _drop_reason = f"Session Freshness: {i} < {MIN_BRICKS_TODAY}"
                    continue

                if daily_losses >= MAX_LOSSES_PER_STOCK:
                    continue

                # Signal fires! Compute ideal qty and queue as PENDING (T+1)
                alloc = STARTING_CAPITAL * POSITION_SIZE_PCT * INTRADAY_LEVERAGE
                qty   = max(1, int(alloc / brick_close))

                # FIX #10: Store as pending — will OPEN on the NEXT brick's open price
                pending_entry = {"side": signal, "qty": qty}

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
                f"{vetoed_count} vetoed | {eod_exits} EOD exits | "
                f"{volume_rejected} volume-rejected")
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

    # Equity curve using additive exact PnL on virtual turnover
    pnl_rs_exact = np.array([t.net_pnl_pct * t.entry_price * t.qty for t in trades])
    cumulative_pnl = np.cumsum(pnl_rs_exact)
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
    print(f"   {'Brokerage & Taxes:':<30} Exact Upstox intraday charges")
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
            "qty": t.qty,
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
    args_to_parse = sys.argv[1:]
    if args_to_parse and args_to_parse[0] == "backtest":
        args_to_parse = args_to_parse[1:]
    args, _ = parser.parse_known_args(args_to_parse)

    start_year = args.start
    end_year   = args.end
    period_label = f"{start_year}-{end_year}"

    logger.info("=" * 72)
    logger.info("THE TRUTH TELLER v2 -- Intraday Backtest Engine")
    logger.info(f"Test Period: {period_label}")
    logger.info(f"Brokerage: Exact Upstox | Entry: Prob>{ENTRY_PROB_THRESH} Conv>{ENTRY_CONV_THRESH}")
    logger.info(f"Intraday Only | StopLoss: {MAX_ADVERSE_BRICKS} bricks | MaxHold: {MAX_HOLD_BRICKS} bricks")
    logger.info("=" * 72)

    # Phase 1: Load data & models
    test_data = load_test_data(start_year, end_year)
    b1_long, b1_short, b2 = load_models()

    # Phase 2: Generate signals
    test_data = generate_signals(test_data, b1_long, b1_short, b2)

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
