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
import keras
import gc
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import config
from src.core.quant_fixes import IsotonicCalibrationWrapper
from src.core.renko import check_path_conflict

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
# BACKTEST CONSTANTS (Synchronized with config.py)
# =============================================================================
LONG_ENTRY_PROB_THRESH  = config.LONG_ENTRY_PROB_THRESH
SHORT_ENTRY_PROB_THRESH = config.SHORT_ENTRY_PROB_THRESH

RAW_LONG_ENTRY_PROB_THRESH  = config.RAW_LONG_ENTRY_PROB_THRESH
RAW_SHORT_ENTRY_PROB_THRESH = config.RAW_SHORT_ENTRY_PROB_THRESH

USE_CALIBRATED_MODELS    = config.USE_CALIBRATED_MODELS

ENTRY_CONV_THRESH       = config.ENTRY_CONV_THRESH
EXIT_CONV_THRESH        = config.EXIT_CONV_THRESH
STARTING_CAPITAL        = config.STARTING_CAPITAL

# Anti-Myopia: Hysteresis Dead-Zone (Probability State Machine)
HYST_LONG_SELL_FLOOR  = config.HYST_LONG_SELL_FLOOR
HYST_SHORT_SELL_CEIL  = config.HYST_SHORT_SELL_CEIL

# Anti-Myopia: Structural Safety Nets
STRUCTURAL_REVERSAL_BRICKS = config.STRUCTURAL_REVERSAL_BRICKS
MAX_ADVERSE_BRICKS         = config.STRUCTURAL_REVERSAL_BRICKS # STOP: Matches structural
MAX_HOLD_BRICKS            = config.MAX_HOLD_BRICKS

# Upstox Intraday Equity Charges & Sizing
POSITION_SIZE_PCT    = config.POSITION_SIZE_PCT
INTRADAY_LEVERAGE    = config.INTRADAY_LEVERAGE
TRANSACTION_COST_PCT = config.TRANSACTION_COST_PCT
T1_SLIPPAGE_PCT      = config.T1_SLIPPAGE_PCT

def calculate_charges(entry_price: float, exit_price: float, qty: int, side: str) -> float:
    """
    Hyper-accurate transaction friction math mirroring Upstox Intraday (MIS) Equity.
    Covers Brokerage, STT, GST, SEBI, and Stamp Duty.
    """
    if qty <= 0: return 0.0
    
    # 0. Turnover Math
    buy_turnover  = (entry_price * qty) if side == "LONG" else (exit_price * qty)
    sell_turnover = (exit_price * qty) if side == "LONG" else (entry_price * qty)
    total_turnover = buy_turnover + sell_turnover

    # 1. Brokerage: Lower of Rs 20 or 0.05% per side
    brok_entry = min(config.SIM_BROKERAGE_MAX, (entry_price * qty) * config.SIM_BROKERAGE_PCT)
    brok_exit  = min(config.SIM_BROKERAGE_MAX, (exit_price * qty)  * config.SIM_BROKERAGE_PCT)
    brokerage = brok_entry + brok_exit

    # 2. STT: 0.025% on Sell Side Only
    stt = sell_turnover * config.SIM_STT_SELL_PCT

    # 3. Stamp Duty: 0.003% on Buy Side Only
    stamp = buy_turnover * config.SIM_STAMP_BUY_PCT

    # 4. Exchange Transaction Charge: 0.00297% on both sides
    exchange = total_turnover * config.SIM_EXCHANGE_PCT

    # 5. SEBI Turnover Fee: Rs 10 per Crore
    sebi = total_turnover * config.SIM_SEBI_PCT

    # 6. GST: 18% on (Brokerage + Exchange)
    gst = (brokerage + exchange) * config.SIM_GST_PCT

    total_friction = brokerage + stt + stamp + exchange + sebi + gst
    return total_friction

# Intraday constraints - matches paper_trader.py exactly
EOD_EXIT_HOUR      = config.EOD_SQUARE_OFF_HOUR
EOD_EXIT_MINUTE    = config.EOD_SQUARE_OFF_MIN
NO_NEW_ENTRY_HOUR  = config.NO_NEW_ENTRY_HOUR
NO_NEW_ENTRY_MIN   = config.NO_NEW_ENTRY_MIN
MAX_OPEN_POSITIONS = config.MAX_OPEN_POSITIONS

# Whipsaw protection
MIN_CONSECUTIVE_BRICKS = config.MIN_CONSECUTIVE_BRICKS
MIN_BRICKS_TODAY       = config.MIN_BRICKS_TODAY
MAX_LOSSES_PER_STOCK   = config.MAX_LOSSES_PER_STOCK

# Entry filters
ENTRY_RS_THRESHOLD = config.ENTRY_RS_THRESHOLD
MAX_ENTRY_WICK     = config.MAX_ENTRY_WICK

# Volume-cap constants
VOLUME_LIMIT_PCT   = config.VOLUME_LIMIT_PCT
MIN_CANDLE_VOLUME  = config.MIN_CANDLE_VOLUME

# Fix #10: T+1 slippage - entry price penalty for API latency
T1_SLIPPAGE_PCT    = config.T1_SLIPPAGE_PCT

# Penny Stock Filter
MIN_PRICE_FILTER   = config.MIN_PRICE_FILTER

# =============================================================================
# PHASE 4: PESSIMISTIC EXECUTION ENGINE CONSTANTS
# =============================================================================
# These three constants implement the "Friction Tax" that makes backtest
# results representative of real-world execution quality.

SLIPPAGE_PCT    = config.T1_SLIPPAGE_PCT
JITTER_SECONDS  = config.JITTER_SECONDS 

PATH_CONFLICT   = config.PATH_CONFLICT_PESSIMISM
                           # hit within the same 1-minute candle's interpolated path,
                           # record the outcome as LOSS (worst-case, pessimistic).
                           # This is important because: on a fast NSE candle, the wick
                           # may touch the SL before the price recovers to hit the target.
                           # Without this check, backtests assume perfect fill order,
                           # which they never have in reality.


# =============================================================================
# FEATURE ORDER SHIELD
# =============================================================================
# CRITICAL: Feature alignment single source of truth
EXPECTED_FEATURES = config.FEATURE_COLS

# Legacy alias kept for Brain2 meta-regressor which uses its own columns
FEATURE_COLS = EXPECTED_FEATURES


META_COLS = config.BRAIN2_FEATURES

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
    test_start = pd.Timestamp(f"{start_year}-01-01", tz="Asia/Kolkata")
    test_end = pd.Timestamp(f"{end_year + 1}-01-01", tz="Asia/Kolkata")

    # 2. Define minimum required columns to prevent OOM
    required_cols = list(set(
        ["brick_timestamp", "brick_open", "brick_close", "brick_high", "brick_low", "brick_size", "direction"] +
        config.FEATURE_COLS + 
        config.BRAIN2_FEATURES
    ))
    # Filter out column aliases that don't exist in early data
    required_cols = [c for c in required_cols if c not in ["brain1_prob_long", "brain1_prob_short", "trade_direction"]]

    frames = []
    for sector_dir in config.FEATURES_DIR.iterdir():
        if not sector_dir.is_dir():
            continue
        for pf in sorted(sector_dir.glob("*.parquet")):
            try:
                # OPTIMIZATION: Load only columns we actually need for the model/PnL
                # and immediately downcast to float32 to save 50% RAM.
                df = pd.read_parquet(pf, columns=[c for c in required_cols if c != "_symbol"])
                
                # Downcast floats to save memory
                float_cols = df.select_dtypes(include=["float64"]).columns
                df[float_cols] = df[float_cols].astype("float32")
                # Downcast direction to save memory
                if "direction" in df.columns:
                    df["direction"] = df["direction"].astype("int8")
                
                # Filter indices before adding symbol/sector to save string memory
                years = df["brick_timestamp"].dt.year
                months = df["brick_timestamp"].dt.month
                
                generic_mask = (years.isin(getattr(config, "HOLDOUT_YEARS", []))) & \
                               (months.isin(getattr(config, "HOLDOUT_MONTHS", [])))
                               
                specific_masks = []
                for yr, m_list in getattr(config, "HOLDOUT_SPECIFIC_YEAR_MONTHS", {}).items():
                    specific_masks.append((years == yr) & (months.isin(m_list)))
                
                if specific_masks:
                    specific_mask = pd.concat(specific_masks, axis=1).any(axis=1)
                else:
                    specific_mask = pd.Series(False, index=df.index)
                    
                test_mask = generic_mask | specific_mask
                
                # Fix 5: CNN Cold-Start lookback (Warmup Buffer)
                test_indices = np.where(test_mask)[0]
                if len(test_indices) == 0:
                    continue
                
                min_idx = test_indices[0]
                max_idx = test_indices[-1]
                # Prepend lookback buffer rows
                start_idx = max(0, min_idx - config.CNN_WINDOW_SIZE)
                
                df = df.iloc[start_idx : max_idx + 1].copy()
                df["_is_warmup"] = True
                # The actual test data starts after the buffer
                df.iloc[(min_idx - start_idx):, df.columns.get_loc("_is_warmup")] = False
                
                df["_sector"] = sector_dir.name
                df["_symbol"] = pf.stem
                
                # Optimization: Convert symbol to category to save massive string RAM
                df["_symbol"] = df["_symbol"].astype("category")
                
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
                f"(Custom Holdout Config)")
    return test


def load_models():
    """
    Load Brain 1 (CNN), Scaler, and Brain 2 (Booster).
    """
    b1_long  = keras.models.load_model(str(config.BRAIN1_CNN_LONG_PATH))
    b1_short = keras.models.load_model(str(config.BRAIN1_CNN_SHORT_PATH))
    scaler   = joblib.load(str(config.BRAIN1_SCALER_PATH))
    
    # Booster for class-agnostic meta-regressor
    b2 = xgb.Booster(); b2.load_model(str(config.BRAIN2_MODEL_PATH))

    logger.info(f"Models loaded (Brain1: CNN, Brain2: JSON, Scaler: Robust)")
    return b1_long, b1_short, b2, scaler


# =============================================================================
# PREDICTION ENGINE
# =============================================================================
def generate_signals(df: pd.DataFrame, brain1_long, brain1_short, brain2, scaler) -> pd.DataFrame:
    """
    Run Brain1 (CNN) + Brain2 (XGBoost Booster) on the test DataFrame.
    """
    # OPTIMIZATION: Removed df.copy() which was doubling RAM for large datasets.
    # We already filtered the columns in load_test_data, so this is safe.
    df["brain1_prob_long"] = np.zeros(len(df), dtype="float32")
    df["brain1_prob_short"] = np.zeros(len(df), dtype="float32")
    df["brain1_signal"] = "FLAT"
    df["brain1_prob"] = np.zeros(len(df), dtype="float32")
    
    window_calc = config.CNN_WINDOW_SIZE
    
    # Process each symbol separately to avoid window overlap
    for (symbol, day), group in df.groupby(["_symbol", "_trade_date"]):
        indices = group.index
        if len(group) < window_calc:
            continue
            
        # 1. Brain 1 (CNN) Inference
        X_raw_df = group[config.FEATURE_COLS].fillna(0)
        # Apply scaling BEFORE 3D windowing
        X_scaled = scaler.transform(X_raw_df)
        
        # Create 3D windows: (num_samples, window_size, num_features)
        X_3d = sliding_window_view(X_scaled, (window_calc, X_scaled.shape[1])).squeeze(1)
        
        # Predictions (CNN models return single value per sample)
        # Note: sliding_window_view returns N-W+1 samples. We align them to the end of each window.
        p_long  = brain1_long.predict(X_3d, verbose=0).flatten()
        p_short = brain1_short.predict(X_3d, verbose=0).flatten()
        
        df.loc[indices[window_calc-1:], "brain1_prob_long"] = p_long
        df.loc[indices[window_calc-1:], "brain1_prob_short"] = p_short

    # 2. Universal Signal Selection Logic
    prob_long = df["brain1_prob_long"].values
    prob_short = df["brain1_prob_short"].values
    
    # Static Threshold Selection
    t_long  = config.LONG_ENTRY_PROB_THRESH 
    t_short = config.SHORT_ENTRY_PROB_THRESH
    
    long_ok  = prob_long >= t_long
    short_ok = prob_short >= t_short
    
    sg = np.full(len(df), "FLAT", dtype=object)
    pb = np.zeros(len(df), dtype=float)
    b1d = np.zeros(len(df), dtype=float)
    
    # Masked operations for speed
    long_only  = long_ok & ~short_ok
    short_only = short_ok & ~long_ok
    both_ok    = long_ok & short_ok
    
    sg[long_only]  = "LONG"
    pb[long_only]  = prob_long[long_only]
    b1d[long_only] = 1.0
    
    sg[short_only]  = "SHORT"
    pb[short_only]  = prob_short[short_only]
    b1d[short_only] = -1.0
    
    # Tie-breaker for both
    l_win = both_ok & (prob_long >= prob_short)
    s_win = both_ok & (prob_long < prob_short)
    
    sg[l_win]  = "LONG"
    pb[l_win]  = prob_long[l_win]
    b1d[l_win] = 1.0
    
    sg[s_win]  = "SHORT"
    pb[s_win]  = prob_short[s_win]
    b1d[s_win] = -1.0
    
    df["brain1_signal"] = sg
    df["brain1_prob"]   = pb

    # 3. Brain 2: Conviction Meta-Regressor (Booster)
    # Build feature matrix dynamically from config
    meta_feats = {}
    for f in config.BRAIN2_FEATURES:
        if f == "brain1_prob_long": meta_feats[f] = df["brain1_prob_long"]
        elif f == "brain1_prob_short": meta_feats[f] = df["brain1_prob_short"]
        elif f == "trade_direction": meta_feats[f] = b1d
        else: meta_feats[f] = df[f].fillna(0)
    
    X_meta = pd.DataFrame(meta_feats)[config.BRAIN2_FEATURES]
    dm = xgb.DMatrix(X_meta, feature_names=config.BRAIN2_FEATURES)
    df["brain2_conviction"] = brain2.predict(dm).clip(0, 100)

    long_count  = (df["brain1_signal"] == "LONG").sum()
    short_count = (df["brain1_signal"] == "SHORT").sum()
    
    logger.info(
        f"Signals generated: LONG={long_count:,}  SHORT={short_count:,}  "
        f"Avg Conviction={df['brain2_conviction'].mean():.1f}"
    )
    
    keras.backend.clear_session()
    gc.collect()
    
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
    if signal == "LONG" and rel_strength < -config.SOFT_VETO_THRESHOLD:
        return False
    if signal == "SHORT" and rel_strength > config.SOFT_VETO_THRESHOLD:
        return False
    return True


# =============================================================================
# CLOSE POSITION HELPER
# =============================================================================
def close_position(position: Trade, price: float, ts: pd.Timestamp,
                   reason: str) -> Trade:
    """
    Calculate PnL and close a position.

    Phase 4: Pessimistic Execution - SLIPPAGE_PCT applied to exit.
      LONG exit:  fill_price = price * (1 - SLIPPAGE_PCT)
                  (We sell at a slightly LOWER price - bid-ask spread, impact)
      SHORT exit: fill_price = price * (1 + SLIPPAGE_PCT)
                  (We buy back at a slightly HIGHER price - spread + urgency)

    This is applied on top of T1_SLIPPAGE on entry, so total round-trip
    friction = 2 * SLIPPAGE_PCT approx.== 0.10%, matching realistic NSE execution costs.
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

    charges = calculate_charges(position.entry_price, effective_exit, position.qty, position.side)
    position.cost_pct = charges / (position.entry_price * position.qty) if position.qty > 0 else 0.0
    position.net_pnl_pct = position.gross_pnl_pct - position.cost_pct
    return position


# =============================================================================
# INTRADAY SIMULATION ENGINE
# =============================================================================
def run_simulation(df: pd.DataFrame) -> List[Trade]:
    """
    Event-driven INTRADAY backtesting loop.
    FIX #10: T+1 Execution - signals fire on brick[i], fill on brick[i+1]'s open.
    FIX #9:  Volume Guard - position capped at 5% of candle volume.
    FIX #13: EOD 3:15PM exit uses brick_open of 3:15 candle (not brick_close).
    """
    all_trades: List[Trade] = []
    trade_counter = 0
    vetoed_count  = 0
    volume_rejected = 0
    low_volume_dropped = 0
    eod_exits = 0
    
    # DROP COUNTERS
    drop_conviction = 0
    drop_rs = 0
    drop_wick = 0
    drop_whipsaw = 0
    drop_time = 0
    drop_fomo = 0
    drop_vwap = 0

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
                
                # Fix 5: Skip warmup bricks for trading logic
                if row.get("_is_warmup", False):
                    continue

                prob      = row["brain1_prob"]
                signal    = row["brain1_signal"]
                conviction= row["brain2_conviction"]
                rel_str   = row["relative_strength"] if pd.notna(row["relative_strength"]) else 0.0
                ts        = row["brick_timestamp"]
                brick_dir = row["direction"]

                # brick_open available for T+1 fills; fall back to close if missing
                brick_open  = row.get("brick_open",  row["brick_close"])
                brick_close = row["brick_close"]

                # -- FIX #13: Force EOD exit at 3:15 PM using brick_open -----
                is_eod_candle = (
                    ts.hour > EOD_EXIT_HOUR or
                    (ts.hour == EOD_EXIT_HOUR and ts.minute >= EOD_EXIT_MINUTE)
                )
                if is_eod_candle:
                    pending_entry = None  # cancel any pending T+1 entry
                    if position is not None:
                        # Exit at brick_open of the 3:15 candle - not close (fiction fix)
                        position = close_position(position, brick_open, ts, "EOD_315")
                        all_trades.append(position)
                        eod_exits += 1
                        position = None
                    continue  # No new entries at or after 3:15

                # -- FIX #10: Execute T+1 pending entry on THIS brick's open --
                if pending_entry is not None and position is None:
                    p = pending_entry
                    # FIX #3: Liquidity Mirage - adjust virtual brick_open by actual market gap
                    gap_pct = row.get("true_gap_pct", 0.0)
                    gap_pct = gap_pct / 100.0 if pd.notna(gap_pct) else 0.0
                    gap_adjusted_open = brick_open * (1.0 + gap_pct)

                    # Apply T+1 slippage on the gap-adjusted OPEN price
                    fill_price = gap_adjusted_open * (1 + T1_SLIPPAGE_PCT) if p["side"] == "LONG" \
                                 else gap_adjusted_open * (1 - T1_SLIPPAGE_PCT)
                    # FIX #9: Volume guard - cap qty at 5% of this candle's volume
                    raw_vol = row.get("volume", 1_000_000_000)
                    if pd.isna(raw_vol):
                        raw_vol = 1_000_000_000
                    candle_vol = max(float(raw_vol or 1_000_000_000), 1.0)
                    if candle_vol < MIN_CANDLE_VOLUME:
                        # Not enough liquidity - reject trade entirely
                        pending_entry = None
                        volume_rejected += 1
                    else:
                        max_qty = max(1, int(candle_vol * VOLUME_LIMIT_PCT))
                        qty     = min(p["qty"], max_qty)
                        
                        # Fix 3: Low-Volume Capital Trap - reject if < 20% of ideal
                        if qty < (0.2 * p["qty"]):
                            pending_entry = None
                            low_volume_dropped += 1
                        elif qty < 1:
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

                # -- If we have an open position, check EXIT rules ------------
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
                            # The maximum adverse bricks allowed from the PEAK (favorable_bricks)
                            dynamic_trail_allowance = config.TRAIL_DISTANCE_BRICKS
                            
                            # Once activated, if we fall back by the trail distance, exit immediately to lock profit
                            if position.adverse_bricks >= dynamic_trail_allowance:
                                exit_reason = "TRAIL_PROFIT_ACTIVATED"

                    # Exit Rule 2: Hysteresis Dead-Zone Trend Reversal
                    # Anti-Myopia: Do NOT exit on any probability nudge.
                    # The model must STRONGLY confirm reversal before exiting.
                    # Dead-Zone [0.40, 0.60] = HOLD - pure noise, stay in.
                    if exit_reason is None:
                        # Exit LONG if model STRONGLY leans SHORT (> 0.60)
                        if position.side == "LONG" and signal == "SHORT" and prob > (1.0 - HYST_LONG_SELL_FLOOR):
                            exit_reason = "TREND_REVERSAL"
                        # Exit SHORT if model STRONGLY leans LONG (> 0.60)
                        elif position.side == "SHORT" and signal == "LONG" and prob > HYST_SHORT_SELL_CEIL:
                            exit_reason = "TREND_REVERSAL"

                    # Exit Rule 3: 2-Brick Structural Trailing Stop (chart override)
                    # Chart structure is unambiguous - exit regardless of XGBoost state.
                    if exit_reason is None and position.adverse_bricks >= STRUCTURAL_REVERSAL_BRICKS:
                        if position.side == "LONG" and brick_dir < 0:
                            exit_reason = "STRUCTURAL_REVERSAL"
                        elif position.side == "SHORT" and brick_dir > 0:
                            exit_reason = "STRUCTURAL_REVERSAL"

                    # Exit Rule 4: Stop-loss
                    if exit_reason is None and position.adverse_bricks >= MAX_ADVERSE_BRICKS:
                        exit_reason = "STOP_LOSS"

                    # Exit Rule 5: Max hold
                    if exit_reason is None and position.bricks_held >= MAX_HOLD_BRICKS:
                        exit_reason = "MAX_HOLD"

                    if exit_reason is None and PATH_CONFLICT:
                        # FIX #14: Path Conflict Realism (Vector 4 Audit)
                        # If the brick's High/Low touched the stop loss level,
                        # assume the trade was killed regardless of the Close.
                        brick_size = row["brick_size"]
                        sl_level = position.entry_price - (MAX_ADVERSE_BRICKS * brick_size) if position.side == "LONG" \
                                   else position.entry_price + (MAX_ADVERSE_BRICKS * brick_size)
                        
                        if position.side == "LONG" and row["brick_low"] <= sl_level:
                            exit_reason = "PATH_CONFLICT_SL"
                        elif position.side == "SHORT" and row["brick_high"] >= sl_level:
                            exit_reason = "PATH_CONFLICT_SL"

                    if exit_reason:
                        exit_price = brick_close
                        
                        # Fix 2: Use exact sl_level for wick-hit (Worst-Case SL)
                        if exit_reason == "PATH_CONFLICT_SL":
                            brick_size = row["brick_size"]
                            exit_price = position.entry_price - (MAX_ADVERSE_BRICKS * brick_size + 0.05) if position.side == "LONG" \
                                         else position.entry_price + (MAX_ADVERSE_BRICKS * brick_size + 0.05)
                        
                        position = close_position(position, exit_price, ts, exit_reason)
                        all_trades.append(position)
                        if position.net_pnl_pct <= 0:
                            daily_losses += 1
                        position = None
                    continue  # Don't open new position on same brick as exit

                # -- No open position, evaluate ENTRY for T+1 fill -----------
                # Strict Morning Time-Lock. Do not enter any trades before 09:30 AM.
                # Option 2: Require Fresh Evidence - check brick_start_time to prevent Gate Rush
                start_ts = row.get("brick_start_time", ts)
                if pd.isna(start_ts):
                    start_ts = ts
                elif isinstance(start_ts, str):
                    start_ts = pd.to_datetime(start_ts)
                    
                is_too_early = (start_ts.hour < config.MARKET_OPEN_HOUR) or \
                               (start_ts.hour == config.MARKET_OPEN_HOUR and start_ts.minute < (config.MARKET_OPEN_MINUTE + config.ENTRY_LOCK_MINUTES))
                
                # No new entries from 3:00 PM onwards (not enough time for T+1 fill)
                is_too_late = (ts.hour > NO_NEW_ENTRY_HOUR) or (ts.hour == NO_NEW_ENTRY_HOUR and ts.minute >= NO_NEW_ENTRY_MIN)
                
                if is_too_early or is_too_late:
                    drop_time += 1
                    continue

                # Prevent entering in the same physical minute twice
                ts_minute = ts.floor("T")
                if last_entry_minute is not None and ts_minute == last_entry_minute:
                    continue

                # Static Threshold Selection
                _eff_long  = LONG_ENTRY_PROB_THRESH
                _eff_short = SHORT_ENTRY_PROB_THRESH
                if signal == "LONG":
                    entry_prob_ok = prob >= _eff_long
                elif signal == "SHORT":
                    entry_prob_ok = prob >= _eff_short
                else:
                    continue

                if not entry_prob_ok or conviction < ENTRY_CONV_THRESH:
                    drop_conviction += 1
                    continue

                veto_bypass = getattr(config, "VETO_BYPASS_CONV", 30.0)
                is_vetoed = not passes_soft_veto(signal, rel_str)

                if is_vetoed and conviction < veto_bypass:
                    vetoed_count += 1
                    continue

                if brick_close < MIN_PRICE_FILTER:
                    continue  # Block penny stocks from generating noise signals

                # Gate: RS Anchor - only trade sector leaders/laggards
                # Bypass if conviction is exceptionally high
                if conviction < veto_bypass:
                    if signal == "LONG" and rel_str < ENTRY_RS_THRESHOLD:
                        drop_rs += 1
                        continue
                    if signal == "SHORT" and rel_str > -ENTRY_RS_THRESHOLD:
                        drop_rs += 1
                        continue

                # Gate: Wick Trap - block absorption candles
                wick_p = row.get("wick_pressure", 0) or 0
                if wick_p > MAX_ENTRY_WICK:
                    drop_wick += 1
                    continue

                # Gate: VWAP Exhaustion (Anti-Peak Gap)
                z_vwap = row.get("vwap_zscore", 0) or 0
                max_vwap = getattr(config, "MAX_VWAP_ZSCORE", 3.0)
                if signal == "LONG" and z_vwap > max_vwap:
                    drop_vwap += 1
                    continue
                if signal == "SHORT" and z_vwap < -max_vwap:
                    drop_vwap += 1
                    continue

                expected_dir = 1 if signal == "LONG" else -1
                if brick_dir != expected_dir or row.get("consecutive_same_dir", 0) < MIN_CONSECUTIVE_BRICKS:
                    drop_whipsaw += 1
                    continue

                # FIX #6: Ghost Momentum FOMO Protection
                if row.get("consecutive_same_dir", 0) >= config.STREAK_LIMIT:
                    drop_fomo += 1
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

                # FIX #10: Store as pending - will OPEN on the NEXT brick's open price
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

    logger.info(f"Simulation complete: {len(all_trades)} trades | {vetoed_count} vetoed | {eod_exits} EOD exits | {volume_rejected} volume-rejected | {low_volume_dropped} capital-traps")
    logger.info(f"Silent Drops -> Conv: {drop_conviction} | RS: {drop_rs} | Wick: {drop_wick} | Whipsaw: {drop_whipsaw} | Time: {drop_time} | FOMO: {drop_fomo}")
    return all_trades


# =============================================================================
# PORTFOLIO CONCURRENCY ENFORCEMENT
# =============================================================================
def enforce_portfolio_limits(trades: List[Trade], max_positions: int) -> List[Trade]:
    """
    Fix 4: Enforce Portfolio Concurrency Limit (Infinite Margin Flaw)
    Filters out trades that overlap in time when the portfolio is already full.
    """
    if not trades or max_positions <= 0:
        return trades

    # Sort by entry_time to process chronologically
    sorted_trades = sorted(trades, key=lambda x: x.entry_time)
    filtered_trades = []
    
    # Maintain a list of exit times for currently open positions
    open_exits: List[pd.Timestamp] = []
    
    for t in sorted_trades:
        # 1. Clear trades that have exited by this trade's entry_time
        open_exits = [et for et in open_exits if et > t.entry_time]
        
        # 2. If we have room, take the trade
        if len(open_exits) < max_positions:
            filtered_trades.append(t)
            if t.exit_time:
                open_exits.append(t.exit_time)
        # else: trade is dropped (Inf Margin Filter)

    logger.info(f"Portfolio Filter: {len(trades):,} -> {len(filtered_trades):,} trades (Max={max_positions})")
    return filtered_trades


# =============================================================================
# PERFORMANCE REPORT
# =============================================================================
def generate_report(trades: List[Trade]) -> dict:
    """Generate the boardroom-quality performance report."""
    if not trades:
        logger.warning("No trades executed. Cannot generate report.")
        return {}

    # Fix 1: Chronological Equity Curve (Reporting Flaw)
    # Sort by exit_time to ensure PnL accumulation follows real time
    trades = sorted(trades, key=lambda x: x.exit_time if x.exit_time else x.entry_time)

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

    # Net ROI - simple sum (not compounded, to avoid overflow with 28K+ trades)
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

    # -- Print Boardroom Report ------------------------------------------
    print("\n")
    print("=" * 72)
    print("   THE TRUTH TELLER v2 -- Backtest Performance Report")
    print(f"   Institutional Fortress | Test Period: {report.get('period', 'N/A')}")
    print("=" * 72)

    print(f"\n   CONFIGURATION")
    print(f"   {'Brokerage & Taxes:':<30} Exact Upstox intraday charges")
    print(f"   {'Long Entry:':<25} {LONG_ENTRY_PROB_THRESH}")
    print(f"   {'Short Entry:':<25} {SHORT_ENTRY_PROB_THRESH}")
    print(f"   {'Conviction:':<25} > {ENTRY_CONV_THRESH}")
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
    """Entry point - parses --start / --end from sys.argv."""
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
    logger.info(f"Brokerage: Exact Upstox | Entry: Prob[L:{LONG_ENTRY_PROB_THRESH}, S:{SHORT_ENTRY_PROB_THRESH}] Conv>{ENTRY_CONV_THRESH}")
    logger.info(f"Intraday Only | StopLoss: {MAX_ADVERSE_BRICKS} bricks | MaxHold: {MAX_HOLD_BRICKS} bricks")
    logger.info("=" * 72)

    # Phase 1: Load data & models
    test_data = load_test_data(start_year, end_year)
    b1_long, b1_short, b2, scaler = load_models()

    # Phase 2: Generate signals
    test_data = generate_signals(test_data, b1_long, b1_short, b2, scaler)

    # Phase 3: Run simulation
    trades = run_simulation(test_data)

    if not trades:
        logger.warning("No trades generated. Check entry thresholds or data.")
        return

    # Phase 4: Enforce Portfolio Limits (Concurrency Fix)
    trades = enforce_portfolio_limits(trades, config.MAX_OPEN_POSITIONS)

    # Phase 5: Save trade log
    save_trade_log(trades)

    # Phase 6: Generate report
    report = generate_report(trades)
    report["period"] = period_label

    logger.info("TRUTH TELLER v2 COMPLETE")
    return report


if __name__ == "__main__":
    run_backtester()
