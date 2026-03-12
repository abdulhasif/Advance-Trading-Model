"""
src/ml/brain_trainer.py — Phase 3: Dual XGBoost GPU Trainer
=============================================================
Brain 1 (Direction Classifier) + Brain 2 (Conviction Meta-Regressor).
Uses tree_method='gpu_hist' + device='cuda' for RTX 3050 acceleration.

Run:  python -m src.ml.brain_trainer
"""

import sys
import pathlib
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")   # headless-safe — no display required
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score, brier_score_loss

import config
from src.core.quant_fixes import (
    purge_overlapping_samples,
    add_triple_barrier_t1,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "brain_trainer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

FEATURE_COLS = config.FEATURE_COLS

# Fix 5: Columns to apply Robust IQR scaling on (excludes binary cols)
ROBUST_SCALE_COLS = config.ROBUST_SCALE_COLS

# Anti-Myopia: Multi-brick horizon for target engineering
# Model predicts majority direction over next H bricks, not just next 1.
TRAINING_HORIZON = config.TRAINING_HORIZON_BRICKS


# ── Data Loading ────────────────────────────────────────────────────────────

def load_all_features() -> pd.DataFrame:
    """Load enriched Parquet files — one sector at a time for RAM safety."""
    if not config.FEATURES_DIR.exists():
        logger.error("Features dir missing. Run feature_engine first."); sys.exit(1)

    frames = []
    total_long = 0
    total_short = 0

    # Formulas for Triple Barrier Synchronization
    STOP_PCT        = config.NATR_BRICK_PERCENT * config.STRUCTURAL_REVERSAL_BRICKS
    LONG_TARGET_PCT  = config.NATR_BRICK_PERCENT * config.TRAINING_HORIZON_BRICKS
    # FIX: SHORT model uses a smaller target to generate more label_short=1 examples.
    # Bear market grinds are slow — a 4-brick LONG target (~0.75%) often isn't reached intraday,
    # so the EOD barrier cuts the label to 0. A 2-brick target fixes this.
    SHORT_TARGET_PCT = config.NATR_BRICK_PERCENT * getattr(config, 'SHORT_TARGET_BRICKS', 2)

    for sector_dir in config.FEATURES_DIR.iterdir():
        if not sector_dir.is_dir():
            continue
        for pf in sorted(sector_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(pf)
                
                # --- MEMORY OPTIMIZATION ---
                # Downcast float64 -> float32 and int64 -> int32 to cut RAM usage by 50%
                float_cols = df.select_dtypes(include=["float64"]).columns
                if len(float_cols) > 0:
                    df[float_cols] = df[float_cols].astype(np.float32)
                
                int_cols = df.select_dtypes(include=["int64"]).columns
                if len(int_cols) > 0:
                    df[int_cols] = df[int_cols].astype(np.int32)

                df["_sector"] = sector_dir.name
                df["_symbol"] = pf.stem
                
                # Apply triple barrier incrementally per file (Synchronized with config)
                # FIX: Pass separate LONG and SHORT target pcts so each model gets appropriately
                # labelled training data. SHORT uses a smaller target (bear grinds are slower).
                df = add_triple_barrier_t1(df, stop_pct=STOP_PCT,
                                           target_pct=LONG_TARGET_PCT,
                                           short_target_pct=SHORT_TARGET_PCT)
                
                if "label_long" in df.columns:
                    total_long += int((df["label_long"] == 1).sum())
                    total_short += int((df["label_short"] == 1).sum())

                frames.append(df)
            except Exception as e:
                logger.warning(f"Skip {pf}: {e}")

    if not frames:
        logger.error("No feature files."); sys.exit(1)

    logger.info(f"Symmetric Triple Barrier Summary: LONG=1 ({total_long:,}), SHORT=1 ({total_short:,})")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["_symbol", "brick_timestamp"], kind="mergesort", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    
    # --- DATA SANITIZATION: Surgical Gap Purge ---
    # We must remove the very first brick of each day because it contains the 
    # "fake" overnight gap (15 hours of move in one brick). 
    # But we keep everything from 9:16 AM onwards so the model learns opening momentum.
    logger.info("Applying Surgical Purge: Dropping only the first brick of the day per symbol...")
    combined["_date"] = combined["brick_timestamp"].dt.date
    # Drop first brick of each (symbol, date) group
    combined["_first_brick"] = combined.groupby(["_symbol", "_date"]).cumcount() == 0
    purged_count = combined["_first_brick"].sum()
    combined = combined[~combined["_first_brick"]].copy()
    
    # Cleanup temp columns
    combined.drop(columns=["_date", "_first_brick"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    
    logger.info(f"Surgical Purge: Dropped {purged_count:,} overnight gap bricks. Opening momentum preserved.")
    logger.info(f"Total bricks loaded after purge: {len(combined):,} (Memory optimized)")
    return combined


# ── Target Engineering ──────────────────────────────────────────────────────

def create_targets(df: pd.DataFrame, horizon: int = TRAINING_HORIZON) -> pd.DataFrame:
    """
    Anti-Myopia: H-Horizon Majority-Vote Target
    ────────────────────────────────────────────
    Instead of predicting the NEXT single brick, the model predicts whether
    the MAJORITY of the next H bricks will be upward-direction.

    This forces the XGBoost model to learn sustained multi-brick trends
    instead of reacting to single-brick noise (the root cause of sub-10-min trades).

    y=1 if mean(direction_h1, ..., direction_hH) > 0.50 (bullish majority)
    y=0 if mean <= 0.50 (bearish or neutral majority)
    """
    # Operate inplace to save RAM (avoids 2.5GB allocation spike on 17.2M rows)
    out = df
    out["_date"] = out["brick_timestamp"].dt.date

    # Sort inplace to ensure deterministic ordering within each (symbol, date) group
    out.sort_values(["_symbol", "brick_timestamp"], kind="mergesort", inplace=True)
    out.reset_index(drop=True, inplace=True)

    # --- Direction Target: H-brick majority vote ---
    # Collect shifted direction columns: +1,+2,...,+H
    horizon_dirs = []
    for h in range(1, horizon + 1):
        shifted = out.groupby(["_symbol", "_date"])["direction"].shift(-h)
        # Convert to bullish flag (1=up, 0=down)
        horizon_dirs.append((shifted > 0).astype(float))

    # Stack: shape (N, H) → take mean across horizon
    horizon_stack = pd.concat(horizon_dirs, axis=1)
    majority_bull = horizon_stack.mean(axis=1)  # fraction of horizon that is bullish

    # The last-H bricks of each (symbol, date) group get NaN from shift → propagate NaN
    any_nan = horizon_stack.isna().any(axis=1)
    out["direction_target"] = np.where(any_nan, np.nan, (majority_bull > 0.50).astype(float))

    # --- Conviction Target: H-brick forward log-return in BASIS POINTS (bps) ---
    # Fix: convert to bps (log_ret × 10,000). Typical 10-brick intraday move = 10–100 bps.
    # Values clipped at TARGET_CLIPPING_BPS, giving a proper range for large trends.
    close_h = out.groupby(["_symbol", "_date"])["brick_close"].shift(-horizon)
    log_ret = np.log(
        close_h.clip(lower=1e-9) / out["brick_close"].clip(lower=1e-9)
    ).abs()
    out["conviction_target"] = (log_ret * 10_000).clip(upper=config.TARGET_CLIPPING_BPS)

    out = out.drop(columns=["_date"])

    n_dropped_before = len(out)
    out = out.dropna(subset=["direction_target", "conviction_target"]).reset_index(drop=True)
    n_pos = int((out["direction_target"] == 1).sum())
    n_neg = int((out["direction_target"] == 0).sum())
    logger.info(
        f"H={horizon} Majority Target: y=1 (bull): {n_pos:,}  y=0 (bear): {n_neg:,}  "
        f"Dropped (NaN): {n_dropped_before - len(out):,}  "
        f"Class ratio: {n_neg / max(n_pos, 1):.2f}:1"
    )
    return out


def create_triple_barrier_targets(df: pd.DataFrame,
                                   stop_pct: float = config.NATR_BRICK_PERCENT * config.STRUCTURAL_REVERSAL_BRICKS,
                                   target_pct: float = config.NATR_BRICK_PERCENT * config.TRAINING_HORIZON_BRICKS,
                                   eod_hour: int = config.EOD_SQUARE_OFF_HOUR,
                                   eod_minute: int = config.EOD_SQUARE_OFF_MIN) -> pd.DataFrame:
    """
    Triple Barrier Method — Synchronized with Central Config.
    """

from numba import njit
import numpy as np

@njit(cache=True)
def _compute_triple_barrier_fast(closes, highs, lows, min_timestamps, stop_pct, target_pct, eod_mins):
    """
    Numba-optimized core for Symmetric Dual Triple Barrier.
    """
    n = len(closes)
    sym_dir_long = np.zeros(n, dtype=np.float32)
    sym_dir_short = np.zeros(n, dtype=np.float32)
    sym_conv = np.full(n, 20.0, dtype=np.float32) # default conviction

    for i in range(n):
        entry = closes[i]
        long_stop   = entry * (1.0 - stop_pct)
        long_target = entry * (1.0 + target_pct)
        short_stop  = entry * (1.0 + stop_pct)
        short_target= entry * (1.0 - target_pct)
        
        label_long = 0.0
        label_short = 0.0
        b_long = -1
        b_short = -1

        for j in range(i + 1, n):
            ts_mins = min_timestamps[j]
            if ts_mins >= eod_mins:
                break
            
            p_high = highs[j]
            p_low = lows[j]
            
            if b_long == -1:
                if p_low <= long_stop:
                    label_long = 0.0
                    b_long = j - i
                elif p_high >= long_target:
                    label_long = 1.0
                    b_long = j - i
                    
            if b_short == -1:
                if p_high >= short_stop:
                    label_short = 0.0
                    b_short = j - i
                elif p_low <= short_target:
                    label_short = 1.0
                    b_short = j - i
                    
            if b_long != -1 and b_short != -1:
                break
                
        sym_dir_long[i] = label_long
        sym_dir_short[i] = label_short
        b_min = -1
        if b_long > 0 and b_short > 0:
            b_min = min(b_long, b_short)
        elif b_long > 0:
            b_min = b_long
        elif b_short > 0:
            b_min = b_short
        if b_min > 0:
            # FIX: If the trade hits the profit target, conviction should be the actual basis point move
            # expected from the target_pct (e.g., 0.0075 -> 75 bps).
            # The old formula (100.0 / b_min) artificially suppressed conviction below 15 for any trade
            # taking > 6 bricks.
            
            # If a trade was successful (label_long == 1 or label_short == 1), we assign it the target bps.
            # If it failed (hit stop loss), we decay the conviction score rapidly.
            if (b_min == b_long and label_long == 1.0) or (b_min == b_short and label_short == 1.0):
                # successful hit prior to decay
                target_bps = target_pct * 10000.0
                sym_conv[i] = min(config.TARGET_CLIPPING_BPS, target_bps)
            else:
                sym_conv[i] = min(config.TARGET_CLIPPING_BPS, config.TARGET_CLIPPING_BPS / b_min)
            
    return sym_dir_long, sym_dir_short, sym_conv


def add_triple_barrier_t1(df: pd.DataFrame, stop_pct=None, target_pct=None,
                          short_target_pct=None,
                          eod_hour=config.EOD_SQUARE_OFF_HOUR, eod_minute=config.EOD_SQUARE_OFF_MIN) -> pd.DataFrame:
    """
    Creates Purge/Embargo Target Logic + 't1' resolution timestamp.
    Target: Symmetric Dual Triple Barrier.
    Synchronized with central config.py.

    FIX: short_target_pct allows SHORT labels to use a smaller target than LONG.
    Bear market grinds are slower — a 4-brick target is rarely hit intraday,
    causing the EOD barrier to cut label_short to 0. A 2-brick SHORT target
    generates far more label_short=1 examples from the training data.
    """
    if stop_pct is None:
        stop_pct = config.NATR_BRICK_PERCENT * config.STRUCTURAL_REVERSAL_BRICKS
    if target_pct is None:
        target_pct = config.NATR_BRICK_PERCENT * config.TRAINING_HORIZON_BRICKS
    if short_target_pct is None:
        short_target_pct = target_pct   # default: same as LONG (backward compatible)

    df = df.copy()
    if not df.index.is_monotonic_increasing:
         df = df.sort_values(["_symbol", "brick_timestamp"], kind="mergesort").reset_index(drop=True)
         
    df["_date"] = df["brick_timestamp"].dt.date
    df["_min_of_day"] = df["brick_timestamp"].dt.hour * 60 + df["brick_timestamp"].dt.minute
    eod_mins = eod_hour * 60 + eod_minute

    n = len(df)
    dir_long = np.zeros(n, dtype=np.float32)
    dir_short = np.zeros(n, dtype=np.float32)
    conviction_labels = np.full(n, 20.0, dtype=np.float32)

    use_asymmetric = (short_target_pct != target_pct)
    grouped = df.groupby(["_symbol", "_date"], sort=False)
    
    for _, grp in grouped:
        idx = grp.index.values
        closes = grp["brick_close"].values.astype(np.float64)
        highs = grp.get("brick_high", grp["brick_close"]).values.astype(np.float64)
        lows = grp.get("brick_low", grp["brick_close"]).values.astype(np.float64)
        mins = grp["_min_of_day"].values.astype(np.int32)

        if use_asymmetric:
            # LONG barrier: standard target_pct
            l_long, _, sym_conv = _compute_triple_barrier_fast(
                closes, highs, lows, mins, stop_pct, target_pct, eod_mins
            )
            # SHORT barrier: smaller short_target_pct — independent computation
            _, l_short, _ = _compute_triple_barrier_fast(
                closes, highs, lows, mins, stop_pct, short_target_pct, eod_mins
            )
        else:
            l_long, l_short, sym_conv = _compute_triple_barrier_fast(
                closes, highs, lows, mins, stop_pct, target_pct, eod_mins
            )
        
        dir_long[idx] = l_long
        dir_short[idx] = l_short
        conviction_labels[idx] = sym_conv

    df["label_long"]  = dir_long
    df["label_short"] = dir_short
    # Ensure backward compatibility with existing codebase
    df["direction_target"] = dir_long
    df["conviction_target"] = conviction_labels
    df["t1"] = df["brick_timestamp"] 
    
    # FIX: Use inplace deletion (del) instead of df.drop() to prevent copying the
    # entire 54-million row DataFrame in memory, which triggers ArrayMemoryError
    if "_date" in df.columns:
        del df["_date"]
    if "_min_of_day" in df.columns:
        del df["_min_of_day"]

    # Use gc to immediately free memory from the dropped arrays
    import gc
    gc.collect()

    # Silenced per-symbol logging to avoid console spam
    return df


# ── Walk-Forward Split ──────────────────────────────────────────────────────

def walk_forward_split(df: pd.DataFrame):
    """Single cutoff split using TEST_START_DATE from config."""
    if hasattr(config, 'TEST_START_DATE'):
        cutoff = pd.Timestamp(config.TEST_START_DATE, tz="Asia/Kolkata")
    else:
        cutoff = pd.Timestamp(f"{getattr(config, 'TEST_START_YEAR', 2025)}-01-01", tz="Asia/Kolkata")
    
    # Fallback to chronological 6-month split if config cutoff is beyond our data
    if cutoff > df["brick_timestamp"].max():
        cutoff = df["brick_timestamp"].max() - pd.DateOffset(months=6)

    train = df[df["brick_timestamp"] < cutoff]
    test  = df[df["brick_timestamp"] >= cutoff]
    logger.info(f"Split -- Train: {len(train):,}  Test: {len(test):,}  Cutoff: {cutoff.date()}")
    
    # Avoid massive memory copy here by returning the slice directly.
    # The caller must delete `df` and gc.collect() immediately after.
    return train, test


def walk_forward_rolling_splits(df: pd.DataFrame,
                                 train_months: int = 18,
                                 test_months: int = 1):
    """
    FIX #7: Rolling Walk-Forward Generator (Proper OOS Validation).
    Yields (train, test) chunks in a rolling window:
      - train_months: size of the training window (default 18 months)
      - test_months:  out-of-sample test window (default 1 month)
    The window shifts forward by test_months on each iteration.
    This ensures the model is NEVER evaluated on data it trained on.
    """
    ts_col  = "brick_timestamp"
    min_ts  = df[ts_col].min()
    max_ts  = df[ts_col].max()

    train_start = min_ts
    test_start  = min_ts + pd.DateOffset(months=train_months)

    fold = 0
    while test_start + pd.DateOffset(months=test_months) <= max_ts:
        test_end = test_start + pd.DateOffset(months=test_months)

        train = df[(df[ts_col] >= train_start) & (df[ts_col] <  test_start)]
        test  = df[(df[ts_col] >= test_start)  & (df[ts_col] <  test_end)]

        if len(train) > 1000 and len(test) > 100:
            fold += 1
            logger.info(f"Walk-Forward Fold {fold}: "
                        f"Train [{train_start.date()} -> {test_start.date()}] {len(train):,} bricks | "
                        f"Test  [{test_start.date()} -> {test_end.date()}]  {len(test):,} bricks")
            yield fold, train.copy(), test.copy()

        # Slide the window forward by 1 test period
        train_start += pd.DateOffset(months=test_months)
        test_start  += pd.DateOffset(months=test_months)

    if fold == 0:
        logger.warning("Not enough data for rolling walk-forward. Falling back to single split.")
        yield from [(1,) + walk_forward_split(df)]


# ── Brain 1: Direction Classifier ───────────────────────────────────────────

def feature_importance_diagnostic(model: xgb.XGBClassifier, feature_names: list) -> None:
    """
    Prints the Top 10 Feature Importances and saves a bar chart to storage/logs/.

    Diagnosis logic:
      - If 'velocity' or 'momentum_acceleration' dominate the top 3, the model is
        over-indexed on raw momentum and the alpha factors have no signal weight.
      - Target state: 'vwap_zscore', 'squeeze_zscore', 'streak_exhaustion' or
        'vpt_acceleration' should appear in the top 5 after a properly calibrated train.
    """
    importances = model.feature_importances_
    feature_importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
          .sort_values("importance", ascending=False)
          .reset_index(drop=True)
    )

    top10 = feature_importance_df.head(10)
    logger.info("\n" + "=" * 50)
    logger.info("TOP 10 FEATURE IMPORTANCES (Brain 1)")
    logger.info("=" * 50)
    for rank, row in top10.iterrows():
        tag = ""
        if row["feature"] in ("vwap_zscore", "vpt_acceleration", "squeeze_zscore", "streak_exhaustion"):
            tag = "  [ALPHA FACTOR - institutional signal]"
        elif row["feature"] in ("velocity", "momentum_acceleration", "velocity_long"):
            tag = "  [RAW MOMENTUM - risk of over-trading if top-3]"
        logger.info(f"  #{rank+1:2d}  {row['feature']:<26}  {row['importance']:.4f}{tag}")

    logger.info("=" * 50)

    # Warn if top-3 are ALL momentum features (model is momentum-blind)
    top3 = set(top10.head(3)["feature"].tolist())
    alpha_features = {"vwap_zscore", "vpt_acceleration", "squeeze_zscore", "streak_exhaustion"}
    if not top3.intersection(alpha_features):
        logger.warning(
            "DIAGNOSTIC WARNING: No institutional alpha factor in top-3. "
            "Model is over-reliant on raw momentum - likely to fire on wick-noise. "
            "Consider: more training data, lower learning rate, or higher alpha feature weights."
        )

    # Save bar chart
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [
            "#2ecc71" if f in alpha_features else "#3498db"
            for f in top10["feature"].tolist()
        ]
        ax.barh(top10["feature"].tolist()[::-1], top10["importance"].tolist()[::-1], color=colors[::-1])
        ax.set_xlabel("XGBoost Feature Importance (gain)")
        ax.set_title("Brain 1 - Top 10 Feature Importances\n(green = institutional alpha, blue = core momentum)")
        plt.tight_layout()
        plot_path = config.LOGS_DIR / "brain1_feature_importance.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Feature importance chart saved -> {plot_path}")
    except Exception as e:
        logger.warning(f"Could not save feature importance chart: {e}")


def train_directional_model(train: pd.DataFrame, test: pd.DataFrame, target_col: str, 
                             model_name: str, model_path: pathlib.Path, 
                             calibrated_path: pathlib.Path) -> tuple[xgb.XGBClassifier, CalibratedClassifierCV]:
    """
    Generic training engine for directional models (LONG or SHORT).
    Uses Isotonic Calibration for true probability mapping.
    """
    logger.info("-" * 50)
    logger.info(f"TRAINING {model_name} -- Target: {target_col}")
    logger.info("-" * 50)

    X_all_tr = train[FEATURE_COLS].fillna(0)
    y_all_tr  = train[target_col].astype(int)
    X_te      = test[FEATURE_COLS].fillna(0)
    y_te      = test[target_col].astype(int)

    n_neg = int((y_all_tr == 0).sum())
    n_pos = int((y_all_tr == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    logger.info(f"Class balance - y=0: {n_neg:,}  y=1: {n_pos:,} | Ratio: {scale_pos_weight:.2f}")

    split_idx = int(len(train) * 0.90)
    X_tr_raw  = X_all_tr.iloc[:split_idx]
    y_tr_raw  = y_all_tr.iloc[:split_idx]
    X_va_raw  = X_all_tr.iloc[split_idx:]
    y_va_raw  = y_all_tr.iloc[split_idx:]

    base_model = xgb.XGBClassifier(
        tree_method        = config.XGBOOST_TREE_METHOD,
        device             = config.XGBOOST_DEVICE,
        max_depth          = config.XGBOOST_MAX_DEPTH,
        learning_rate      = config.XGBOOST_LEARNING_RATE,
        n_estimators       = config.XGBOOST_N_ESTIMATORS,
        objective          = "binary:logistic",
        eval_metric        = "aucpr",
        early_stopping_rounds = config.XGBOOST_EARLY_STOPPING,
        subsample          = config.XGBOOST_SUBSAMPLE,
        colsample_bytree   = config.XGBOOST_COLSAMPLE_BYTREE,
        min_child_weight   = 1,             # Phase 2: Terminal leaves for rare gap events without pruning
        reg_lambda         = config.XGBOOST_REG_LAMBDA,
        scale_pos_weight   = scale_pos_weight,
        verbosity          = 1,
    )

    base_model.fit(X_tr_raw, y_tr_raw, eval_set=[(X_va_raw, y_va_raw)], verbose=50)

    # FIX: Change device back to CPU for inference to avoid 'mismatched devices' 
    # warning when passing Pandas DataFrames to predict_proba() and Calibration.
    base_model.set_params(device="cpu")

    # Phase 3: The "Gain" Diagnostic
    # Explaining "Gain" over "Weight": When dealing with extremely sparse/rare events (like explosive 
    # market gaps < 2%), relying on 'weight' (the raw count of splits) will artificially rank gap features 
    # near the bottom because they mathematically cannot split the trees frequently.
    # We must evaluate by 'gain', which measures the actual loss reduction (contribution to accuracy)
    # when the rare feature IS used. A temporal gap feature might split a tree only twice, but it could 
    # perfectly isolate a high-momentum regime, capturing massive PnL.
    try:
        import matplotlib.pyplot as plt
        xgb.plot_importance(base_model, importance_type='gain', max_num_features=15, title=f"{model_name} Feature Gain")
        plt.tight_layout()
        plt.savefig(str(model_path.parent / f"{model_name}_gain_importance.png"))
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to generate gain diagnostic plot: {e}")

    # Diagnostics
    raw_proba = base_model.predict_proba(X_te)[:, 1]
    raw_brier = brier_score_loss(y_te, raw_proba)
    logger.info(f"Raw {model_name} Brier Score: {raw_brier:.4f}")

    # Calibration
    from src.core.quant_fixes import IsotonicCalibrationWrapper
    calibrator = IsotonicCalibrationWrapper()
    
    # We use a subsample for calibration to be fast
    calib_X = pd.concat([X_va_raw, X_te], ignore_index=True)
    calib_y = pd.concat([y_va_raw, y_te], ignore_index=True)
    
    if len(calib_X) > config.CALIBRATION_SAMPLE_LIMIT:
        calib_X = calib_X.sample(config.CALIBRATION_SAMPLE_LIMIT, random_state=42)
        calib_y = calib_y.loc[calib_X.index]

    # FIX: Weighted isotonic calibration.
    # Without weights, isotonic calibration on unbalanced data squashes SHORT probabilities
    # toward 0.5 because bear-market label_short=1 examples are scarce.
    # Equalising class weights forces the calibration curve to map both y=0 and y=1 evenly.
    calib_weights = None
    if getattr(config, 'CALIBRATION_CLASS_WEIGHT', False):
        n_total = len(calib_y)
        n_pos   = max((calib_y == 1).sum(), 1)
        n_neg   = max((calib_y == 0).sum(), 1)
        calib_weights = np.where(
            calib_y == 1,
            n_total / (2.0 * n_pos),   # up-weight minority class
            n_total / (2.0 * n_neg),   # down-weight majority class
        ).astype(np.float32)
        logger.info(f"Calibration: class weights applied (pos_w={n_total/(2.0*n_pos):.2f}, neg_w={n_total/(2.0*n_neg):.2f})")

    calibrator.fit_on_validation(base_model, calib_X, calib_y, sample_weight=calib_weights)
    
    # Save
    base_model.save_model(str(model_path))
    calibrator.save(calibrated_path)
    logger.info(f"Saved {model_name} -> {calibrated_path}")

    return base_model, calibrator


# ── Brain 2: Conviction Meta-Regressor ─────────────────────────────────────

def train_brain2(train, test, b1_long, b1_short) -> xgb.XGBRegressor:
    logger.info("-" * 50 + "\nTRAINING BRAIN 2 -- Conviction Meta-Regressor")

    def meta(X_base, df_orig):
        prob_long = b1_long.predict_proba(X_base)[:, 1]
        prob_short = b1_short.predict_proba(X_base)[:, 1]
        prob_max = np.maximum(prob_long, prob_short)
        
        # Meta-Model now sees the full context (Trend, Alpha, etc.)
        meta_df = X_base.copy()
        meta_df["brain1_prob"] = prob_max
        return meta_df

    split_idx = int(len(train) * 0.90)
    train_set = train.iloc[:split_idx]
    val_set = train.iloc[split_idx:]

    X_tr = meta(train_set[FEATURE_COLS].fillna(0), train_set)
    X_va = meta(val_set[FEATURE_COLS].fillna(0), val_set)
    X_te = meta(test[FEATURE_COLS].fillna(0), test)
    y_tr, y_va, y_te = train_set["conviction_target"], val_set["conviction_target"], test["conviction_target"]

    m = xgb.XGBRegressor(
        tree_method=config.XGBOOST_TREE_METHOD, device=config.XGBOOST_DEVICE,
        max_depth=config.XGBOOST_MAX_DEPTH, learning_rate=config.XGBOOST_LEARNING_RATE,
        n_estimators=config.XGBOOST_N_ESTIMATORS, objective="reg:squarederror",
        eval_metric="mae", early_stopping_rounds=config.XGBOOST_EARLY_STOPPING, 
        subsample=config.XGBOOST_SUBSAMPLE, reg_lambda=config.XGBOOST_REG_LAMBDA,
        verbosity=1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)

    # FIX: Switch to CPU for inference predictions
    m.set_params(device="cpu")

    logger.info(f"Brain 2 MAE: {mean_absolute_error(y_te, m.predict(X_te)):.2f} | R2: {r2_score(y_te, m.predict(X_te)):.4f}")
    m.save_model(str(config.BRAIN2_MODEL_PATH))
    logger.info(f"Saved -> {config.BRAIN2_MODEL_PATH}")
    return m


# ── Orchestrator ────────────────────────────────────────────────────────────

def run_brain_trainer():
    """
    Orchestrator: uses the single walk_forward_split for production training.
    Applies:
      - Fix 3: Purge/Embargo overlapping training samples (t1 gate FIXED)
      - Phase 3: Isotonic Calibration after Brain1 training
      - Phase 3: colsample_bytree=0.7 + subsample=0.7 (pessimistic hyperparams)
    Call walk_forward_rolling_splits() for research / OOS validation.
    """
    logger.info("=" * 70)
    logger.info(f"BRAIN TRAINER -- GPU: {config.XGBOOST_TREE_METHOD} / {config.XGBOOST_DEVICE}")
    logger.info(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    logger.info("=" * 70)

    df = load_all_features()

    train, test = walk_forward_split(df)
    
    # FIX: Explicitly free the massive 54M row master DataFrame before XGBoost 
    # allocates its internal DMatrix structure.
    del df
    import gc
    gc.collect()

    if "t1" in train.columns and getattr(config, "ENABLE_PURGE_EMBARGO", True):
        logger.info("Applying Purge/Embargo to remove overlapping training samples (Zero-Copy Optimization)...")
        # ZERO-COPY MEMORY OPTIMIZATION for Bug #11
        # Instead of heavy set_index() which forces 50GB copies, use direct masking
        test_start = test["brick_timestamp"].min()
        test_end = test["brick_timestamp"].max()
        
        n_embargo = int(len(test) * config.EMBARGO_PCT)
        embargo_cutoff = test_end + pd.Timedelta(minutes=n_embargo)
        
        # Purge: t1 >= test_start ensures training sample exit doesn't bleed into test window
        # Embargo: timestamp >= embargo_cutoff drops all future training data until after the embargo
        drop_mask = (train["t1"] >= test_start) & (train["brick_timestamp"] < embargo_cutoff)
        
        purged = drop_mask.sum()
        train = train[~drop_mask].reset_index(drop=True)
        logger.info(f"Purge/Embargo removed {purged} rows. Remaining: {len(train):,}")
    else:
        logger.info("Purge/Embargo skipped.")

    b1_long, b1_long_calib = train_directional_model(
        train, test, "label_long", "Brain1 (LONG)", 
        config.BRAIN1_MODEL_LONG_PATH, config.BRAIN1_CALIBRATED_LONG_PATH
    )
    b1_short, b1_short_calib = train_directional_model(
        train, test, "label_short", "Brain1 (SHORT)", 
        config.BRAIN1_MODEL_SHORT_PATH, config.BRAIN1_CALIBRATED_SHORT_PATH
    )
    
    train_brain2(train, test, b1_long_calib, b1_short_calib)
    logger.info("BRAIN TRAINER COMPLETE")
    logger.info(f"Calibrated LONG model saved at: {config.BRAIN1_CALIBRATED_LONG_PATH}")
    logger.info(f"Calibrated SHORT model saved at: {config.BRAIN1_CALIBRATED_SHORT_PATH}")


if __name__ == "__main__":
    run_brain_trainer()
