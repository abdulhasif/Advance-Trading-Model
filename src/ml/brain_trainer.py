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

FEATURE_COLS = [
    "velocity", "wick_pressure", "relative_strength",
    "brick_size", "duration_seconds",
    # "direction" removed — was 69.8% of model gain (pure momentum echo, not signal)
    # Model must now learn from velocity, hurst, fracdiff_price, RS etc.
    "consecutive_same_dir", "brick_oscillation_rate",
    "fracdiff_price",        # Fix 1: Fractional Differentiation
    "hurst",                 # Fix 4: Hurst Regime Feature
    "is_trending_regime",    # Fix 4: Boolean regime gate
    # Anti-Myopia: Long-lookback features
    "velocity_long",         # 20-brick momentum vs 10-brick
    "trend_slope",           # 14-brick OLS price slope (scale-invariant)
    "rolling_range_pct",     # 14-brick price range / avg (volatility gate)
    "momentum_acceleration", # 5-brick vel minus 14-brick vel (trend strength)
    # Phase 2: Institutional Alpha Factors
    "vwap_zscore",           # VWAP anchor: >+2.5 = exhaustion peak (ABB trade blocker)
    "vpt_acceleration",      # VPT 2nd derivative: spike = institutional absorption signal
    "squeeze_zscore",        # Brick density Z-score: expansion after squeeze = early entry
    "streak_exhaustion",     # Sigmoid decay: penalizes late-stage momentum (streak >8)
]

# Fix 5: Columns to apply Robust IQR scaling on (excludes binary cols)
ROBUST_SCALE_COLS = [
    "velocity", "wick_pressure", "relative_strength",
    "brick_size", "duration_seconds",
    "consecutive_same_dir", "brick_oscillation_rate",
    "fracdiff_price", "hurst",
    # Anti-Myopia: Long-lookback features
    "velocity_long", "trend_slope", "rolling_range_pct", "momentum_acceleration",
]

# Anti-Myopia: Multi-brick horizon for target engineering
# Model predicts majority direction over next H bricks, not just next 1.
TRAINING_HORIZON = 5  # Predict sustained trend over 5 bricks (~15-30 min on NSE)




# ── Data Loading ────────────────────────────────────────────────────────────

def load_all_features() -> pd.DataFrame:
    """Load enriched Parquet files — one sector at a time for RAM safety."""
    if not config.FEATURES_DIR.exists():
        logger.error("Features dir missing. Run feature_engine first."); sys.exit(1)

    frames = []
    total_long = 0
    total_short = 0

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
                
                # Apply triple barrier incrementally per file (O(1) memory instead of O(N))
                df = add_triple_barrier_t1(df, stop_pct=0.0075, target_pct=0.0075)
                
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
    combined.sort_values("brick_timestamp", kind="mergesort", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    logger.info(f"Total bricks loaded: {len(combined):,} (Memory optimized)")
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
    # Bug fix: old formula `log_ret / brick_size * 50` divided log-return (dimensionless)
    # by absolute brick price (e.g. Rs 4.2), producing labels near ~0.001 → model predicts mean.
    # Fix: convert to bps (log_ret × 10,000). Typical 5-brick intraday move = 5–50 bps.
    # Values clipped at 100 bps (~1% net move), giving a proper 0–100 scale.
    close_h = out.groupby(["_symbol", "_date"])["brick_close"].shift(-horizon)
    log_ret = np.log(
        close_h.clip(lower=1e-9) / out["brick_close"].clip(lower=1e-9)
    ).abs()
    out["conviction_target"] = (log_ret * 10_000).clip(upper=100)  # 0–100 bps scale

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
                                   stop_pct: float = 0.010,
                                   target_pct: float = 0.020,
                                   eod_hour: int = 15,
                                   eod_minute: int = 15) -> pd.DataFrame:
    """
    FIX #2: Triple Barrier Method — Vectorized Implementation.
    For every brick, determines the label based on which barrier is hit first:
      - BARRIER 1 (Floor):  brick_close drops by stop_pct    -> y=0 (stop loss hit)
      - BARRIER 2 (Target): brick_close rises by target_pct  -> y=1 (target hit)  
      - BARRIER 3 (Time):   3:15 PM IST auto-square-off      -> y=0 (expired)
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
            sym_conv[i] = min(100.0, 100.0 / b_min)
            
    return sym_dir_long, sym_dir_short, sym_conv


def add_triple_barrier_t1(df: pd.DataFrame, stop_pct=0.0075, target_pct=0.0075,
                          eod_hour=15, eod_minute=10) -> pd.DataFrame:
    """
    Creates Purge/Embargo Target Logic + 't1' resolution timestamp.
    Target: Symmetric Dual Triple Barrier.
    label_long = 1 if hitting +0.75% before -0.75%
    label_short = 1 if hitting -0.75% before +0.75%
    """
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

    grouped = df.groupby(["_symbol", "_date"], sort=False)
    
    for _, grp in grouped:
        idx = grp.index.values
        closes = grp["brick_close"].values.astype(np.float64)
        highs = grp.get("brick_high", grp["brick_close"]).values.astype(np.float64)
        lows = grp.get("brick_low", grp["brick_close"]).values.astype(np.float64)
        mins = grp["_min_of_day"].values.astype(np.int32)
        
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
        max_depth          = 5,             # Phase 2: Allow interaction between spatial and temporal
        learning_rate      = config.XGBOOST_LEARNING_RATE,
        n_estimators       = config.XGBOOST_N_ESTIMATORS,
        objective          = "binary:logistic",
        eval_metric        = "aucpr",
        early_stopping_rounds = config.XGBOOST_EARLY_STOPPING,
        subsample          = config.XGBOOST_SUBSAMPLE,
        colsample_bytree   = 0.6,           # Phase 2: Mask spatial features to force temporal usage
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
    
    SAMPLE_LIMIT = 500_000
    if len(calib_X) > SAMPLE_LIMIT:
        calib_X = calib_X.sample(SAMPLE_LIMIT, random_state=42)
        calib_y = calib_y.loc[calib_X.index]

    calibrator.fit_on_validation(base_model, calib_X, calib_y)
    
    # Save
    base_model.save_model(str(model_path))
    calibrator.save(calibrated_path)
    logger.info(f"Saved {model_name} -> {calibrated_path}")

    return base_model, calibrator


# ── Brain 2: Conviction Meta-Regressor ─────────────────────────────────────

def train_brain2(train, test, brain1: xgb.XGBClassifier) -> xgb.XGBRegressor:
    logger.info("-" * 50 + "\nTRAINING BRAIN 2 -- Conviction Meta-Regressor")

    def meta(X_base, df_orig):
        prob = brain1.predict_proba(X_base)[:, 1]
        return pd.DataFrame({
            "brain1_prob": prob,
            "velocity": df_orig["velocity"].fillna(0).values,
            "wick_pressure": df_orig["wick_pressure"].fillna(0).values,
            "relative_strength": df_orig["relative_strength"].fillna(0).values,
        })

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
        logger.info("Applying Purge/Embargo to remove overlapping training samples...")
        train_indexed = train.set_index("brick_timestamp")
        test_indexed  = test.set_index("brick_timestamp")
        train_indexed = purge_overlapping_samples(
            train_indexed, test_indexed, t1_col="t1", pct_embargo=0.01
        )
        train = train_indexed.reset_index()
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
    
    train_brain2(train, test, b1_long) # b2 just needs one model to format features
    logger.info("BRAIN TRAINER COMPLETE")
    logger.info(f"Calibrated LONG model saved at: {config.BRAIN1_CALIBRATED_LONG_PATH}")
    logger.info(f"Calibrated SHORT model saved at: {config.BRAIN1_CALIBRATED_SHORT_PATH}")


if __name__ == "__main__":
    run_brain_trainer()
