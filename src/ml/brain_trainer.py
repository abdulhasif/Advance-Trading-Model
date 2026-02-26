"""
src/ml/brain_trainer.py — Phase 3: Dual XGBoost GPU Trainer
=============================================================
Brain 1 (Direction Classifier) + Brain 2 (Conviction Meta-Regressor).
Uses tree_method='gpu_hist' + device='cuda' for RTX 3050 acceleration.

Run:  python -m src.ml.brain_trainer
"""

import sys
import logging
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

import config
from src.core.quant_fixes import (
    purge_overlapping_samples,
    RobustFeatureScaler,
    apply_quantile_transformer,
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
]

# Fix 5: Columns to apply Robust IQR scaling on (excludes binary cols)
ROBUST_SCALE_COLS = [
    "velocity", "wick_pressure", "relative_strength",
    "brick_size", "duration_seconds",
    "consecutive_same_dir", "brick_oscillation_rate",
    "fracdiff_price", "hurst",
]



# ── Data Loading ────────────────────────────────────────────────────────────

def load_all_features() -> pd.DataFrame:
    """Load enriched Parquet files — one sector at a time for RAM safety."""
    if not config.FEATURES_DIR.exists():
        logger.error("Features dir missing. Run feature_engine first."); sys.exit(1)

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
        logger.error("No feature files."); sys.exit(1)

    combined = pd.concat(frames, ignore_index=True).sort_values("brick_timestamp", kind="mergesort").reset_index(drop=True)
    logger.info(f"Total bricks: {len(combined):,}")
    return combined


# ── Target Engineering ──────────────────────────────────────────────────────

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_date"] = out["brick_timestamp"].dt.date
    
    # Sort first to ensure deterministic ordering
    out = out.sort_values(["_symbol", "brick_timestamp"], kind="mergesort").reset_index(drop=True)

    # Vectorized shift grouped by symbol AND date strictly isolates days
    # The last brick of the day will properly receive NaN and be dropped
    next_dir = out.groupby(["_symbol", "_date"])["direction"].shift(-1)
    out["direction_target"] = np.where(next_dir.isna(), np.nan, (next_dir > 0).astype(float))

    next_close = out.groupby(["_symbol", "_date"])["brick_close"].shift(-1)
    move = (next_close - out["brick_close"]).abs()
    out["conviction_target"] = (move / out["brick_size"].clip(lower=1e-9) * 50).clip(upper=100)

    out = out.drop(columns=["_date"])
    return out.dropna(subset=["direction_target", "conviction_target"]).reset_index(drop=True)


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
    y=1 ONLY if Barrier 2 is hit before Barrier 1 AND before 3:15 PM.

    Uses only future bricks within the SAME day (day-boundary safe).
    Operates in log-return space for consistency with Renko's percentage basis.
    """
    df = df.copy()
    df = df.sort_values(["_symbol", "brick_timestamp"], kind="mergesort").reset_index(drop=True)
    df["_date"] = df["brick_timestamp"].dt.date

    direction_labels = []
    conviction_labels = []

    for (sym, date), grp in df.groupby(["_symbol", "_date"], sort=False):
        grp = grp.reset_index(drop=True)
        closes = grp["brick_close"].values
        times  = grp["brick_timestamp"].values  # numpy datetime64

        sym_dir  = []
        sym_conv = []

        for i in range(len(grp)):
            entry = closes[i]
            stop_level   = entry * (1 - stop_pct)
            target_level = entry * (1 + target_pct)

            label     = 0    # default: barrier 3 (time) hit or stop hit
            bricks_to = np.nan

            # Look forward within the same day after the entry brick
            for j in range(i + 1, len(grp)):
                ts = pd.Timestamp(times[j])
                # Barrier 3: end of day
                if ts.hour > eod_hour or (ts.hour == eod_hour and ts.minute >= eod_minute):
                    break
                price = closes[j]
                if price <= stop_level:
                    label = 0         # stop loss hit first
                    bricks_to = j - i
                    break
                if price >= target_level:
                    label = 1         # target hit first
                    bricks_to = j - i
                    break

            sym_dir.append(float(label))
            # conviction = inverse of bricks_to_hit (faster = higher conviction)
            sym_conv.append(min(100.0, 100.0 / max(bricks_to, 1)) if not np.isnan(bricks_to) else 20.0)

        direction_labels.extend(sym_dir)
        conviction_labels.extend(sym_conv)

    df["direction_target"]  = direction_labels
    df["conviction_target"] = conviction_labels
    df = df.drop(columns=["_date"])

    n_pos = int((df["direction_target"] == 1).sum())
    n_neg = int((df["direction_target"] == 0).sum())
    logger.info(f"Triple Barrier: y=1 (target hit): {n_pos:,}  y=0 (stop/time): {n_neg:,}  "
                f"Ratio: {n_neg/max(n_pos,1):.1f}:1")
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
    return train.copy(), test.copy()


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

def train_brain1(train, test) -> xgb.XGBClassifier:
    logger.info("-" * 50 + "\nTRAINING BRAIN 1 -- Direction Classifier")
    split_idx = int(len(train) * 0.90)
    train_set = train.iloc[:split_idx]
    val_set   = train.iloc[split_idx:]

    X_tr, y_tr = train_set[FEATURE_COLS].fillna(0), train_set["direction_target"]
    X_va, y_va = val_set[FEATURE_COLS].fillna(0),   val_set["direction_target"]
    X_te, y_te = test[FEATURE_COLS].fillna(0),      test["direction_target"]

    # FIX #12: Class imbalance correction via scale_pos_weight
    # Computed STRICTLY from training data — never touches the test set
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)  # e.g., 3.0 if 75/25 split
    logger.info(f"Brain1 class balance — y=0: {n_neg:,}  y=1: {n_pos:,}  "
                f"scale_pos_weight: {scale_pos_weight:.2f}")

    m = xgb.XGBClassifier(
        tree_method=config.XGBOOST_TREE_METHOD, device=config.XGBOOST_DEVICE,
        max_depth=config.XGBOOST_MAX_DEPTH, learning_rate=config.XGBOOST_LEARNING_RATE,
        n_estimators=config.XGBOOST_N_ESTIMATORS, objective="binary:logistic",
        eval_metric="logloss", early_stopping_rounds=config.XGBOOST_EARLY_STOPPING,
        subsample=config.XGBOOST_SUBSAMPLE, reg_lambda=config.XGBOOST_REG_LAMBDA,
        scale_pos_weight=scale_pos_weight,   # FIX #12: Balances classes without SMOTE
        use_label_encoder=False, verbosity=1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)

    acc = accuracy_score(y_te, m.predict(X_te))
    logger.info(f"Brain 1 Accuracy: {acc:.4f}")
    logger.info(f"\n{classification_report(y_te, m.predict(X_te), target_names=['Down','Up'])}")

    m.save_model(str(config.BRAIN1_MODEL_PATH))
    logger.info(f"Saved -> {config.BRAIN1_MODEL_PATH}")
    return m


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

    logger.info(f"Brain 2 MAE: {mean_absolute_error(y_te, m.predict(X_te)):.2f} | R2: {r2_score(y_te, m.predict(X_te)):.4f}")
    m.save_model(str(config.BRAIN2_MODEL_PATH))
    logger.info(f"Saved -> {config.BRAIN2_MODEL_PATH}")
    return m


# ── Orchestrator ────────────────────────────────────────────────────────────

def run_brain_trainer():
    """
    Orchestrator: uses the single walk_forward_split for production training.
    Applies:
      - Fix 3: Purge/Embargo overlapping training samples
      - Fix 5: Robust IQR scaling fitted ONLY on training set
    Call walk_forward_rolling_splits() for research / OOS validation.
    """
    logger.info("=" * 70)
    logger.info(f"BRAIN TRAINER -- GPU: {config.XGBOOST_TREE_METHOD} / {config.XGBOOST_DEVICE}")
    logger.info(f"Features: {FEATURE_COLS}")
    logger.info("=" * 70)

    df = create_targets(load_all_features())

    # Log class distribution for transparency
    n_pos = int((df["direction_target"] == 1).sum())
    n_neg = int((df["direction_target"] == 0).sum())
    logger.info(f"Full dataset class balance \u2014 y=1: {n_pos:,}  y=0: {n_neg:,}  "
                f"Imbalance ratio: {n_neg/max(n_pos,1):.2f}")

    train, test = walk_forward_split(df)
    del df

    # ── Fix 3: Purge/Embargo ─────────────────────────────────────────────
    # Drop training samples whose Triple Barrier window overlaps the test set.
    # Requires 't1' column produced by add_triple_barrier_t1().
    if "t1" in train.columns and isinstance(train.index, pd.DatetimeIndex):
        logger.info("Applying Purge/Embargo to remove overlapping training samples...")
        train = purge_overlapping_samples(train, test, t1_col="t1", pct_embargo=0.01)
    else:
        logger.info("Skipping Purge/Embargo: 't1' column absent or index not DatetimeIndex.")

    # ── Fix 5: Robust IQR Scaling ────────────────────────────────────────
    # Fit scaler ONLY on training data, then transform both train and test.
    # This is the mathematically correct way to prevent future-mean leakage.
    scale_cols = [c for c in ROBUST_SCALE_COLS if c in train.columns]
    if scale_cols:
        scaler = RobustFeatureScaler(quantile_range=(25.0, 75.0))
        train = scaler.fit_transform(train, scale_cols)
        test  = scaler.transform(test,  scale_cols)
        logger.info(f"Robust IQR scaling applied to {len(scale_cols)} features.")

    b1 = train_brain1(train, test)
    train_brain2(train, test, b1)
    logger.info("BRAIN TRAINER COMPLETE")


if __name__ == "__main__":
    run_brain_trainer()
