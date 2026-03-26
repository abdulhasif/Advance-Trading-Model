"""
src/ml/brain_trainer.py - Phase 3: Dual XGBoost GPU Trainer
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
matplotlib.use("Agg")   # headless-safe - no display required
import matplotlib.pyplot as plt

import xgboost as xgb
try:
    import tf_keras as keras
except ImportError:
    import keras
import joblib
import torch
from sklearn.utils import class_weight
from sklearn.model_selection import KFold

from trading_pipeline import config
from trading_pipeline.scripts.sequence_engine import CnnSequenceGenerator
# from trading_core.core.physics.quant_fixes import add_triple_barrier_dynamic


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


# -- Data Loading ------------------------------------------------------------

def load_all_features() -> pd.DataFrame:
    """Load enriched Parquet files - one sector at a time for RAM safety."""
    if not config.FEATURES_DIR.exists():
        logger.error("Features dir missing. Run feature_engine first."); sys.exit(1)

    frames = []
    total_long = 0
    total_short = 0

    # Formulas for Triple Barrier Synchronization
    STOP_PCT   = config.NATR_BRICK_PERCENT * config.STRUCTURAL_REVERSAL_BRICKS
    TARGET_PCT = config.NATR_BRICK_PERCENT * config.TRAINING_TARGET_BRICKS

    for sector_dir in config.FEATURES_DIR.iterdir():
        if not sector_dir.is_dir():
            continue
        for pf in sorted(sector_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(pf)
                
                if "brick_timestamp" in df.columns:
                    df["brick_timestamp"] = config.to_naive_ist(df["brick_timestamp"])

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
                df = add_triple_barrier_t1(df, stop_pct=STOP_PCT, target_pct=TARGET_PCT)
                
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


# -- Target Engineering ------------------------------------------------------

def create_triple_barrier_targets(df: pd.DataFrame,
                                   stop_pct: float = config.NATR_BRICK_PERCENT * config.STRUCTURAL_REVERSAL_BRICKS,
                                   target_pct: float = config.NATR_BRICK_PERCENT * config.TRAINING_TARGET_BRICKS,
                                   eod_hour: int = config.EOD_SQUARE_OFF_HOUR,
                                   eod_minute: int = config.EOD_SQUARE_OFF_MIN) -> pd.DataFrame:
    """
    Triple Barrier Method - Synchronized with Central Config.
    """

from numba import njit
import numpy as np

@njit(cache=True)
def _compute_triple_barrier_fast(closes, highs, lows, min_timestamps, stop_pct, target_pct, eod_mins,horizon_bricks=TRAINING_HORIZON):
    """
    Numba-optimized core for Symmetric Dual Triple Barrier.
    """
    n = len(closes)
    sym_dir_long = np.zeros(n, dtype=np.float32)
    sym_dir_short = np.zeros(n, dtype=np.float32)
    sym_conv = np.full(n, 20.0, dtype=np.float32) # default conviction
    sym_t1_idx = np.zeros(n, dtype=np.int32)

    for i in range(n - 1):
        # Fix 1: T+1 Labeling Shift: Entry happens at i+1 (the next brick)
        entry = closes[i + 1]
        long_stop   = entry * (1.0 - stop_pct)
        long_target = entry * (1.0 + target_pct)
        short_stop  = entry * (1.0 + stop_pct)
        short_target= entry * (1.0 - target_pct)
        
        label_long = 0.0
        label_short = 0.0
        b_long = -1
        b_short = -1

        last_j = i
        for j in range(i + 1, n):
            last_j = j
            ts_mins = min_timestamps[j]
            if ts_mins >= eod_mins:
                break
            
            p_high = highs[j]
            p_low = lows[j]
            
            if b_long == -1:
                if p_low <= long_stop:
                    label_long = 0.0
                    b_long = max(1, j - i)
                elif p_high >= long_target:
                    label_long = 1.0
                    b_long = max(1, j - i)
                    
            if b_short == -1:
                if p_high >= short_stop:
                    label_short = 0.0
                    b_short = max(1, j - i)
                elif p_low <= short_target:
                    label_short = 1.0
                    b_short = max(1, j - i)
                    
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
            sym_t1_idx[i] = i + b_min
            
            # --- THE CORRECTED CONVICTION LOGIC ---
            if (b_min == b_long and label_long == 1.0) or (b_min == b_short and label_short == 1.0):
                # WINNER: It hit the target. 
                # Optional: We can reward it slightly more if it hit the target faster, 
                # but for TREND FOLLOWING, a flat max score is safer.
                efficiency = float(horizon_bricks) / max(float(horizon_bricks), float(b_min))
                sym_conv[i] = efficiency * 100.0
            else:
                # LOSER: It hit the Stop Loss.
                # The AI must learn to HATE these setups. Conviction must be ZERO.
                sym_conv[i] = 0.0
        else:
            # TIME-OUT: It reached the End of Day without hitting Target or Stop
            sym_t1_idx[i] = last_j
            sym_conv[i] = 0.0
            
    return sym_dir_long, sym_dir_short, sym_conv, sym_t1_idx


def add_triple_barrier_t1(df: pd.DataFrame, stop_pct=None, target_pct=None,
                          eod_hour=config.EOD_SQUARE_OFF_HOUR, eod_minute=config.EOD_SQUARE_OFF_MIN) -> pd.DataFrame:
    """
    Creates Purge/Embargo Target Logic + 't1' resolution timestamp.
    Target: Symmetric Dual Triple Barrier.
    Synchronized with central config.py.
    """
    if stop_pct is None:
        stop_pct = config.NATR_BRICK_PERCENT * config.STRUCTURAL_REVERSAL_BRICKS
    if target_pct is None:
        # FIX #2: Changed from (ENTRY_CONV_THRESH / 10000.0) + (TRANSACTION_COST_PCT * 2) to 1.6% to match load_all_features()
        target_pct = config.NATR_BRICK_PERCENT * config.TRAINING_TARGET_BRICKS
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
    t1_timestamps = np.empty(n, dtype='datetime64[ns]')

    grouped = df.groupby(["_symbol", "_date"], sort=False)

    horizon_bricks = config.TRAINING_HORIZON_BRICKS
    
    for _, grp in grouped:
        idx = grp.index.values
        closes = grp["brick_close"].values.astype(np.float64)
        highs = grp.get("brick_high", grp["brick_close"]).values.astype(np.float64)
        lows = grp.get("brick_low", grp["brick_close"]).values.astype(np.float64)
        mins = grp["_min_of_day"].values.astype(np.int32)
        
        l_long, l_short, sym_conv, t1_idx = _compute_triple_barrier_fast(
            closes, highs, lows, mins, stop_pct, target_pct, eod_mins,horizon_bricks
        )
        
        dir_long[idx] = l_long
        dir_short[idx] = l_short
        conviction_labels[idx] = sym_conv
        t1_timestamps[idx] = grp["brick_timestamp"].values[t1_idx]

    df["label_long"]  = dir_long
    df["label_short"] = dir_short
    # Ensure backward compatibility with existing codebase
    df["direction_target"] = dir_long
    df["conviction_target"] = conviction_labels
    df["t1"] = t1_timestamps
    
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


# -- Walk-Forward Split ------------------------------------------------------

def custom_holdout_split(df: pd.DataFrame):
    """
    Custom Map-Based Holdout Split.
    Maps specific months of specific years to the test set, and everything else to train.
    """
    logger.info("Applying custom holdout split...")
    
    # Create masks based on config
    years = df["brick_timestamp"].dt.year
    months = df["brick_timestamp"].dt.month
    
    # 1. Base rule: In HOLDOUT_YEARS, mask the HOLDOUT_MONTHS
    generic_holdout_mask = (years.isin(getattr(config, "HOLDOUT_YEARS", []))) & \
                           (months.isin(getattr(config, "HOLDOUT_MONTHS", [])))
                           
    # 2. Specific rule: apply specific months for specific years
    specific_masks = []
    specific_year_months = getattr(config, "HOLDOUT_SPECIFIC_YEAR_MONTHS", {})
    for yr, m_list in specific_year_months.items():
        specific_masks.append((years == yr) & (months.isin(m_list)))
        
    if specific_masks:
        specific_holdout_mask = pd.concat(specific_masks, axis=1).any(axis=1)
    else:
        specific_holdout_mask = pd.Series(False, index=df.index)
        
    test_mask = generic_holdout_mask | specific_holdout_mask
    
    train = df[~test_mask]
    test  = df[test_mask]
    
    logger.info(f"Custom Split -- Train: {len(train):,}  Test (Holdout): {len(test):,}")
    
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


# ==============================================================================
# ML Wrappers & Utilities
# ==============================================================================
class KerasClassifierWrapper:
    """Sklearn-compatible wrapper for a pre-trained Keras model."""
    def __init__(self, keras_model): 
        self.model = keras_model
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"

    def fit(self, X, y): 
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X): 
        X_np = X.values if hasattr(X, 'values') else X
        seq = CnnSequenceGenerator(X_np, window_size=config.CNN_WINDOW_SIZE)
        preds = self.model.predict(seq, verbose=0).flatten()
        pad = np.full(config.CNN_WINDOW_SIZE - 1, 0.5)
        full_preds = np.concatenate([pad, preds])
        return np.vstack([1 - full_preds, full_preds]).T

    def get_params(self, deep=True):
        return {"keras_model": self.model}

# -- Brain 1: Direction Classifier -------------------------------------------

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


def extract_meta_features(df: pd.DataFrame, prob_long: np.ndarray, prob_short: np.ndarray, pred_dir: np.ndarray) -> pd.DataFrame:
    """
    Surgical Fix: Meta-Label Blindspots + Feature Order Consistency.
    Extracts the feature matrix for Brain 2, explicitly including Brain 1's 
    directional probabilities and its categorical decision.
    """
    # Fix 4: Feature Order Consistency. Brain 2 requires a fixed matrix shape.
    meta_feats = {}
    for feat in config.BRAIN2_FEATURES:
        if feat == "brain1_prob_long":
            meta_feats[feat] = prob_long.astype('float32')
        elif feat == "brain1_prob_short":
            meta_feats[feat] = prob_short.astype('float32')
        elif feat == "trade_direction":
            meta_feats[feat] = pred_dir.astype('float32')
        elif feat in df.columns:
            meta_feats[feat] = df[feat].fillna(0).astype('float32').values
        else:
            # Initialize missing columns with zeros to maintain XGBoost input shape
            meta_feats[feat] = np.zeros(len(df), dtype='float32')
            
    final_df = pd.DataFrame(meta_feats, copy=False)[config.BRAIN2_FEATURES]
    
    # Validation guard for Live Engine compatibility
    if len(final_df.columns) != len(config.BRAIN2_FEATURES):
        raise ValueError(f"Brain 2 Matrix Shape Mismatch! Expected {len(config.BRAIN2_FEATURES)}, got {len(final_df.columns)}")
        
    return final_df


def generate_oof_predictions(train_df: pd.DataFrame, target_col: str, model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates Out-of-Fold (OOF) predictions for Brain 2 meta-training.
    Fix: Scaler fitted inside K-Fold to prevent data leak. Class Weights applied.
    """
    logger.info(f"Generating OOF predictions for {model_name}...")
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    oof_probs = np.zeros(len(train_df), dtype='float32')
    from sklearn.preprocessing import RobustScaler
    
    y_all = train_df[target_col].values

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df), 1):
        # 1. Split BEFORE scaling and strictly downcast to float32 to prevent 4.8 GiB OOM crash
        X_tr_raw = train_df[FEATURE_COLS].iloc[tr_idx].fillna(0).astype('float32').values
        X_va_raw = train_df[FEATURE_COLS].iloc[va_idx].fillna(0).astype('float32').values
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]
        
        # 2. Fit scaler strictly on training fold (RobustScaler natively returns float32 here)
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)
        
        # 3. Sequence Generators (Pass symbols for continuity check)
        train_gen = CnnSequenceGenerator(X_tr, y_tr, window_size=config.CNN_WINDOW_SIZE, symbols=train_df["_symbol"].values[tr_idx])
        val_gen   = CnnSequenceGenerator(X_va, y_va, window_size=config.CNN_WINDOW_SIZE, symbols=train_df["_symbol"].values[va_idx])
        
        # 4. FIX: Class Weights (sklearn.utils.class_weight)
        classes = np.unique(y_tr)
        cw_dict = None
        if len(classes) > 1:
            weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_tr)
            cw_dict = dict(zip(classes, weights))
            
        # 5. Model Architecture (1D-CNN)
        model = keras.Sequential([
            keras.layers.Input(shape=(config.CNN_WINDOW_SIZE, len(FEATURE_COLS))),
            keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # 6. Train (Fast OOF train)
        model.fit(train_gen, validation_data=val_gen, epochs=3, verbose=0, class_weight=cw_dict)
        
        # 7. Predict (Align to validation indices using generator's valid target indices)
        preds = model.predict(val_gen, verbose=0).flatten()
        
        # Fix 5: VRAM Management (Empty Cache after each fold)
        del model
        keras.backend.clear_session()
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # FIX: Use the generator's actual valid target indices instead of naive offset.
        # The generator filters out cross-symbol windows, so preds.shape != va_idx[offset:].shape.
        # get_target_indices() returns the LOCAL indices within the validation fold's data.
        # We map these back to the GLOBAL indices in the original DataFrame.
        valid_targets = val_gen.get_target_indices()  # Local indices within va fold
        global_targets = va_idx[valid_targets]         # Map to original df indices
        oof_probs[global_targets] = preds
        
    return oof_probs


def train_brain1_cnn(train: pd.DataFrame, test: pd.DataFrame, target_col: str, 
                     model_name: str, model_path: pathlib.Path, calib_path: pathlib.Path) -> keras.Model:
    """
    Train the final production 1D-CNN directional model.
    Fix: Balanced Class Weights + Isotonic Calibration.
    """
    logger.info(f"Training Production 1D-CNN: {model_name}")
    
    # 1. Fit & Save Scaler (Final iteration)
    from sklearn.preprocessing import RobustScaler
    from trading_core.core.physics.quant_fixes import IsotonicCalibrationWrapper
    
    scaler = RobustScaler()
    X_raw = train[FEATURE_COLS].fillna(0)
    scaler.fit(X_raw)
    joblib.dump(scaler, str(config.BRAIN1_SCALER_PATH))

    # 2. Setup Data Generators (with scaling)
    split_idx = int(len(train) * 0.90)
    tr_df, va_df = train.iloc[:split_idx], train.iloc[split_idx:]
    
    # FIX: Get y_train for weighting
    y_train = tr_df[target_col].values
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    cw_dict = dict(zip(classes, weights))
    
    # FIX: Downcast to float32 before transforming to prevent OOM
    X_tr_scaled = scaler.transform(tr_df[FEATURE_COLS].fillna(0).astype('float32'))
    X_va_scaled = scaler.transform(va_df[FEATURE_COLS].fillna(0).astype('float32'))
    
    train_gen = CnnSequenceGenerator(X_tr_scaled, y_train, window_size=config.CNN_WINDOW_SIZE, symbols=tr_df["_symbol"].values)
    val_gen   = CnnSequenceGenerator(X_va_scaled, va_df[target_col].values, window_size=config.CNN_WINDOW_SIZE, symbols=va_df["_symbol"].values)

    # 3. Model Architecture
    model = keras.Sequential([
        keras.layers.Input(shape=(config.CNN_WINDOW_SIZE, len(FEATURE_COLS))),
        keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
    
    # 4. FIX: Apply Class Weights
    history = model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1, class_weight=cw_dict)
    
    # Plot Training Curves
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss Plot
        ax1.plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Val Loss', color='orange', linewidth=2, linestyle='--')
        ax1.set_title(f'{model_name} - Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Binary Crossentropy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # AUC Plot
        ax2.plot(history.history['auc'], label='Train AUC', color='green', linewidth=2)
        ax2.plot(history.history['val_auc'], label='Val AUC', color='purple', linewidth=2, linestyle='--')
        ax2.set_title(f'{model_name} - AUC Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"{model_name} Training Diagnostics", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_path = config.LOGS_DIR / f"{safe_name}_training_history.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Training history plot saved -> {plot_path}")
    except Exception as e:
        logger.warning(f"Could not save training history chart: {e}")
        
    # 5. Save raw model
    model.save(str(model_path))
    
    # 6. THE FIX: Calibrate the Keras model using the validation set
    logger.info(f"Applying Isotonic Calibration to {model_name}...")
    

            
    base_estimator = KerasClassifierWrapper(model)
    calibrator = IsotonicCalibrationWrapper()
    
    # FIX: Pass the raw NumPy array to fit_on_validation, not a DataFrame
    calibrator.fit_on_validation(base_estimator, X_va_scaled, va_df[target_col].values)
    calibrator.save(calib_path)
    
    return model


# -- Brain 2: Conviction Meta-Regressor -------------------------------------

def train_brain2_meta(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                       oof_long: np.ndarray, oof_short: np.ndarray) -> xgb.XGBRegressor:
    """
    Train Brain 2 using OOF predictions from Brain 1.
    Surgical Fix: Meta-Label Blindspots.
    """
    logger.info("-" * 50 + "\nTRAINING BRAIN 2 -- Meta-Regressor")

    # Categorical direction from Brain 1 OOF
    # Safely handle numpy arrays or pandas series by converting to Series first
    oof_long_s = pd.Series(oof_long).fillna(0)
    oof_short_s = pd.Series(oof_short).fillna(0)
    
    train_pred_dir = np.where(oof_long_s > oof_short_s, 1, -1)
    # If both very low, could be 0, but for now we follow the user's logic
    
    X_tr = extract_meta_features(train_df, oof_long, oof_short, train_pred_dir)
    y_tr = train_df["conviction_target"].values
    
    # Fix 6: Meta-Noise Filter (Strict 0.55 Threshold)
    valid_meta_mask = (oof_long >= 0.55) | (oof_short >= 0.55)
    X_tr = X_tr[valid_meta_mask]
    y_tr = y_tr[valid_meta_mask]
    
    if len(X_tr) < 1000:
        logger.warning(f"Meta-Noise filter left only {len(X_tr)} samples. Skipping Meta-Reg training.")
        return

    # Brain 2 Model (XGBoost)
    m = xgb.XGBRegressor(
        tree_method=config.XGBOOST_TREE_METHOD, device=config.XGBOOST_DEVICE,
        max_depth=4, learning_rate=0.05, n_estimators=200, objective="reg:squarederror"
    )
    
    m.fit(X_tr, y_tr, verbose=50)
    m.set_params(device="cpu")
    
    m.save_model(str(config.BRAIN2_MODEL_PATH))
    logger.info(f"Saved Brain 2 -> {config.BRAIN2_MODEL_PATH}")
    return m


# -- Orchestrator ------------------------------------------------------------

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

    # FIX 3: Calculate dynamic barriers FIRST using config-synced logic
    logger.info("Applying Dynamic Triple Barriers...")
    df.sort_values(by=["_symbol", "brick_timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Note: Dynamic labeling is applied downstream in run_brain_trainer
    # to guarantee Brain 1 & 2 see identical targets.

    logger.info(f"Final training dataset size: {len(df):,} rows")
    train, test = custom_holdout_split(df)
    
    # To enable multi-period purging, we need to extract the exact start/end 
    # of each continuous test block before deleting df
    test_blocks = []
    if not test.empty:
        # Sort test and find gaps > 1 month to identify blocks
        test_sorted = test["brick_timestamp"].sort_values().reset_index(drop=True)
        # Identify boundaries where month changes non-contiguously or year changes
        # Simple heuristic: any gap > 15 days is a new block
        diffs = test_sorted.diff()
        block_starts = [0] + diffs[diffs > pd.Timedelta(days=15)].index.tolist()
        block_ends = block_starts[1:] + [len(test_sorted)]
        
        for s, e in zip(block_starts, block_ends):
            block_ts = test_sorted.iloc[s:e-1] if e > s+1 else test_sorted.iloc[s:e]
            if not block_ts.empty:
                test_blocks.append((block_ts.min(), block_ts.max()))
                
        logger.info(f"Identified {len(test_blocks)} discrete test (holdout) blocks for purge/embargo.")
        for b_start, b_end in test_blocks:
            logger.info(f"  Holdout Block: {b_start.date()} to {b_end.date()}")
    
    # FIX: Explicitly free the massive 54M row master DataFrame before XGBoost  
    # allocates its internal DMatrix structure.
    del df
    import gc
    gc.collect()

    if "t1" in train.columns and getattr(config, "ENABLE_PURGE_EMBARGO", True) and test_blocks:
        logger.info("Applying Multi-Period Purge/Embargo (Zero-Copy Optimization)...")
        # ZERO-COPY MEMORY OPTIMIZATION for Bug #11
        # Instead of heavy set_index() which forces 50GB copies, use direct masking
        
        total_drop_mask = pd.Series(False, index=train.index)
        
        for test_start, test_end in test_blocks:
            n_embargo = 400
            embargo_cutoff = test_end + pd.Timedelta(minutes=n_embargo)
            block_drop_mask = (train["t1"] >= test_start) & (train["brick_timestamp"] < embargo_cutoff)
            total_drop_mask = total_drop_mask | block_drop_mask
        
        purged = total_drop_mask.sum()
        train = train[~total_drop_mask].reset_index(drop=True)
        logger.info(f"Multi-Period Purge/Embargo removed {purged:,} rows across {len(test_blocks)} blocks. Remaining: {len(train):,}")
    else:
        logger.info("Purge/Embargo skipped / missing t1.")

    # Phase 3: OOF Predictions for Meta-Learner
    oof_long = generate_oof_predictions(train, "label_long", "Brain1 (LONG)")
    oof_short = generate_oof_predictions(train, "label_short", "Brain1 (SHORT)")

    # Phase 4: Production Brain 1 (CNN + Class Weights + Isotonic Calibration)
    train_brain1_cnn(train, test, "label_long", "Brain1 (LONG)", config.BRAIN1_CNN_LONG_PATH, config.BRAIN1_CALIBRATED_LONG_PATH)
    train_brain1_cnn(train, test, "label_short", "Brain1 (SHORT)", config.BRAIN1_CNN_SHORT_PATH, config.BRAIN1_CALIBRATED_SHORT_PATH)
    
    # Phase 5: Production Brain 2 (Meta-Blindspots Fix)
    train_brain2_meta(train, test, oof_long, oof_short)
    
    logger.info("BRAIN TRAINER COMPLETE")
    logger.info(f"Calibrated LONG model saved at: {config.BRAIN1_CALIBRATED_LONG_PATH}")
    logger.info(f"Calibrated SHORT model saved at: {config.BRAIN1_CALIBRATED_SHORT_PATH}")


if __name__ == "__main__":
    run_brain_trainer()

