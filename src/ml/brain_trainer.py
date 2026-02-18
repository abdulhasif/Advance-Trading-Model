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
    "brick_size", "duration_seconds", "direction",
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

    combined = pd.concat(frames, ignore_index=True).sort_values("brick_timestamp").reset_index(drop=True)
    logger.info(f"Total bricks: {len(combined):,}")
    return combined


# ── Target Engineering ──────────────────────────────────────────────────────

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dirs, convs = [], []

    for _, grp in out.groupby("_symbol"):
        grp = grp.sort_values("brick_timestamp")
        next_dir = grp["direction"].shift(-1)
        dirs.append((next_dir > 0).astype(float))

        next_close = grp["brick_close"].shift(-1)
        move = (next_close - grp["brick_close"]).abs()
        convs.append((move / grp["brick_size"].clip(lower=1e-9) * 50).clip(upper=100))

    out["direction_target"] = pd.concat(dirs)
    out["conviction_target"] = pd.concat(convs)
    return out.dropna(subset=["direction_target", "conviction_target"]).reset_index(drop=True)


# ── Walk-Forward Split ──────────────────────────────────────────────────────

def walk_forward_split(df: pd.DataFrame, test_months: int = 6):
    cutoff = df["brick_timestamp"].max() - pd.DateOffset(months=test_months)
    train = df[df["brick_timestamp"] < cutoff]
    test = df[df["brick_timestamp"] >= cutoff]
    logger.info(f"Split — Train: {len(train):,}  Test: {len(test):,}  Cutoff: {cutoff.date()}")
    return train.copy(), test.copy()


# ── Brain 1: Direction Classifier ───────────────────────────────────────────

def train_brain1(train, test) -> xgb.XGBClassifier:
    logger.info("─" * 50 + "\nTRAINING BRAIN 1 — Direction Classifier")
    X_tr, y_tr = train[FEATURE_COLS].fillna(0), train["direction_target"]
    X_te, y_te = test[FEATURE_COLS].fillna(0), test["direction_target"]

    m = xgb.XGBClassifier(
        tree_method=config.XGBOOST_TREE_METHOD, device=config.XGBOOST_DEVICE,
        max_depth=config.XGBOOST_MAX_DEPTH, learning_rate=config.XGBOOST_LEARNING_RATE,
        n_estimators=config.XGBOOST_N_ESTIMATORS, objective="binary:logistic",
        eval_metric="logloss", early_stopping_rounds=config.XGBOOST_EARLY_STOPPING,
        use_label_encoder=False, verbosity=1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=50)

    acc = accuracy_score(y_te, m.predict(X_te))
    logger.info(f"Brain 1 Accuracy: {acc:.4f}")
    logger.info(f"\n{classification_report(y_te, m.predict(X_te), target_names=['Down','Up'])}")

    m.save_model(str(config.BRAIN1_MODEL_PATH))
    logger.info(f"Saved → {config.BRAIN1_MODEL_PATH}")
    return m


# ── Brain 2: Conviction Meta-Regressor ─────────────────────────────────────

def train_brain2(train, test, brain1: xgb.XGBClassifier) -> xgb.XGBRegressor:
    logger.info("─" * 50 + "\nTRAINING BRAIN 2 — Conviction Meta-Regressor")

    def meta(X_base, df_orig):
        prob = brain1.predict_proba(X_base)[:, 1]
        return pd.DataFrame({
            "brain1_prob": prob,
            "velocity": df_orig["velocity"].fillna(0).values,
            "wick_pressure": df_orig["wick_pressure"].fillna(0).values,
            "relative_strength": df_orig["relative_strength"].fillna(0).values,
        })

    X_tr = meta(train[FEATURE_COLS].fillna(0), train)
    X_te = meta(test[FEATURE_COLS].fillna(0), test)
    y_tr, y_te = train["conviction_target"], test["conviction_target"]

    m = xgb.XGBRegressor(
        tree_method=config.XGBOOST_TREE_METHOD, device=config.XGBOOST_DEVICE,
        max_depth=config.XGBOOST_MAX_DEPTH, learning_rate=config.XGBOOST_LEARNING_RATE,
        n_estimators=config.XGBOOST_N_ESTIMATORS, objective="reg:squarederror",
        eval_metric="mae", early_stopping_rounds=config.XGBOOST_EARLY_STOPPING, verbosity=1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=50)

    logger.info(f"Brain 2 MAE: {mean_absolute_error(y_te, m.predict(X_te)):.2f} | R²: {r2_score(y_te, m.predict(X_te)):.4f}")
    m.save_model(str(config.BRAIN2_MODEL_PATH))
    logger.info(f"Saved → {config.BRAIN2_MODEL_PATH}")
    return m


# ── Orchestrator ────────────────────────────────────────────────────────────

def run_brain_trainer():
    logger.info("=" * 70)
    logger.info(f"BRAIN TRAINER — GPU: {config.XGBOOST_TREE_METHOD} / {config.XGBOOST_DEVICE}")
    logger.info("=" * 70)

    df = create_targets(load_all_features())
    train, test = walk_forward_split(df)
    del df

    b1 = train_brain1(train, test)
    train_brain2(train, test, b1)
    logger.info("BRAIN TRAINER COMPLETE")


if __name__ == "__main__":
    run_brain_trainer()
