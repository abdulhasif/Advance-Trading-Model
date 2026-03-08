"""
config.py — Central Configuration for the Institutional Fortress Trading System
================================================================================
All shared constants, paths, and hyperparameters live here.
Paths are relative to PROJECT_ROOT (the repo root, not src/).
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Storage layer — all runtime artefacts live here
STORAGE_DIR  = PROJECT_ROOT / "storage"
DATA_DIR     = STORAGE_DIR  / "data"
FEATURES_DIR = STORAGE_DIR  / "features"
MODELS_DIR   = STORAGE_DIR  / "models"
LOGS_DIR     = STORAGE_DIR  / "logs"
BACKUP_DIR   = STORAGE_DIR  / "backup"     # permanent append-only data archive

# Config
CONFIG_DIR   = PROJECT_ROOT / "config_data"

# Ensure directories exist
for d in [DATA_DIR, FEATURES_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR, BACKUP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UPSTOX API CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
UPSTOX_API_BASE     = "https://api.upstox.com/v3"
UPSTOX_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI2R0I1OTUiLCJqdGkiOiI2OWFiNzBiMGMxOTk0NjNjOTY2MjY1YTkiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3Mjg0MzE4NCwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzcyOTIwODAwfQ.9rnHNRIKpmEqar8I9WYUEYmYNxZWrvIm4xM_8UiOrCY"
UPSTOX_WS_AUTHORIZE  = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
# NOTE: The actual wss:// URL is dynamic — obtained from the authorize endpoint above.
# The upstox-python-sdk MarketDataStreamerV3 handles this automatically.

# API rate-limit safety
API_MAX_WORKERS         = 4       # concurrent download threads
API_DELAY_BETWEEN_CALLS = 0.35   # seconds between API hits per thread

# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOURS (IST — Indian Standard Time)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_WAKE_HOUR     = 8;   SYSTEM_WAKE_MINUTE     = 50
WARMUP_HOUR          = 9;   WARMUP_MINUTE          = 8
MARKET_OPEN_HOUR     = 9;   MARKET_OPEN_MINUTE     = 15
MARKET_CLOSE_HOUR    = 15;  MARKET_CLOSE_MINUTE    = 30
SYSTEM_SHUTDOWN_HOUR = 15;  SYSTEM_SHUTDOWN_MINUTE = 35

# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD RANGE (4 years of history)
# ─────────────────────────────────────────────────────────────────────────────
DOWNLOAD_START_YEAR = 2022
DOWNLOAD_END_YEAR   = 2026   # inclusive

# Train/Test Split
TEST_START_DATE     = "2025-07-01"   # Train < this date, Test >= this date



# ─────────────────────────────────────────────────────────────────────────────
# RENKO PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
NATR_BRICK_PERCENT      = 0.0040   # 0.40% of price
ATR_PERIOD              = 14       # ATR lookback for normalised brick size
GAP_FILTER_MULTIPLIER   = 2       # Gap > 2× brick_size triggers teleport

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
VELOCITY_LOOKBACK        = 10     # average of last N bricks for velocity
RS_ROLLING_WINDOW        = 50     # rolling Z-score window for RS
WICK_REJECTION_THRESHOLD = 0.6    # wick ratio above this -> rejection/trap

# ── Alpha Factor Hyperparameters (Institutional Features) ─────────────────────
VWAP_WINDOW              = 20     # rolling bricks for VWAP Z-Score (Phase 2)
VPT_ACCEL_DIFF           = 2      # 2nd order difference lag for VPT acceleration
SQUEEZE_WINDOW           = 20     # rolling window for volatility squeeze density
STREAK_EXHAUSTION_ONSET  = 8     # consecutive bricks after which decay kicks in
STREAK_EXHAUSTION_SCALE  = 0.5   # sigmoid steepness of exhaustion penalty

# --- PERFORMANCE OPTIONS ---
FEATURE_OPTIMIZATION_ENABLED  = True    # Set to False to revert to iterative loops (for debugging)
FEATURE_INCREMENTAL_ENABLED   = False # Set to False to force a clean recompute from scratch
FEATURE_PARALLEL_WORKERS      = -1      # -1 = auto-detect CPU count

# ─────────────────────────────────────────────────────────────────────────────
# MODEL (XGBOOST) CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
XGBOOST_TREE_METHOD   = "hist"           # newer xgboost: use "hist" + device="cuda"
XGBOOST_DEVICE        = "cuda"
XGBOOST_MAX_DEPTH     = 4                # Reduced from 6 to prevent over-fitting/memorization
XGBOOST_LEARNING_RATE = 0.05
XGBOOST_N_ESTIMATORS  = 500
XGBOOST_EARLY_STOPPING = 30
XGBOOST_SUBSAMPLE        = 0.7           # Pessimistic: harder regularization on 17.8M rows
XGBOOST_COLSAMPLE_BYTREE = 0.7          # Randomly drop features per tree (prevents velocity dominance)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY PROBABILITY THRESHOLDS
# LONG and SHORT use separate thresholds because short model is harder to
# calibrate (fewer training examples). Lower SHORT thresh unlocks more trades.
# ─────────────────────────────────────────────────────────────────────────────
LONG_ENTRY_PROB_THRESH  = 0.68   # Long: high-conviction institutional breakout
SHORT_ENTRY_PROB_THRESH = 0.65   # Short: slightly lower bar (model is tighter)

# ─────────────────────────────────────────────────────────────────────────────
# FRACDIFF WARMUP
# Number of prior bricks prepended before computing FracDiff in live mode,
# to avoid NaN in the first ~50 bricks of the session (which kills the feature).
# ─────────────────────────────────────────────────────────────────────────────
FRACDIFF_WARMUP_BRICKS = 60   # Bricks of prior-day history prepended at session start

XGBOOST_REG_LAMBDA       = 10.0          # L2 regularization term on weights

BRAIN1_MODEL_LONG_PATH        = MODELS_DIR / "brain1_long.json"
BRAIN1_MODEL_SHORT_PATH       = MODELS_DIR / "brain1_short.json"
BRAIN2_MODEL_PATH             = MODELS_DIR / "brain2_conviction.json"
BRAIN1_CALIBRATED_LONG_PATH   = MODELS_DIR / "brain1_calibrated_long.pkl"
BRAIN1_CALIBRATED_SHORT_PATH  = MODELS_DIR / "brain1_calibrated_short.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION TRAILING STOP
# ─────────────────────────────────────────────────────────────────────────────
TRAIL_ACTIVATION_BRICKS = 5       # Lock trailing stop at Break-Even when +5 hit
TRAIL_DISTANCE_BRICKS   = 2.0      # Distance to trail behind price once activated
STRONG_CONVICTION_THRESH= 10.0      # Disable tight trail for massive conviction runs

# ─────────────────────────────────────────────────────────────────────────────
# LIVE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
LIVE_STATE_FILE      = PROJECT_ROOT / "live_state.json"
LIVE_LOG_FILE        = LOGS_DIR / "live_engine.log"
TRADE_CONTROL_FILE   = LOGS_DIR / "trade_control.json"
STATE_WRITE_INTERVAL = 1.0    # seconds between live_state.json writes

ENABLE_PURGE_EMBARGO = False  # Disabled for 250-stock train (causes OOM on 48M rows)

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE FILE
# ─────────────────────────────────────────────────────────────────────────────
UNIVERSE_CSV = CONFIG_DIR / "sector_universe.csv"
