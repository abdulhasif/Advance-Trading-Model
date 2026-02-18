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

# Config
CONFIG_DIR   = PROJECT_ROOT / "config"

# Ensure directories exist
for d in [DATA_DIR, FEATURES_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UPSTOX API CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
UPSTOX_API_BASE     = "https://api.upstox.com/v3"
UPSTOX_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN_HERE"          # <-- USER MUST SET THIS
UPSTOX_WEBSOCKET_URL = "wss://api.upstox.com/v2/feed/market-data-feed"

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

# ─────────────────────────────────────────────────────────────────────────────
# RENKO PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
NATR_BRICK_PERCENT      = 0.0015   # 0.15% of price
ATR_PERIOD              = 14       # ATR lookback for normalised brick size
GAP_FILTER_MULTIPLIER   = 2       # Gap > 2× brick_size triggers teleport

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
VELOCITY_LOOKBACK        = 10     # average of last N bricks for velocity
RS_ROLLING_WINDOW        = 50     # rolling Z-score window for RS
WICK_REJECTION_THRESHOLD = 0.6    # wick ratio above this → rejection/trap

# ─────────────────────────────────────────────────────────────────────────────
# MODEL (XGBOOST) CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
XGBOOST_TREE_METHOD   = "gpu_hist"
XGBOOST_DEVICE        = "cuda"
XGBOOST_MAX_DEPTH     = 6
XGBOOST_LEARNING_RATE = 0.05
XGBOOST_N_ESTIMATORS  = 500
XGBOOST_EARLY_STOPPING = 30

BRAIN1_MODEL_PATH = MODELS_DIR / "brain1_direction.json"
BRAIN2_MODEL_PATH = MODELS_DIR / "brain2_conviction.json"

# ─────────────────────────────────────────────────────────────────────────────
# RISK FORTRESS
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_PENALTY    = 25      # penalty points when stock ≠ sector direction
TOP_N_SIGNALS     = 3       # leaderboard shows top N
DRIFT_WINDOW      = 50      # rolling window for accuracy drift
DRIFT_THRESHOLD   = 0.50    # below 50% accuracy → yellow alert

# ─────────────────────────────────────────────────────────────────────────────
# LIVE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
LIVE_STATE_FILE      = PROJECT_ROOT / "live_state.json"
LIVE_LOG_FILE        = LOGS_DIR / "live_engine.log"
STATE_WRITE_INTERVAL = 1.0    # seconds between live_state.json writes

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE FILE
# ─────────────────────────────────────────────────────────────────────────────
UNIVERSE_CSV = CONFIG_DIR / "sector_universe.csv"
