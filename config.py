"""
config.py — Central Configuration for the Institutional Fortress Trading System
================================================================================
All shared constants, paths, and hyperparameters live here.
Paths are relative to PROJECT_ROOT (the repo root, not src/).

ORGANIZATION:
1. System Paths & Directories
2. Upstox API & Connection Settings
3. Market Hours & Trading Windows
4. Data & Download Settings
5. ML Model Constants (XGBoost & Training)
6. Trading Strategy & Execution (Sniper Settings)
7. Core Physics (Renko & Alpha Features)
8. Risk Management & Guardrails
9. Simulator & Friction Mechanics
10. UI & Dashboard Aesthetics
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYSTEM PATHS & DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
# WHERE: Used by every module to resolve absolute paths.
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Storage layer — all runtime artefacts live here
STORAGE_DIR  = PROJECT_ROOT / "storage"
DATA_DIR     = STORAGE_DIR  / "data"
FEATURES_DIR = STORAGE_DIR  / "features"
MODELS_DIR   = STORAGE_DIR  / "models"
LOGS_DIR     = STORAGE_DIR  / "logs"
BACKUP_DIR   = STORAGE_DIR  / "backup"     # Permanent append-only data archive
CONFIG_DIR   = PROJECT_ROOT / "config_data"

# Ensure directories exist
for d in [DATA_DIR, FEATURES_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR, BACKUP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

UNIVERSE_CSV = CONFIG_DIR / "sector_universe.csv"
LIVE_STATE_FILE = PROJECT_ROOT / "live_state.json"
LIVE_LOG_FILE   = LOGS_DIR / "live_engine.log"
TRADE_CONTROL_FILE = LOGS_DIR / "trade_control.json"


# ─────────────────────────────────────────────────────────────────────────────
# 2. UPSTOX API & CONNECTION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# WHERE: src/live/tick_provider.py, src/live/engine.py
UPSTOX_API_BASE       = "https://api.upstox.com/v3"
UPSTOX_WS_AUTHORIZE    = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
UPSTOX_ACCESS_TOKEN   = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI2R0I1OTUiLCJqdGkiOiI2OWIzNTEwYTJhOGUyMTA4ZGIzZTMwODIiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3MzM1OTM3MCwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzczNDM5MjAwfQ.WNvEiw217vBiSTRt8gjzdhlLRSG6MJuTSXNE1NppxpA"

# API Rate-Limiting & Safety
API_MAX_WORKERS         = 4       # Concurrent download threads
API_DELAY_BETWEEN_CALLS = 0.35    # Seconds between API hits per thread
TICK_RECONNECT_DELAYS   = [5, 10, 20, 40, 60] # Exponential backoff for WebSocket
TICK_FLUSH_INTERVAL     = 1.0     # Seconds before flushing raw ticks to disk


# ─────────────────────────────────────────────────────────────────────────────
# 3. MARKET HOURS & TRADING WINDOWS (IST)
# ─────────────────────────────────────────────────────────────────────────────
# WHERE: src/live/engine.py, src/live/paper_trader.py
SYSTEM_WAKE_HOUR       = 8;   SYSTEM_WAKE_MINUTE       = 50
WARMUP_HOUR            = 9;   WARMUP_MINUTE            = 8
MARKET_OPEN_HOUR       = 9;   MARKET_OPEN_MINUTE       = 15
MARKET_CLOSE_HOUR      = 15;  MARKET_CLOSE_MINUTE      = 30
SYSTEM_SHUTDOWN_HOUR   = 15;  SYSTEM_SHUTDOWN_MINUTE   = 35

# Sniper Entry/Exit Windows
ENTRY_LOCK_MINUTES     = 2   # Morning filter (Wait for range to set: 2 mins after 09:15)
NO_NEW_ENTRY_HOUR      = 14   # Stop taking new trades at 02:30 PM
NO_NEW_ENTRY_MIN       = 30           
EOD_SQUARE_OFF_HOUR    = 15   # Force close everything at 03:14 PM
EOD_SQUARE_OFF_MIN     = 14         


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA & DOWNLOAD SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# WHERE: src/data/feature_engine.py, src/data/backup_pipeline.py
DOWNLOAD_START_YEAR = 2022
DOWNLOAD_END_YEAR   = 2026
MAX_BACKUP_STOCKS   = 2000      # Limit archival to top N stocks

# Custom Holdout Backtesting (OOS Validation)
HOLDOUT_MONTHS = [3, 11]  # Months to keep hidden for the backtester (e.g., Nov & Dec)
HOLDOUT_YEARS  = [2022, 2023, 2024, 2025,2026] # Years to apply the generic holdout months

# # Additional specific month logic (e.g. hold out March 2026 only, rest of 2026 is train)
# HOLDOUT_SPECIFIC_YEAR_MONTHS = {
#     2026: [3]
# }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ML MODEL CONSTANTS (XGBOOST & TRAINING)
# ─────────────────────────────────────────────────────────────────────────────
# WHERE: src/ml/brain_trainer.py, src/ml/backtester.py
BRAIN1_MODEL_LONG_PATH        = MODELS_DIR / "brain1_long.json"
BRAIN1_MODEL_SHORT_PATH       = MODELS_DIR / "brain1_short.json"
BRAIN1_MODEL_PATH             = MODELS_DIR / "brain1_direction.json" # Legacy reference
BRAIN2_MODEL_PATH             = MODELS_DIR / "brain2_conviction.json"
BRAIN1_CALIBRATED_LONG_PATH   = MODELS_DIR / "brain1_calibrated_long.pkl"
BRAIN1_CALIBRATED_SHORT_PATH  = MODELS_DIR / "brain1_calibrated_short.pkl"

# Model Selection Toggle
USE_CALIBRATED_MODELS      = True   # Set to False to use Raw .json models with higher thresholds

XGBOOST_TREE_METHOD      = "hist"
XGBOOST_DEVICE           = "cuda"
XGBOOST_MAX_DEPTH        = 4      # Prevent over-fitting
XGBOOST_LEARNING_RATE    = 0.05
XGBOOST_N_ESTIMATORS     = 500
XGBOOST_EARLY_STOPPING   = 30
XGBOOST_SUBSAMPLE        = 0.7           
XGBOOST_COLSAMPLE_BYTREE = 0.7          
XGBOOST_REG_LAMBDA       = 1.0     # FIX #1: Reduced from 10.0. High lambda squashes predictions to 0.50 (noise), causing inverted predictions after isotonic calibration.
CALIBRATION_SAMPLE_LIMIT = 500_000 # Samples for Isotonic probability calibration

# Target Horizons
TRAINING_HORIZON_BRICKS  = 4      # Model predicts likelihood of move within 4 bricks
TARGET_CLIPPING_BPS      = 250.0  # Caps conviction at 2.5% to normalize outliers


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRADING STRATEGY & EXECUTION (SNIPER SETTINGS)
# ─────────────────────────────────────────────────────────────────────────────
# WHERE: src/live/engine.py, src/ml/backtester.py, src/live/paper_trader.py
LONG_ENTRY_PROB_THRESH   = 0.65   # Calibrated Probability (0.35 maps to ~62% Raw confidence on the Isotonic Curve)
SHORT_ENTRY_PROB_THRESH  = 0.60 

RAW_LONG_ENTRY_PROB_THRESH  = 0.72  # Balanced threshold for Raw scores
RAW_SHORT_ENTRY_PROB_THRESH = 0.72
   # 68% probability requirement for SHORTs
ENTRY_CONV_THRESH        = 75   # FIX #7: Reduced from 5.0. 5.0 blocked almost all entries because Brain2 outputs are squashed.
STRONG_CONVICTION_THRESH = 1.0   # FIX #8: Reduced from 5.0. 5.0 caused trailing stops to hit immediately for nearly all trades.
BIAS_ENTRY_THRESHOLD     = 0.65   # Prob threshold when manual bias is set
VETO_BYPASS_CONV         = 100.0   # Conviction score high enough to bypass soft vetos (like sector weakness)

# Sniper Entry Gates
ENTRY_RS_THRESHOLD     = -0.5      # |RS| > 1.0 (Only trade relative leaders/laggards)
MAX_VWAP_ZSCORE        = 3.0      # Hard block for overextended exhaustion peaks
MAX_ENTRY_WICK         = 0.50     # Block if wick > 40% (Avoid absorption traps)
MIN_PRICE_FILTER       = 100.0    # No penny stocks
MIN_CONSECUTIVE_BRICKS = 1        # Requirement for momentum strength
MIN_BRICKS_TODAY       = 0        # Ensure symbol has formed at least one brick today
STREAK_LIMIT           = 7        # Max same-dir bricks (Anti-FOMO protection)
BRICK_COOLDOWN         = 3        # Bricks to wait after exit before re-entry

# Volume Filters
VOLUME_LIMIT_PCT       = 0.05     # Trade < 5% of candle volume (Anti-Impact)
MIN_CANDLE_VOLUME      = 500      # Minimum raw ticks in candle to trust signal
# Exit Rules & Hysteresis
STRUCTURAL_REVERSAL_BRICKS = 5    # Stop-loss: Exit if price reverses 5 bricks (2.0% leeway)
TRAIL_ACTIVATION_BRICKS    = 3    # Move to break-even after +5 bricks
TRAIL_DISTANCE_BRICKS      = 1.0  # Trail behind the peak by 1 brick
MAX_HOLD_BRICKS            = 300  # Kill-switch to prevent infinite bag-holding
HYST_LONG_SELL_FLOOR       = 0.40 # Exit LONG if prob falls below 0.40
HYST_SHORT_SELL_CEIL       = 0.60 # Exit SHORT if prob rises above 0.60
EXIT_CONV_THRESH           = 0.0  # Soft exit threshold for conviction


# ─────────────────────────────────────────────────────────────────────────────
# 7. CORE PHYSICS (RENKO & ALPHA FEATURES)
# ─────────────────────────────────────────────────────────────────────────────
# WHERE: src/core/renko.py, src/core/features.py
NATR_BRICK_PERCENT       = 0.0040 # 0.40% institutional brick size
RENKO_HISTORY_LIMIT      = 100    # History bricks to load on startup
RENKO_BRIDGE_STEPS       = 10     # Sub-tick points for Brownian Bridge
RENKO_BRIDGE_TRIGGER_MULTIPLIER = 2.0
GAP_FILTER_MULTIPLIER    = 2.0    # Teleport threshold for 9:15 gaps

# Alpha Factor Hyperparameters
VELOCITY_LOOKBACK          = 10
VELOCITY_LONG_LOOKBACK     = 20
VELOCITY_MIN_DURATION      = 1.0  # Clip to prevent infinite velocity
VELOCITY_LONG_MIN_DURATION = 15.0
MIN_BRICK_DURATION         = 15.0 # Global floor for duration math
MAX_BRICK_DURATION_SECONDS = 300  # Cap formation time to prevent outliers
RS_ROLLING_WINDOW          = 20   # Window for Relative Strength Z-score
RS_SMOOTHING_WINDOW        = 15   # Noise-reduction filter for SRF-style shakeouts
WICK_REJECTION_THRESHOLD   = 0.6  # Trap detector sensitivity
VWAP_WINDOW                = 10   # Institutional anchor window
VPT_ACCEL_DIFF             = 2    # Lag for volume acceleration logic
VPT_ACCEL_LAG              = 2    # Feature engine reference
SQUEEZE_WINDOW             = 10   # Window for time-density squeeze
STREAK_EXHAUSTION_ONSET    = 8    # Brick streak where momentum decay starts
STREAK_EXHAUSTION_SCALE    = 0.5  # Sigmoid steepness for exhaustion penalty

# Physics Math (FracDiff & Hurst)
FRACDIFF_D               = 0.4    # Differentiation order (Stationarity + Memory)
FRACDIFF_WARMUP_BRICKS   = 30     # Warm up memory before opening bell
FRACDIFF_THRESHOLD       = 1e-4   # Weight truncation limit
FRACDIFF_MAX_WINDOW      = 100    # Hard cap on weights
HURST_WINDOW             = 30     # Lookback for trending vs mean-reverting
HURST_THRESHOLD          = 0.45   # H > Threshold = Trending Regime (Consolidated)
TREND_THRESHOLD          = 0.45   # Synchronized with HURST_THRESHOLD
ADF_THRESHOLD            = 0.05   # Dickey-Fuller p-value for stationarity
EMBARGO_PCT              = 0.01   # Purging window to prevent leakage
ENABLE_PURGE_EMBARGO      = True  # FIX #11: Re-enabled memory safety gate after fixing the zero-copy OOM crash in brain_trainer.py.
# Execution Realism (Slippage)
T1_SLIPPAGE_PCT          = 0.0005 # 5 bps slippage per trade
TRANSACTION_COST_PCT     = 0.00075 # 15 bps (Brokerage + GST + STT)
JITTER_SECONDS           = 1.0    # Random delay for realistic OOS backtesting
PATH_CONFLICT_PESSIMISM  = True   # If wick touches SL/Target in same candle, assume SL
# Feature Engineering Optimization
FEATURE_OPTIMIZATION_ENABLED = True
FEATURE_INCREMENTAL_ENABLED  = False # Enable fast delta-computes
FEATURE_PARALLEL_WORKERS     = -1   # -1 = All CPUs
FEATURE_LOOKBACK_CONTEXT     = 100  # Bricks needed for full indicator warmup

# Single Source of Truth for Feature Order
FEATURE_COLS = [
    "velocity", "wick_pressure", "relative_strength", "brick_size",
    "duration_seconds", "consecutive_same_dir", "brick_oscillation_rate",
    "fracdiff_price", "hurst", "is_trending_regime", "velocity_long",
    "trend_slope", "rolling_range_pct", "momentum_acceleration",
    "vwap_zscore", "vpt_acceleration", "squeeze_zscore", "streak_exhaustion",
    "true_gap_pct", "time_to_form_seconds", "volume_intensity_per_sec",
    "is_opening_drive",
    "feature_tib_zscore", "feature_vpb_roc",
    "regime_morning", "regime_midday", "regime_afternoon",
    "feature_brick_volume_delta", "feature_cvd_divergence",
]

ROBUST_SCALE_COLS = [
    "velocity", "wick_pressure", "relative_strength", "brick_size",
    "duration_seconds", "consecutive_same_dir", "brick_oscillation_rate",
    "fracdiff_price", "hurst", "velocity_long", "trend_slope",
    "rolling_range_pct", "momentum_acceleration",
    "vwap_zscore", "vpt_acceleration", "squeeze_zscore", "streak_exhaustion",
    "true_gap_pct", "time_to_form_seconds", "volume_intensity_per_sec",
    "feature_tib_zscore", "feature_vpb_roc",
    "feature_brick_volume_delta", "feature_cvd_divergence",
]
BRAIN2_FEATURES = [
    "brain1_prob", "velocity", "wick_pressure", "relative_strength",
    "feature_tib_zscore", "feature_vpb_roc",
    "regime_morning", "regime_midday", "regime_afternoon",
    "feature_brick_volume_delta", "feature_cvd_divergence",
]



# ─────────────────────────────────────────────────────────────────────────────
# 8. RISK MANAGEMENT & GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────
# WHERE: src/core/risk.py, src/live/execution_guard.py
STARTING_CAPITAL      = 20000   
INTRADAY_LEVERAGE     = 5         
MAX_OPEN_POSITIONS    = 10        # Global diversification limit
MAX_LOSSES_PER_STOCK  = 1         # Stop trading symbol after 1 loss
POSITION_SIZE_PCT     = 0.10      # 10% of buying power per stock

CIRCUIT_BREAKER_STALE_SEC = 30.0   # Engine freeze if market delay > 5s
HEARTBEAT_INJECT_SEC      = 60.0  # Synthetic tick after 1m of silence
ORDER_LOCK_TIMEOUT_SEC    = 30    # Max time a symbol can be "blocked" pending
MAX_BUFFER_SIZE           = 2000   # O(1) rolling indicator memory limit
DRIFT_WINDOW               = 50    # Rolling lookback for drift history
DRIFT_WARMUP_WINDOW       = 10    # Minimum sample for drift detection
DRIFT_ACCURACY_THRESHOLD  = 0.5   # Sigma alert for Out-of-Distribution features
DRIFT_THRESHOLD           = 0.50  # Risk fortress gate threshold
SECTOR_PENALTY            = 25.0  # Score reduction for sector mismatch
TOP_N_SIGNALS             = 5     # Max signals to permit per rank cycle
SOFT_VETO_THRESHOLD       = 0.9   # Required sector-stock RS correlation


# ─────────────────────────────────────────────────────────────────────────────
# 9. SIMULATOR & FRICTION MECHANICS
# ─────────────────────────────────────────────────────────────────────────────
# WHERE: src/live/upstox_simulator.py
SIM_STARTING_CAPITAL = 100000.0
SIM_LEVERAGE         = 5.0
SIM_BROKERAGE_MAX    = 20.0       # Rs 20 per order max
SIM_BROKERAGE_PCT    = 0.0005     # 0.05% turnover limit
SIM_STT_SELL_PCT     = 0.00025    # STT on sell side only
SIM_STAMP_BUY_PCT    = 0.00003    # Stamp on buy side only
SIM_EXCHANGE_PCT     = 0.0000297  # NSE charge
SIM_SEBI_PCT         = 0.000001   # SEBI turnover fee
SIM_GST_PCT          = 0.18       # 18% on (Brokerage + Exchange)


# ─────────────────────────────────────────────────────────────────────────────
# 10. UI & DASHBOARD AESTHETICS
# ─────────────────────────────────────────────────────────────────────────────
# WHERE: src/ui/dashboard.py, src/ui/paper_dashboard.py
JET_THEME_PRIMARY     = "#00f2ff" # Cyber Cyan
JET_THEME_SECONDARY   = "#7000ff" # Electric Purple
DASHBOARD_REFRESH_SEC = 30        # Auto-refresh interval
STATE_WRITE_INTERVAL  = 1.0       # Interval for live_state.json update

# Market Regime Viz
REGIME_WINDOW         = 40
REGIME_MIN_SIGNALS    = 10
REGIME_BIAS_TRENDING  = 60
REGIME_BIAS_VOLATILE  = 40
REGIME_CONV_TRENDING  = 60
REGIME_CONV_VOLATILE  = 45

# News & Sentiment
SENTIMENT_THRESHOLD   = 0.5
NEWS_POLL_INTERVAL    = 300
NEWS_CACHE_LIMIT      = 2000
NEWS_RSS_FEEDS        = [
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/2146842.cms",
    "https://www.business-standard.com/rss/markets-106.rss"
]
