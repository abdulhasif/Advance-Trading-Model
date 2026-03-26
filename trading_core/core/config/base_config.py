import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYSTEM PATHS & DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Storage layer — all runtime artefacts live here
STORAGE_DIR  = PROJECT_ROOT / "storage"
DATA_DIR     = STORAGE_DIR  / "data"
FEATURES_DIR = STORAGE_DIR  / "features"
MODELS_DIR   = STORAGE_DIR  / "models"
LOGS_DIR     = STORAGE_DIR  / "logs"
BACKUP_DIR   = STORAGE_DIR  / "backup"
CONFIG_DIR   = PROJECT_ROOT / "config_data"

# Ensure directories exist
for d in [DATA_DIR, FEATURES_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR, BACKUP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

UNIVERSE_CSV = CONFIG_DIR / "sector_universe.csv"
LIVE_STATE_FILE = PROJECT_ROOT / "live_state.json"
LIVE_LOG_FILE   = LOGS_DIR / "live_engine.log"
TRADE_CONTROL_FILE = LOGS_DIR / "trade_control.json"

# ─────────────────────────────────────────────────────────────────────────────
# 7. CORE PHYSICS (RENKO & ALPHA FEATURES)
# ─────────────────────────────────────────────────────────────────────────────
NATR_BRICK_PERCENT       = 0.0015
RENKO_HISTORY_LIMIT      = 100
RENKO_BRIDGE_STEPS       = 10
RENKO_BRIDGE_TRIGGER_MULTIPLIER = 2.0
GAP_FILTER_MULTIPLIER    = 2.0
VOL_MULT                 = 1e-4

# Alpha Factor Hyperparameters
VELOCITY_LOOKBACK          = 10
VELOCITY_LONG_LOOKBACK     = 20
VELOCITY_MIN_DURATION      = 1.0
VELOCITY_LONG_MIN_DURATION = 15.0
MIN_BRICK_DURATION         = 15.0
MAX_BRICK_DURATION_SECONDS = 300
RS_ROLLING_WINDOW          = 20
RS_SMOOTHING_WINDOW        = 15
WICK_REJECTION_THRESHOLD   = 0.6
VWAP_WINDOW                = 10
VPT_ACCEL_DIFF             = 2
VPT_ACCEL_LAG              = 2
SQUEEZE_WINDOW             = 10
STREAK_EXHAUSTION_ONSET    = 8
STREAK_EXHAUSTION_SCALE    = 0.5
STRUCTURAL_WINDOW          = 50

# Physics Math (FracDiff & Hurst)
FRACDIFF_D               = 0.4
FRACDIFF_WARMUP_BRICKS   = 30
FRACDIFF_THRESHOLD       = 1e-4
FRACDIFF_MAX_WINDOW      = 100
HURST_WINDOW             = 30
HURST_THRESHOLD          = 0.45
TREND_THRESHOLD          = 0.45
ADF_THRESHOLD            = 0.05
EMBARGO_PCT              = 0.01
ENABLE_PURGE_EMBARGO      = True

# Feature Engineering Optimization
FEATURE_OPTIMIZATION_ENABLED = True
FEATURE_INCREMENTAL_ENABLED  = True
FEATURE_PARALLEL_WORKERS     = -1
FEATURE_LOOKBACK_CONTEXT     = 100

# Feature Order (CNN Streamlined - 17 Features)
FEATURE_COLS = [
    "velocity", "momentum_acceleration", "feature_tib_zscore",
    "vwap_zscore", "feature_vpb_roc", "feature_cvd_divergence",
    "vpt_acceleration", "relative_strength", "fracdiff_price",
    "wick_pressure", "hurst", "consecutive_same_dir",
    "streak_exhaustion", "true_gap_pct", "regime_morning",
    "regime_midday", "regime_afternoon",
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

# Meta-Regressor Features
BRAIN2_FEATURES = [
    "brain1_prob_long", "brain1_prob_short", "trade_direction",
    "velocity", "momentum_acceleration", "feature_tib_zscore",
    "wick_pressure", "relative_strength", "feature_vpb_roc",
    "regime_morning", "regime_midday", "regime_afternoon",
    "feature_cvd_divergence"
]

# ─────────────────────────────────────────────────────────────────────────────
# 8. RISK MANAGEMENT & GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────
MAX_OPEN_POSITIONS    = 10
MAX_LOSSES_PER_STOCK  = 1
CIRCUIT_BREAKER_STALE_SEC = 30.0
HEARTBEAT_INJECT_SEC      = 60.0
ORDER_LOCK_TIMEOUT_SEC    = 30
MAX_BUFFER_SIZE           = 2000
DRIFT_WINDOW               = 50
DRIFT_WARMUP_WINDOW       = 10
DRIFT_ACCURACY_THRESHOLD  = 0.5
DRIFT_THRESHOLD           = 0.50
SECTOR_PENALTY            = 25.0
TOP_N_SIGNALS             = 5
SOFT_VETO_THRESHOLD       = 0.9

# ─────────────────────────────────────────────────────────────────────────────
# 11. GLOBAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def to_naive_ist(ts):
    import pandas as pd
    if ts is None: return None
    if hasattr(ts, "dt"):
        if ts.dt.tz is None: return ts
        return ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    ts_scalar = pd.to_datetime(ts)
    if ts_scalar.tz is None: return ts_scalar
    return ts_scalar.tz_convert("Asia/Kolkata").tz_localize(None)

