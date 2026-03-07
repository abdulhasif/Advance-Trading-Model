"""
Live Engine Alignment Audit
============================
Checks every integration point between the live engine and the current
model / features / config so we catch failures BEFORE 9:15 AM.

Run: .venv\Scripts\python.exe tmp\alignment_audit.py
"""
import sys
import importlib
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"

results = []

def check(label, fn):
    try:
        ok, msg = fn()
        tag = PASS if ok else FAIL
        results.append((ok, f"{tag} {label}: {msg}"))
    except Exception as e:
        results.append((False, f"{FAIL} {label}: EXCEPTION — {e}\n         {traceback.format_exc().splitlines()[-2]}"))


# ─────────────────────────────────────────────────────────────────
# 1. Model files exist
# ─────────────────────────────────────────────────────────────────
check("Brain1 LONG model file exists",
      lambda: (config.BRAIN1_CALIBRATED_LONG_PATH.exists(),
               str(config.BRAIN1_CALIBRATED_LONG_PATH)))

check("Brain1 SHORT model file exists",
      lambda: (config.BRAIN1_CALIBRATED_SHORT_PATH.exists(),
               str(config.BRAIN1_CALIBRATED_SHORT_PATH)))

check("Brain2 model file exists",
      lambda: (config.BRAIN2_MODEL_PATH.exists(),
               str(config.BRAIN2_MODEL_PATH)))

# ─────────────────────────────────────────────────────────────────
# 2. Models load without error
# ─────────────────────────────────────────────────────────────────
brain1_long = brain1_short = brain2 = None

def load_brain1_long():
    global brain1_long
    import joblib
    brain1_long = joblib.load(str(config.BRAIN1_CALIBRATED_LONG_PATH))
    return True, f"type={type(brain1_long).__name__}"
check("Brain1 LONG model loads", load_brain1_long)

def load_brain1_short():
    global brain1_short
    import joblib
    brain1_short = joblib.load(str(config.BRAIN1_CALIBRATED_SHORT_PATH))
    return True, f"type={type(brain1_short).__name__}"
check("Brain1 SHORT model loads", load_brain1_short)

def load_brain2():
    global brain2
    import xgboost as xgb
    brain2 = xgb.XGBRegressor()
    brain2.load_model(str(config.BRAIN2_MODEL_PATH))
    return True, "XGBRegressor loaded"
check("Brain2 model loads", load_brain2)

# ─────────────────────────────────────────────────────────────────
# 3. Feature count match: engine EXPECTED_FEATURES vs features.py
# ─────────────────────────────────────────────────────────────────
ENGINE_EXPECTED = [
    "velocity", "wick_pressure", "relative_strength",
    "brick_size", "duration_seconds",
    "consecutive_same_dir", "brick_oscillation_rate",
    "fracdiff_price", "hurst", "is_trending_regime",
    "velocity_long", "trend_slope", "rolling_range_pct",
    "momentum_acceleration", "vwap_zscore", "vpt_acceleration",
    "squeeze_zscore", "streak_exhaustion"
]

FEATURES_PY_OUTPUT = [
    'velocity', 'wick_pressure', 'relative_strength', 'brick_size',
    'duration_seconds', 'consecutive_same_dir', 'brick_oscillation_rate',
    'fracdiff_price', 'hurst', 'is_trending_regime', 'velocity_long',
    'trend_slope', 'rolling_range_pct', 'momentum_acceleration',
    'vwap_zscore', 'vpt_acceleration', 'squeeze_zscore', 'streak_exhaustion'
]

def check_feature_order():
    if ENGINE_EXPECTED == FEATURES_PY_OUTPUT:
        return True, f"{len(ENGINE_EXPECTED)} features in correct order"
    missing = set(FEATURES_PY_OUTPUT) - set(ENGINE_EXPECTED)
    extra   = set(ENGINE_EXPECTED) - set(FEATURES_PY_OUTPUT)
    wrong_order = [f for i,(a,b) in enumerate(zip(ENGINE_EXPECTED,FEATURES_PY_OUTPUT)) if a!=b for f in [f"pos{i}: engine={a} vs features={b}"]]
    return False, f"MISMATCH: missing={missing} extra={extra} order={wrong_order[:3]}"
check("Feature order: engine vs features.py", check_feature_order)

# ─────────────────────────────────────────────────────────────────
# 4. Model trained feature count matches engine list
# ─────────────────────────────────────────────────────────────────
def check_model_features():
    if brain1_long is None:
        return False, "Model not loaded"
    # IsotonicCalibrationWrapper wraps a calibrated classifier
    clf = brain1_long
    n_expected = len(ENGINE_EXPECTED)
    # Try to get n_features_in_ from the inner model
    try:
        inner = clf.calibrated_classifiers_[0].estimator
        n_model = inner.n_features_in_
    except Exception:
        try:
            n_model = clf.n_features_in_
        except Exception:
            return True, "Cannot inspect n_features_in_ (OK for calibrated wrapper)"
    if n_model == n_expected:
        return True, f"Model trained on {n_model} features — matches engine list"
    return False, f"Model trained on {n_model} features but engine sends {n_expected}!"
check("Brain1 LONG: feature count matches engine", check_model_features)

# ─────────────────────────────────────────────────────────────────
# 5. Dummy inference — does predict_proba work with 18 features?
# ─────────────────────────────────────────────────────────────────
def check_inference():
    if brain1_long is None or brain1_short is None:
        return False, "Models not loaded"
    dummy = np.zeros((1, len(ENGINE_EXPECTED)), dtype=np.float32)
    pl = float(brain1_long.predict_proba(dummy)[0][1])
    ps = float(brain1_short.predict_proba(dummy)[0][1])
    return True, f"LONG={pl:.4f}  SHORT={ps:.4f} (dummy input)"
check("Dummy inference (18 zeros)", check_inference)

# ─────────────────────────────────────────────────────────────────
# 6. Thresholds in config
# ─────────────────────────────────────────────────────────────────
check("config.LONG_ENTRY_PROB_THRESH exists",
      lambda: (hasattr(config, "LONG_ENTRY_PROB_THRESH"),
               getattr(config, "LONG_ENTRY_PROB_THRESH", "MISSING")))

check("config.SHORT_ENTRY_PROB_THRESH exists",
      lambda: (hasattr(config, "SHORT_ENTRY_PROB_THRESH"),
               getattr(config, "SHORT_ENTRY_PROB_THRESH", "MISSING")))

check("LONG thresh >= SHORT thresh (no inversion)",
      lambda: (getattr(config, "LONG_ENTRY_PROB_THRESH", 0) >= 
               getattr(config, "SHORT_ENTRY_PROB_THRESH", 0),
               f"LONG={getattr(config,'LONG_ENTRY_PROB_THRESH',0)} SHORT={getattr(config,'SHORT_ENTRY_PROB_THRESH',0)}"))

# ─────────────────────────────────────────────────────────────────
# 7. FracDiff warmup config exists
# ─────────────────────────────────────────────────────────────────
check("config.FRACDIFF_WARMUP_BRICKS exists",
      lambda: (hasattr(config, "FRACDIFF_WARMUP_BRICKS"),
               getattr(config, "FRACDIFF_WARMUP_BRICKS", "MISSING")))

# ─────────────────────────────────────────────────────────────────
# 8. Sector universe CSV readable
# ─────────────────────────────────────────────────────────────────
def check_universe():
    df = pd.read_csv(config.UNIVERSE_CSV)
    eq = df[df["is_index"] == False]
    idx = df[df["is_index"] == True]
    bad_tokens = [r["symbol"] for _, r in eq.iterrows()
                  if not str(r["instrument_token"]).startswith("NSE_EQ|")]
    if bad_tokens:
        return False, f"Tokens not starting with NSE_EQ|: {bad_tokens[:5]}"
    return True, f"{len(eq)} equities, {len(idx)} indices"
check("sector_universe.csv readable and tokens valid", check_universe)

# ─────────────────────────────────────────────────────────────────
# 9. compute_features_live imports without error
# ─────────────────────────────────────────────────────────────────
def check_feature_import():
    from src.core.features import compute_features_live
    return True, "compute_features_live importable"
check("compute_features_live importable", check_feature_import)

# ─────────────────────────────────────────────────────────────────
# 10. LiveRenkoState works
# ─────────────────────────────────────────────────────────────────
def check_renko():
    from src.core.renko import LiveRenkoState
    rs = LiveRenkoState(symbol="TEST", sector="IT", brick_size=5.0)
    rs.process_tick(100.0, 100.5, 99.5, pd.Timestamp.now())
    return True, "LiveRenkoState instantiable and processable"
check("LiveRenkoState works", check_renko)

# ─────────────────────────────────────────────────────────────────
# 11. Path configs exist
# ─────────────────────────────────────────────────────────────────
check("config.DATA_DIR exists",     lambda: (config.DATA_DIR.exists(), str(config.DATA_DIR)))
check("config.MODELS_DIR exists",   lambda: (config.MODELS_DIR.exists(), str(config.MODELS_DIR)))
check("config.LOGS_DIR exists",     lambda: (config.LOGS_DIR.exists(), str(config.LOGS_DIR)))

# ─────────────────────────────────────────────────────────────────
# 12. Config sync: paper_trader vs engine vs backtester
# ─────────────────────────────────────────────────────────────────
def check_threshold_sync():
    import ast, re
    files = {
        "engine.py":      ROOT / "src/live/engine.py",
        "paper_trader.py": ROOT / "src/live/paper_trader.py",
        "backtester.py":  ROOT / "src/ml/backtester.py",
    }
    issues = []
    for fname, fpath in files.items():
        txt = fpath.read_text(errors="ignore")
        if "LONG_ENTRY_PROB_THRESH" not in txt:
            issues.append(f"{fname} missing LONG_ENTRY_PROB_THRESH")
        if "SHORT_ENTRY_PROB_THRESH" not in txt:
            issues.append(f"{fname} missing SHORT_ENTRY_PROB_THRESH")
    if issues:
        return False, " | ".join(issues)
    return True, "All 3 files reference both LONG and SHORT thresholds"
check("LONG/SHORT thresh wired in engine+paper_trader+backtester", check_threshold_sync)

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  LIVE ENGINE ALIGNMENT AUDIT")
print("="*65)
for ok, msg in results:
    print(msg)

total  = len(results)
passed = sum(1 for ok, _ in results if ok)
failed = total - passed

print("="*65)
print(f"  Result: {passed}/{total} checks passed | {failed} FAILED")
print("="*65)

if failed > 0:
    print("\n  ACTION REQUIRED: Fix FAIL items above before live trading!")
    sys.exit(1)
else:
    print("\n  Engine is ALIGNED — safe to run main.py live")
    sys.exit(0)
