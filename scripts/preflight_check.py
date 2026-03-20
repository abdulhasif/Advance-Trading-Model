"""
scripts/preflight_check.py - Pre-market System Connectivity Check
=================================================================
Run BEFORE starting the trading system the next morning.
Checks: models, features, API server, Upstox token, logs dirs.

Usage:
    .venv\\Scripts\\python.exe scripts/preflight_check.py
"""

import sys
import os
import json
import socket
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

PASS = "  [OK]  "
FAIL = "  [!!]  "
WARN = "  [??]  "

results = []


def check(label, ok, detail="", warn=False):
    tag = WARN if (warn and not ok) else (PASS if ok else FAIL)
    line = f"{tag} {label:<45} {detail}"
    print(line)
    results.append((ok or warn, label))


print("\n" + "=" * 65)
print("  PRE-FLIGHT CHECK - Institutional Fortress Trading System")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
print("=" * 65 + "\n")

# -- 1. Models --------------------------------------------------------------
print("[ MODELS ]")
use_cal = getattr(config, "USE_CALIBRATED_MODELS", False)
if use_cal:
    model_paths = [
        ("Brain1 LONG (Calibrated)",  config.BRAIN1_CALIBRATED_LONG_PATH),
        ("Brain1 SHORT (Calibrated)", config.BRAIN1_CALIBRATED_SHORT_PATH),
        ("Brain2 Conviction",         config.BRAIN2_MODEL_PATH),
    ]
else:
    model_paths = [
        ("Brain1 LONG (Raw)",  config.BRAIN1_MODEL_LONG_PATH),
        ("Brain1 SHORT (Raw)", config.BRAIN1_MODEL_SHORT_PATH),
        ("Brain2 Conviction",  config.BRAIN2_MODEL_PATH),
    ]

for name, path in model_paths:
    check(f"{name} model exists", path.exists(), str(path.name))
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        check(f"{name} freshness", age_days < 30, f"trained {age_days:.1f} days ago", warn=True)

# -- 2. Features ------------------------------------------------------------
print("\n[ FEATURES ]")
feat_files = list(config.FEATURES_DIR.rglob("*.parquet"))
check("Feature parquet files exist", len(feat_files) > 0, f"{len(feat_files)} files found")
if feat_files:
    newest = max(feat_files, key=lambda f: f.stat().st_mtime)
    age_h = (time.time() - newest.stat().st_mtime) / 3600
    check("Feature data freshness", age_h < 72, f"newest: {newest.name} ({age_h:.0f}h ago)", warn=True)

# -- 3. Model loads correctly ----------------------------------------------
print("\n[ MODEL LOAD ]")
try:
    import numpy as np
    import xgboost as xgb
    import joblib

    expected = list(getattr(config, "FEATURE_COLS", []))
    dummy = np.zeros((1, len(expected)), dtype=np.float32)

    if use_cal:
        b1_long = joblib.load(str(config.BRAIN1_CALIBRATED_LONG_PATH))
        b1_short = joblib.load(str(config.BRAIN1_CALIBRATED_SHORT_PATH))
        _ = b1_long.predict_proba(dummy)
        _ = b1_short.predict_proba(dummy)
        check("Brain1 LONG/SHORT load & infer (calibrated)", True, f"{len(expected)} features")
    else:
        b1_long = xgb.XGBClassifier(); b1_long.load_model(str(config.BRAIN1_MODEL_LONG_PATH))
        b1_short = xgb.XGBClassifier(); b1_short.load_model(str(config.BRAIN1_MODEL_SHORT_PATH))
        _ = b1_long.predict_proba(dummy)
        _ = b1_short.predict_proba(dummy)
        feat_names = b1_long.get_booster().feature_names or []
        check("Brain1 LONG/SHORT load & infer (raw)", True, f"{len(feat_names)} features")
        check("No stale 'direction' feature in model",
              "direction" not in feat_names,
              "OK" if "direction" not in feat_names else "OLD MODEL - retrain!")

    b2m = xgb.XGBRegressor(); b2m.load_model(str(config.BRAIN2_MODEL_PATH))
    b2_expected = list(getattr(config, "BRAIN2_FEATURES", ["brain1_prob"]))
    b2_dummy = np.zeros((1, len(b2_expected)), dtype=np.float32)
    _ = b2m.predict(b2_dummy)
    check("Brain2 Conviction loads & infer", True)
except Exception as e:
    check("Brain models load", False, str(e)[:60])

# -- 4. Upstox API token ----------------------------------------------------
print("\n[ UPSTOX API TOKEN ]")
# Read from config.py first (primary source), env var as fallback only
token = getattr(config, "UPSTOX_ACCESS_TOKEN", "") or os.environ.get("UPSTOX_ACCESS_TOKEN", "")
check("UPSTOX_ACCESS_TOKEN set", bool(token),
      f"length={len(token)}" if token else "NOT SET - define in config.py!", warn=not bool(token))
if token:
    try:
        import base64
        import json as _json
        from datetime import datetime as _dt
        # Decode JWT expiry locally (Upstox API can return 403 before 9:00 AM even with a valid token)
        parts = token.split(".")
        payload = _json.loads(base64.b64decode(parts[1] + "==").decode())
        exp_ts = payload.get("exp", 0)
        exp_dt = _dt.fromtimestamp(exp_ts)
        still_valid = exp_ts > _dt.now().timestamp()
        check("Upstox token valid (JWT expiry check)", still_valid,
              f"Expires: {exp_dt.strftime('%Y-%m-%d %H:%M')} IST" if still_valid
              else f"EXPIRED at {exp_dt.strftime('%Y-%m-%d %H:%M')} - refresh now!")
    except Exception as e:
        check("Upstox token valid (JWT expiry check)", False, str(e)[:60])

# -- 5. Internet / Upstox reachability -------------------------------------
print("\n[ NETWORK ]")
for host, label in [("api.upstox.com", "Upstox API"), ("8.8.8.8", "Internet (DNS)")]:
    try:
        socket.setdefaulttimeout(4)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, 443))
        check(f"{label} reachable", True, host)
    except Exception as e:
        check(f"{label} reachable", False, str(e)[:50])

# -- 6. Port 8000 availability ---------------------------------------------
print("\n[ PORT ]")
try:
    s = socket.socket(); s.bind(("0.0.0.0", 8000)); s.close()
    check("Port 8000 free for FastAPI", True)
except OSError:
    check("Port 8000 free for FastAPI", False, "Something already using port 8000")

# -- 7. Log directories writable -------------------------------------------
print("\n[ LOG DIRS ]")
debug_dir = config.LOGS_DIR.parent.parent / "logs" / "paper_debug"
debug_dir.mkdir(parents=True, exist_ok=True)
for d, label in [
    (config.LOGS_DIR, "storage/logs"),
    (debug_dir, "logs/paper_debug"),
]:
    try:
        test = d / ".preflight_test"
        test.touch(); test.unlink()
        check(f"{label} writable", True)
    except Exception as e:
        check(f"{label} writable", False, str(e)[:50])

# -- 8. Config: sector universe --------------------------------------------
print("\n[ CONFIG ]")
try:
    import pandas as pd
    univ = pd.read_csv(config.UNIVERSE_CSV)
    check("sector_universe.csv readable", True, f"{len(univ)} rows, {univ['sector'].nunique()} sectors")
except Exception as e:
    check("sector_universe.csv readable", False, str(e)[:60])

# -- 9. Python dependencies -------------------------------------------------
print("\n[ DEPENDENCIES ]")
for pkg in ["xgboost", "pandas", "numpy", "fastapi", "uvicorn",
            "streamlit", "upstox_client", "websockets"]:
    try:
        __import__(pkg.replace("-", "_"))
        check(f"{pkg}", True)
    except ImportError:
        check(f"{pkg}", False, "pip install " + pkg)

# -- SUMMARY ---------------------------------------------------------------
passed = sum(1 for ok, _ in results if ok)
failed = len(results) - passed
print("\n" + "=" * 65)
if failed == 0:
    print(f"  [OK]  ALL {passed} CHECKS PASSED - System ready for tomorrow morning")
else:
    print(f"  [!!]  {failed} CHECK(S) FAILED - Fix before starting live trading")
    for ok, label in results:
        if not ok:
            print(f"       -> {label}")
print("=" * 65 + "\n")

sys.exit(0 if failed == 0 else 1)
