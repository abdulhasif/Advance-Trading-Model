"""
main.py — Unified Entry Point for the Institutional Fortress
==============================================================
Usage:
  python main.py download       # Phase 1: batch download + Renko transform
  python main.py features       # Phase 2: compute features
  python main.py train          # Phase 3: train dual XGBoost models
  python main.py live           # Phase 4: start live trading engine
  python main.py dashboard      # Phase 5: launch Streamlit dashboard
   python main.py backup         # Archive: download ALL ~2000 NSE stocks
   python main.py backtest       # Backtest: Truth Teller simulation
   # python main.py paper          # [DEPRECATED] Use Android App
   # python main.py dashboard      # [DEPRECATED] Use Android App
   # python main.py paper_dashboard # [DEPRECATED] Use Android App
   python main.py --help         # show this help
"""

import sys
import os
os.environ["KERAS_BACKEND"] = "torch"


def _run_preflight_audit() -> list:
    """
    Pre-flight alignment audit. Returns a list of failure messages (empty = all OK).
    Checks: model files, model loads, feature count, config thresholds, token format.
    """
    import numpy as np
    import pandas as pd

    failures = []

    try:
        import config
        import joblib
        import xgboost as xgb

        # 1. Model files exist & Loadability
        EXPECTED_B1 = config.FEATURE_COLS
        EXPECTED_B2 = config.BRAIN2_FEATURES
        
        dummy_b1 = np.zeros((1, len(EXPECTED_B1)), dtype=np.float32)
        dummy_b2 = np.zeros((1, len(EXPECTED_B2)), dtype=np.float32)

        if config.USE_CALIBRATED_MODELS:
            models_to_check = [
                ("Brain1 LONG (Calibrated)",  config.BRAIN1_CALIBRATED_LONG_PATH),
                ("Brain1 SHORT (Calibrated)", config.BRAIN1_CALIBRATED_SHORT_PATH),
                ("Brain2 Conviction",         config.BRAIN2_MODEL_PATH),
            ]
        else:
            models_to_check = [
                ("Brain1 LONG (Raw)",  config.BRAIN1_MODEL_LONG_PATH),
                ("Brain1 SHORT (Raw)", config.BRAIN1_MODEL_SHORT_PATH),
                ("Brain2 Conviction",  config.BRAIN2_MODEL_PATH),
            ]

        for name, path in models_to_check:
            if not path.exists():
                failures.append(f"{name} model file missing: {path}")
            else:
                try:
                    if str(path).endswith(".keras"):
                        # CNN Model Loader (3D Input)
                        import keras
                        m = keras.models.load_model(str(path))
                        # Use a 3D dummy input [batch, window, features]
                        dummy_3d = np.zeros((1, config.CNN_WINDOW_SIZE, len(EXPECTED_B1)), dtype=np.float32)
                        _ = m.predict(dummy_3d, verbose=0)
                    elif ".pkl" in str(path):
                        m = joblib.load(str(path))
                        _ = m.predict_proba(dummy_b1)
                    else:
                        if "Brain2" in name:
                            # Use Booster for class-agnostic JSON loading
                            m = xgb.Booster(); m.load_model(str(path))
                            # Crucial: DMatrix needs names if model has them
                            _ = m.predict(xgb.DMatrix(dummy_b2, feature_names=EXPECTED_B2))
                        else:
                            m = xgb.XGBClassifier(); m.load_model(str(path))
                            _ = m.predict_proba(dummy_b1)
                except Exception as e:
                    failures.append(f"{name} load/inference error: {e}")

        # 3. Config thresholds
        for key in ("LONG_ENTRY_PROB_THRESH", "SHORT_ENTRY_PROB_THRESH", "FRACDIFF_WARMUP_BRICKS"):
            if not hasattr(config, key):
                failures.append(f"config.{key} missing — add to config.py")

        # 4. Threshold sanity
        lt = getattr(config, "LONG_ENTRY_PROB_THRESH", 0.55)
        st = getattr(config, "SHORT_ENTRY_PROB_THRESH", 0.50)
        if lt < st:
            failures.append(f"LONG_ENTRY_PROB_THRESH ({lt}) < SHORT_ENTRY_PROB_THRESH ({st}) — inverted!")

        # 5. Universe CSV and token format
        df = pd.read_csv(config.UNIVERSE_CSV)
        eq = df[df["is_index"] == False]
        bad = [r["symbol"] for _, r in eq.iterrows()
               if not str(r["instrument_token"]).startswith("NSE_EQ|")]
        if bad:
            failures.append(f"Bad tokens (not NSE_EQ|...): {bad[:5]}")

        # 6. Feature import
        from src.core.features import compute_features_live  # noqa: F401

    except Exception as e:
        failures.append(f"Audit exception: {e}")

    return failures


def main():
    commands = {
        "download":  "src.data.batch_factory",
        "features":  "src.data.feature_engine",
        "train":     "src.ml.brain_trainer",
        "live":      "src.live.engine",
        "backup":    "src.data.backup_pipeline",
        "backtest":  "src.ml.backtester",
        "spoofer":   "offline_spoofer", 
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    # --- DEPRECATED DASHBOARDS ---
    # if cmd == "dashboard":
    #     import subprocess
    #     subprocess.run([
    #         sys.executable, "-m", "streamlit", "run",
    #         "src/ui/dashboard.py", "--server.port", "8501",
    #     ])
    #     return
    # 
    # if cmd == "paper_dashboard":
    #     import subprocess
    #     subprocess.run([
    #         sys.executable, "-m", "streamlit", "run",
    #         "src/ui/paper_dashboard.py", "--server.port", "8502",
    #     ])
    #     return
    # -----------------------------

    if cmd not in commands:
        print(f"Unknown command: {cmd}\n{__doc__}")
        sys.exit(1)

    # ── PRE-FLIGHT ALIGNMENT AUDIT (live only) ───────────────────────────────
    # Catches model/feature/config mismatches BEFORE market open, not at 9:15.
    if cmd == "live":
        import importlib as _il
        _audit = _il.import_module("tmp.alignment_audit") if False else None
        # Inline audit — no import path issues
        _failures = _run_preflight_audit()
        if _failures:
            print("\n" + "="*60)
            print("  PRE-FLIGHT AUDIT FAILED — Engine blocked from starting!")
            print("="*60)
            for f in _failures:
                print(f"  [FAIL] {f}")
            print("="*60)
            print("  Fix the issues above then restart main.py live")
            sys.exit(1)
        print("  [PRE-FLIGHT] All checks passed — starting engine...\n")

    # Dynamic import + run
    import importlib
    mod = importlib.import_module(commands[cmd])


    runners = {
        "download":  "run_batch_factory",
        "features":  "run_feature_engine",
        "train":     "run_brain_trainer",
        "live":      "run_live_engine",
        "backup":    "run_backup_pipeline",
        "backtest":  "run_backtester",
        # "paper":     "run_paper_trader",
    }
    getattr(mod, runners[cmd])()


if __name__ == "__main__":
    main()
