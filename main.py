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
  python main.py paper          # Paper Trading: virtual execution mode
  python main.py --help         # show this help
"""

import sys


def main():
    commands = {
        "download":  "src.data.batch_factory",
        "features":  "src.data.feature_engine",
        "train":     "src.ml.brain_trainer",
        "live":      "src.live.engine",
        "backup":    "src.data.backup_pipeline",
        "backtest":  "src.ml.backtester",
        "paper":     "src.live.paper_trader",
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "dashboard":
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/ui/dashboard.py", "--server.port", "8501",
        ])
        return

    if cmd not in commands:
        print(f"Unknown command: {cmd}\n{__doc__}")
        sys.exit(1)

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
        "paper":     "run_paper_trader",
    }
    getattr(mod, runners[cmd])()


if __name__ == "__main__":
    main()
