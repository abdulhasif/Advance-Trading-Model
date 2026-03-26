import os
import sys
import subprocess

# Add project directories to PYTHONPATH automatically
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECTS = ["trading_core", "trading_pipeline", "trading_engine", "trading_api"]

for p in PROJECTS:
    sys.path.append(os.path.join(PROJECT_ROOT, p))
sys.path.append(PROJECT_ROOT)

def run_script(script_path, args=[]):
    python_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable
    
    env = os.environ.copy()
    env["PYTHONPATH"] = ";".join([os.path.join(PROJECT_ROOT, p) for p in PROJECTS] + [PROJECT_ROOT])
    
    cmd = [python_exe, os.path.join(PROJECT_ROOT, script_path)] + args
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [features|train|backtest|api|engine]")
        return

    cmd = sys.argv[1].lower()
    
    if cmd == "features":
        run_script("trading_pipeline/scripts/build_features.py")
    elif cmd == "train":
        run_script("trading_pipeline/scripts/train_models.py")
    elif cmd == "backtest":
        run_script("trading_pipeline/scripts/backtest.py")
    elif cmd == "api":
        run_script("trading_api/src/main.py")
    elif cmd == "engine":
        run_script("trading_engine/src/engine_main.py")
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()

