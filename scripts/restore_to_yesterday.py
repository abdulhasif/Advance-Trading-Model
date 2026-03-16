import pandas as pd
from pathlib import Path
import os
import sys

# Define target directories
PROJECT_ROOT = Path(r"c:\Trading Platform\Advance Trading Model")
FEATURES_DIR = PROJECT_ROOT / "storage" / "features"
DATA_DIR     = PROJECT_ROOT / "storage" / "data"

# The cutoff is the start of March 13th. 
# We want to keep everything before this.
CUTOFF_DATE = "2026-03-13"

def restore_directory(target_dir: Path):
    if not target_dir.exists():
        print(f"Skipping {target_dir} (not found)")
        return

    print(f"Restoring {target_dir} to pre-{CUTOFF_DATE} state...")
    
    # Iterate through all parquet files recursively
    count = 0
    for pf in target_dir.rglob("*.parquet"):
        try:
            df = pd.read_parquet(pf)
            if "brick_timestamp" not in df.columns:
                continue
                
            original_len = len(df)
            
            # Standardize to Naive IST before comparison
            if df["brick_timestamp"].dt.tz is not None:
                if df["brick_timestamp"].dt.tz is not None:
                    df["brick_timestamp"] = df["brick_timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                else:
                    df["brick_timestamp"] = df["brick_timestamp"].dt.tz_localize(None) if hasattr(df["brick_timestamp"].dt, 'tz_localize') else df["brick_timestamp"]
            else:
                df["brick_timestamp"] = df["brick_timestamp"].dt.tz_localize(None)
                
            cutoff_ts = pd.Timestamp(CUTOFF_DATE)
            df = df[df["brick_timestamp"] < cutoff_ts]
            
            if len(df) < original_len:
                df.to_parquet(pf)
                count += 1
                
        except Exception as e:
            pass
            
    print(f"Successfully truncated {count} files in {target_dir}.")

if __name__ == "__main__":
    # In agentic mode, we might want to skip the input or handle it via command
    if len(sys.argv) > 1 and sys.argv[1] == "--yes":
        pass
    else:
        confirm = input("This will DELETE all data from 2026-03-13 onwards in storage/features and storage/data. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        
    restore_directory(FEATURES_DIR)
    restore_directory(DATA_DIR)
    print("\nRestoration complete. Parquets are now in 'Yesterday's State'.")
