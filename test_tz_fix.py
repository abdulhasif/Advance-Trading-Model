import pandas as pd
import numpy as np
import pathlib
import sys
import os

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(os.getcwd())
sys.path.append(str(PROJECT_ROOT))

import config

def test_tz_stripping():
    # Find one parquet file
    pf = None
    for p in config.FEATURES_DIR.glob("**/*.parquet"):
        pf = p
        break
    
    if pf is None:
        print("No parquet files found.")
        return

    print(f"Testing on {pf}")
    df = pd.read_parquet(pf)
    print(f"Original dtype: {df['brick_timestamp'].dtype}")
    
    def _strip_tz(col):
        return pd.to_datetime(col).apply(lambda t: t.tz_localize(None) if hasattr(t, "tzinfo") and t.tzinfo else t)
    
    stripped = _strip_tz(df['brick_timestamp'])
    print(f"Stripped dtype: {stripped.dtype}")
    
    # Check one value
    val = stripped.iloc[0]
    print(f"First value: {val} (type: {type(val)})")
    
    # Simulate a comparison
    test_start = val + pd.Timedelta(days=1)
    print(f"Comparison value: {test_start} (type: {type(test_start)})")
    
    # This is what failed in the user's run
    try:
        mask = (stripped >= test_start)
        print("Comparison successful!")
    except Exception as e:
        print(f"Comparison failed: {e}")

if __name__ == "__main__":
    test_tz_stripping()
