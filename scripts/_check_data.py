"""Quick check: verify 2026 data includes today's candles."""
import glob
import pandas as pd

files = sorted(glob.glob("storage/data/*/*/2026.parquet"))
print(f"Total 2026 parquet files: {len(files)}")
print()

# Check 5 samples
for f in files[:5]:
    parts = f.replace("\\", "/").split("/")
    sym = parts[-2]
    df = pd.read_parquet(f)
    print(f"{sym}: {len(df)} bricks, cols={list(df.columns)}")
    # Find date-like column
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower() or "ts" in col.lower():
            print(f"  {col}: {df[col].min()} to {df[col].max()}")
    print()
