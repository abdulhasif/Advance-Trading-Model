import pandas as pd
import numpy as np
from pathlib import Path
import random
import warnings

# Suppress pandas warning about timezone arithmetic
warnings.simplefilter(action='ignore', category=FutureWarning)

features_dir = Path("storage/features")
if not features_dir.exists():
    print("FATAL: No storage/features directory found.")
    exit(1)

all_files = list(features_dir.rglob("*.parquet"))
if not all_files:
    print("FATAL: No parquet files found in storage/features.")
    exit(1)

print(f"==================================================")
print(f"  DATA QUALITY DIAGNOSTIC — {len(all_files)} feature files")
print(f"==================================================")

# Pick 3 random, non-empty files
random.seed(42)  # for reproducibility
sample_files = random.sample(all_files, min(3, len(all_files)))

for pf in sample_files:
    print(f"\n[ FILE ] {pf.parent.name} / {pf.name}")
    try:
        df = pd.read_parquet(pf)
    except Exception as e:
        print(f"  ❌ ERROR loading parquet: {e}")
        continue
        
    print(f"  -> Rows: {len(df):,} | Columns: {len(df.columns)}")
    
    # Check 1: Temporal Order (critical for leakage prevention)
    if "brick_timestamp" in df.columns:
        is_sorted = df["brick_timestamp"].is_monotonic_increasing
        ts_status = "✅ YES" if is_sorted else "❌ NO (CRITICAL LEAKAGE RISK)"
        print(f"  -> Time Monotonically Sorted: {ts_status}")
        
        # Check start and end dates
        try:
            start_date = df["brick_timestamp"].iloc[0].strftime("%Y-%m-%d")
            end_date   = df["brick_timestamp"].iloc[-1].strftime("%Y-%m-%d")
            print(f"  -> Date Range: {start_date} to {end_date}")
        except:
            pass
    else:
        print("  ❌ ERROR: Missing 'brick_timestamp' column!")

    # Check 2: Missing Values (NaNs) & Infinities
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    # Filter out columns we expect to be entirely NaN placeholders
    expected_nans = {"whale_oi_score", "sentiment_score"}
    active_cols = [c for c in num_cols if c not in expected_nans]
    
    total_bricks = len(df)
    nans = df[active_cols].isna().sum()
    infs = df[active_cols].isin([np.inf, -np.inf]).sum()
    
    if nans.sum() == 0 and infs.sum() == 0:
        print("  -> Cleanliness: ✅ 0 unexpected NaNs, 0 Infinities")
    else:
        print("  -> Cleanliness: ⚠️ Issues Found")
        for col, count in nans[nans > 0].items():
             pct = (count / total_bricks) * 100
             print(f"      - {col}: {count:,} NaNs ({pct:.1f}%)")
        for col, count in infs[infs > 0].items():
             pct = (count / total_bricks) * 100
             print(f"      - {col}: {count:,} Infinities ({pct:.1f}%)")

    # Check 3: Statistical distribution of key engineered features
    print("  -> Feature Profiling:")
    for f in ["velocity", "wick_pressure", "duration_seconds", "fracdiff_price", "hurst", "brick_size"]:
        if f in df.columns:
            s_min = df[f].min()
            s_max = df[f].max()
            s_med = df[f].median()
            s_std = df[f].std()
            
            # Simple sanity check diagnostics
            status = "✅"
            if f == "hurst" and (s_med < 0.35 or s_med > 0.65): status = "⚠️ (abnormal regime)"
            if f == "wick_pressure" and s_med > 0.5: status = "⚠️ (abnormally high wick)"
            if f == "duration_seconds" and s_med < 2: status = "⚠️ (abnormally fast bricks)"
            
            print(f"      {status} {f:<17} | med: {s_med:8.3f} | min: {s_min:8.3f} | max: {s_max:8.3f} | std: {s_std:8.3f}")

print("\n==================================================")
print("  ✓ Diagnostic complete.")
