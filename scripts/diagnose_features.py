import pandas as pd
from src.core.features import compute_features_live

# Load historical batch features for SIEMENS (which traded today)
df_hist = pd.read_parquet("storage/features/Infrastructure/SIEMENS.parquet")
print("--- Historical Features (Last 5 bricks) ---")
print(df_hist[["brick_timestamp", "brick_close", "velocity", "wick_pressure", "relative_strength"]].dropna().tail(5))
print()

# Simulate live computation on the same raw bricks
df_raw = pd.read_parquet("storage/data/Infrastructure/SIEMENS/2026.parquet")
df_sec = pd.read_parquet("storage/data/Infrastructure/NIFTY INFRA/2026.parquet")

# Since live engine only sees data *up to now*, let's compute on the full raw history to match
df_live = compute_features_live(df_raw, df_sec)
print("--- Live Features (Last 5 bricks) ---")
print(df_live[["brick_timestamp", "brick_close", "velocity", "wick_pressure", "relative_strength"]].dropna().tail(5))
