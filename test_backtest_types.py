import pandas as pd
import pathlib
import joblib
import xgboost as xgb
import numpy as np

data_path = pathlib.Path(r"C:\Trading Platform\Advance Trading Model\storage\features\IT\TCS.parquet")
df = pd.read_parquet(data_path)
df = df.sort_values("brick_timestamp").reset_index(drop=True)

EXPECTED_FEATURES = [
    "velocity", "wick_pressure", "relative_strength", "brick_size",
    "duration_seconds", "consecutive_same_dir", "brick_oscillation_rate",
    "fracdiff_price", "hurst", "is_trending_regime", "velocity_long",
    "trend_slope", "rolling_range_pct", "momentum_acceleration",
    "vwap_zscore", "vpt_acceleration", "squeeze_zscore", "streak_exhaustion",
    "true_gap_pct", "time_to_form_seconds", "volume_intensity_per_sec", "is_opening_drive"
]

X = df[EXPECTED_FEATURES].fillna(0)

b1_long = joblib.load(r"C:\Trading Platform\Advance Trading Model\storage\models\brain1_calibrated_long.pkl")
b1_short = joblib.load(r"C:\Trading Platform\Advance Trading Model\storage\models\brain1_calibrated_short.pkl")
b2 = xgb.XGBRegressor()
b2.load_model(r"C:\Trading Platform\Advance Trading Model\storage\models\brain2_conviction.json")

df["pl"] = b1_long.predict_proba(X)[:, 1]
df["ps"] = b1_short.predict_proba(X)[:, 1]

print("Max PL:", df["pl"].max())
print("Max PS:", df["ps"].max())
print("PL > 0.62 count:", (df["pl"] > 0.62).sum())
print("PS > 0.62 count:", (df["ps"] > 0.62).sum())

pb = np.zeros(len(df), dtype=float)
for i in range(len(df)):
    pl = df["pl"].iloc[i]
    ps = df["ps"].iloc[i]
    long_ok  = pl >= 0.55
    short_ok = ps >= 0.50
    p = 0.0
    if long_ok and short_ok:
        p = max(pl, ps)
    elif long_ok:
        p = pl
    elif short_ok:
        p = ps
    pb[i] = p

df["brain1_prob"] = pb

X_meta = df[["brain1_prob", "velocity", "wick_pressure", "relative_strength"]].fillna(0)
df["brain2_conviction"] = b2.predict(X_meta).clip(0, 100)

print("Rows where pb > 0.62:", (df["brain1_prob"] > 0.62).sum())
print("Rows where Conv > 18:", (df["brain2_conviction"] >= 18.0).sum())
valid = df[(df["brain1_prob"] > 0.62) & (df["brain2_conviction"] >= 18.0)]
print("Rows passing BOTH:", len(valid))

# Dump a few high prob rows to see their conviction
print("\nHigh Prob rows conviction:")
high = df[df["brain1_prob"] > 0.60]
if len(high) > 0:
    print(high[["brain1_prob", "brain2_conviction"]].head(10))
