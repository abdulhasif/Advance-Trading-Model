"""
src/core/legacy_logic.py — ORIGINAL Iterative Algorithms
===========================================================
This file contains the original, loop-based implementations of the 
feature engine math. Use this if you need to revert for debugging.

NOTE: This entire file is commented out because the vectorized versions in
features.py and quant_fixes.py are faster and actively used in production.
These are preserved here ONLY as a debugging fallback reference.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FUTURE: Iterative Debug Fallback Algorithms
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: If the numpy-vectorized versions of compute_consecutive_same_dir()
#          or compute_hurst_exponent() in quant_fixes.py ever give unexpected
#          results, these slow-but-readable iterative versions can be used for
#          debugging to mathematically verify the vectorized output.
# HOW TO ACTIVATE:
#   1. Import this module in features.py (uncomment the existing import).
#   2. Temporarily call the _ITERATIVE variants instead of the vectorized ones.
#   3. Compare outputs row-by-row to identify any numerical discrepancy.
# ─────────────────────────────────────────────────────────────────────────────

# import numpy as np
# import pandas as pd
#
# def compute_consecutive_same_dir_ITERATIVE(df: pd.DataFrame) -> pd.Series:
#     dirs = df["direction"].values
#     counts = np.ones(len(dirs), dtype=float)
#     for i in range(1, len(dirs)):
#         if dirs[i] == dirs[i - 1]:
#             counts[i] = counts[i - 1] + 1
#         else:
#             counts[i] = 1
#     return pd.Series(counts, index=df.index)
#
# def compute_hurst_exponent_ITERATIVE(series: pd.Series,
#                                     min_lag: int = 2,
#                                     max_lag: int = 50) -> float:
#     lags = range(min_lag, min(max_lag, len(series) // 2))
#     ts = np.log(series.values + 1e-9)
#     rs = []
#
#     for lag in lags:
#         sub_rs = []
#         for start in range(0, len(ts) - lag, lag):
#             sub = ts[start : start + lag]
#             mean = sub.mean()
#             dev = np.cumsum(sub - mean)
#             r = dev.max() - dev.min()
#             s = sub.std(ddof=1)
#             if s > 1e-9:
#                 sub_rs.append(r / s)
#         if sub_rs:
#             rs.append(np.mean(sub_rs))
#
#     if len(rs) < 2:
#         return 0.5
#
#     log_lags = np.log(list(lags)[:len(rs)])
#     log_rs = np.log(np.array(rs) + 1e-9)
#     h, _ = np.polyfit(log_lags, log_rs, 1)
#     return float(np.clip(h, 0.0, 1.0))
