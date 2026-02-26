"""
src/core/legacy_logic.py — ORIGINAL Iterative Algorithms
===========================================================
This file contains the original, loop-based implementations of the 
feature engine math. Use this if you need to revert for debugging.
"""
import numpy as np
import pandas as pd

def compute_consecutive_same_dir_ITERATIVE(df: pd.DataFrame) -> pd.Series:
    dirs = df["direction"].values
    counts = np.ones(len(dirs), dtype=float)
    for i in range(1, len(dirs)):
        if dirs[i] == dirs[i - 1]:
            counts[i] = counts[i - 1] + 1
        else:
            counts[i] = 1
    return pd.Series(counts, index=df.index)

def compute_hurst_exponent_ITERATIVE(series: pd.Series, 
                                    min_lag: int = 2, 
                                    max_lag: int = 50) -> float:
    lags = range(min_lag, min(max_lag, len(series) // 2))
    ts = np.log(series.values + 1e-9)
    rs = []
    
    for lag in lags:
        sub_rs = []
        # Original exact range: range(0, len(ts) - lag, lag)
        for start in range(0, len(ts) - lag, lag):
            sub = ts[start : start + lag]
            mean = sub.mean()
            dev = np.cumsum(sub - mean)
            r = dev.max() - dev.min()
            s = sub.std(ddof=1)
            if s > 1e-9:
                sub_rs.append(r / s)
        if sub_rs:
            rs.append(np.mean(sub_rs))
            
    if len(rs) < 2:
        return 0.5
        
    log_lags = np.log(list(lags)[:len(rs)])
    log_rs = np.log(np.array(rs) + 1e-9)
    h, _ = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(h, 0.0, 1.0))
