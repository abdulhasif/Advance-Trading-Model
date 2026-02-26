import numpy as np
import pandas as pd

def debug_hurst_rs(ts, lag):
    # --- ORIGINAL ---
    orig_rs = []
    orig_steps = []
    for start in range(0, len(ts) - lag, lag):
        sub = ts[start : start + lag]
        mean = sub.mean()
        dev = np.cumsum(sub - mean)
        r = dev.max() - dev.min()
        s = sub.std(ddof=1)
        if s > 1e-9:
            orig_rs.append(r / s)
            orig_steps.append({'r': r, 's': s, 'rs': r/s})
    orig_mean = np.mean(orig_rs) if orig_rs else np.nan

    # --- OPTIMIZED ---
    n = len(ts)
    n_sub = (n - lag) // lag + 1
    sub_series = ts[:n_sub * lag].reshape(n_sub, lag)
    means = sub_series.mean(axis=1, keepdims=True)
    devs = np.cumsum(sub_series - means, axis=1)
    rs_vec = devs.max(axis=1) - devs.min(axis=1) # R
    ss_vec = sub_series.std(axis=1, ddof=1)      # S
    mask = ss_vec > 1e-9
    opt_rs = rs_vec[mask] / ss_vec[mask]
    opt_mean = np.mean(opt_rs) if np.any(mask) else np.nan

    # --- CALCULATION COMPARISON ---
    rs_vec_filtered = rs_vec[mask] / ss_vec[mask]
    
    # Original iterative mean
    print(f"\nFinal Statistics for Lag {lag}:")
    print(f"Original R/S List Length: {len(orig_rs)}")
    print(f"Optimized R/S List Length: {len(rs_vec_filtered)}")
    
    # Check if THE LISTS ARE IDENTICAL
    if len(orig_rs) == len(rs_vec_filtered):
        l_orig = np.array(orig_rs)
        l_opt = rs_vec_filtered
        diffs = np.abs(l_orig - l_opt)
        print(f"Max difference between individual R/S values: {np.max(diffs)}")
        print(f"Original Mean:  {np.mean(orig_rs)}")
        print(f"Optimized Mean: {np.mean(rs_vec_filtered)}")
        
    else:
        print("LIST LENGTHS DO NOT MATCH")

# Test Data
np.random.seed(42)
prices = np.random.normal(100, 1, 1000)
log_prices = np.log(prices)

debug_hurst_rs(log_prices, 10)
debug_hurst_rs(log_prices, 50)
