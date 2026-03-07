import numpy as np
import pandas as pd

# 1. CONSECUTIVE DIRECTION LOGIC
def original_consecutive(dirs):
    counts = np.ones(len(dirs), dtype=float)
    for i in range(1, len(dirs)):
        if dirs[i] == dirs[i - 1]:
            counts[i] = counts[i - 1] + 1
        else:
            counts[i] = 1
    return counts

def optimized_consecutive(dirs):
    if len(dirs) == 0: return np.array([])
    changes = np.diff(dirs, prepend=dirs[0] + 1) != 0
    group_ids = np.cumsum(changes)
    group_starts = np.where(changes)[0]
    counts = np.arange(len(dirs)) - group_starts[group_ids - 1] + 1
    return counts.astype(float) # Ensure float type match

# 2. HURST EXPONENT LOGIC (Core R/S)
def original_hurst_rs(ts, lag):
    sub_rs = []
    for start in range(0, len(ts) - lag, lag):
        sub  = ts[start : start + lag]
        mean = sub.mean()
        dev  = np.cumsum(sub - mean)
        r    = dev.max() - dev.min()
        s    = sub.std(ddof=1)
        if s > 1e-9:
            sub_rs.append(r / s)
    return np.mean(sub_rs) if sub_rs else np.nan

def optimized_hurst_rs(ts, lag):
    n = len(ts)
    # Match refined logic: (n - lag - 1) // lag + 1
    n_sub = (n - lag - 1) // lag + 1 if n > lag else 0
    if n_sub == 0: return np.nan
    sub_series = ts[:n_sub * lag].reshape(n_sub, lag)
    means = sub_series.mean(axis=1, keepdims=True)
    devs = np.cumsum(sub_series - means, axis=1)
    r = devs.max(axis=1) - devs.min(axis=1)
    s = sub_series.std(axis=1, ddof=1)
    valid_mask = s > 1e-9
    if np.any(valid_mask):
        return np.mean(r[valid_mask] / s[valid_mask])
    return np.nan

# TEST DATA
np.random.seed(42)
prices = np.random.normal(100, 1, 1000)
log_prices = np.log(prices)
directions = np.where(np.diff(prices, prepend=prices[0]) > 0, 1, -1)

# VERIFY 1
res1_old = original_consecutive(directions)
res1_new = optimized_consecutive(directions)
match1 = np.array_equal(res1_old, res1_new)
print(f"Consecutive Logic Match: {match1}")

# VERIFY 2
res2_old = [original_hurst_rs(log_prices, l) for l in range(2, 20)]
res2_new = [optimized_hurst_rs(log_prices, l) for l in range(2, 20)]
# Use allclose for floats to account for epsilon differences in parallel sum orders
match2 = np.allclose(res2_old, res2_new, rtol=1e-13, atol=1e-13)
diff = np.max(np.abs(np.array(res2_old) - np.array(res2_new)))
print(f"Hurst R/S Logic Match: {match2} (Max Diff: {diff})")

if match1 and match2:
    print("SUCCESS: Logic is EXACTLY the same.")
else:
    print("FAILURE: Divergence detected.")
