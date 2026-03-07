import numpy as np
import pandas as pd

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
    return counts

# Test Data
test_dirs = np.array([1, 1, 1, -1, -1, 1, -1, -1, -1, -1])
res_orig = original_consecutive(test_dirs)
res_opt = optimized_consecutive(test_dirs)

print(f"Original:  {res_orig}")
print(f"Optimized: {res_opt}")
assert np.array_equal(res_orig, res_opt), "Consecutive Logic Mismatch!"
print("✓ Consecutive Direction logic is identical.")
