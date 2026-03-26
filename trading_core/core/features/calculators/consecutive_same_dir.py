import numpy as np
import pandas as pd
from trading_core.core.config import base_config as config

def compute_consecutive_same_dir(df: pd.DataFrame) -> pd.Series:
    """Count of consecutive bricks in the same direction."""
    dirs = df["direction"].values
    if len(dirs) == 0: return pd.Series([], index=df.index)
    changes = np.diff(dirs, prepend=dirs[0] + 1) != 0
    group_ids = np.cumsum(changes)
    group_starts = np.where(changes)[0]
    counts = np.arange(len(dirs)) - group_starts[group_ids - 1] + 1
    return pd.Series(counts, index=df.index)

