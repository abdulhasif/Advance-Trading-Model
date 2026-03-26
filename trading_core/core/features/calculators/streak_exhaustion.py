import numpy as np
import pandas as pd
from core.config import base_config as config

def compute_streak_exhaustion(df: pd.DataFrame, onset: int = config.STREAK_EXHAUSTION_ONSET, scale: float = config.STREAK_EXHAUSTION_SCALE) -> pd.Series:
    """Streak Exhaustion - Mathematical Momentum Decay Filter"""
    streak = compute_consecutive_same_dir(df).clip(lower=0)
    x = (streak - onset) * scale
    sigmoid = 1.0 / (1.0 + np.exp(-x.clip(lower=-50, upper=50)))
    return (-sigmoid * 0.5).where(streak >= onset, 0.0)

def compute_consecutive_same_dir(df: pd.DataFrame) -> pd.Series:
    """Count of consecutive bricks in the same direction."""
    dirs = df["direction"].values
    if len(dirs) == 0: return pd.Series([], index=df.index)
    changes = np.diff(dirs, prepend=dirs[0] + 1) != 0
    group_ids = np.cumsum(changes)
    group_starts = np.where(changes)[0]
    counts = np.arange(len(dirs)) - group_starts[group_ids - 1] + 1
    return pd.Series(counts, index=df.index)
