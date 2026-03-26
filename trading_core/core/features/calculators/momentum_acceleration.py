import numpy as np
import pandas as pd
from .feature_utils import _normalize_ts
from trading_core.core.config import base_config as config

def compute_momentum_acceleration(df: pd.DataFrame,
                                    fast: int = config.VELOCITY_LOOKBACK,
                                    slow: int = config.VELOCITY_LONG_LOOKBACK) -> pd.Series:
    """Momentum Acceleration (fast - slow velocity diff)"""
    durations = df["duration_seconds"].copy()
    ts = _normalize_ts(df["brick_timestamp"])
    identicals = ts.groupby(ts).transform('count')
    durations = np.where(identicals > 1, (60.0 / identicals).clip(lower=config.VELOCITY_LONG_MIN_DURATION), durations)
    # Fix 3: Clip BEFORE rolling mean
    s_dur = pd.Series(durations, index=df.index).clip(lower=config.VELOCITY_LONG_MIN_DURATION)
    avg_fast = s_dur.rolling(window=fast, min_periods=1).mean()
    avg_slow = s_dur.rolling(window=slow, min_periods=1).mean()
    vel_fast = np.log10((avg_fast / s_dur).clip(lower=1e-9))
    vel_slow = np.log10((avg_slow / s_dur).clip(lower=1e-9))
    return vel_fast - vel_slow

