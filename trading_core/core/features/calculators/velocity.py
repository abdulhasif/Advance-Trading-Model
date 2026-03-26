import numpy as np
import pandas as pd
from .feature_utils import _normalize_ts
from trading_core.core.config import base_config as config

def compute_velocity(df: pd.DataFrame, lookback: int = config.VELOCITY_LOOKBACK) -> pd.Series:
    """
    Renko Velocity (Momentum)
    -------------------------
    Formula:  log10( avg_dur_last_N / current_dur )
    Positive -> explosive (institutional) | Negative -> grinding (retail)
    """
    durations = df["duration_seconds"].copy()
    ts = _normalize_ts(df["brick_timestamp"])
    identicals = ts.groupby(ts).transform('count')
    durations = np.where(identicals > 1, (60.0 / identicals).clip(lower=config.MIN_BRICK_DURATION), durations)
    # Fix 3: Clip BEFORE rolling mean
    s_durations = pd.Series(durations, index=df.index).clip(lower=config.MIN_BRICK_DURATION)
    avg_dur = s_durations.rolling(window=lookback, min_periods=1).mean()
    ratio = avg_dur / s_durations
    return np.log10(ratio.clip(lower=1e-9))

def compute_velocity_long(df: pd.DataFrame, lookback: int = config.VELOCITY_LONG_LOOKBACK) -> pd.Series:
    """Long-Period Renko Velocity (20-brick momentum)"""
    durations = df["duration_seconds"].copy()
    ts = _normalize_ts(df["brick_timestamp"])
    identicals = ts.groupby(ts).transform('count')
    durations = np.where(identicals > 1, (60.0 / identicals).clip(lower=config.VELOCITY_LONG_MIN_DURATION), durations)
    # Fix 3: Clip BEFORE rolling mean
    s_durations = pd.Series(durations, index=df.index).clip(lower=config.VELOCITY_LONG_MIN_DURATION)
    avg_dur = s_durations.rolling(window=lookback, min_periods=max(1, lookback // 4)).mean()
    ratio = avg_dur / s_durations
    return np.log10(ratio.clip(lower=1e-9))

