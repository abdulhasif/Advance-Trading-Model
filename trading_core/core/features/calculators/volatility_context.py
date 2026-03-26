import numpy as np
import pandas as pd
from .feature_utils import _normalize_ts
from .feature_utils import compute_zscore
from trading_core.core.config import base_config as config

def compute_tib_zscore(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """Time-in-Brick (TiB) Z-Score - Formation Speed Normalizer"""
    if "duration_seconds" in df.columns:
        dur = df["duration_seconds"].clip(lower=config.VELOCITY_LONG_MIN_DURATION).fillna(60.0)
    else:
        ts = _normalize_ts(df["brick_timestamp"])
        dur = ts.diff().dt.total_seconds().fillna(60.0).clip(lower=config.VELOCITY_LONG_MIN_DURATION)
    mu = dur.rolling(window=window, min_periods=1).mean().shift(1)
    sigma = dur.rolling(window=window, min_periods=1).std().shift(1).clip(lower=1.0)
    return ((dur - mu) / sigma).fillna(0.0).clip(lower=-5.0, upper=5.0)

def compute_vpb_roc(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume-per-Brick (VpB) Rate of Change - Local Volume Spike Detector"""
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return pd.Series(0.0, index=df.index)
    vol = df["volume"].fillna(0.0).clip(lower=0.0)
    mu_vol = vol.rolling(window=window, min_periods=1).mean().shift(1)
    safe_denominator = mu_vol.clip(lower=1.0)
    return ((vol - mu_vol) / safe_denominator).fillna(0.0).clip(lower=-10.0, upper=10.0)

def compute_squeeze_zscore(df: pd.DataFrame, window: int = config.SQUEEZE_WINDOW) -> pd.Series:
    """Volatility Squeeze Z-Score - Coil & Breakout Detector"""
    dur = df["duration_seconds"].clip(lower=config.VELOCITY_LONG_MIN_DURATION).fillna(60.0)
    density = 1.0 / dur
    return compute_zscore(density, window=window).clip(lower=-4.0, upper=4.0)

