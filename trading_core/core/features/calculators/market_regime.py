import numpy as np
import pandas as pd
from .feature_utils import _normalize_ts
from core.config import base_config as config

def compute_market_regime_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical Market Regime (IST) - One-Hot Encoded Time Buckets"""
    ts = _normalize_ts(df["brick_timestamp"])
    decimal_hour = ts.dt.hour + ts.dt.minute / 60.0
    regime = pd.Series(1, index=df.index, dtype=int)
    regime = regime.where(~(decimal_hour < 10.5), 0)
    regime = regime.where(~(decimal_hour >= 13.5), 2)
    return pd.DataFrame({
        "regime_morning":   (regime == 0).astype(int),
        "regime_midday":    (regime == 1).astype(int),
        "regime_afternoon": (regime == 2).astype(int),
    }, index=df.index)
