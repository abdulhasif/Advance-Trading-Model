import numpy as np
import pandas as pd
from trading_core.core.config import base_config as config

def compute_vwap_zscore(df: pd.DataFrame, window: int = config.VWAP_WINDOW) -> pd.Series:
    """VWAP Z-Score - The Institutional Anchor"""
    has_volume = ("volume" in df.columns and "typical_price" in df.columns and df["volume"].sum() > 0)
    if has_volume:
        tp  = df["typical_price"].fillna(df["brick_close"])
        vol = df["volume"].fillna(0.0).clip(lower=0.0)
        tp_vol = (tp * vol).rolling(window=window, min_periods=1).sum()
        vol_sum = vol.rolling(window=window, min_periods=1).sum().clip(lower=1e-9)
        vwap = tp_vol / vol_sum
    else:
        vwap = df["brick_close"].rolling(window=window, min_periods=1).mean()
    close  = df["brick_close"]
    sigma  = close.rolling(window=window, min_periods=1).std().clip(lower=1e-9)
    return ((close - vwap) / sigma).clip(lower=-5.0, upper=5.0)

