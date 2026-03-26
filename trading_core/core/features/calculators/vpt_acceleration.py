import numpy as np
import pandas as pd
from .feature_utils import compute_zscore
from trading_core.core.config import base_config as config

def compute_vpt_acceleration(df: pd.DataFrame, diff_lag: int = config.VPT_ACCEL_DIFF) -> pd.Series:
    """VPT Acceleration - Institutional Footprint Detector"""
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return pd.Series(0.0, index=df.index)
    close = df["brick_close"]
    vol   = df["volume"].fillna(0.0)
    price_ret = close.pct_change().fillna(0.0)
    vpt = (vol * price_ret).cumsum()
    vpt_accel = vpt.diff(diff_lag).diff(diff_lag)
    return compute_zscore(vpt_accel.fillna(0.0), window=20)

