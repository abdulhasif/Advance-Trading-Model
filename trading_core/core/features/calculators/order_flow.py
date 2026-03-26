import numpy as np
import pandas as pd
from .feature_utils import _normalize_ts
from core.config import base_config as config

def compute_order_flow_delta(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Microstructure Delta - Patched for Renko Geometry"""
    result = pd.DataFrame(index=df.index)
    
    if "volume" not in df.columns:
        result["feature_brick_volume_delta"] = 0.0
        result["feature_cvd_divergence"] = 0.0
        return result
        
    # Check if true order flow was aggregated during the 1-minute candle ingestion
    if "true_volume_delta" in df.columns:
        raw_delta = df["true_volume_delta"].fillna(0.0)
    else:
        # Graceful Degradation: Standard geometric delta is binary on Renko.
        # Fallback to pure directional volume to prevent mathematical distortion.
        direction = np.sign(df["brick_close"] - df["brick_open"])
        
        # If your DataFrame already has a 'direction' column (1 or -1), you can use:
        # direction = df["direction"] 
        
        raw_delta = df["volume"].fillna(0.0) * direction

    result["feature_brick_volume_delta"] = raw_delta
    
    # CVD resets daily to prevent macro-drift across trading sessions
    ts = _normalize_ts(df["brick_timestamp"])
    trading_day = ts.dt.date
    cvd = result["feature_brick_volume_delta"].groupby(trading_day).cumsum()
    
    # NEW: Safely group the rolling Z-score by day without NaN poisoning
    def rolling_z(x):
        mu = x.rolling(window=window, min_periods=1).mean()
        sigma = x.rolling(window=window, min_periods=1).std().clip(lower=1e-9)
        z = (x - mu.shift(1).bfill()) / sigma.shift(1).bfill()
        return z
        
    result["feature_cvd_divergence"] = cvd.groupby(trading_day, group_keys=False).apply(rolling_z).fillna(0.0).clip(-5, 5)
    
    return result
