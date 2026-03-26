import pandas as pd
import numpy as np

def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling Z-Score: (x - mu) / sigma."""
    mu = series.rolling(window=window, min_periods=1).mean()
    sigma = series.rolling(window=window, min_periods=1).std().clip(lower=1e-9)
    return (series - mu) / sigma

def _normalize_ts(ts_series: pd.Series) -> pd.Series:
    """Helper to convert to naive IST for grouping."""
    if ts_series.dt.tz is not None:
        return ts_series.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return ts_series

