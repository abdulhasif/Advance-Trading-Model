import numpy as np
import pandas as pd
from .feature_utils import compute_zscore
from core.config import base_config as config

def compute_squeeze_zscore(df: pd.DataFrame, window: int = config.SQUEEZE_WINDOW) -> pd.Series:
    """Volatility Squeeze Z-Score - Coil & Breakout Detector"""
    dur = df["duration_seconds"].clip(lower=config.VELOCITY_LONG_MIN_DURATION).fillna(60.0)
    density = 1.0 / dur
    return compute_zscore(density, window=window).clip(lower=-4.0, upper=4.0)
