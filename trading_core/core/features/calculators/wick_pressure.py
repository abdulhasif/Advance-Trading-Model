import numpy as np
import pandas as pd
from trading_core.core.config import base_config as config

def compute_wick_pressure(df: pd.DataFrame) -> pd.Series:
    """Wick Pressure (Hidden Flow) - Direction-Aware"""
    is_long = df["direction"] > 0
    long_wick  = (df["brick_high"] - df["brick_close"]) / df["brick_size"].clip(lower=1e-9)
    short_wick = (df["brick_close"] - df["brick_low"])  / df["brick_size"].clip(lower=1e-9)
    return long_wick.where(is_long, short_wick).clip(lower=0.0)

