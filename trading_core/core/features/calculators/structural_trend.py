import numpy as np
import pandas as pd
from trading_core.core.config import base_config as config

def compute_structural_score(df: pd.DataFrame, window: int = config.STRUCTURAL_WINDOW) -> pd.Series:
    """
    Structural Trend Score (Phase 3)
    --------------------------------
    Percentage of bricks in the same direction over the last N bricks.
    High score (>0.85) indicates a 'Safe Grind'.
    """
    dirs = df["direction"]
    is_up = (dirs > 0).astype(float)
    is_down = (dirs < 0).astype(float)
    
    # Fast vectorized rolling sums
    up_count = is_up.rolling(window=window, min_periods=1).sum()
    down_count = is_down.rolling(window=window, min_periods=1).sum()
    
    # Return the percentage of the dominant direction
    score = np.maximum(up_count, down_count) / window
    return score.clip(upper=1.0)

