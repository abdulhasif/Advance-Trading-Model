import numpy as np
import pandas as pd
from .feature_utils import compute_zscore
from core.config import base_config as config

class RelativeStrengthCalculator:
    """RS = Stock_Z - Sector_Z (rolling 20-brick)."""
    def __init__(self, window: int = config.RS_ROLLING_WINDOW):
        self.window = window
    def _strip_tz(self, col: pd.Series) -> pd.Series:
        return config.to_naive_ist(col)
    def compute_rs(self, stock_df: pd.DataFrame, sector_bricks_df: pd.DataFrame) -> pd.Series:
        if sector_bricks_df.empty: return pd.Series(0.0, index=stock_df.index)
        stock_z = compute_zscore(stock_df["brick_close"], self.window)
        sector_z = compute_zscore(sector_bricks_df["brick_close"], self.window)
        ts = pd.DataFrame({"brick_timestamp": self._strip_tz(stock_df["brick_timestamp"]), "stock_z": stock_z.values})
        ss = pd.DataFrame({"brick_timestamp": self._strip_tz(sector_bricks_df["brick_timestamp"]), "sector_z": sector_z.values})
        m = pd.merge_asof(
            ts.sort_values("brick_timestamp"), 
            ss.sort_values("brick_timestamp"), 
            on="brick_timestamp", 
            direction="backward",
            tolerance=pd.Timedelta(minutes=15)
        )
        return (m["stock_z"] - m["sector_z"].fillna(0)).values
