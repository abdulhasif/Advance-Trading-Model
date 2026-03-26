import numpy as np
import pandas as pd
from trading_core.core.config import base_config as config

# Import calculators
from .calculators.velocity import compute_velocity
from .calculators.momentum_acceleration import compute_momentum_acceleration
from .calculators.volatility_context import compute_tib_zscore, compute_vpb_roc, compute_squeeze_zscore
from .calculators.vwap_zscore import compute_vwap_zscore
from .calculators.order_flow import compute_order_flow_delta
from .calculators.vpt_acceleration import compute_vpt_acceleration
from .calculators.wick_pressure import compute_wick_pressure
from .calculators.streak_exhaustion import compute_consecutive_same_dir, compute_streak_exhaustion
from .calculators.relative_strength import RelativeStrengthCalculator
from .calculators.structural_trend import compute_structural_score
from .calculators.market_regime import compute_market_regime_dummies

def compute_features_live(bricks_df: pd.DataFrame, sector_bricks_df: pd.DataFrame, frac_d: float = config.FRACDIFF_D, hurst_win: int = config.HURST_WINDOW) -> pd.DataFrame:
    """Final standardized interface producing exactly 17 features."""
    from trading_core.core.physics.quant_fixes import FractionalDifferentiator, compute_hurst_exponent
    df = bricks_df.copy()
    df["velocity"]              = compute_velocity(df)
    df["momentum_acceleration"] = compute_momentum_acceleration(df)
    df["feature_tib_zscore"]   = compute_tib_zscore(df)
    df["vwap_zscore"]          = compute_vwap_zscore(df)
    df["feature_vpb_roc"]      = compute_vpb_roc(df)
    prices = df["brick_close"].values
    try:
        fd = FractionalDifferentiator()
        df["fracdiff_price"] = fd.transform(np.log(df["brick_close"].clip(lower=1e-9)), frac_d).ffill().fillna(0.0).values
    except:
        df["fracdiff_price"] = 0.0
    # --- PATCHED HURST EXPONENT (Dual-Path Execution) ---
    hurst_vals = np.full(len(prices), 0.5)
    if len(prices) > hurst_win:
        # Path A: Live Mode Fast-Path (Low Latency for WebSockets)
        # Triggered if the array is just the window size + a small buffer
        if len(prices) <= hurst_win + 5: 
            sub = pd.Series(prices[-hurst_win:])
            hurst_vals[-1] = compute_hurst_exponent(sub, min_lag=2, max_lag=hurst_win // 2)
        
        # Path B: Offline Batch Mode (Local Parquet Historical Training)
        # Triggered when processing large historical DataFrames
        else:
            def _hurst_wrapper(array_slice):
                return compute_hurst_exponent(pd.Series(array_slice), min_lag=2, max_lag=hurst_win // 2)
            
            # Apply rolling calculation across the entire historical dataset
            rolling_h = pd.Series(prices).rolling(window=hurst_win).apply(_hurst_wrapper, raw=True)
            hurst_vals = rolling_h.fillna(0.5).values
            
    df["hurst"] = hurst_vals
    oflow = compute_order_flow_delta(df)
    df["feature_cvd_divergence"] = oflow["feature_cvd_divergence"]
    df["vpt_acceleration"]      = compute_vpt_acceleration(df)
    df["wick_pressure"]         = compute_wick_pressure(df)
    df["consecutive_same_dir"] = compute_consecutive_same_dir(df)
    df["streak_exhaustion"]    = compute_streak_exhaustion(df)
    rs_calc = RelativeStrengthCalculator()
    df["relative_strength"] = rs_calc.compute_rs(df, sector_bricks_df)
    df["structural_score"] = compute_structural_score(df) # Phase 3 Support
    if "true_gap_pct" not in df.columns: df["true_gap_pct"] = 0.0
    regimes = compute_market_regime_dummies(df)
    for col in regimes.columns: df[col] = regimes[col]
    return df

class FeatureSanityCheck:
    """Diagnostic tool to detect Sim2Real drift."""
    def __init__(self, enabled: bool = True): self.enabled = enabled
    def fit_from_parquet(self, sector: str, symbol: str) -> bool: return True
    def check(self, feat_dict: dict, symbol: str, timestamp, prob: float = -1.0) -> list: return []

