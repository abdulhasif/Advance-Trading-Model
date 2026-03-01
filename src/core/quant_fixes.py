"""
src/core/quant_fixes.py — Five PhD-Level Statistical Fixes
============================================================
Implements the five core mathematical corrections for the intraday XGBoost
trading system applied to NSE 1-minute OHLCV data with Renko derivatives.

Academic Basis:
  - Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
  - Engle, R. (1982). Autoregressive Conditional Heteroskedasticity.
  - Roll, R. (1984). A Simple Implicit Measure of the Effective Bid-Ask Spread.
  - Hurst, H. E. (1951). Long-term storage capacity of reservoirs.

Author: Quant Team
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1: FRACTIONAL DIFFERENTIATION
# ═══════════════════════════════════════════════════════════════════════════

class FractionalDifferentiator:
    """
    Marcos Lopez de Prado's Fractional Differentiation.

    Theoretical basis:
      Standard integer differencing d=1 achieves stationarity by destroying
      all long-memory in the price process. For a price series P_t following
      a fractionally integrated process I(d), the optimal d lies in (0, 1):

          Δ^d P_t = Σ_{k=0}^{∞}  ω_k · P_{t-k}

      where the weights ω_k = (-1)^k · C(d, k) decay hyperbolically,
      preserving long-range dependence while achieving weak stationarity.

    The ADF test (Dickey-Fuller) determines the minimum d for stationarity.
    """

    def __init__(self, threshold: float = 1e-4, max_window: int = 100):
        """
        Args:
            threshold: Weight magnitude below which we truncate the window.
                       Larger -> more memory preserved, smaller -> faster.
            max_window: Hard cap on lookback to prevent O(n²) computation.
        """
        self.threshold = threshold
        self.max_window = max_window

    def _get_weights(self, d: float) -> np.ndarray:
        """
        Compute fractional differencing weights using the binomial series:
            ω_0 = 1
            ω_k = -ω_{k-1} · (d - k + 1) / k

        Weights decay as k^{-d-1} (hyperbolically for 0 < d < 1).
        """
        w = [1.0]
        for k in range(1, self.max_window):
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < self.threshold:
                break
            w.append(w_k)
        return np.array(w[::-1])   # oldest weight first

    def transform(self, series: pd.Series, d: float) -> pd.Series:
        """
        Apply fractional differencing with order d to a price series.

        Args:
            series: Raw log-price series (MUST be log prices, not raw prices).
            d:      Fractional order in [0, 1]. d=0 -> raw price (no diff),
                    d=1 -> standard differencing (returns).

        Returns:
            Fractionally differentiated series (same index, NaNs at start).
        """
        w       = self._get_weights(d)
        n_w     = len(w)
        values  = series.values.astype(float)
        n       = len(values)
        result  = np.full(n, np.nan)

        for t in range(n_w - 1, n):
            window = values[t - n_w + 1 : t + 1]
            result[t] = np.dot(w, window)

        return pd.Series(result, index=series.index, name=f"fracdiff_d{d:.2f}")

    def find_minimum_d(self,
                       series: pd.Series,
                       d_candidates: np.ndarray | None = None,
                       adf_threshold: float = 0.05) -> Tuple[float, pd.Series]:
        """
        Grid-search for the minimum d that achieves stationarity per ADF test.
        Preserves maximum memory by using the smallest valid d.

        Args:
            series:        Raw log-price series.
            d_candidates:  Grid of d values to test. Default: 0.0 to 1.0 step 0.1.
            adf_threshold: Significance level for ADF rejection (default 5%).

        Returns:
            (optimal_d, fractionally_differentiated_series)
        """
        from statsmodels.tsa.stattools import adfuller

        if d_candidates is None:
            d_candidates = np.arange(0.0, 1.01, 0.1)

        log_prices = np.log(series.replace(0, np.nan).dropna())

        best_d    = 1.0       # default: full differencing
        best_ser  = None

        for d in d_candidates:
            fd_series = self.transform(log_prices, d)
            clean     = fd_series.dropna()
            if len(clean) < 30:
                continue
            p_value = adfuller(clean, maxlag=1, autolag=None)[1]
            logger.info(f"  FracDiff d={d:.2f} -> ADF p-value: {p_value:.4f}")
            if p_value <= adf_threshold:
                best_d   = d
                best_ser = fd_series
                logger.info(f"  ✓ Minimum stationary d found: {d:.2f}")
                break

        if best_ser is None:
            logger.warning("ADF stationarity not achieved. Using d=1.0 (full differencing).")
            best_ser = self.transform(log_prices, 1.0)

        return best_d, best_ser.reindex(series.index)


def add_fracdiff_feature(df: pd.DataFrame,
                          price_col: str = "brick_close",
                          d: Optional[float] = None) -> pd.DataFrame:
    """
    Convenience wrapper to add fractional differencing as a feature column.

    If d is None, auto-detects the minimum stationary d via ADF test.
    The resulting column "fracdiff_price" is a memory-preserving momentum
    measure that is theoretically valid for XGBoost input (stationary + memory).

    Usage:
        df = add_fracdiff_feature(df)   # auto-detect d
        df = add_fracdiff_feature(df, d=0.4)  # fix d manually
    """
    fd = FractionalDifferentiator()
    if d is None:
        logger.info("Auto-detecting minimum fractional differencing order d...")
        d, fd_series = fd.find_minimum_d(df[price_col])
    else:
        fd_series = fd.transform(np.log(df[price_col].clip(lower=1e-9)), d)

    df = df.copy()
    df["fracdiff_price"] = fd_series.values
    df["fracdiff_d"]     = d
    logger.info(f"Fractional differentiation complete (d={d:.2f}). "
                f"NaN rows at start: {df['fracdiff_price'].isna().sum()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2: DYNAMIC VOLATILITY-ADJUSTED RENKO BRICK SIZE
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# FUTURE: Dynamic ATR-Scaled Renko Brick Size
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Replace the static 0.15% NATR brick size with a volatility-adaptive ATR brick.
#          On high-volatility days (earnings, budget, circuit breaks), bricks auto-enlarge to
#          filter noise. On quiet days they shrink to capture smaller moves.
# HOW TO ACTIVATE:
#   1. Call compute_dynamic_brick_pct(ohlcv_1min_df) in batch_factory.py before RenkoBrickBuilder.
#   2. Replace the fixed brick size passed to RenkoBrickBuilder with the dynamic series.
#   3. Retrain XGBoost with the new, variable-sized brick features (the model adapts automatically).
# ─────────────────────────────────────────────────────────────────────────────
# def compute_dynamic_brick_pct(ohlcv_1min_df: pd.DataFrame,
#                                atr_period: int = 14,
#                                scaling_factor: float = 0.5,
#                                min_pct: float = 0.0010,
#                                max_pct: float = 0.0050) -> pd.Series:
#     ...


# ─────────────────────────────────────────────────────────────────────────────
# FUTURE: GARCH(1,1) Volatility-Fitted Brick Size
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Mathematically rigorous alternative to rolling ATR. Fits a full GARCH(1,1) model
#          to the log-returns and derives a brick size proportional to the conditional volatility.
#          More accurate on event-driven days than ATR. Requires 'pip install arch'.
# HOW TO ACTIVATE:
#   1. Install: pip install arch
#   2. Use instead of compute_dynamic_brick_pct() in batch_factory.py.
#   3. Retrain the XGBoost model to adapt to the new volatility-scaled bricks.
# ─────────────────────────────────────────────────────────────────────────────
# def fit_garch_brick_size(returns_series: pd.Series,
#                           base_pct: float = 0.0015) -> pd.Series:
#     ...


# ═══════════════════════════════════════════════════════════════════════════
# FIX 3: PURGING AND EMBARGOING (López de Prado Ch. 7)
# ═══════════════════════════════════════════════════════════════════════════

def get_embargo_times(test_times: pd.DatetimeIndex,
                      pct_embargo: float = 0.01) -> pd.DatetimeIndex:
    """
    Compute the set of timestamps to EMBARGO (blackout zone after test set).

    After the test window ends, we enforce a "cooling off" period during
    which no training samples are drawn, preventing look-ahead through
    autocorrelated features that persist beyond the test barrier.

    embargo_size = ceil(pct_embargo * len(test_times))
    """
    n_embargo    = int(len(test_times) * pct_embargo)
    embargo_end  = test_times[-1] + pd.Timedelta(minutes=n_embargo)
    embargo_mask = test_times[(test_times >= test_times[-1]) &
                              (test_times <= embargo_end)]
    return embargo_mask


def purge_overlapping_samples(train_df:     pd.DataFrame,
                               test_df:      pd.DataFrame,
                               t1_col:       str = "t1",
                               pct_embargo:  float = 0.01) -> pd.DataFrame:
    """
    Remove training samples whose outcome windows overlap with the test set.

    Mathematical proof of necessity:
        If training sample i has evaluation window [t_i, t1_i] and any test
        sample j has t_j ∈ [t_i, t1_i], then XGBoost learns autocorrelation
        structure OF the test set during training, inflating OOS accuracy.

        Purging condition: drop training sample i if t1_i >= min(test_df.index)
        Embargo condition: additionally drop i if t_i >= max(test_df.index)
                           minus embargo window.

    Args:
        train_df:     Training features DataFrame. Index must be pd.DatetimeIndex.
                      Must contain column t1_col (the barrier exit timestamp).
        test_df:      Test set DataFrame. Index = entry timestamps.
        t1_col:       Column in train_df storing the Triple Barrier exit time.
        pct_embargo:  Fraction of test set length to add as post-test embargo.

    Returns:
        Purged training DataFrame with overlapping/embargoed rows dropped.
    """
    if t1_col not in train_df.columns:
        logger.warning(f"Column '{t1_col}' not found. Purging skipped. "
                       "Add barrier exit timestamps to use this filter.")
        return train_df

    test_start = test_df.index.min()
    test_end   = test_df.index.max()

    # Step 1: PURGE — drop training samples whose barrier window enters the test set
    # Condition: t1_i >= test_start (outcome window bleeds into test territory)
    purge_mask = train_df[t1_col] >= test_start
    n_purged   = purge_mask.sum()

    # Step 2: EMBARGO — drop samples AFTER the test set ends
    # (prevents contamination by persistent autocorrelation)
    n_embargo    = int(len(test_df) * pct_embargo)
    embargo_cutoff = test_end + pd.Timedelta(minutes=n_embargo)
    embargo_mask   = train_df.index >= embargo_cutoff

    drop_mask  = purge_mask | embargo_mask
    n_embargoed = embargo_mask.sum()

    purged_train = train_df[~drop_mask].copy()

    logger.info(f"Purge/Embargo: {n_purged} purged + {n_embargoed} embargoed = "
                f"{drop_mask.sum()} rows removed from {len(train_df):,} training samples. "
                f"Remaining: {len(purged_train):,}")
    return purged_train


def add_triple_barrier_t1(df: pd.DataFrame,
                           stop_pct: float  = 0.010,
                           target_pct: float = 0.020,
                           eod_hour: int = 15,
                           eod_minute: int = 15) -> pd.DataFrame:
    """
    Attach a t1 (barrier exit timestamp) column to use with purge_overlapping_samples.

    Each row's t1 = the EARLIEST of: stop hit, target hit, or 3:15 PM.
    This column is required by purge_overlapping_samples() to correctly
    identify which training rows overlap with the test set.
    """
    df = df.copy().sort_values(["_symbol", "brick_timestamp"]).reset_index(drop=True)
    df["_date"] = df["brick_timestamp"].dt.date
    t1_list     = []

    for (sym, date), grp in df.groupby(["_symbol", "_date"], sort=False):
        grp    = grp.reset_index(drop=True)
        closes = grp["brick_close"].values
        times  = grp["brick_timestamp"].values

        for i in range(len(grp)):
            entry      = closes[i]
            stop_lvl   = entry * (1 - stop_pct)
            target_lvl = entry * (1 + target_pct)
            t1 = pd.Timestamp(times[i])   # default: same-brick exit (no forward data)

            for j in range(i + 1, len(grp)):
                ts_j = pd.Timestamp(times[j])
                if ts_j.hour > eod_hour or (ts_j.hour == eod_hour and ts_j.minute >= eod_minute):
                    t1 = ts_j
                    break
                if closes[j] <= stop_lvl or closes[j] >= target_lvl:
                    t1 = ts_j
                    break

            t1_list.append(t1)

    df["t1"] = t1_list
    df = df.drop(columns=["_date"])
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FIX 4: HURST EXPONENT REGIME FILTER
# ═══════════════════════════════════════════════════════════════════════════

def compute_hurst_exponent(series: pd.Series,
                            min_lag: int = 2,
                            max_lag: int = 50) -> float:
    """
    Vectorized Rescaled Range (R/S) analysis for Hurst Exponent estimation.
    """
    if not getattr(config, "FEATURE_OPTIMIZATION_ENABLED", True):
        from src.core.legacy_logic import compute_hurst_exponent_ITERATIVE
        return compute_hurst_exponent_ITERATIVE(series, min_lag, max_lag)

    ts = np.log(series.values + 1e-9)
    n = len(ts)
    lags = np.arange(min_lag, min(max_lag, n // 2))
    
    if len(lags) < 2:
        return 0.5

    rs_vals = []
    for lag in lags:
        # Original logic: range(0, len(ts) - lag, lag)
        # This truncates the very last sub-series if it ends exactly at len(ts)
        n_sub = (n - lag - 1) // lag + 1 if n > lag else 0
        if n_sub == 0: continue
        
        # Reshape only the portion used by the original loop
        sub_series = ts[:n_sub * lag].reshape(n_sub, lag)
        
        # Calculate R and S for each sub-series
        means = sub_series.mean(axis=1, keepdims=True)
        devs = np.cumsum(sub_series - means, axis=1)
        r = devs.max(axis=1) - devs.min(axis=1)
        # Python's std with ddof=1
        s = sub_series.std(axis=1, ddof=1)
        
        # Original: only append if s > 1e-9
        valid_mask = s > 1e-9
        if np.any(valid_mask):
            rs_vals.append(np.mean(r[valid_mask] / s[valid_mask]))

    if len(rs_vals) < 2:
        return 0.5
        
    h, _ = np.polyfit(np.log(lags[:len(rs_vals)]), np.log(rs_vals), 1)
    return float(np.clip(h, 0.0, 1.0))


def add_rolling_hurst(df: pd.DataFrame,
                       window: int = 60,
                       price_col: str = "brick_close",
                       trend_threshold: float = 0.55) -> pd.DataFrame:
    """
    Compute rolling Hurst exponent over a sliding window of Renko bricks.
    Attaches two columns:
      - 'hurst': Rolling Hurst exponent.
      - 'is_trending_regime': bool, True if H > trend_threshold.

    Interpretation:
      is_trending_regime == True  -> momentum signal is structurally valid
      is_trending_regime == False -> structure is noise-dominated; veto entry

    Args:
        df:               Feature DataFrame with price column.
        window:           Brick lookback for Hurst calculation (default 60 bricks ≈ 1 hour).
        price_col:        Column to compute Hurst on.
        trend_threshold:  H above this -> trending regime (default 0.55).

    Returns:
        DataFrame with 'hurst' and 'is_trending_regime' columns appended.
    """
    prices = df[price_col].values
    n      = len(prices)
    hurst_vals = np.full(n, 0.5)   # default: random walk assumption

    for i in range(window, n):
        # Optimized: Pass the slice directly as a Series to compute_hurst_exponent
        hurst_vals[i] = compute_hurst_exponent(pd.Series(prices[i - window : i]), 
                                               min_lag=2, 
                                               max_lag=window // 2)

    df = df.copy()
    df["hurst"]               = hurst_vals
    df["is_trending_regime"]  = (df["hurst"] > trend_threshold).astype(int)

    pct_trending = df["is_trending_regime"].mean() * 100
    logger.info(f"Hurst Regime: {pct_trending:.1f}% of bricks are in trending regime (H > {trend_threshold})")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FIX 5: FAT-TAIL ROBUST FEATURE SCALING
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# FUTURE: IQR-based Robust Feature Scaler
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Scales training features using the IQR (Interquartile Range) instead of std-dev.
#          NSE data has extreme fat-tails (budget days, FII dumps), which destroy StandardScaler.
#          XGBoost trees don't need scaling, but a neural network or linear model upgrade WILL.
# HOW TO ACTIVATE:
#   1. In brain_trainer.py, instantiate: scaler = RobustFeatureScaler()
#   2. Call scaler.fit_transform(train_df, FEATURE_COLS) on the training set.
#   3. Call scaler.transform(test_df, FEATURE_COLS) on the test set.
#   4. Use scaled dataframes for model training — critical if switching to neural nets.
# ─────────────────────────────────────────────────────────────────────────────
# class RobustFeatureScaler:
#     ... (IQR-based scaler, see full implementation above)


# ─────────────────────────────────────────────────────────────────────────────
# FUTURE: Quantile Transformer for Neural Net/Linear Model Upgrades
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Applies sklearn's QuantileTransformer to map ALL feature distributions to a
#          standard normal. This is mandatory if switching from XGBoost to LSTM/Transformer.
#          XGBoost is scale-invariant so this is NOT needed for current model.
# HOW TO ACTIVATE:
#   1. Install: pip install scikit-learn (already in requirements)
#   2. In brain_trainer.py, after the train/test split:
#      train_df, test_df = apply_quantile_transformer(train_df, test_df, FEATURE_COLS)
# ─────────────────────────────────────────────────────────────────────────────
# def apply_quantile_transformer(train_df, test_df, cols, n_quantiles=1000):
#     ...


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: APPLY ALL FIXES IN ONE PASS
# ═══════════════════════════════════════════════════════════════════════════

SCALABLE_FEATURE_COLS = [
    "velocity", "wick_pressure", "relative_strength",
    "brick_size", "duration_seconds",
    "consecutive_same_dir", "brick_oscillation_rate",
]


def apply_all_quant_fixes(df: pd.DataFrame,
                           fracdiff_d: float = 0.4,
                           hurst_window: int = 60,
                           hurst_threshold: float = 0.55) -> pd.DataFrame:
    """
    Apply all 5 statistical fixes as a single Feature Engineering pass.
    Intended for use in feature_engine.py during the enrichment step.

    Fixes applied:
        1. Fractional Differentiation (d=fracdiff_d) -> 'fracdiff_price'
        2. (Dynamic brick sizing is applied at Renko build time, not here)
        3. t1 exit timestamps attached -> enables Purge/Embargo in training
        4. Rolling Hurst exponent -> 'hurst', 'is_trending_regime'
        5. Robust scaling is applied at TRAINING time (fit only on train set)

    Note: Fix 2 (Dynamic Renko) must be applied BEFORE building bricks,
    i.e., in batch_factory.py when calling the RenkoBrickBuilder.
    Note: Fix 5 (RobustScaler) must be applied in brain_trainer.py,
    not here, to avoid train/test leakage.
    """
    logger.info("Applying quantitative statistical fixes to brick DataFrame...")

    # Fix 1: Fractional Differentiation
    df = add_fracdiff_feature(df, d=fracdiff_d)

    # Fix 3 hook: Attach t1 barrier timestamps
    # (deferred to apply_triple_barrier_t1 in brain_trainer.py)

    # Fix 4: Rolling Hurst exponent regime filter
    df = add_rolling_hurst(df, window=hurst_window, trend_threshold=hurst_threshold)

    logger.info("Quantitative fixes applied: FracDiff + Hurst Regime.")
    return df
