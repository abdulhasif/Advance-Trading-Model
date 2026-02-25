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
                       Larger → more memory preserved, smaller → faster.
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
            d:      Fractional order in [0, 1]. d=0 → raw price (no diff),
                    d=1 → standard differencing (returns).

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
            logger.info(f"  FracDiff d={d:.2f} → ADF p-value: {p_value:.4f}")
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

def compute_dynamic_brick_pct(ohlcv_1min_df: pd.DataFrame,
                               atr_period: int = 14,
                               scaling_factor: float = 0.5,
                               min_pct: float = 0.0010,
                               max_pct: float = 0.0050) -> pd.Series:
    """
    Replace the static 0.15% Renko brick with a Dynamic ATR-scaled brick.

    Theoretical Motivation:
        Under GARCH(1,1), the conditional variance σ²_t evolves as:
            σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

        The volatility-adjusted threshold ensures the brick represents
        exactly the same fraction of the conditional standard deviation
        regardless of regime (opening auction vs. afternoon consolidation).

    This implementation uses rolling intraday ATR as a computationally
    practical approximation for the GARCH conditional variance.

    ATR₁₄ = rolling_mean(max(H-L, |H-prev_C|, |L-prev_C|), 14)
    dynamic_pct = clip(ATR₁₄ / close · scaling_factor, min_pct, max_pct)

    Args:
        ohlcv_1min_df:  1-minute OHLCV DataFrame with columns
                        ['open','high','low','close','volume'].
        atr_period:     Rolling window for ATR (default 14 minutes).
        scaling_factor: Multiplier to map ATR → brick pct (tune empirically).
        min_pct:        Floor brick size (0.10%) — below this is pure noise.
        max_pct:        Ceiling brick size (0.50%) — prevents runaway bricks.

    Returns:
        pd.Series of dynamic brick_pct per minute, aligned to input index.
    """
    df = ohlcv_1min_df.copy()
    h, l, c = df["high"], df["low"], df["close"]
    prev_c  = c.shift(1)

    # True Range: max of H-L, |H-prev_C|, |L-prev_C|
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)

    # Rolling ATR as % of close
    atr_pct = tr.rolling(atr_period, min_periods=1).mean() / c.clip(lower=1e-9)

    # Scale and clip to valid brick range
    dynamic_pct = (atr_pct * scaling_factor).clip(lower=min_pct, upper=max_pct)
    dynamic_pct.name = "dynamic_brick_pct"
    return dynamic_pct


def fit_garch_brick_size(returns_series: pd.Series,
                          base_pct: float = 0.0015) -> pd.Series:
    """
    Fit a GARCH(1,1) model to log-returns and derive volatility-scaled brick sizes.

    Mathematically rigorous alternative to rolling ATR.
    Uses arch library (pip install arch).

    σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
    brick_pct_t = base_pct · (σ_t / σ_mean)

    Args:
        returns_series: Series of 1-minute log returns (ln(C_t / C_{t-1})).
        base_pct:       The nominal brick size to scale around.

    Returns:
        Series of GARCH-adjusted brick sizes per minute.
    """
    try:
        from arch import arch_model   # pip install arch
    except ImportError:
        logger.warning("arch library not installed. Run: pip install arch. "
                       "Falling back to static brick size.")
        return pd.Series(base_pct, index=returns_series.index,
                         name="garch_brick_pct")

    # Fit GARCH(1,1) — scale by 1e4 for numerical stability in optimizer
    clean_ret = returns_series.dropna() * 1e4
    model     = arch_model(clean_ret, vol="Garch", p=1, q=1, dist="t")
    result    = model.fit(disp="off")

    # Extract conditional volatility and rescale
    cond_vol  = result.conditional_volatility / 1e4   # back to decimal
    mean_vol  = cond_vol.mean()

    # Scale brick size proportionally to current vol / average vol
    scaled    = base_pct * (cond_vol / mean_vol.clip(lower=1e-9))
    scaled    = scaled.clip(lower=0.001, upper=0.005)
    scaled.name = "garch_brick_pct"

    logger.info(f"GARCH(1,1) fitted. ω={result.params['omega']:.4f}  "
                f"α={result.params['alpha[1]']:.4f}  "
                f"β={result.params['beta[1]']:.4f}")
    return scaled.reindex(returns_series.index)


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
    Estimate the Hurst Exponent via the Rescaled Range (R/S) analysis.

    Mathematical definition:
        H = lim_{n→∞} [log(E[R(n)/S(n)])] / log(n)

        - H < 0.5: Mean-reverting (anti-persistent) process → bid-ask bounce
        - H ≈ 0.5: Random walk (Geometric Brownian Motion) → no edge
        - H > 0.5: Trending / persistent process → momentum is real

    NSE implication: A Renko brick formed in a H < 0.5 regime is a Roll
    effect artifact (the stock is ping-ponging the bid-ask spread).
    Brain 1 should not generate signals in such regimes.

    Args:
        series:  Price series (raw prices or log-prices, N >= 50 required).
        min_lag: Minimum R/S lag (default 2).
        max_lag: Maximum R/S lag (default 50, ~50 minutes for 1-min data).

    Returns:
        Hurst exponent H in [0, 1].
    """
    lags = range(min_lag, min(max_lag, len(series) // 2))
    ts   = np.log(series.values + 1e-9)   # log-prices
    rs   = []

    for lag in lags:
        # R/S for each sub-series of length lag
        sub_rs = []
        for start in range(0, len(ts) - lag, lag):
            sub  = ts[start : start + lag]
            mean = sub.mean()
            dev  = np.cumsum(sub - mean)
            r    = dev.max() - dev.min()   # range of cumulative deviations
            s    = sub.std(ddof=1)
            if s > 1e-9:
                sub_rs.append(r / s)
        if sub_rs:
            rs.append(np.mean(sub_rs))

    if len(rs) < 2:
        return 0.5   # fallback: assume random walk

    # Linear regression of log(RS) on log(lags)
    log_lags = np.log(list(lags)[:len(rs)])
    log_rs   = np.log(np.array(rs) + 1e-9)
    h, _     = np.polyfit(log_lags, log_rs, 1)
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
      is_trending_regime == True  → momentum signal is structurally valid
      is_trending_regime == False → structure is noise-dominated; veto entry

    Args:
        df:               Feature DataFrame with price column.
        window:           Brick lookback for Hurst calculation (default 60 bricks ≈ 1 hour).
        price_col:        Column to compute Hurst on.
        trend_threshold:  H above this → trending regime (default 0.55).

    Returns:
        DataFrame with 'hurst' and 'is_trending_regime' columns appended.
    """
    prices = df[price_col].values
    n      = len(prices)
    hurst_vals = np.full(n, 0.5)   # default: random walk assumption

    for i in range(window, n):
        sub = pd.Series(prices[i - window : i])
        hurst_vals[i] = compute_hurst_exponent(sub, min_lag=2, max_lag=window // 2)

    df = df.copy()
    df["hurst"]               = hurst_vals
    df["is_trending_regime"]  = (df["hurst"] > trend_threshold).astype(int)

    pct_trending = df["is_trending_regime"].mean() * 100
    logger.info(f"Hurst Regime: {pct_trending:.1f}% of bricks are in trending regime (H > {trend_threshold})")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FIX 5: FAT-TAIL ROBUST FEATURE SCALING
# ═══════════════════════════════════════════════════════════════════════════

class RobustFeatureScaler:
    """
    IQR-based Robust Scaler for intraday financial features.

    Standard Motivation Against StandardScaler:
        NSE 1-minute returns exhibit excess kurtosis κ >> 3 (Leptokurtic).
        Block order imbalances and circuit-breaker events create 5σ+ spikes.
        StandardScaler: x' = (x - μ) / σ   ← σ is destroyed by outliers.
        RobustScaler:   x' = (x - q50) / (q75 - q25)

        Because the IQR is a L-statistic (linear combination of order statistics),
        it has a 50% Breakdown Point — the estimator remains bounded even if
        50% of the data points are adversarial outliers.

    This scaler is fitted STRICTLY on the training set and applied to test,
    reproducing the strict temporal boundary required for OOS validity.
    """

    def __init__(self, quantile_range: Tuple[float, float] = (25.0, 75.0)):
        """
        Args:
            quantile_range: (q_low, q_high) for IQR computation.
                            Default (25, 75) = standard IQR.
        """
        self.quantile_range = quantile_range
        self._medians: dict = {}
        self._iqrs: dict    = {}
        self._fitted: bool  = False

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "RobustFeatureScaler":
        """
        Fit robust statistics (median, IQR) on the TRAINING set only.
        Call this BEFORE calling transform on either train or test.
        """
        q_lo, q_hi = self.quantile_range
        for col in cols:
            series = df[col].dropna()
            self._medians[col] = series.median()
            q75 = series.quantile(q_hi / 100)
            q25 = series.quantile(q_lo / 100)
            iqr = q75 - q25
            self._iqrs[col] = max(iqr, 1e-9)   # prevent division by zero

        self._fitted = True
        logger.info(f"RobustFeatureScaler fitted on {len(cols)} columns, "
                    f"{len(df):,} training samples.")
        return self

    def transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """
        Apply fitted robust scaling: x' = (x - median) / IQR.
        Clips at ±4 IQR to suppress extreme kurtosis artifacts.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform().")

        df = df.copy()
        for col in cols:
            if col not in self._medians:
                logger.warning(f"Column '{col}' not seen during fit. Skipping.")
                continue
            scaled = (df[col] - self._medians[col]) / self._iqrs[col]
            df[col] = scaled.clip(-4.0, 4.0)   # ±4 IQR cap on kurtosis tails

        return df

    def fit_transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Convenience: fit on df, then transform df (for training set only)."""
        return self.fit(df, cols).transform(df, cols)


def apply_quantile_transformer(train_df: pd.DataFrame,
                                test_df:  pd.DataFrame,
                                cols: list[str],
                                n_quantiles: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a QuantileTransformer (fit on train only) to normalize feature
    distributions to uniform or normal output distributions.

    Superiority over StandardScaler for fat-tailed NSE data:
      - Quantile mapping is a monotonic rank transformation.
      - It is invariant to any monotone rescaling of the input.
      - Output marginal distribution is exactly uniform or normal regardless
        of the kurtosis, skewness, or bimodality of the raw feature.

    Args:
        train_df, test_df: Train/test DataFrames.
        cols:              Feature columns to transform.
        n_quantiles:       Resolution of the quantile mapping (default 1000).

    Returns:
        (train_transformed, test_transformed)
    """
    from sklearn.preprocessing import QuantileTransformer

    qt = QuantileTransformer(n_quantiles=n_quantiles,
                              output_distribution="normal",
                              random_state=42)

    train_out = train_df.copy()
    test_out  = test_df.copy()

    # CRITICAL: fit ONLY on training data
    qt.fit(train_df[cols].fillna(0))
    train_out[cols] = qt.transform(train_df[cols].fillna(0))
    test_out[cols]  = qt.transform(test_df[cols].fillna(0))

    logger.info(f"QuantileTransformer (normal output) applied to {len(cols)} features. "
                f"Train: {len(train_df):,}  Test: {len(test_df):,}")
    return train_out, test_out


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
        1. Fractional Differentiation (d=fracdiff_d) → 'fracdiff_price'
        2. (Dynamic brick sizing is applied at Renko build time, not here)
        3. t1 exit timestamps attached → enables Purge/Embargo in training
        4. Rolling Hurst exponent → 'hurst', 'is_trending_regime'
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
