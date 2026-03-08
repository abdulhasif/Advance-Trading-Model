"""
src/data/feature_engine.py — Phase 2: Batch Feature Computation
================================================================
Loads Renko Parquet files -> computes Velocity, Wick Pressure,
Relative Strength -> saves enriched Parquet to storage/features/.

Run:  python -m src.data.feature_engine
"""

import sys
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import config
from src.core.features import (
    compute_velocity,
    compute_wick_pressure,
    compute_consecutive_same_dir,
    compute_brick_oscillation_rate,
    RelativeStrengthCalculator,
    # Anti-Myopia: Long-lookback features
    compute_velocity_long,
    compute_trend_slope,
    compute_rolling_range_pct,
    compute_momentum_acceleration,
    # Phase 2: Institutional Alpha Factors
    compute_vwap_zscore,
    compute_vpt_acceleration,
    compute_squeeze_zscore,
    compute_streak_exhaustion,
)
from src.core.quant_fixes import apply_all_quant_fixes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "feature_engine.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def enrich_stock(symbol: str, sector: str, rs_calc: RelativeStrengthCalculator) -> str:
    """
    Enriches a single stock with features. 
    Supports INCREMENTAL updates if the feature file already exists.
    """
    out_dir = config.FEATURES_DIR / sector
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.parquet"
    
    # --- 1. Load Existing Features (if any) ---
    existing_df = pd.DataFrame()
    last_ts = None
    incremental_enabled = getattr(config, "FEATURE_INCREMENTAL_ENABLED", True)
    
    if out_path.exists() and incremental_enabled:
        try:
            existing_df = pd.read_parquet(out_path)
            if not existing_df.empty:
                last_ts = existing_df["brick_timestamp"].max()
        except Exception as e:
            logger.warning(f"Could not read existing features for {symbol}: {e}")

    # --- 2. Load Raw Bricks ---
    stock_dir = config.DATA_DIR / sector / symbol
    if not stock_dir.exists():
        return f"SKIP  {symbol} — raw data dir not found"

    parquets = sorted(stock_dir.glob("*.parquet"))
    if not parquets:
        return f"SKIP  {symbol} — no raw parquet files"

    all_raw_dfs = []
    for f in parquets:
        chunk = pd.read_parquet(f)
        # Normalize timezones
        for col in chunk.select_dtypes(include=["datetime64", "datetimetz"]).columns:
            if chunk[col].dt.tz is None:
                chunk[col] = chunk[col].dt.tz_localize("Asia/Kolkata")
            else:
                chunk[col] = chunk[col].dt.tz_convert("Asia/Kolkata")
        
        # Incremental filter
        if last_ts is not None:
            chunk = chunk[chunk["brick_timestamp"] > last_ts]
            
        if not chunk.empty:
            all_raw_dfs.append(chunk)

    if not all_raw_dfs:
        return f"SKIP  {symbol} — no NEW bricks since {last_ts}"

    new_raw_df = pd.concat(all_raw_dfs, ignore_index=True).sort_values("brick_timestamp", kind="mergesort")
    
    # We need ~X previous bricks to calculate rolling features (Velocity, RS, Hurst) accurately
    lookback_context = config.FEATURE_LOOKBACK_CONTEXT
    if existing_df.empty:
        # Full re-run
        compute_df = new_raw_df
    else:
        # Take the tail of existing to provide context for the new bricks
        context_df = existing_df.tail(lookback_context).copy()
        # Drop feature columns from context so they are re-calculated fresh with the new data
        feature_cols = [
            "velocity", "wick_pressure", "relative_strength", "consecutive_same_dir",
            "brick_oscillation_rate", "fracdiff_price", "hurst", "is_trending_regime",
            "whale_oi_score", "sentiment_score",
            # Anti-Myopia: Long-lookback features
            "velocity_long", "trend_slope", "rolling_range_pct", "momentum_acceleration",
            # Phase 2: Institutional Alpha Factors
            "vwap_zscore", "vpt_acceleration", "squeeze_zscore", "streak_exhaustion",
            # Temporal Alpha Features
            "true_gap_pct", "time_to_form_seconds", "volume_intensity_per_sec", "is_opening_drive",
        ]
        context_df = context_df.drop(columns=[c for c in feature_cols if c in context_df.columns])
        
        compute_df = pd.concat([context_df, new_raw_df], ignore_index=True)

    if len(compute_df) < 2:
        return f"SKIP  {symbol} — too few bricks for math"

    # --- 4. Calculate Features ---
    compute_df["velocity"] = compute_velocity(compute_df)
    compute_df["wick_pressure"] = compute_wick_pressure(compute_df)
    compute_df["relative_strength"] = rs_calc.compute_rs(compute_df, sector)
    compute_df["consecutive_same_dir"] = compute_consecutive_same_dir(compute_df)
    compute_df["brick_oscillation_rate"] = compute_brick_oscillation_rate(compute_df)
    # Anti-Myopia: Long-lookback features
    compute_df["velocity_long"] = compute_velocity_long(compute_df)
    compute_df["trend_slope"] = compute_trend_slope(compute_df)
    compute_df["rolling_range_pct"] = compute_rolling_range_pct(compute_df)
    compute_df["momentum_acceleration"] = compute_momentum_acceleration(compute_df)

    # Phase 2: Institutional Alpha Factors
    compute_df["vwap_zscore"] = compute_vwap_zscore(
        compute_df, window=config.VWAP_WINDOW
    )
    compute_df["vpt_acceleration"] = compute_vpt_acceleration(
        compute_df, diff_lag=config.VPT_ACCEL_LAG
    )
    compute_df["squeeze_zscore"] = compute_squeeze_zscore(
        compute_df, window=config.SQUEEZE_WINDOW
    )
    compute_df["streak_exhaustion"] = compute_streak_exhaustion(
        compute_df,
        onset=config.STREAK_EXHAUSTION_ONSET,
        scale=config.STREAK_EXHAUSTION_SCALE,
    )

    compute_df["whale_oi_score"] = float("nan")
    compute_df["sentiment_score"] = float("nan")

    try:
        compute_df = apply_all_quant_fixes(compute_df, fracdiff_d=config.FRACDIFF_D, hurst_window=config.HURST_WINDOW)
    except Exception as e:
        logger.warning(f"quant_fixes skipped for {symbol}: {e}")

    # --- 5. Merge and Save ---
    if existing_df.empty:
        final_df = compute_df
    else:
        # Only keep the NEWLY calculated part (exclude the context rows)
        new_features_df = compute_df.iloc[len(context_df):].copy()
        final_df = pd.concat([existing_df, new_features_df], ignore_index=True)

    final_df.to_parquet(out_path, engine="pyarrow", index=False)
    return f"OK    {symbol} -> +{len(new_raw_df)} new bricks -> {out_path.name}"


def run_feature_engine():
    logger.info("=" * 70)
    logger.info("FEATURE ENGINE — Starting (Parallel + Incremental)")
    logger.info("=" * 70)

    universe = pd.read_csv(config.UNIVERSE_CSV)
    universe["is_index"] = universe["is_index"].astype(str).str.lower().isin(["true", "1", "yes"])
    stocks = universe[~universe["is_index"]]
    rs_calc = RelativeStrengthCalculator()

    ok = skip = fail = 0
    
    # Use ProcessPoolExecutor for parallel processing
    config_workers = getattr(config, "FEATURE_PARALLEL_WORKERS", -1)
    if config_workers == -1:
        num_workers = max(1, cpu_count() - 1)
    else:
        num_workers = max(1, config_workers)
        
    logger.info(f"Using {num_workers} CPU cores for parallel processing...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for _, row in stocks.iterrows():
            futures.append(executor.submit(enrich_stock, row["symbol"], row["sector"], rs_calc))
            
        for future in futures:
            try:
                r = future.result()
                logger.info(r)
                ok += r.startswith("OK")
                skip += r.startswith("SKIP")
            except Exception as e:
                logger.error(f"Worker FAIL: {e}")
                fail += 1

    logger.info(f"DONE — OK: {ok}  Skip: {skip}  Fail: {fail}")


if __name__ == "__main__":
    run_feature_engine()
