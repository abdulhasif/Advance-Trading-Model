"""
src/data/feature_engine.py — Phase 2: Batch Feature Computation
================================================================
Loads Renko Parquet files → computes Velocity, Wick Pressure,
Relative Strength → saves enriched Parquet to storage/features/.

Run:  python -m src.data.feature_engine
"""

import sys
import logging
import numpy as np
import pandas as pd

import config
from src.core.features import (
    compute_velocity,
    compute_wick_pressure,
    compute_consecutive_same_dir,
    compute_brick_oscillation_rate,
    RelativeStrengthCalculator,
    add_whale_oi_placeholder,
    add_sentiment_placeholder,
)

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
    stock_dir = config.DATA_DIR / sector / symbol
    if not stock_dir.exists():
        return f"SKIP  {symbol} — dir not found"

    parquets = sorted(stock_dir.glob("*.parquet"))
    if not parquets:
        return f"SKIP  {symbol} — no files"

    # Load and normalize timezone per file before concat
    # (yfinance 2019-2020 is tz-naive, Upstox 2022+ is tz-aware)
    dfs = []
    for f in parquets:
        chunk = pd.read_parquet(f)
        # Normalize ALL datetime columns to Asia/Kolkata
        for col in chunk.select_dtypes(include=["datetime64", "datetimetz"]).columns:
            if chunk[col].dt.tz is None:
                chunk[col] = chunk[col].dt.tz_localize("Asia/Kolkata")
            else:
                chunk[col] = chunk[col].dt.tz_convert("Asia/Kolkata")
        dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("brick_timestamp")

    if len(df) < 2:
        return f"SKIP  {symbol} — too few bricks"

    df["velocity"] = compute_velocity(df)
    df["wick_pressure"] = compute_wick_pressure(df)
    df["relative_strength"] = rs_calc.compute_rs(df, sector)
    df["consecutive_same_dir"] = compute_consecutive_same_dir(df)
    df["brick_oscillation_rate"] = compute_brick_oscillation_rate(df)
    df = add_whale_oi_placeholder(df)
    df = add_sentiment_placeholder(df)

    out_dir = config.FEATURES_DIR / sector
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.parquet"
    df.to_parquet(out_path, engine="pyarrow", index=False)
    return f"OK    {symbol} -> {len(df)} bricks -> {out_path.name}"


def run_feature_engine():
    logger.info("=" * 70)
    logger.info("FEATURE ENGINE — Starting")
    logger.info("=" * 70)

    universe = pd.read_csv(config.UNIVERSE_CSV)
    universe["is_index"] = universe["is_index"].astype(str).str.lower().isin(["true", "1", "yes"])
    stocks = universe[~universe["is_index"]]
    rs_calc = RelativeStrengthCalculator()

    ok = skip = fail = 0
    for _, row in stocks.iterrows():
        try:
            r = enrich_stock(row["symbol"], row["sector"], rs_calc)
            logger.info(r)
            ok += r.startswith("OK")
            skip += not r.startswith("OK")
        except Exception as e:
            logger.error(f"FAIL  {row['symbol']}: {e}")
            fail += 1

    logger.info(f"DONE — OK: {ok}  Skip: {skip}  Fail: {fail}")


if __name__ == "__main__":
    run_feature_engine()
