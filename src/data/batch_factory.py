"""
src/data/batch_factory.py — Phase 1: Batch Download + Renko Pipeline
=====================================================================
Reads sector_universe.csv, downloads 4 years of 1-min OHLC from Upstox,
converts to Renko bricks (with 9:15 AM Gap Filter), and saves clean
bricks to Parquet partitioned by Sector/Stock/Year.

Run:  python -m src.data.batch_factory
"""

import sys
import shutil
import logging
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, time

import config
from src.data.downloader import UpstoxHistoricalFetcher
from src.core.renko import RenkoBrickBuilder

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "batch_factory.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def sanitize_ohlc(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Strict data sanitization middleware:
    1. Enforces NSE Market Hours (09:15-15:30).
    2. Drops entire Days with >15% inter-minute price jumps (Corporate Actions).
    3. Fills 1-minute gaps with forward-filled prices to preserve Renko continuity.
    """
    if df.empty:
        return df

    # --- 1. Basic Formatting & Sorting ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["date"] = df["timestamp"].dt.date

    # --- 2. Market Hours Enforcement (09:15 - 15:30 IST) ---
    df = df[df["timestamp"].dt.time >= time(9, 15)]
    df = df[df["timestamp"].dt.time <= time(15, 30)]

    # --- 3. Split/Corporate Action Detection ---
    # 3a. Intra-day spikes (>15%)
    df["prev_close_intra"] = df.groupby("date")["close"].shift(1)
    df["jump_intra"] = (df["open"] - df["prev_close_intra"]).abs() / df["prev_close_intra"].clip(lower=1e-9)
    dirty_dates_intra = df[df["jump_intra"] > 0.15]["date"].unique()

    # 3b. Inter-day (overnight) splits (>25%)
    # We compare the first 'open' of a day with the last 'close' of the previous day
    day_bounds = df.groupby("date").agg({"open": "first", "close": "last"}).sort_index()
    day_bounds["prev_close"] = day_bounds["close"].shift(1)
    day_bounds["overnight_jump"] = (day_bounds["open"] - day_bounds["prev_close"]).abs() / day_bounds["prev_close"].clip(lower=1e-9)
    dirty_dates_inter = day_bounds[day_bounds["overnight_jump"] > 0.25].index.unique()

    dirty_dates = set(dirty_dates_intra) | set(dirty_dates_inter)

    if dirty_dates:
        logger.warning(f"PURGE: {symbol} has large price jumps/splits on {sorted(list(dirty_dates))}. Dropping segments.")
        df = df[~df["date"].isin(dirty_dates)]

    if df.empty:
        return pd.DataFrame()

    # --- 4. 1-Minute Gap Filling ---
    # Ensures the Renko builder receives a continuous time series.
    processed_days = []
    for d, day_df in df.groupby("date"):
        full_range = pd.date_range(
            start=pd.Timestamp.combine(d, time(9, 15)),
            end=pd.Timestamp.combine(d, time(15, 30)),
            freq="1min"
        )
        # Ensure day_df is naive for reindexing with naive full_range (Robust Naive IST Pattern)
        if day_df["timestamp"].dt.tz is not None:
            if day_df["timestamp"].dt.tz is not None:
                day_df["timestamp"] = day_df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            else:
                day_df["timestamp"] = day_df["timestamp"].dt.tz_localize(None) if hasattr(day_df["timestamp"].dt, 'tz_localize') else day_df["timestamp"]
        else:
            # If already naive, we assume it's IST (as per Upstox/Project convention)
            pass
            
        day_df = day_df.set_index("timestamp").reindex(full_range)
        
        # Forward fill prices, zero-fill volume
        day_df["close"] = day_df["close"].ffill()
        day_df["open"] = day_df["open"].fillna(day_df["close"])
        day_df["high"] = day_df["high"].fillna(day_df["close"])
        day_df["low"] = day_df["low"].fillna(day_df["close"])
        day_df["volume"] = day_df["volume"].fillna(0)
        
        # Only keep days that have at least some real starts (ignore early morning gaps)
        day_df = day_df.dropna(subset=["close"])
        if not day_df.empty:
            processed_days.append(day_df.reset_index().rename(columns={"index": "timestamp"}))

    if not processed_days:
        return pd.DataFrame()

    final_df = pd.concat(processed_days, ignore_index=True)
    return final_df.drop(columns=["date", "prev_close", "jump"], errors="ignore")


def load_universe(csv_path: Path = config.UNIVERSE_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["is_index"] = df["is_index"].astype(str).str.lower().isin(["true", "1", "yes"])
    logger.info(f"Universe loaded: {len(df)} instruments")
    return df


def process_instrument_year(
    symbol: str, instrument_key: str, sector: str, year: int,
    fetcher: UpstoxHistoricalFetcher,
) -> str:
    from src.core.renko import RenkoBrickBuilder
    builder = RenkoBrickBuilder()
    """Download -> Renko -> Parquet for ONE stock x ONE year."""
    out_dir = config.DATA_DIR / sector / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}.parquet"

    # Support for forcing a refresh to apply new sanitization logic
    force_refresh = getattr(config, "FORCE_REFRESH", False)

    if out_path.exists() and year < date.today().year and not force_refresh:
        return f"SKIP  {symbol}/{year} (exists)"

    # Current year or forced refresh: re-download to capture latest/sanitized data
    if out_path.exists() and (year == date.today().year or force_refresh):
        logger.info(f"REFRESH {symbol}/{year} (applying sanitization)")

    ohlc = fetcher.fetch_year(instrument_key, year)
    if ohlc.empty:
        return f"EMPTY {symbol}/{year} (no data)"

    # --- Sanitization Middleware ---
    ohlc = sanitize_ohlc(ohlc, symbol)
    if ohlc.empty:
        return f"DROP  {symbol}/{year} (all days failed sanitization)"

    # --- Tick-Size Safety Check ---
    # Brick size is usually NATR_BRICK_PERCENT * price.
    # If this is < 0.05, features like 'streak_exhaustion' will break.
    avg_price = ohlc["close"].mean()
    brick_size_est = avg_price * config.NATR_BRICK_PERCENT
    if brick_size_est < 0.05:
        logger.warning(f"  [{symbol}] RISK: Estimated brick size ({brick_size_est:.4f}) is smaller than NSE tick size (0.05). "
                       f"Features for this low-priced stock may be noisy or zero.")

    bricks = builder.transform(ohlc)
    if bricks.empty:
        return f"EMPTY {symbol}/{year} (no bricks)"

    bricks.to_parquet(out_path, engine="pyarrow", index=False)

    # -- Backup (append-only archive) -----------------------------------------
    bkp_dir = config.BACKUP_DIR / sector / symbol
    bkp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, bkp_dir / f"{year}.parquet")

    return f"OK    {symbol}/{year} -> {len(bricks)} bricks"


def run_batch_factory():
    """Main orchestrator — downloads entire universe across all years."""
    logger.info("=" * 70)
    logger.info("BATCH FACTORY — Starting full download pipeline")
    logger.info("=" * 70)

    universe = load_universe()
    fetcher = UpstoxHistoricalFetcher()
    years = list(range(config.DOWNLOAD_START_YEAR, config.DOWNLOAD_END_YEAR + 1))

    work_items = [
        (row["symbol"], row["instrument_token"], row["sector"], yr)
        for _, row in universe.iterrows()
        for yr in years
    ]
    logger.info(f"Total work items: {len(work_items)}")

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=config.API_MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_instrument_year, s, k, sec, y, fetcher): (s, y)
            for s, k, sec, y in work_items
        }
        for future in as_completed(futures):
            sym, yr = futures[future]
            try:
                result = future.result()
                logger.info(result)
                ok += 1 if result.startswith("OK") else 0
                skip += 1 if not result.startswith("OK") else 0
            except Exception as exc:
                logger.error(f"FAIL  {sym}/{yr}: {exc}")
                fail += 1

    logger.info(f"DONE — OK: {ok}  Skip: {skip}  Fail: {fail}")


if __name__ == "__main__":
    run_batch_factory()
