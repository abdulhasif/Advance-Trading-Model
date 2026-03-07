"""
src/data/backup_pipeline.py -- Full NSE Archive Pipeline (Separate from Trading)
==================================================================================
Downloads 1-min OHLC for ALL ~2000 NSE equities from Upstox, converts to Renko
bricks, and stores in storage/backup/. This is an append-only archive for
future model upgrades.

Completely independent from the trading pipeline (storage/data/).

Run:  python main.py backup
"""

import sys
import gzip
import json
import shutil
import logging
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from src.data.downloader import UpstoxHistoricalFetcher
from src.core.renko import RenkoBrickBuilder

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "backup_pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

NSE_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
BACKUP_UNIVERSE_PATH = config.BACKUP_DIR / "_nse_universe.csv"


# -- Fetch All NSE Equities ---------------------------------------------------

def fetch_nse_universe(max_stocks: int = 2000) -> pd.DataFrame:
    """
    Download the full NSE instrument list from Upstox and filter to
    top equity stocks (NSE_EQ segment, not derivatives/indices).
    Saves a local CSV cache for re-runs.
    """
    logger.info("Fetching NSE instrument master from Upstox...")
    try:
        resp = requests.get(NSE_INSTRUMENTS_URL, timeout=60)
        resp.raise_for_status()
        raw = gzip.decompress(resp.content)
        instruments = json.loads(raw)
    except Exception as e:
        logger.error(f"Failed to fetch instrument list: {e}")
        # Fall back to cached version if available
        if BACKUP_UNIVERSE_PATH.exists():
            logger.info("Using cached universe file")
            return pd.read_csv(BACKUP_UNIVERSE_PATH)
        sys.exit(1)

    # Filter: only NSE equities (not F&O, not indices, not ETFs/debt)
    equities = []
    for inst in instruments:
        seg = inst.get("segment", "")
        itype = inst.get("instrument_type", "")
        exchange = inst.get("exchange", "")
        name = inst.get("name", "")
        ikey = inst.get("instrument_key", "")
        symbol = inst.get("trading_symbol", "")

        # Keep only NSE equity stocks
        if exchange == "NSE" and seg == "NSE_EQ" and itype == "EQUITY":
            equities.append({
                "symbol": symbol,
                "instrument_key": ikey,
                "name": name,
            })

    df = pd.DataFrame(equities)
    if df.empty:
        logger.error("No NSE equities found in instrument master")
        sys.exit(1)

    # Remove duplicates by symbol, keep first
    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    # Take top N (sorted alphabetically for consistency)
    df = df.sort_values("symbol").head(max_stocks).reset_index(drop=True)

    # Cache locally
    df.to_csv(BACKUP_UNIVERSE_PATH, index=False)
    logger.info(f"NSE universe: {len(df)} equities fetched and cached")
    return df


# -- Process One Stock-Year ---------------------------------------------------

def backup_instrument_year(
    symbol: str, instrument_key: str, year: int,
    fetcher: UpstoxHistoricalFetcher,
) -> str:
    from src.core.renko import RenkoBrickBuilder
    builder = RenkoBrickBuilder()
    """Download -> Renko -> Parquet for ONE stock x ONE year into backup/."""
    out_dir = config.BACKUP_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}.parquet"

    # Skip if already downloaded (append-only)
    if out_path.exists():
        return f"SKIP  {symbol}/{year} (exists)"

    ohlc = fetcher.fetch_year(instrument_key, year)
    if ohlc.empty:
        return f"EMPTY {symbol}/{year} (no data)"

    bricks = builder.transform(ohlc)
    if bricks.empty:
        return f"EMPTY {symbol}/{year} (no bricks)"

    bricks.to_parquet(out_path, engine="pyarrow", index=False)
    return f"OK    {symbol}/{year} -> {len(bricks)} bricks"


# -- Main Orchestrator --------------------------------------------------------

def run_backup_pipeline():
    logger.info("=" * 70)
    logger.info("BACKUP ARCHIVE PIPELINE -- Full NSE Download")
    logger.info("=" * 70)

    universe = fetch_nse_universe(max_stocks=2000)
    fetcher = UpstoxHistoricalFetcher()
    years = list(range(config.DOWNLOAD_START_YEAR, config.DOWNLOAD_END_YEAR + 1))

    work_items = [
        (row["symbol"], row["instrument_key"], yr)
        for _, row in universe.iterrows()
        for yr in years
    ]
    logger.info(f"Total work items: {len(work_items)} ({len(universe)} stocks x {len(years)} years)")

    ok = skip = empty = fail = 0
    with ThreadPoolExecutor(max_workers=config.API_MAX_WORKERS) as pool:
        futures = {
            pool.submit(backup_instrument_year, s, k, y, fetcher): (s, y)
            for s, k, y in work_items
        }
        for future in as_completed(futures):
            sym, yr = futures[future]
            try:
                result = future.result()
                logger.info(result)
                if result.startswith("OK"):
                    ok += 1
                elif result.startswith("SKIP"):
                    skip += 1
                else:
                    empty += 1
            except Exception as exc:
                logger.error(f"FAIL  {sym}/{yr}: {exc}")
                fail += 1

            # Progress every 100 items
            total_done = ok + skip + empty + fail
            if total_done % 100 == 0:
                logger.info(f"  Progress: {total_done}/{len(work_items)} "
                            f"(OK:{ok} Skip:{skip} Empty:{empty} Fail:{fail})")

    logger.info("=" * 70)
    logger.info(f"BACKUP COMPLETE -- OK:{ok}  Skip:{skip}  Empty:{empty}  Fail:{fail}")
    logger.info(f"Archive location: {config.BACKUP_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_backup_pipeline()
