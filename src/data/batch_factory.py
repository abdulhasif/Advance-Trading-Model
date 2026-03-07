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
from datetime import date

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

    if out_path.exists() and year < date.today().year:
        return f"SKIP  {symbol}/{year} (exists)"

    # Current year: re-download to capture latest data
    if out_path.exists() and year == date.today().year:
        logger.info(f"UPDATE {symbol}/{year} (re-downloading current year)")

    ohlc = fetcher.fetch_year(instrument_key, year)
    if ohlc.empty:
        return f"EMPTY {symbol}/{year} (no data)"

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
