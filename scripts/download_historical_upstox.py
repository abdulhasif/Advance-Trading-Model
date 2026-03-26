"""
scripts/download_historical_upstox.py
======================================
Downloads 1-minute OHLCV data from Upstox Historical API,
filters to the requested year range (e.g. 2022-2026), runs through the
fixed Renko pipeline, and saves Parquet files.
"""

import sys, os, time, logging, argparse, shutil
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from trading_api.src.core.renko import RenkoBrickBuilder
from trading_api.src.data.downloader import UpstoxHistoricalFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def download_and_process(
    symbol: str, instrument_token: str, sector: str, start_year: int, end_year: int,
    builder: RenkoBrickBuilder, fetcher: UpstoxHistoricalFetcher
) -> str:
    """Download from Upstox API, Renko transform, save Parquet."""
    if not instrument_token or pd.isna(instrument_token):
        return f"SKIP  {symbol} (No instrument token)"

    out_dir = config.DATA_DIR / sector / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    all_exist = True
    for yr in range(start_year, end_year + 1):
        out_path = out_dir / f"{yr}.parquet"
        if not out_path.exists():
            all_exist = False
            break
    if all_exist:
        return f"SKIP  {symbol} (all years exist)"

    results = []
    for yr in range(start_year, end_year + 1):
        out_path = out_dir / f"{yr}.parquet"

        if out_path.exists():
            results.append(f"SKIP  {symbol}/{yr}")
            continue
            
        logger.info(f"Downloading {symbol} for {yr} from Upstox...")
        year_df = fetcher.fetch_year(instrument_token, yr)

        if year_df.empty:
            results.append(f"EMPTY {symbol}/{yr}")
            continue

        # Renko transform with our NEW LEAK-FREE builder
        bricks = builder.transform(year_df)
        if bricks.empty:
            results.append(f"EMPTY {symbol}/{yr} (no bricks)")
            continue

        bricks.to_parquet(out_path, engine="pyarrow", index=False)
        results.append(f"OK    {symbol}/{yr} -> {len(bricks)} bricks")

    return " | ".join(results) if results else f"EMPTY {symbol}"


def main():
    parser = argparse.ArgumentParser(description="Download historical 1-min data from Upstox")
    parser.add_argument("--start", type=int, default=2022, help="Start year")
    parser.add_argument("--end", type=int, default=2026, help="End year inclusive")
    parser.add_argument("--workers", type=int, default=2, help="Parallel downloads (default: 2)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"UPSTOX HISTORICAL DOWNLOAD — {args.start} to {args.end}")
    logger.info("=" * 70)

    # Load universe
    universe = pd.read_csv(config.UNIVERSE_CSV)
    universe["is_index"] = universe["is_index"].astype(str).str.lower().isin(["true", "1", "yes"])
    stocks = universe[~universe["is_index"]].drop_duplicates(subset=["symbol"])
    logger.info(f"Universe: {len(stocks)} stocks")

    builder = RenkoBrickBuilder()
    
    # Needs valid UPSTOX_ACCESS_TOKEN in env or .env
    fetcher = UpstoxHistoricalFetcher()

    ok = skip = miss = fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_and_process,
                row["symbol"], row.get("instrument_token"), row["sector"], args.start, args.end, builder, fetcher
            ): row["symbol"]
            for _, row in stocks.iterrows()
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                logger.info(result)
                if "OK" in result:
                    ok += 1
                elif "SKIP" in result:
                    skip += 1
                elif "MISS" in result:
                    miss += 1
                else:
                    fail += 1
            except Exception as exc:
                logger.error(f"FAIL  {sym}: {exc}")
                fail += 1

    logger.info("=" * 70)
    logger.info(f"DONE — OK: {ok}  Skip: {skip}  Miss: {miss}  Fail: {fail}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

