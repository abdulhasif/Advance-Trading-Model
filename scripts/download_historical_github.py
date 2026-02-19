"""
scripts/download_historical_github.py
======================================
Downloads 1-minute OHLCV data from ShabbirHasan1/NSE-Data GitHub repo,
filters to the requested year range (default 2019-2020), runs through the
Renko pipeline, and saves Parquet files into data/Sector/Symbol/YEAR.parquet.

This script does NOT touch trained models or trading logic.
It ONLY adds Renko-processed data for additional years.

Usage:
    python -m scripts.download_historical_github                  # 2019-2020
    python -m scripts.download_historical_github --start 2017 --end 2020
"""

import sys, os, time, logging, argparse, io, shutil
import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.core.renko import RenkoBrickBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "github_download.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── GitHub source ───────────────────────────────────────────────────────────
GITHUB_BASE = (
    "https://raw.githubusercontent.com/ShabbirHasan1/NSE-Data/main/"
    "NSE%20Minute%20Data/NSE_Stocks_Data"
)

# Map our symbol names to GitHub file names
# Most are SYMBOL__EQ__NSE__NSE__MINUTE.csv
# Special cases: M&M -> M_M, M&MFIN -> M_MFIN, BAJAJ-AUTO -> BAJAJ_AUTO, etc.
SYMBOL_TO_GITHUB = {
    "M&M":       "M_M",
    "M&MFIN":    "M_MFIN",
    "BAJAJ-AUTO": "BAJAJ_AUTO",
}


def github_filename(symbol: str) -> str:
    """Convert our symbol to the GitHub CSV filename."""
    gh_sym = SYMBOL_TO_GITHUB.get(symbol, symbol)
    return f"{gh_sym}__EQ__NSE__NSE__MINUTE.csv"


def download_and_process(
    symbol: str, sector: str, start_year: int, end_year: int,
    builder: RenkoBrickBuilder,
) -> str:
    """Download CSV from GitHub, filter years, Renko transform, save Parquet."""
    fname = github_filename(symbol)
    url = f"{GITHUB_BASE}/{fname}"

    # Check if ALL years already exist
    all_exist = True
    for yr in range(start_year, end_year + 1):
        out_path = config.DATA_DIR / sector / symbol / f"{yr}.parquet"
        if not out_path.exists():
            all_exist = False
            break
    if all_exist:
        return f"SKIP  {symbol} (all years exist)"

    # Download full CSV
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            return f"MISS  {symbol} (not in GitHub repo)"
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"FAIL  {symbol}: {e}"

    # Parse CSV
    try:
        df = pd.read_csv(io.StringIO(resp.text))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
        df = df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        return f"FAIL  {symbol} (parse): {e}"

    results = []
    for yr in range(start_year, end_year + 1):
        out_dir = config.DATA_DIR / sector / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{yr}.parquet"

        if out_path.exists():
            results.append(f"SKIP  {symbol}/{yr}")
            continue

        # Filter to this year
        mask = df["timestamp"].dt.year == yr
        year_df = df[mask].copy()

        if year_df.empty:
            results.append(f"EMPTY {symbol}/{yr}")
            continue

        # Renko transform
        bricks = builder.transform(year_df)
        if bricks.empty:
            results.append(f"EMPTY {symbol}/{yr} (no bricks)")
            continue

        bricks.to_parquet(out_path, engine="pyarrow", index=False)

        # Backup
        bkp_dir = config.BACKUP_DIR / sector / symbol
        bkp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, bkp_dir / f"{yr}.parquet")

        results.append(f"OK    {symbol}/{yr} -> {len(bricks)} bricks")

    return " | ".join(results) if results else f"EMPTY {symbol}"


def main():
    parser = argparse.ArgumentParser(description="Download historical 1-min data from GitHub")
    parser.add_argument("--start", type=int, default=2019, help="Start year (default: 2019)")
    parser.add_argument("--end", type=int, default=2020, help="End year inclusive (default: 2020)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads (default: 4)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"GITHUB HISTORICAL DOWNLOAD — {args.start} to {args.end}")
    logger.info("=" * 70)

    # Load universe (stocks only, skip indices)
    universe = pd.read_csv(config.UNIVERSE_CSV)
    universe["is_index"] = universe["is_index"].astype(str).str.lower().isin(["true", "1", "yes"])
    stocks = universe[~universe["is_index"]].drop_duplicates(subset=["symbol"])
    logger.info(f"Universe: {len(stocks)} stocks (indices skipped — no GitHub data)")

    builder = RenkoBrickBuilder()
    ok = skip = miss = fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_and_process,
                row["symbol"], row["sector"], args.start, args.end, builder,
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
