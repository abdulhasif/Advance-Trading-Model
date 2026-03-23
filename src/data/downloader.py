"""
src/data/downloader.py - Upstox V3 Historical Data Fetcher
============================================================
Downloads 1-minute OHLCV candles, chunked by month.
"""

import time
import logging
import requests
import pandas as pd
from datetime import date, timedelta
from urllib.parse import quote
from pathlib import Path

import config

logger = logging.getLogger(__name__)


class UpstoxHistoricalFetcher:
    """Downloads 1-min OHLCV from the Upstox V3 Historical REST API."""

    def __init__(self, access_token: str = config.UPSTOX_ACCESS_TOKEN):
        self.base_url = config.UPSTOX_API_BASE
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

    def fetch_candles(
        self, instrument_key: str, from_date: str, to_date: str,
        unit: str = "minutes", interval: int = 1,
    ) -> pd.DataFrame:
        """
        Upstox V3 Historical Candle endpoint:
          GET /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}

        unit     = minutes | hours | days | weeks | months
        interval = 1, 2, 3 ... (e.g. 1 for 1-minute candles)
        """
        encoded_key = quote(instrument_key, safe="")
        url = (
            f"{self.base_url}/historical-candle/{encoded_key}"
            f"/{unit}/{interval}/{to_date}/{from_date}"
        )
        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            candles = resp.json().get("data", {}).get("candles", [])
            if not candles:
                return pd.DataFrame()

            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume", "oi"],
            )
            # 1. Force Naive IST via String Truncation
            # Upstox returns e.g. "2024-01-01T09:15:00+05:30" or "Z".
            # Slicing the first 19 chars guarantees we parse the exact nominal digits "09:15:00" 
            # as a pure naive timezone format matching the live execution engine without double-shifting.
            df["timestamp"] = pd.to_datetime(df["timestamp"].str[:19])
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df[["timestamp", "open", "high", "low", "close", "volume"]]

        except requests.exceptions.RequestException as e:
            logger.warning(f"API error {instrument_key} [{from_date}->{to_date}]: {e}")
            return pd.DataFrame()

    def fetch_year(self, instrument_key: str, year: int) -> pd.DataFrame:
        """Download a full year of 1-min data, chunked by month."""
        frames = []
        for month in range(1, 13):
            from_dt = date(year, month, 1)
            to_dt = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)

            if from_dt > date.today():
                break
            if to_dt > date.today():
                to_dt = date.today()

            chunk = self.fetch_candles(instrument_key, from_dt.isoformat(), to_dt.isoformat())
            if not chunk.empty:
                frames.append(chunk)
            time.sleep(config.API_DELAY_BETWEEN_CALLS)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        # 2. Gap Detection (Ignoring overnight gaps > 6h)
        diffs = df["timestamp"].diff()
        gaps = diffs[(diffs > pd.Timedelta(minutes=5)) & (diffs < pd.Timedelta(hours=6))]
        if not gaps.empty:
            logger.warning(f"PROVIDER GAP: {instrument_key} has {len(gaps)} discrete gap(s) > 5 mins. First instance at {df.loc[gaps.index[0], 'timestamp']}")

        # 3. Type Optimization (Reduce size by 50%)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype("float32")
        df["volume"] = df["volume"].astype("int32")

        return df

    def save_parquet(self, df: pd.DataFrame, out_path: Path):
        """Saves with Metadata tagging for synchronization safety."""
        from datetime import datetime
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # 4. Metadata Tagging
        custom_meta = {
            "last_synced_utc": datetime.utcnow().isoformat(),
            "source": "upstox_v3_historical"
        }
        
        table = pa.Table.from_pandas(df)
        existing_meta = table.schema.metadata or {}
        # Merge custom metadata (must be bytes)
        merged_meta = {**existing_meta, **{k.encode(): v.encode() for k, v in custom_meta.items()}}
        table = table.replace_schema_metadata(merged_meta)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)
        logger.info(f"Saved {len(df)} candles to {out_path}")
