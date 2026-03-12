"""
src/data/data_verifier.py — Data Verification & Cleanup Pipeline
=================================================================
Performs 9 categories of checks across both the raw brick Parquets
(storage/data/) and the enriched feature Parquets (storage/features/).

Checks performed:
  1.  Schema completeness  — required columns present
  2.  Dtype sanity         — timestamps are datetime, prices are numeric
  3.  Price spike detector — flags moves > spike_threshold% in a single brick
  4.  Split/demerger anomaly — detects unrealistic price jumps (like HINDUNILVR -88%)
  5.  Duplicate removal    — drops duplicate brick_timestamp rows per symbol
  6.  Chronological order  — ensures bricks are time-sorted
  7.  NaN audit            — reports and optionally forward-fills feature NaNs
  8.  Feature completeness — checks all 22 FEATURE_COLS are present and non-zero
  9.  Brick size sanity    — flags bricks whose size deviates from the configured NATR%

Outputs:
  • storage/logs/data_verification_report.txt — full human-readable report
  • Optionally overwrites cleaned Parquets in-place (when --fix is passed)

Run:
    python -m src.data.data_verifier            # report only
    python -m src.data.data_verifier --fix      # report + clean in-place
    python -m src.data.data_verifier --fix --features-only  # only clean feature files
    python -m src.data.data_verifier --fix --raw-only       # only clean raw files
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

# ── Bootstrap path so we can import config ────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
import config

# ── Logging ───────────────────────────────────────────────────────────────────
_REPORT_PATH = config.LOGS_DIR / "data_verification_report.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "data_verifier.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_RAW_COLS = [
    "brick_timestamp", "brick_open", "brick_high", "brick_low", "brick_close",
    "direction", "volume",
]
REQUIRED_FEATURE_COLS = config.FEATURE_COLS   # 22 features

# An individual brick moving more than 5% in price is a spike (absolute %)
SPIKE_THRESHOLD_PCT = 5.0

# A brick moving more than 20% from close to next open = split/demerger anomaly
SPLIT_ANOMALY_PCT = 20.0


@dataclass
class FileReport:
    """Per-file summary accumulator."""
    path: Path
    symbol: str
    sector: str
    kind: str                          # 'raw' or 'feature'
    rows_total: int = 0
    duplicates_removed: int = 0
    price_spikes: List[str] = field(default_factory=list)
    split_anomalies: List[str] = field(default_factory=list)
    nan_counts: dict = field(default_factory=dict)
    missing_columns: List[str] = field(default_factory=list)
    out_of_order_fixed: bool = False
    bricks_removed: int = 0            # rows dropped (duplicates + anomalies)
    status: str = "OK"

    @property
    def has_issues(self) -> bool:
        return bool(
            self.duplicates_removed or self.price_spikes or self.split_anomalies
            or self.nan_counts or self.missing_columns or self.out_of_order_fixed
        )


class DataVerifier:
    """
    Orchestrates all data quality checks.
    
    Parameters
    ----------
    fix : bool
        If True, clean and overwrite the Parquet files in-place.
    check_raw : bool
        Whether to verify files in storage/data/.
    check_features : bool
        Whether to verify files in storage/features/.
    """

    def __init__(self, fix: bool = False, check_raw: bool = True, check_features: bool = True):
        self.fix = fix
        self.check_raw = check_raw
        self.check_features = check_features
        self.reports: List[FileReport] = []
        self.global_stats = {
            "files_checked": 0,
            "files_with_issues": 0,
            "total_duplicates_removed": 0,
            "total_split_anomalies": 0,
            "total_price_spikes": 0,
            "total_nan_cells_fixed": 0,
            "total_rows_removed": 0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 1: Schema completeness
    # ─────────────────────────────────────────────────────────────────────────
    def _check_schema(self, df: pd.DataFrame, required: List[str], report: FileReport) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            report.missing_columns = missing
            report.status = "WARN"
            logger.warning(f"  [{report.symbol}] Missing columns: {missing}")

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 2: Dtype sanity
    # ─────────────────────────────────────────────────────────────────────────
    def _fix_dtypes(self, df: pd.DataFrame, report: FileReport) -> pd.DataFrame:
        # Ensure brick_timestamp is datetime
        if "brick_timestamp" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["brick_timestamp"]):
                df["brick_timestamp"] = pd.to_datetime(df["brick_timestamp"], utc=True, errors="coerce")

        # Ensure price columns are float32
        for col in ["brick_open", "brick_high", "brick_low", "brick_close"]:
            if col in df.columns and not pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 3 & 4: Price spike + split/demerger anomaly
    # ─────────────────────────────────────────────────────────────────────────
    def _check_price_anomalies(self, df: pd.DataFrame, report: FileReport) -> pd.DataFrame:
        if "brick_close" not in df.columns:
            return df

        close = df["brick_close"].astype(float)
        returns_pct = close.pct_change().abs() * 100

        # CHECK 3: Within-brick spike (open to close range)
        if "brick_open" in df.columns:
            intra_range = ((df["brick_high"].astype(float) - df["brick_low"].astype(float))
                           / df["brick_open"].astype(float).clip(lower=1e-9) * 100)
            spikes = df[intra_range > SPIKE_THRESHOLD_PCT]
            if len(spikes) > 0:
                for idx, row in spikes.iterrows():
                    ts = str(row.get("brick_timestamp", idx))
                    report.price_spikes.append(f"Spike {intra_range[idx]:.1f}% at {ts}")
                if len(report.price_spikes) > 0:
                    logger.warning(f"  [{report.symbol}] {len(report.price_spikes)} price spikes detected")

        # CHECK 4: Inter-brick split/demerger anomaly
        split_mask = returns_pct > SPLIT_ANOMALY_PCT
        split_rows = df[split_mask]
        if len(split_rows) > 0:
            for idx, row in split_rows.iterrows():
                ts = str(row.get("brick_timestamp", idx))
                pct = returns_pct[idx]
                report.split_anomalies.append(f"{pct:.1f}% jump at {ts}")
                report.status = "CLEANED" if self.fix else "ERROR"
            logger.warning(
                f"  [{report.symbol}] {len(split_rows)} split/demerger anomalies "
                f"(price gap > {SPLIT_ANOMALY_PCT}%) found!"
            )
            if self.fix:
                # HOTFIX: replace anomalous close values with forward-fill from previous brick
                df.loc[split_mask, "brick_close"] = np.nan
                df["brick_close"] = df["brick_close"].ffill()
                report.bricks_removed += len(split_rows)
                self.global_stats["total_split_anomalies"] += len(split_rows)
                logger.info(f"  [{report.symbol}] Split anomalies patched via ffill.")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 5: Duplicate rows
    # ─────────────────────────────────────────────────────────────────────────
    def _remove_duplicates(self, df: pd.DataFrame, report: FileReport) -> pd.DataFrame:
        """
        IMPORTANT — Renko bricks can legally share the same brick_timestamp.
        If price moves 3 brick-widths inside one minute, you get 3 bricks all
        stamped with that minute. They are DIFFERENT rows (different open/close/
        direction/volume) and must NOT be dropped.

        We only remove rows where EVERY column is identical — true copy-paste
        duplicates caused by appending the same file twice, etc.
        """
        original_len = len(df)
        df = df.drop_duplicates(keep="first")   # all-column equality check
        removed = original_len - len(df)
        if removed > 0:
            report.duplicates_removed = removed
            report.bricks_removed += removed
            self.global_stats["total_duplicates_removed"] += removed
            logger.info(f"  [{report.symbol}] Removed {removed} fully-identical duplicate rows.")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 6: Chronological order
    # ─────────────────────────────────────────────────────────────────────────
    def _ensure_sorted(self, df: pd.DataFrame, report: FileReport) -> pd.DataFrame:
        if "brick_timestamp" not in df.columns:
            return df
        if not df["brick_timestamp"].is_monotonic_increasing:
            df = df.sort_values("brick_timestamp", kind="mergesort").reset_index(drop=True)
            report.out_of_order_fixed = True
            logger.info(f"  [{report.symbol}] Re-sorted brick_timestamp (was out-of-order).")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 7: NaN audit (feature files)
    # ─────────────────────────────────────────────────────────────────────────
    def _check_nans(self, df: pd.DataFrame, report: FileReport, feature_cols: List[str]) -> pd.DataFrame:
        nan_counts = {}
        for col in feature_cols:
            if col in df.columns:
                n_nan = int(df[col].isna().sum())
                if n_nan > 0:
                    nan_counts[col] = n_nan

        if nan_counts:
            report.nan_counts = nan_counts
            total_nan = sum(nan_counts.values())
            logger.info(
                f"  [{report.symbol}] NaN cells: {total_nan} "
                f"across {len(nan_counts)} columns"
            )
            if self.fix:
                # Forward-fill then backward-fill, then fill remaining with 0
                df[feature_cols] = (
                    df[feature_cols]
                    .ffill()
                    .bfill()
                    .fillna(0)
                )
                self.global_stats["total_nan_cells_fixed"] += total_nan
                logger.info(f"  [{report.symbol}] NaNs patched (ffill → bfill → 0).")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 8: Feature completeness (all 22 columns present + non-zero)
    # ─────────────────────────────────────────────────────────────────────────
    def _check_feature_completeness(self, df: pd.DataFrame, report: FileReport) -> None:
        for col in REQUIRED_FEATURE_COLS:
            if col not in df.columns:
                if col not in report.missing_columns:
                    report.missing_columns.append(col)
            else:
                # Warn if the column is all-zero (it may never have been computed)
                non_zero_pct = (df[col].fillna(0) != 0).mean() * 100
                if non_zero_pct < 0.1:
                    logger.warning(
                        f"  [{report.symbol}] Column '{col}' is {non_zero_pct:.2f}% non-zero "
                        f"(looks like it was never computed!)"
                    )
                    if col not in report.missing_columns:
                        report.missing_columns.append(f"{col}[zero]")
                    report.status = "WARN"

    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 9: Brick size sanity
    # ─────────────────────────────────────────────────────────────────────────
    def _check_brick_size(self, df: pd.DataFrame, report: FileReport) -> None:
        if "brick_size" not in df.columns or "brick_close" not in df.columns:
            return
        expected_pct = config.NATR_BRICK_PERCENT
        # brick_size is absolute price; we compare as % of close
        actual_pct = df["brick_size"] / df["brick_close"].clip(lower=1e-9)
        deviation = (actual_pct - expected_pct).abs()
        # Flag rows where brick size is 10x bigger than expected (likely data error)
        extreme = (deviation > expected_pct * 10).sum()
        if extreme > 5:
            logger.warning(
                f"  [{report.symbol}] {extreme} bricks with extreme size deviation "
                f"(possible corporate action or data error)"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Main Per-File Processor
    # ─────────────────────────────────────────────────────────────────────────
    def _process_file(self, path: Path, symbol: str, sector: str, kind: str) -> FileReport:
        report = FileReport(path=path, symbol=symbol, sector=sector, kind=kind)
        try:
            df = pd.read_parquet(path)
            report.rows_total = len(df)

            required_cols = REQUIRED_RAW_COLS if kind == "raw" else REQUIRED_RAW_COLS
            self._check_schema(df, required_cols, report)
            df = self._fix_dtypes(df, report)
            df = self._remove_duplicates(df, report)
            df = self._ensure_sorted(df, report)
            df = self._check_price_anomalies(df, report)

            if kind == "feature":
                df = self._check_nans(df, report, REQUIRED_FEATURE_COLS)
                self._check_feature_completeness(df, report)

            self._check_brick_size(df, report)

            if self.fix and (report.duplicates_removed or report.out_of_order_fixed
                             or report.split_anomalies or report.nan_counts):
                df.to_parquet(path, engine="pyarrow", index=False)
                if report.status not in ("ERROR", "WARN"):
                    report.status = "CLEANED"

        except Exception as e:
            report.status = "FAIL"
            logger.error(f"  [{symbol}] FAILED: {e}")

        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Run raw verification
    # ─────────────────────────────────────────────────────────────────────────
    def _run_raw(self):
        logger.info("=" * 70)
        logger.info("VERIFYING RAW BRICK DATA  (storage/data/)")
        logger.info("=" * 70)
        if not config.DATA_DIR.exists():
            logger.warning("Raw data directory not found — skipping.")
            return
        for sector_dir in sorted(config.DATA_DIR.iterdir()):
            if not sector_dir.is_dir():
                continue
            for symbol_dir in sorted(sector_dir.iterdir()):
                if not symbol_dir.is_dir():
                    continue
                symbol = symbol_dir.name
                for pf in sorted(symbol_dir.glob("*.parquet")):
                    report = self._process_file(pf, symbol, sector_dir.name, "raw")
                    self.reports.append(report)
                    self.global_stats["files_checked"] += 1
                    if report.has_issues:
                        self.global_stats["files_with_issues"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Run feature verification
    # ─────────────────────────────────────────────────────────────────────────
    def _run_features(self):
        logger.info("=" * 70)
        logger.info("VERIFYING ENRICHED FEATURE DATA  (storage/features/)")
        logger.info("=" * 70)
        if not config.FEATURES_DIR.exists():
            logger.warning("Features directory not found — skipping.")
            return
        for sector_dir in sorted(config.FEATURES_DIR.iterdir()):
            if not sector_dir.is_dir():
                continue
            for pf in sorted(sector_dir.glob("*.parquet")):
                symbol = pf.stem
                report = self._process_file(pf, symbol, sector_dir.name, "feature")
                self.reports.append(report)
                self.global_stats["files_checked"] += 1
                if report.has_issues:
                    self.global_stats["files_with_issues"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Build and write report
    # ─────────────────────────────────────────────────────────────────────────
    def _write_report(self):
        lines = []
        lines.append("=" * 72)
        lines.append("DATA VERIFICATION REPORT")
        lines.append("=" * 72)
        lines.append(f"Mode      : {'AUTO-FIX enabled' if self.fix else 'Report-only (run --fix to clean)'}")
        lines.append(f"Files     : {self.global_stats['files_checked']}")
        lines.append(f"Issues    : {self.global_stats['files_with_issues']}")
        lines.append(f"Duplicates: {self.global_stats['total_duplicates_removed']}")
        lines.append(f"Split Anom: {self.global_stats['total_split_anomalies']}")
        lines.append(f"NaN Fixed : {self.global_stats['total_nan_cells_fixed']}")
        lines.append("")

        # Group by status
        for status_filter in ("ERROR", "WARN", "CLEANED", "FAIL"):
            group = [r for r in self.reports if r.status == status_filter]
            if not group:
                continue
            lines.append(f"── {status_filter} ({len(group)} files) " + "─" * 40)
            for r in group:
                lines.append(f"  {r.sector}/{r.symbol}  [{r.rows_total:,} rows]")
                if r.missing_columns:
                    lines.append(f"    Missing/Zero columns : {r.missing_columns}")
                if r.duplicates_removed:
                    lines.append(f"    Duplicates removed   : {r.duplicates_removed}")
                if r.split_anomalies:
                    lines.append(f"    Split anomalies      : {len(r.split_anomalies)}")
                    for a in r.split_anomalies[:5]:
                        lines.append(f"      * {a}")
                    if len(r.split_anomalies) > 5:
                        lines.append(f"      ... and {len(r.split_anomalies) - 5} more")
                if r.price_spikes:
                    lines.append(f"    Price spikes (>{SPIKE_THRESHOLD_PCT}%) : {len(r.price_spikes)}")
                if r.nan_counts:
                    top = sorted(r.nan_counts.items(), key=lambda x: -x[1])[:5]
                    lines.append(f"    Top NaN cols         : {dict(top)}")
                if r.out_of_order_fixed:
                    lines.append(f"    Out-of-order         : FIXED")
            lines.append("")

        lines.append("=" * 72)
        lines.append("RECOMMENDATION SUMMARY")
        lines.append("=" * 72)
        split_stocks = [r.symbol for r in self.reports if r.split_anomalies]
        if split_stocks:
            lines.append(
                f"  WARNING: {len(split_stocks)} stocks had split/demerger anomalies: {split_stocks}"
            )
            lines.append(
                "  ACTION : Consider excluding these from training or manually verifying "
                "their corporate action adjustments."
            )
        zero_feature_stocks = [
            r.symbol for r in self.reports
            if any("[zero]" in c for c in r.missing_columns)
        ]
        if zero_feature_stocks:
            lines.append(
                f"  WARNING: {len(zero_feature_stocks)} stocks have all-zero feature columns: {zero_feature_stocks}"
            )
            lines.append(
                "  ACTION : If these are penny stocks (e.g., IDEA), the brick size may be smaller than "
                "the ₹0.05 NSE tick size, breaking momentum features. Otherwise, re-run with incremental=False."
            )
        if not split_stocks and not zero_feature_stocks:
            lines.append("  ✓ No critical issues detected. Data appears clean for training.")
        lines.append("=" * 72)

        report_txt = "\n".join(lines)
        _REPORT_PATH.write_text(report_txt, encoding="utf-8")
        logger.info(f"\n{report_txt}")
        logger.info(f"\nFull report written -> {_REPORT_PATH}")

    # ─────────────────────────────────────────────────────────────────────────
    # Entrypoint
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        logger.info("=" * 70)
        logger.info("DATA VERIFIER  —  Starting")
        if self.fix:
            logger.info("Mode: AUTO-FIX (files will be cleaned in-place)")
        else:
            logger.info("Mode: REPORT-ONLY (pass --fix to auto-clean)")
        logger.info("=" * 70)

        if self.check_raw:
            self._run_raw()
        if self.check_features:
            self._run_features()

        self._write_report()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Data Verification & Cleanup — Institutional Fortress Trading System"
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Auto-clean issues in-place (duplicate removal, split patches, NaN fills)",
    )
    parser.add_argument(
        "--features-only", action="store_true",
        help="Only verify feature Parquets (storage/features/)",
    )
    parser.add_argument(
        "--raw-only", action="store_true",
        help="Only verify raw brick Parquets (storage/data/)",
    )
    args = parser.parse_args()

    check_raw = not args.features_only
    check_features = not args.raw_only

    verifier = DataVerifier(
        fix=args.fix,
        check_raw=check_raw,
        check_features=check_features,
    )
    verifier.run()


if __name__ == "__main__":
    main()
