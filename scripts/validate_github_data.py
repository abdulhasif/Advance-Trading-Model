"""
Cross-validate GitHub 1-min data against Yahoo Finance daily data.
This checks if the GitHub source is credible by comparing OHLC values.
"""
import io, requests, sys
import pandas as pd
import numpy as np

# Symbols to validate (pick a few liquid ones)
SYMBOLS = ["SBIN", "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY"]
YEAR = 2020

GITHUB_BASE = (
    "https://raw.githubusercontent.com/ShabbirHasan1/NSE-Data/main/"
    "NSE%20Minute%20Data/NSE_Stocks_Data"
)

def validate_symbol(symbol):
    """Compare GitHub 1-min aggregated to daily vs Yahoo Finance daily."""
    print(f"\n{'='*60}")
    print(f"  VALIDATING: {symbol} ({YEAR})")
    print(f"{'='*60}")

    # 1. Download GitHub 1-min data
    fname = f"{symbol}__EQ__NSE__NSE__MINUTE.csv"
    url = f"{GITHUB_BASE}/{fname}"
    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        print(f"  SKIP - not in GitHub repo")
        return None

    gh = pd.read_csv(io.StringIO(resp.text))
    gh["timestamp"] = pd.to_datetime(gh["timestamp"])
    gh = gh[gh["timestamp"].dt.year == YEAR].copy()
    gh["date"] = gh["timestamp"].dt.date

    # Aggregate 1-min to daily OHLC
    gh_daily = gh.groupby("date").agg(
        gh_open=("open", "first"),
        gh_high=("high", "max"),
        gh_low=("low", "min"),
        gh_close=("close", "last"),
        gh_volume=("volume", "sum"),
    )
    gh_daily.index = pd.to_datetime(gh_daily.index)

    # 2. Get Yahoo Finance daily data
    try:
        import yfinance as yf
        yf_sym = f"{symbol}.NS"
        yf_data = yf.download(yf_sym, start=f"{YEAR}-01-01", end=f"{YEAR+1}-01-01",
                              interval="1d", progress=False)
        if yf_data.empty:
            print(f"  SKIP - no Yahoo Finance data")
            return None

        # Handle MultiIndex columns from yfinance
        if isinstance(yf_data.columns, pd.MultiIndex):
            yf_data.columns = yf_data.columns.get_level_values(0)

        yf_daily = yf_data[["Open", "High", "Low", "Close", "Volume"]].copy()
        yf_daily.columns = ["yf_open", "yf_high", "yf_low", "yf_close", "yf_volume"]
        yf_daily.index = yf_daily.index.tz_localize(None)
    except Exception as e:
        print(f"  SKIP Yahoo Finance: {e}")
        return None

    # 3. Merge on date
    merged = gh_daily.join(yf_daily, how="inner")
    if merged.empty:
        print(f"  NO OVERLAP between sources")
        return None

    print(f"  Trading days in GitHub: {len(gh_daily)}")
    print(f"  Trading days in Yahoo:  {len(yf_daily)}")
    print(f"  Overlapping days:       {len(merged)}")

    # 4. Compare prices
    # Note: Yahoo Finance may include adjusted prices, so we compare ratios
    results = {}
    for col in ["open", "high", "low", "close"]:
        gh_col = f"gh_{col}"
        yf_col = f"yf_{col}"
        diff_pct = ((merged[gh_col] - merged[yf_col]) / merged[yf_col] * 100).abs()
        results[col] = {
            "mean_diff_pct": diff_pct.mean(),
            "max_diff_pct": diff_pct.max(),
            "median_diff_pct": diff_pct.median(),
            "within_1pct": (diff_pct < 1.0).mean() * 100,
            "within_2pct": (diff_pct < 2.0).mean() * 100,
        }

    print(f"\n  PRICE COMPARISON (% difference: GitHub vs Yahoo Finance)")
    print(f"  {'Field':<8} {'Mean Diff%':<12} {'Max Diff%':<12} {'Median':<10} {'Within 1%':<12} {'Within 2%'}")
    print(f"  {'-'*70}")
    for col, stats in results.items():
        print(f"  {col:<8} {stats['mean_diff_pct']:>8.3f}%   {stats['max_diff_pct']:>8.3f}%   "
              f"{stats['median_diff_pct']:>6.3f}%   {stats['within_1pct']:>8.1f}%     {stats['within_2pct']:>6.1f}%")

    # 5. Show COVID crash days specifically (March 12-23, 2020)
    crash = merged.loc["2020-03-09":"2020-03-27"]
    if not crash.empty:
        print(f"\n  COVID CRASH DAYS (March 9-27, 2020): {len(crash)} days")
        print(f"  {'Date':<12} {'GH Close':>10} {'YF Close':>10} {'Diff%':>8}")
        print(f"  {'-'*45}")
        for idx, row in crash.iterrows():
            diff = (row["gh_close"] - row["yf_close"]) / row["yf_close"] * 100
            print(f"  {idx.strftime('%Y-%m-%d'):<12} {row['gh_close']:>10.2f} {row['yf_close']:>10.2f} {diff:>+7.2f}%")

    # Volume comparison
    vol_corr = merged["gh_volume"].corr(merged["yf_volume"])
    print(f"\n  Volume Correlation: {vol_corr:.4f}")

    return results


if __name__ == "__main__":
    all_results = {}
    for sym in SYMBOLS:
        r = validate_symbol(sym)
        if r:
            all_results[sym] = r

    # Summary
    print(f"\n{'='*60}")
    print(f"  OVERALL CREDIBILITY SUMMARY")
    print(f"{'='*60}")
    if all_results:
        avg_close_diff = np.mean([r["close"]["mean_diff_pct"] for r in all_results.values()])
        avg_within_1 = np.mean([r["close"]["within_1pct"] for r in all_results.values()])
        print(f"  Average Close Price Difference: {avg_close_diff:.3f}%")
        print(f"  Average Days Within 1% Accuracy: {avg_within_1:.1f}%")
        print(f"  Stocks Validated: {len(all_results)}")
        if avg_close_diff < 1.0:
            print(f"\n  VERDICT: ✅ DATA IS CREDIBLE (avg diff < 1%)")
        elif avg_close_diff < 5.0:
            print(f"\n  VERDICT: ⚠️  DATA HAS MINOR DISCREPANCIES (< 5%)")
        else:
            print(f"\n  VERDICT: ❌ DATA HAS SIGNIFICANT ISSUES (> 5%)")
    else:
        print(f"  No validation results available")
