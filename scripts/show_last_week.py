"""scripts/show_last_week.py — Show last 5 trading days from backtest_trades.csv"""
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
csv = ROOT / "storage" / "logs" / "backtest_trades.csv"
if not csv.exists():
    print("ERROR: Run backtest first: python main.py backtest"); sys.exit(1)

df = pd.read_csv(csv)
df["entry_time"] = pd.to_datetime(df["entry_time"])
df["date"] = df["entry_time"].dt.date

last5 = sorted(df["date"].unique())[-5:]
week  = df[df["date"].isin(last5)].copy()

wins   = week[week["net_pnl_pct"] > 0]
losses = week[week["net_pnl_pct"] <= 0]
wr     = len(wins) / len(week) * 100 if len(week) else 0
pf     = abs(wins["net_pnl_pct"].sum()) / max(abs(losses["net_pnl_pct"].sum()), 1e-9)

print(f"\n{'='*60}")
print(f"  LAST WEEK: {last5[0]}  ->  {last5[-1]}")
print(f"{'='*60}")
print(f"  Total Trades    : {len(week)}")
print(f"  Win Rate        : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
print(f"  Profit Factor   : {pf:.2f}")
print(f"  Net ROI         : {week['net_pnl_pct'].sum():+.2f}%")
print(f"  Avg PnL/Trade   : {week['net_pnl_pct'].mean():+.4f}%")
print(f"  Best Trade      : {week['net_pnl_pct'].max():+.4f}%")
print(f"  Worst Trade     : {week['net_pnl_pct'].min():+.4f}%")
print(f"  LONG / SHORT    : {(week['side']=='LONG').sum()} / {(week['side']=='SHORT').sum()}")

print(f"\n  {'Date':<14} {'Trades':>6} {'Win%':>7} {'Net PnL':>10}  Sector Best")
print(f"  {'-'*58}")
for d in last5:
    day  = week[week["date"] == d]
    dw   = (day["net_pnl_pct"] > 0).sum()
    dwr  = dw / len(day) * 100 if len(day) else 0
    best = day.groupby("sector")["net_pnl_pct"].sum().idxmax() if len(day) else "-"
    flag = " ✓" if day["net_pnl_pct"].sum() > 0 else " ✗"
    print(f"  {str(d):<14} {len(day):>6} {dwr:>6.0f}% {day['net_pnl_pct'].sum():>+9.2f}%  {best}{flag}")

print(f"\n  {'--- Exit Reasons ---'}")
for reason, cnt in week["exit_reason"].value_counts().items():
    print(f"  {reason:<25} {cnt:>5}  ({cnt/len(week)*100:.1f}%)")

print(f"\n  {'Sector':<14} {'Trades':>7} {'Win%':>7} {'Total PnL':>12}")
print(f"  {'-'*44}")
sec = (
    week.groupby("sector")
    .agg(trades=("net_pnl_pct", "count"),
         win_rate=("net_pnl_pct", lambda x: (x > 0).mean() * 100),
         total_pnl=("net_pnl_pct", "sum"))
    .sort_values("total_pnl", ascending=False)
)
for s, row in sec.iterrows():
    print(f"  {s:<14} {int(row['trades']):>7} {row['win_rate']:>6.1f}% {row['total_pnl']:>+11.2f}%")

print(f"\n{'='*60}\n")
