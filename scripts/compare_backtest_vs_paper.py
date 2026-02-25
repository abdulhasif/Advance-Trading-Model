"""Compare backtest results vs today's paper trading."""
import pandas as pd

# Backtest results
bt = pd.read_csv("storage/logs/backtest_trades.csv")
print("=" * 60)
print("BACKTEST (2019-2020, 497 days, 66 stocks)")
print("=" * 60)
total = len(bt)
wins = (bt["net_pnl_pct"] > 0).sum()
losses = total - wins
print(f"Total trades: {total:,}")
print(f"Win rate: {wins:,}/{total:,} = {wins/total*100:.1f}%")
print(f"Avg net PnL: {bt['net_pnl_pct'].mean():.4f}%")
print(f"Avg win:     {bt[bt['net_pnl_pct']>0]['net_pnl_pct'].mean():.4f}%")
print(f"Avg loss:    {bt[bt['net_pnl_pct']<=0]['net_pnl_pct'].mean():.4f}%")
print(f"Avg bricks held: {bt['bricks_held'].mean():.1f}")
print()
print("Exit reasons:")
print(bt["exit_reason"].value_counts().to_string())
print()
print("Direction split:")
print(bt["side"].value_counts().to_string())
print()

daily = bt.groupby(bt["entry_time"].str[:10])["net_pnl_pct"].agg(["count", "sum"])
prof = (daily["sum"] > 0).sum()
print(f"Avg trades/day: {daily['count'].mean():.0f}")
print(f"Profitable days: {prof}/{len(daily)} ({prof/len(daily)*100:.0f}%)")
print(f"Avg daily PnL: {daily['sum'].mean():.4f}%")

# Paper trading results
print()
print("=" * 60)
print("=" * 60)
print("PAPER TRADING (Feb 20, 2026 — 1 day)")
print("=" * 60)
pt = pd.read_csv("storage/logs/paper_trades.csv", header=None)
pt.columns = ["trade_id", "symbol", "sector", "side", "entry_time", "entry_price",
               "exit_time", "exit_price", "qty", "bricks_held", "favorable_bricks",
               "adverse_bricks", "gross_pnl", "cost", "net_pnl", "exit_reason"]
today = pt[pt["entry_time"].astype(str).str.startswith("2026-02-20", na=False)].copy()
for col in ["net_pnl", "bricks_held", "gross_pnl", "cost"]:
    today[col] = pd.to_numeric(today[col], errors="coerce").fillna(0)
total_p = len(today)
wins_p = (today["net_pnl"] > 0).sum()
print(f"Total trades: {total_p}")
print(f"Win rate: {wins_p}/{total_p} = {wins_p/total_p*100:.1f}%")
print(f"Total net PnL: Rs {today['net_pnl'].sum():.2f}")
print(f"Avg net PnL: Rs {today['net_pnl'].mean():.2f}")
print(f"Avg bricks held: {today['bricks_held'].mean():.1f}")
print()
print("Exit reasons:")
print(today["exit_reason"].value_counts().to_string())
print()
print("Direction split:")
print(today["side"].value_counts().to_string())
