import pandas as pd

df = pd.read_csv('storage/logs/backtest_trades.csv')
wins = df[df['net_pnl_pct'] > 0]
win_rate = (len(wins) / len(df)) * 100
avg_pnl = df['net_pnl_pct'].mean() * 100
total_pnl = df['net_pnl_pct'].sum() * 100

print(f"Total Trades: {len(df)}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Avg Net PnL: {avg_pnl:.4f}%")
print(f"Total Net PnL (Uncompounded): {total_pnl:.4f}%")
