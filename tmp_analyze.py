import pandas as pd
df = pd.read_csv('storage/logs/backtest_trades.csv')
print(f"Total Trades: {len(df)}")
if len(df) > 0:
    print(f"Win Rate: {(df['net_pnl_pct'] > 0).mean():.2%}")
    print(f"Total Net PnL (%): {df['net_pnl_pct'].sum():.2f}")
    print(f"Avg Net PnL (%): {df['net_pnl_pct'].mean():.2f}")
    if 'exit_reason' in df.columns:
        print("\nExit Reasons:")
        print(df['exit_reason'].value_counts())
else:
    print("No trades found.")
