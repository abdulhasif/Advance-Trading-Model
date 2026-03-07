import pandas as pd
import numpy as np
from datetime import datetime

def analyze():
    trades_path = r'storage/logs/backtest_trades.csv'
    try:
        df = pd.read_csv(trades_path)
    except Exception as e:
        print(f"Error loading trades: {e}")
        return

    if df.empty:
        print("No trades found in log.")
        return

    # Basic stats
    total_trades = len(df)
    wins = df[df['net_pnl_pct'] > 0]
    win_rate = (len(wins) / total_trades) * 100
    
    total_pnl = df['net_pnl_pct'].sum()
    avg_pnl = df['net_pnl_pct'].mean()
    
    long_trades = len(df[df['side'] == 'LONG'])
    short_trades = len(df[df['side'] == 'SHORT'])
    
    # Calculate daily profit
    # Period reported as 169 days in logs
    total_days = 169
    daily_profit_pct = total_pnl / total_days
    
    # Investment and ROI
    # Assuming each trade uses 1/N of capital or we look at cumulative growth
    # For simplicity, if we risk 100% of capital per trade (sequential):
    # Growth = Product of (1 + pnl_pct/100)
    cumulative_growth = (1 + df['net_pnl_pct']/100).prod()
    total_roi_pct = (cumulative_growth - 1) * 100

    print("====================================================")
    print("      DUAL-BRAIN BACKTEST ANALYSIS (STRICT)")
    print("====================================================")
    print(f"Total Trades:      {total_trades}")
    print(f"Win Rate:          {win_rate:.2f}%")
    print(f"LONG Trades:       {long_trades}")
    print(f"SHORT Trades:      {short_trades}")
    print("----------------------------------------------------")
    print(f"Total Net PnL:     {total_pnl:.4f}% (Sum)")
    print(f"Compounded ROI:    {total_roi_pct:.4f}%")
    print(f"Avg PnL per Trade: {avg_pnl:.4f}%")
    print(f"Daily Profit:      {daily_profit_pct:.4f}% per day")
    print("----------------------------------------------------")
    
    # Sector performance
    print("\nSector Performance:")
    sector_perf = df.groupby('sector')['net_pnl_pct'].agg(['count', 'mean', 'sum']).sort_values(by='sum', ascending=False)
    print(sector_perf)
    
    # Best and Worst trades
    print("\nBest Trade:")
    print(df.loc[df['net_pnl_pct'].idxmax()][['symbol', 'side', 'net_pnl_pct', 'entry_time']])
    
    print("\nWorst Trade:")
    print(df.loc[df['net_pnl_pct'].idxmin()][['symbol', 'side', 'net_pnl_pct', 'entry_time']])
    print("====================================================")

if __name__ == "__main__":
    analyze()
