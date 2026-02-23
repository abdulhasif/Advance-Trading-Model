
import pandas as pd
from pathlib import Path

trade_log = Path("storage/logs/paper_trades.csv")
if not trade_log.exists():
    print("No trades found.")
    exit()

df = pd.read_csv(trade_log)
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])
df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60

total = len(df)
winners = len(df[df['net_pnl'] > 0])
win_rate = (winners / total * 100) if total > 0 else 0
avg_dur = df['duration'].mean()
median_dur = df['duration'].median()

print(f"--- DAILY SUMMARY (Feb 23) ---")
print(f"Total Trades: {total}")
print(f"Winners:      {winners}")
print(f"Win Rate:     {win_rate:.2f}%")
print(f"Avg Duration: {avg_dur:.2f} mins")
print(f"Median Dur:   {median_dur:.2f} mins")
print(f"\n--- EXIT REASONS ---")
print(df['exit_reason'].value_counts())

# Check for rapid re-entries
df['prev_exit'] = df.groupby('symbol')['exit_time'].shift(1)
df['reentry_gap'] = (df['entry_time'] - df['prev_exit']).dt.total_seconds() / 60
rapid = len(df[df['reentry_gap'] < 10])
print(f"\nRapid Re-entries (<10 mins): {rapid}")
