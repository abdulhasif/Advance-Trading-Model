
import pandas as pd
from datetime import datetime

def analyze_signals(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True).dt.tz_localize(None)
    
    # Filter for signals after 12:00
    afternoon_df = df[df['timestamp'].dt.hour >= 12].copy()
    
    if afternoon_df.empty:
        print("No afternoon signals found in the log.")
        return

    print(f"Total Afternoon Signals (>= 12:00): {len(afternoon_df)}")
    
    # Action breakdown
    print("\n--- Action Breakdown ---")
    print(afternoon_df['action'].value_counts())
    
    with open("afternoon_analysis.txt", "w") as f:
        f.write(f"Total Afternoon Signals (>= 12:00): {len(afternoon_df)}\n")
        
        # Action breakdown
        f.write("\n--- Action Breakdown ---\n")
        f.write(afternoon_df['action'].value_counts().to_string() + "\n")
        
        # Skip reason breakdown
        f.write("\n--- Skip Reason Breakdown (for status='SKIP' or 'DROP') ---\n")
        skips = afternoon_df[afternoon_df['action'].isin(['SKIP', 'DROP', 'EXIT'])]
        if not skips.empty:
            f.write(skips['reason'].value_counts().to_string() + "\n")
        
        # Check Brain 1 Probability distribution in afternoon
        f.write("\n--- Brain 1 Probability Distribution (Afternoon) ---\n")
        f.write(afternoon_df['brain1_prob'].describe().to_string() + "\n")
        
        # Detailed look at signals with Brain 1 probability > 0.40
        f.write("\n--- Afternoon Signals with Brain1 Prob > 0.40 ---\n")
        afternoon_signals = afternoon_df[
            (afternoon_df['brain1_prob'] >= 0.10)
        ].sort_values('brain1_prob', ascending=False)
        
        if not afternoon_signals.empty:
            cols = ['timestamp', 'symbol', 'direction', 'brain1_prob', 'brain2_conviction', 'rel_strength', 'action', 'reason']
            f.write(afternoon_signals[cols].head(100).to_string(index=False) + "\n")
        else:
            f.write("No afternoon signals found with Brain1 Prob >= 0.10\n")

    print("Analysis saved to afternoon_analysis.txt")

if __name__ == "__main__":
    analyze_signals("spoofer_logs/spoofer_signals.csv")
