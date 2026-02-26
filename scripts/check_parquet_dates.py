import pandas as pd
import sys

def check_date(sym):
    try:
        df = pd.read_parquet(f"storage/market_data/{sym}/{sym}.parquet")
        last_date = df["brick_timestamp"].dt.date.max() if "brick_timestamp" in df.columns else df["timestamp"].dt.date.max()
        print(f"Latest data in {sym}.parquet: {last_date}")
    except Exception as e:
        print(f"Error reading {sym}: {e}")

if __name__ == "__main__":
    check_date("RELIANCE")
    check_date("TCS")
