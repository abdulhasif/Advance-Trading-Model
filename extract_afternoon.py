
import pandas as pd

def extract_afternoon(input_file, output_file):
    print(f"Reading {input_file}...")
    # Read in chunks to save memory
    chunk_size = 1000000
    first_chunk = True
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='ISO8601', utc=True).dt.tz_localize(None)
        # Filter for after 12:00
        afternoon_chunk = chunk[chunk['timestamp'].dt.hour >= 12]
        
        if not afternoon_chunk.empty:
            mode = 'w' if first_chunk else 'a'
            header = True if first_chunk else False
            afternoon_chunk.to_csv(output_file, mode=mode, header=header, index=False)
            first_chunk = False
            print(f"Wrote {len(afternoon_chunk)} rows to {output_file}")

if __name__ == "__main__":
    extract_afternoon("storage/data/raw_ticks/raw_ticks_dump_2026-03-13.csv", "afternoon_ticks_subset.csv")
