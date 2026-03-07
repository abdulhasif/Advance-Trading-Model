"""
Fixes all 31 invalid tokens in sector_universe.csv using the correct
instrument keys from the live Upstox NSE instrument master.
Run: .venv\Scripts\python.exe tmp\fix_tokens.py
"""
import pandas as pd
from pathlib import Path

CSV = Path("config_data/sector_universe.csv")
df = pd.read_csv(CSV)

# Correct token map: our_symbol -> correct_instrument_key
FIXES = {
    "CESC":       "NSE_EQ|INE486A01021",
    "CANBK":      "NSE_EQ|INE476A01022",
    "MASTEK":     "NSE_EQ|INE759A01021",
    "ZYDUSWELL":  "NSE_EQ|INE768C01028",
    "IPCA":       "NSE_EQ|INE571A01038",   # IPCA LABORATORIES
    "LAURUSLABS": "NSE_EQ|INE947Q01028",
    "METROPOLIS": "NSE_EQ|INE112L01020",
    "JBCHEPHARM": "NSE_EQ|INE572A01036",
    "MANKIND":    "NSE_EQ|INE634S01028",   # MANKIND PHARMA
    "ERIS":       "NSE_EQ|INE406M01024",
    "SUNDRMFAST": "NSE_EQ|INE387A01021",
    "HEIDELBERG": "NSE_EQ|INE578A01017",
    "JSWINFRA":   "NSE_EQ|INE880J01026",
    "MAZDOCK":    "NSE_EQ|INE249Z01020",
    "MOTILALOFS": "NSE_EQ|INE338I01027",
    "ANGELONE":   "NSE_EQ|INE732I01021",
    "CAMS":       "NSE_EQ|INE596I01020",
    "MCX":        "NSE_EQ|INE745G01043",
    "MFSL":       "NSE_EQ|INE180A01020",
    "VEDANTFASH": "NSE_EQ|INE825V01034",   # Manyavar
    "NYKAA":      "NSE_EQ|INE388Y01029",
    "METROBRAND": "NSE_EQ|INE317I01021",   # Metro Brands (corrected trading symbol)
    "DELHIVERY":  "NSE_EQ|INE148O01028",
    "BHARAT":     "NSE_EQ|INE171Z01026",   # Bharat Dynamics (BDL)
    "ELECON":     "NSE_EQ|INE205B01031",
    "MGL":        "NSE_EQ|INE002S01010",
    "PFIZER":     "NSE_EQ|INE182A01018",
    "UJJIVANSFB": "NSE_EQ|INE551W01018",
    "EQUITASBNK": "NSE_EQ|INE063P01018",
    # KALPATPOWR: renamed to KPIL on exchange
    "KALPATPOWR": "NSE_EQ|INE220B01022",
}

# Remove WABCOINDIA (delisted from NSE)
before = len(df)
df = df[df["symbol"] != "WABCOINDIA"]
print(f"Removed WABCOINDIA (delisted). Rows: {before} -> {len(df)}")

# Apply token fixes
fixed = 0
for sym, correct_token in FIXES.items():
    mask = df["symbol"] == sym
    if mask.any():
        df.loc[mask, "instrument_token"] = correct_token
        print(f"  Fixed {sym:<15}: -> {correct_token}")
        fixed += 1
    else:
        print(f"  SKIP  {sym:<15}: not in CSV")

print(f"\nTotal fixed: {fixed}")

# Write back
df.to_csv(CSV, index=False, lineterminator="\n")
print(f"\nWritten: {CSV} ({len(df)} rows)")

# Verify no more bad tokens using the downloaded master
import gzip, json
with gzip.open('/tmp/nse_instruments.json.gz', 'rt') as f:
    instruments = json.load(f)
token_set = {inst['instrument_key'] for inst in instruments}

equities = df[df["is_index"] == False]
bad = [(r["symbol"], r["instrument_token"]) for _, r in equities.iterrows()
       if r["instrument_token"] not in token_set]
if bad:
    print(f"\nStill bad ({len(bad)}):")
    for s, t in bad:
        print(f"  {s}: {t}")
else:
    print(f"\nAll {len(equities)} equity tokens verified OK!")
