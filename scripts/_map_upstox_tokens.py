import pandas as pd
import requests

url = 'https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz'
print(f"Downloading Upstox NSE Master from {url}...")
df_upstox = pd.read_csv(url)

# Keep only Equity instruments
df_upstox = df_upstox[df_upstox['instrument_type'] == 'EQ']

universe_path = 'config_data/sector_universe.csv'
u = pd.read_csv(universe_path)

# Map exact symbol to Upstox tradingsymbol
m = u.merge(df_upstox[['tradingsymbol', 'instrument_key']], left_on='symbol', right_on='tradingsymbol', how='left')

# Some symbols might have different names in Upstox e.g M&M vs M_M
# Let's clean up those if they missed
missing = m[m['instrument_key'].isna()]
if not missing.empty:
    print(f"Missing direct mapping for {len(missing)} symbols. Attempting fuzzy match...")
    for idx, row in missing.iterrows():
        sym = row['symbol']
        # Try replacing & with _, - with _
        fuzzy_sym = sym.replace('&', '_').replace('-', '_')
        match = df_upstox[df_upstox['tradingsymbol'] == fuzzy_sym]
        if not match.empty:
            m.at[idx, 'instrument_key'] = match.iloc[0]['instrument_key']

m['instrument_token'] = m['instrument_key']

# Ensure instrument_token is exactly copied to config_data/sector_universe.csv
# Some might still be missing, we'll drop the helper columns
if 'instrument_key' in m.columns:
    m = m.drop(columns=['tradingsymbol', 'instrument_key'])

m.to_csv(universe_path, index=False)
print(f"Mapped {m['instrument_token'].notna().sum()} out of {len(m)} symbols!")
