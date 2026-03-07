import sys
import logging
from src.data.batch_factory import process_instrument_year
from src.data.downloader import UpstoxHistoricalFetcher
from src.core.renko import RenkoBrickBuilder

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

fetcher = UpstoxHistoricalFetcher()
builder = RenkoBrickBuilder()

# Run just NATCOPHARM 2026 to see if it hangs!
print("Starting debug process...")
try:
    result = process_instrument_year('NATCOPHARM', 'NSE_EQ|INE987B01026', 'Pharma', 2026, fetcher, builder)
    print("FINISHED:", result)
except Exception as e:
    import traceback
    traceback.print_exc()
