import time
import upstox_client
from upstox_client.feeder import MarketDataStreamerV3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI2R0I1OTUiLCJqdGkiOiI2OWJjYzNhOGFlYjRmYjE5NWE3NDQ0MjkiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3Mzk3ODUzNiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzc0MDQ0MDAwfQ.CbUhLk5GoybY2qZlJlSvD9fQXrFmFwCXQDH9AHVDG_4"

def on_open(): print("SUCCESS: Connection Opened!")
def on_error(error): print(f"ERROR: {error}")

def test_indices():
    print("--- Testing Index Symbols (with spaces) ---")
    configuration = upstox_client.Configuration()
    configuration.access_token = ACCESS_TOKEN
    api_client = upstox_client.ApiClient(configuration)

    # Indices from the CSV
    instrument_keys = [
        "NSE_INDEX|Nifty Energy",
        "NSE_INDEX|Nifty Bank",
        "NSE_INDEX|Nifty IT",
        "NSE_INDEX|Nifty FMCG",
        "NSE_INDEX|Nifty Pharma",
        "NSE_INDEX|Nifty Auto",
        "NSE_INDEX|Nifty Metal",
        "NSE_INDEX|Nifty MS Fin Serv",
        "NSE_INDEX|Nifty Infra",
        "NSE_INDEX|Nifty Consumption",
        "NSE_INDEX|Nifty Multi Infra",
        "NSE_INDEX|Nifty Serv Sector"
    ]
    
    streamer = MarketDataStreamerV3(api_client, instrument_keys, mode="ltpc")
    streamer.on("open", on_open); streamer.on("error", on_error)

    print("Connecting...")
    streamer.connect()
    time.sleep(5)
    streamer.disconnect()

if __name__ == "__main__":
    test_indices()
