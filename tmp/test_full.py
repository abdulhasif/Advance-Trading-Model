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

def test_full_mode():
    print("--- Testing FULL market data mode ---")
    configuration = upstox_client.Configuration()
    configuration.access_token = ACCESS_TOKEN
    api_client = upstox_client.ApiClient(configuration)

    instrument_keys = ["NSE_EQ|INE062A01020"] # SBI
    
    streamer = MarketDataStreamerV3(
        api_client,
        instrumentKeys=instrument_keys,
        mode="full" # TEST MODE CHANGE
    )

    streamer.on("open", on_open); streamer.on("error", on_error)

    print("Connecting...")
    streamer.connect()
    time.sleep(5)
    streamer.disconnect()

if __name__ == "__main__":
    test_full_mode()
