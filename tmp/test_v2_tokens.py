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

def test_v2_numerical():
    print("--- Testing Numeric Tokens (V2 Style) ---")
    configuration = upstox_client.Configuration()
    configuration.access_token = ACCESS_TOKEN
    api_client = upstox_client.ApiClient(configuration)

    # Some V2 Numeric Tokens for SBI, RELIANCE, NIFTY
    instrument_keys = ["3045", "2885", "Nifty 50"] # Hypothetical numeric V2 tokens
    
    # We still use StreamerV3 because the SDK might bridge them
    streamer = MarketDataStreamerV3(api_client, instrument_keys, mode="ltpc")
    streamer.on("open", on_open); streamer.on("error", on_error)

    print("Connecting...")
    streamer.connect()
    time.sleep(5)
    streamer.disconnect()

if __name__ == "__main__":
    test_v2_numerical()
