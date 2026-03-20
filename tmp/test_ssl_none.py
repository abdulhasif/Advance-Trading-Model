import time
import threading
import ssl
import websocket
import upstox_client
from upstox_client.feeder import MarketDataStreamerV3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI2R0I1OTUiLCJqdGkiOiI2OWJjYzNhOGFlYjRmYjE5NWE3NDQ0MjkiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3Mzk3ODUzNiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzc0MDQ0MDAwfQ.CbUhLk5GoybY2qZlJlSvD9fQXrFmFwCXQDH9AHVDG_4"

def on_open(): print("SUCCESS: Connection Opened!")
def on_error(error): print(f"ERROR: {error}")

def test_ssl_none():
    print("--- Testing SDK with SSL_CERT_NONE ---")
    
    # We monkey-patch the websocket.create_connection globally for this test
    original_create = websocket.create_connection
    def patched_create(*args, **kwargs):
        kwargs['sslopt'] = {"cert_reqs": ssl.CERT_NONE}
        return original_create(*args, **kwargs)
    
    websocket.create_connection = patched_create

    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = ACCESS_TOKEN
        api_client = upstox_client.ApiClient(configuration)

        instrument_keys = ["NSE_EQ|INE062A01020"]
        streamer = MarketDataStreamerV3(api_client, instrument_keys, mode="ltpc")
        streamer.on("open", on_open); streamer.on("error", on_error)

        print("Connecting...")
        thread = threading.Thread(target=streamer.connect, daemon=True)
        thread.start()
        time.sleep(10)
        streamer.disconnect()
    finally:
        websocket.create_connection = original_create

if __name__ == "__main__":
    test_ssl_none()
