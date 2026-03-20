import requests
import websocket
import ssl
import json
import time

ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI2R0I1OTUiLCJqdGkiOiI2OWJjYzNhOGFlYjRmYjE5NWE3NDQ0MjkiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3Mzk3ODUzNiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzc0MDQ0MDAwfQ.CbUhLk5GoybY2qZlJlSvD9fQXrFmFwCXQDH9AHVDG_4"

def test_handshake_with_headers():
    print("1. Fetching Authorized Redirect URI...")
    url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        print(f"FAILED: {res.text}")
        return
    
    ws_uri = res.json()["data"]["authorizedRedirectUri"]
    print(f"SUCCESS: URI fetched.")

    origins = [
        "https://api.upstox.com",
        "https://upstox.com",
        "http://localhost",
        "http://127.0.0.1"
    ]

    for origin in origins:
        print(f"\n2. Attempting Handshake with Origin: {origin}")
        try:
            ws = websocket.create_connection(
                ws_uri, 
                sslopt={"cert_reqs": ssl.CERT_NONE},
                header=[f"Origin: {origin}"]
            )
            print(f"SUCCESS: Handshake complete with Origin: {origin}")
            ws.close()
            return
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    test_handshake_with_headers()
