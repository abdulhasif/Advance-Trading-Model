from typing import Optional, List
from fastapi import WebSocket
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        message_str = json.dumps(message, default=str)
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception:
                pass

manager = ConnectionManager()
_simulator_ref = None

def set_simulator_ref(simulator):
    global _simulator_ref
    _simulator_ref = simulator

