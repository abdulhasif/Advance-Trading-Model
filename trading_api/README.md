# trading-api

FastAPI-powered telemetry and control bridge for the Advanced Trading Model. Provides real-time WebSocket updates and REST endpoints for system management.

## 🛠 Features
- **Telemetry WebSocket**: Real-time broadcast of Renko bricks, PnL, and model conviction.
- **Trade Control**: Remote pause/resume/kill switches for the live execution engine.
- **Sentiment Analytics**: Dedicated endpoint for VADER-based news mood aggregation.
- **State Interface**: Shared-memory bridge for monitoring the market heartbeat.

## 🚀 Installation & Setup

1. **Install Core Dependency**:
   Ensure `trading-core` is installed in your environment:
   ```bash
   pip install -e ../trading_core
   ```

2. **Install Project Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔄 Execution

To start the API server:
```bash
python src/main.py
```

The API will be available at `http://localhost:8000`.
Check the `/docs` endpoint for the interactive Swagger documentation.
