# Advanced Trading Model - Modular Build

This repository is a production-grade refactor of the advanced trading model, modularized into four independent services with 100% bit-perfect logic parity.

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Module (via CLI wrapper)**:
   ```bash
   python main.py download   # Ingest historical data
   python main.py train      # Brain 1 & 2 ML training
   python main.py backtest   # Large-scale simulation
   python main.py engine     # Live market heartbeat
   python main.py api        # Telemetry & REST API
   ```

## 🏗 Modular Architecture

| Project | Responsibility | Primary Stack |
|---------|----------------|---------------|
| **`trading-core`** | Renko Physics, Alpha Features, Strategy Gates | Python, NumPy, Pandas |
| **`trading-pipeline`** | Data Ingestion, ML Training (CNN/XGB) | TensorFlow, Keras, XGBoost |
| **`trading-engine`** | Upstox Live Connectivity, Execution | WebSockets, SDK |
| **`trading-api`** | Telemetry, UI State, REST Control | FastAPI, Pydantic |

## 🔄 Synchronization Logic
The projects are **synced via a shared directory structure** and a **decentralized configuration** system:

1. **Data Sync**: All modules read/write to `data/` (Models and Parquets) defined in `trading_core/core/config/paths.py`.
2. **Logic Sync**: All simulation and live execution gates are imported from `trading_core.core.risk.strategy`.
3. **Config Sync**: Each module maintains its own `config.py` that inherits from the core `base_config`.
