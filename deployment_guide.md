# Deployment & Synchronization Guide

This document explains how the modular components of the Advanced Trading Model interact and stay synchronized during live execution.

## 1. Synchronization Mechanism

The four projects (`trading-core`, `trading-pipeline`, `trading-engine`, `trading-api`) stay in sync through **Shared State** and **Unified Imports**.

### 🔄 Data Synchronization
- **Primary Storage**: `c:\Trading Platform\Advance Trading Model\data\`
- **Models**: `trading-pipeline` trains models and saves them to `data/brain/`. `trading-engine` loads these exact `.keras` and `.json` files for live inference.
- **Features**: `trading-core` defines the calculation logic. Both `trading-pipeline` (batch) and `trading-engine` (live) use the same code from `trading_core.core.features` to ensure zero training-serving skew.

### 📐 Logic Synchronization (Bit-Perfect)
The system uses **Single-Source Logic (SSL)**:
- **Renko Physics**: Defined in `trading_core.core.physics.renko`.
- **Strategy Gates**: Defined in `trading_core.core.risk.strategy`.
Any logic change in `trading-core` automatically updates both the **Backtester** (simulation) and the **Execution Engine** (live).

## 2. Running in Production

To run the full suite across multiple terminals:

### Terminal 1: Telemetry API (Status Monitor)
```bash
python main.py api
```

### Terminal 2: Market Heartbeat (Execution Engine)
```bash
python main.py engine
```

### Terminal 3: Periodic Re-training (Optional)
```bash
python main.py train --inc
```

## 3. Remote Deployment (Syncing code)
When deploying to a remote server:
1. Ensure the directory structure is preserved.
2. The `PYTHONPATH` should be set to the root folder to resolve `trading_core` imports across all projects.
3. Use a process manager like **PM2** or **Systemd** to keep the `engine` and `api` services alive.
