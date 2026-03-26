# trading-core

Core Renko Physics Engine, Alpha Features, and Strategic Risk Gates for the Institutional Trading System.

## 🛠 Features
- **Institutional Renko Physics**: Bit-perfect 1-minute OHLC to Renko conversion with Brownian Bridge interpolation.
- **Alpha Factory**: 17+ quantitative features (VWAP Z-Score, VPT Acceleration, Structural Trend).
- **Strategy Gates**: Centralized entry/exit logic ensuring parity between live and simulation environments.
- **Risk Fortress**: Real-time position sizing and margin management.

## 🚀 Installation & Setup

This project is a pip-installable dependency for other modules in the ecosystem.

### Install in Editable Mode (Local Development)
```bash
pip install -e .
```

### Key Modules
- `trading_core.core.physics.renko`: Renko Brick Builder v3.0
- `trading_core.core.risk.strategy`: Entry/Exit Logic Gates
- `trading_core.core.features.features`: Global Feature Engine

## 🔄 Standalone Usage
While this project contains the logic, it is typically used as a dependency for `trading-pipeline` or `trading-engine`.
Pip-pind versions ensure no training-serving skew across repositories.
