---
description: How to run the modular trading platform
---

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. To run a historical backtest:
   ```bash
   python main.py backtest
   ```

3. To start the live trading engine:
   ```bash
   python main.py engine
   ```

4. To start the telemetry API:
   ```bash
   python main.py api
   ```
