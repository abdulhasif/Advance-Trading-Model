# trading-pipeline

Institutional-grade data ingestion and ML training pipeline for the Advanced Trading Model.

## 🛠 Features
- **Async Downloader**: Multi-threaded 1-minute OHLC ingestion from yfinance/Upstox.
- **Brain 1 (CNN) Trainer**: 1D-CNN training with triple-barrier labeling.
- **Brain 2 (XGBoost) Meta-Model**: Conviction scoring for false-positive reduction.
- **Vectorized Backtester**: Large-scale simulation with portfolio concurrency limits.

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

## 🔄 Execution Flow

### 1. Data Ingestion
```bash
python scripts/download_history.py
```

### 2. Feature Building
```bash
python scripts/build_features.py
```

### 3. Model Training
```bash
python scripts/train_models.py
```

### 4. Simulation
```bash
python scripts/backtest.py
```
