import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "torch"

import pandas as pd
import numpy as np
import logging
from config import FEATURE_COLS
import config

# Setup minimal logging
logging.basicConfig(level=logging.INFO)
print("--- STARTING SURGICAL ML SMOKE TEST ---")

# 1. Create a tiny mock dataframe that mimics the real data structure
n_samples = 1500
mock_data = {
    "_symbol": np.random.choice(["AAPL", "MSFT"], size=n_samples),
    "_date": pd.date_range(start="2023-01-01", periods=n_samples, freq="5min"),
    "label_long": np.random.choice([0, 1], size=n_samples),
    "label_short": np.random.choice([0, 1], size=n_samples),
    "conviction_target": np.random.choice([0, 1], size=n_samples),
}

for col in FEATURE_COLS:
    if col not in mock_data:
        mock_data[col] = np.random.randn(n_samples)

df = pd.DataFrame(mock_data)
df.sort_values(by=["_symbol", "_date"], inplace=True)
df.reset_index(drop=True, inplace=True)

# 2. Patch config paths
config.CNN_WINDOW_SIZE = 15
temp_model_path = config.MODELS_DIR / "smoke_brain1_long.keras"
temp_calib_path = config.MODELS_DIR / "smoke_calib_long.pkl"

# 3. Import specific components directly to bypass load_training_data
from src.ml.brain_trainer import generate_oof_predictions, train_brain1_cnn, train_brain2_meta

try:
    print("\n[TEST 1] Testing OOF Generation (CnnSequenceGenerator)...")
    oof_long = generate_oof_predictions(df, "label_long", "Smoke Brain1")
    print(f"OOF Generation Output Shape: {oof_long.shape} (Expected: {n_samples})")
    
    print("\n[TEST 2] Testing Brain1 CNN Training + Isotonic Calibration...")
    # Mock Keras fit to be fast
    import keras
    original_fit = keras.Sequential.fit
    def fast_fit(self, *args, **kwargs):
        kwargs['epochs'] = 1
        kwargs['verbose'] = 0
        return original_fit(self, *args, **kwargs)
    keras.Sequential.fit = fast_fit
    
    train_brain1_cnn(df, df.copy(), "label_long", "Smoke Brain1", temp_model_path, temp_calib_path)
    print("Brain 1 Training & Calibration Completed without crash.")
    
    print("\n[TEST 3] Testing Brain2 XGBoost Training...")
    # Give fake OOF outputs
    oof_fake_short = pd.Series(np.random.rand(n_samples), index=df.index)
    oof_long_series = pd.Series(oof_long, index=df.index)
    train_brain2_meta(df, df.copy(), oof_long_series, oof_fake_short)
    
    print("\n✅ --- ALL SMOKE TESTS PASSED SUCCESSFULLY --- ✅")
    
except Exception as e:
    import traceback
    print("\n❌ --- SMOKE TEST FAILED --- ❌")
    traceback.print_exc()
