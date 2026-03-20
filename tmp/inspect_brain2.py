import xgboost as xgb
import sys
sys.path.append('.')
import config

try:
    m = xgb.Booster()
    m.load_model(str(config.BRAIN2_MODEL_PATH))
    print("START_FEATURES")
    for name in m.feature_names:
        print(name)
    print("END_FEATURES")
except Exception as e:
    print(f"ERROR: {e}")
