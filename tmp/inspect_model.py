import os
import sys
sys.path.append('.')
os.environ["KERAS_BACKEND"] = "torch"
import keras
import config

try:
    m = keras.models.load_model(str(config.BRAIN1_CNN_LONG_PATH))
    print(f"MODEL_INPUT_SHAPE: {m.input_shape}")
except Exception as e:
    print(f"ERROR: {e}")
