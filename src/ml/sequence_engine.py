import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import logging

logger = logging.getLogger(__name__)

import keras
from keras import utils

class CnnSequenceGenerator(utils.Sequence):
    """
    Memory-efficient Keras data generator for large datasets.
    Yields 3D sliding windows on-the-fly to avoid RAM exhaustion.
    """
    def __init__(self, df: pd.DataFrame, target_col: str, 
                 window_size: int = 15, batch_size: int = 2048, 
                 feature_cols: list = None, indices: np.ndarray = None, **kwargs):
        super().__init__(**kwargs)
        
        if feature_cols is None:
            import config
            feature_cols = config.FEATURE_COLS
            
        self.vol_idx = None
        if "feature_vpb_roc" in feature_cols:
            self.vol_idx = feature_cols.index("feature_vpb_roc")
        elif "volume" in feature_cols:
            self.vol_idx = feature_cols.index("volume")

        self.window_size = window_size
        self.batch_size = batch_size
        
        # 1. Extract flat features and labels
        self.X_flat = df[feature_cols].copy().fillna(0).values.astype(np.float32)
        self.y_flat = df[target_col].values.astype(np.float32) if target_col else None
        
        # 2. Identify valid sequence indices (same symbol/day overlap)
        if indices is not None:
            self.valid_indices = indices
        else:
            n_total = len(df)
            if n_total < window_size:
                self.valid_indices = np.array([], dtype=np.int32)
            else:
                sym_array = df["_symbol"].values
                sym_start = sym_array[:-window_size + 1]
                sym_end = sym_array[window_size - 1:]

                date_array = df["brick_timestamp"].dt.date.values
                date_start = date_array[:-window_size + 1]
                date_end   = date_array[window_size - 1:]
                
                # FIX 2: Overnight Stitching Guard
                raw_valid = (sym_start == sym_end) & (date_start == date_end)
                self.valid_indices = np.where(raw_valid)[0]

        # 3. Create the lazy view (zero-copy)
        self.X_views = sliding_window_view(
            self.X_flat, window_shape=(window_size, self.X_flat.shape[1])
        ).squeeze(axis=1)

    def __len__(self):
        return int(np.ceil(len(self.valid_indices) / self.batch_size))

    def __getitem__(self, index):
        """Returns one batch of (X, y)"""
        batch_indices = self.valid_indices[index * self.batch_size : (index + 1) * self.batch_size]
        
        # X shape: (batch_size, window_size, features)
        X_batch = self.X_views[batch_indices].copy() # Copy to avoid modifying the view
        
        # Apply volume scaling to help CNN convergence (Fix from Memory Opt turn)
        if self.vol_idx is not None:
            import config
            X_batch[:, :, self.vol_idx] *= config.VOL_MULT

        if self.y_flat is not None:
            # Label at index i+14 for window starting at index i
            y_batch = self.y_flat[batch_indices + self.window_size - 1]
            return X_batch, y_batch
            
        return X_batch

    @property
    def shape(self):
        """Mock shape for architecture building compatibility"""
        return (len(self.valid_indices), self.window_size, self.X_flat.shape[1])

