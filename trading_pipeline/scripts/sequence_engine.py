import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import logging

logger = logging.getLogger(__name__)

# Note: Explicit tensorflow.keras import is generally safer for modern TF2/TF3 environments
from keras import utils

class CnnSequenceGenerator(utils.Sequence):
    """
    Memory-efficient Keras data generator for large datasets.
    Yields 3D sliding windows on-the-fly to avoid RAM exhaustion.
    Hardened for Hybrid CNN-XGBoost Alignment.
    """
    def __init__(self, x_data, y=None, target_col: str = None, 
                 window_size: int = 15, batch_size: int = 2048, 
                 feature_cols: list = None, indices: np.ndarray = None, 
                 symbols: np.ndarray = None, **kwargs):
        super().__init__(**kwargs)
        
        from trading_pipeline import config
        if feature_cols is None:
            feature_cols = config.FEATURE_COLS

        self.window_size = window_size
        self.batch_size = batch_size
        
        # 1. Extract flat features and labels (supports DataFrame or NumPy)
        if isinstance(x_data, pd.DataFrame):
            self.X_flat = x_data[feature_cols].copy().fillna(0).values.astype(np.float32)
            self.y_flat = x_data[target_col].values.astype(np.float32) if target_col else None
            curr_symbols = x_data["_symbol"].values if "_symbol" in x_data.columns else None
        else:
            self.X_flat = x_data.astype(np.float32)
            self.y_flat = y.values.astype(np.float32) if hasattr(y, "values") else (y.astype(np.float32) if y is not None else None)
            curr_symbols = symbols
        
        # 2. Identify valid sequence indices (same symbol/day overlap)
        if indices is not None:
            self.valid_indices = indices
        else:
            n_total = len(self.X_flat)
            if n_total < window_size:
                self.valid_indices = np.array([], dtype=np.int32)
            elif curr_symbols is not None:
                sym_start = curr_symbols[:-window_size + 1]
                sym_end = curr_symbols[window_size - 1:]

                # Symbol Continuity Guard: ensures the entire window belongs to the same symbol.
                # Note: We now allow crossing day boundaries (overnight stitching) for the same stock
                # to ensure the CNN sees the context of "Yesterday" to predict "Today."
                raw_valid = (sym_start == sym_end)
                self.valid_indices = np.where(raw_valid)[0]
            else:
                self.valid_indices = np.arange(n_total - window_size + 1)
        # 3. Create the lazy view (zero-copy)
        self.X_views = sliding_window_view(
            self.X_flat, window_shape=(window_size, self.X_flat.shape[1])
        ).squeeze(axis=1)

    def __len__(self):
        return int(np.ceil(len(self.valid_indices) / self.batch_size))

    def __getitem__(self, index):
        """
        Returns one batch of (X, y) or X if no targets are provided.
        Shape: (Batch, Time, Features) -> (batch_size, window_size, len(feature_cols))
        """
        batch_indices = self.valid_indices[index * self.batch_size : (index + 1) * self.batch_size]
        
        # X shape: (batch_size, window_size, features)
        X_batch = self.X_views[batch_indices].copy() # Copy to avoid modifying the contiguous view

        # CRITICAL FIX: Removed rogue volume scaling here. 
        # All scaling must occur in the central feature engineering pipeline to prevent Train-Serve Skew.

        if self.y_flat is not None:
            # Label at index i + window_size - 1 for window starting at index i
            y_batch = self.y_flat[batch_indices + self.window_size - 1]
            return X_batch, y_batch
            
        return X_batch

    def get_target_indices(self) -> np.ndarray:
        """
        Retrieves the exact absolute DataFrame indices corresponding to the generated targets.
        CRITICAL FOR HYBRID XGBOOST ALIGNMENT: 
        Use this to slice the original DataFrame so the CNN embeddings align perfectly with the tabular features.
        
        Usage: aligned_df = original_df.iloc[generator.get_target_indices()]
        """
        return self.valid_indices + self.window_size - 1

    @staticmethod
    def get_warmup_padding(df: pd.DataFrame, symbol: str, target_time, window_size: int) -> pd.DataFrame:
        """
        Retrieves the necessary historical rows (warmup) to ensure a signal 
        can be generated at exactly target_time.
        """
        sym_df = df[df["_symbol"] == symbol].sort_values("brick_timestamp")
        past_df = sym_df[sym_df["brick_timestamp"] <= target_time]
        return past_df.tail(window_size)

    @property
    def shape(self):
        """Mock shape for architecture building compatibility"""
        return (len(self.valid_indices), self.window_size, self.X_flat.shape[1])

