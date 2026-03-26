"""
src/core/risk.py - The Risk Fortress (Gatekeeper)
====================================================
Soft Veto, Ranking Engine, and Drift Detector.
"""

from collections import deque
import json
import os
import config


class RiskFortress:
    """
    Gatekeeper that filters and ranks signals.

      - Soft Veto   : penalises stock != sector direction
      - Ranking     : keeps Top N signals by composite score
      - Drift Guard : rolling accuracy of last N alerts -> yellow alert if < 50%
    """

    def __init__(self, state_file: str = "risk_state.json"):
        self.alert_history: deque = deque(maxlen=config.DRIFT_WINDOW)
        self.yellow_alert = False
        self.state_file = state_file
        
        # Load memory on startup to survive engine crashes
        self._load_state()

    # -- scoring -------------------------------------------------------------
    def score_signal(
        self,
        brain1_prob: float,
        brain2_conviction: float,
        stock_direction: int,
        sector_direction: int,
    ) -> float:
        raw = brain1_prob * brain2_conviction
        if stock_direction != sector_direction:
            # Note: Ensure config.SECTOR_PENALTY is scaled correctly (e.g., 0.25, not 25)
            # relative to the prob * conviction product.
            raw -= config.SECTOR_PENALTY
        return max(0.0, raw)

    # -- ranking -------------------------------------------------------------
    def rank_signals(self, signals: list[dict]) -> list[dict]:
        ranked = sorted(signals, key=lambda s: s["score"], reverse=True)
        return ranked[: config.TOP_N_SIGNALS]

    # -- drift ---------------------------------------------------------------
    def update_drift(self, predicted_dir: int, actual_dir: int | None = None) -> float | None:
        """
        Asynchronous Drift Guard with State Persistence and Active Alerting.
        """
        if actual_dir is not None:
            self.alert_history.append(1.0 if predicted_dir == actual_dir else 0.0)
            self._save_state()  # Persist memory immediately
            
        current_accuracy = self.drift_accuracy
        
        if current_accuracy is None:
            self.yellow_alert = False
            return 1.0  # Default during warmup
            
        # Actively evaluate and trigger the Yellow Alert protocol
        threshold = getattr(config, 'DRIFT_THRESHOLD', 0.50)
        self.yellow_alert = current_accuracy < threshold
            
        return current_accuracy

    @property
    def drift_accuracy(self) -> float | None:
        if len(self.alert_history) < config.DRIFT_WARMUP_WINDOW:
            return None
        return sum(self.alert_history) / len(self.alert_history)
        
    # -- persistence ---------------------------------------------------------
    def _save_state(self):
        """Saves the rolling drift history to disk to survive application crashes."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({"alert_history": list(self.alert_history)}, f)
        except Exception as e:
            print(f"RiskFortress Warning: Could not save state - {e}")

    def _load_state(self):
        """Restores memory from disk on startup."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    for item in data.get("alert_history", []):
                        self.alert_history.append(item)
                
                # Re-evaluate yellow alert state immediately upon waking up
                acc = self.drift_accuracy
                if acc is not None:
                    threshold = getattr(config, 'DRIFT_THRESHOLD', 0.50)
                    self.yellow_alert = acc < threshold
            except Exception as e:
                print(f"RiskFortress Warning: Could not load state - {e}")
