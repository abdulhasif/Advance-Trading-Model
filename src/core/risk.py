"""
src/core/risk.py — The Risk Fortress (Gatekeeper)
====================================================
Soft Veto, Ranking Engine, and Drift Detector.
"""

from collections import deque
import config


class RiskFortress:
    """
    Gatekeeper that filters and ranks signals.

      • Soft Veto   : penalises stock ≠ sector direction (−25 pts)
      • Ranking     : keeps Top N signals by composite score
      • Drift Guard : rolling accuracy of last 50 alerts -> yellow alert if < 50%
    """

    def __init__(self):
        self.alert_history: deque = deque(maxlen=config.DRIFT_WINDOW)
        self.yellow_alert = False

    # ── scoring ─────────────────────────────────────────────────────────────
    def score_signal(
        self,
        brain1_prob: float,
        brain2_conviction: float,
        stock_direction: int,
        sector_direction: int,
    ) -> float:
        raw = brain1_prob * brain2_conviction
        if stock_direction != sector_direction:
            raw -= config.SECTOR_PENALTY
        return max(0.0, raw)

    # ── ranking ─────────────────────────────────────────────────────────────
    def rank_signals(self, signals: list[dict]) -> list[dict]:
        ranked = sorted(signals, key=lambda s: s["score"], reverse=True)
        return ranked[: config.TOP_N_SIGNALS]

    # ── drift ───────────────────────────────────────────────────────────────
    def update_drift(self, predicted_dir: int, actual_dir: int):
        self.alert_history.append(1 if predicted_dir == actual_dir else 0)
        if len(self.alert_history) >= config.DRIFT_WINDOW:
            acc = sum(self.alert_history) / len(self.alert_history)
            self.yellow_alert = acc < config.DRIFT_THRESHOLD
            return acc
        return None

    @property
    def drift_accuracy(self) -> float | None:
        if len(self.alert_history) < config.DRIFT_WARMUP_WINDOW:
            return None
        return sum(self.alert_history) / len(self.alert_history)
