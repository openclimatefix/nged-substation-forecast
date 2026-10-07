"""Naive baseline forecasters that a trained model has to beat.

``ManualHeuristicForecaster`` is the analogue-ensemble method that distribution network operators
use today. See the package README for what each baseline emits.
"""

from baseline_forecasters.manual_heuristic import ManualHeuristicForecaster

__all__ = ["ManualHeuristicForecaster"]
