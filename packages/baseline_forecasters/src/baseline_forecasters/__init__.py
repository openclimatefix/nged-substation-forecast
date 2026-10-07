"""Naive baseline forecasters that a trained forecasting model has to beat.

``ManualHeuristicForecaster`` is the analogue-ensemble method that was, until recently, the
normal forecasting approach among distribution network operators. See the package README for what
each baseline emits.
"""

from baseline_forecasters.manual_heuristic import ManualHeuristicForecaster

__all__ = ["ManualHeuristicForecaster"]
