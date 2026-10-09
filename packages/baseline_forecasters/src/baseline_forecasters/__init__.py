"""Naive baseline forecasters that a trained forecasting model has to beat.

``ManualHeuristicForecaster`` is the analogue-ensemble method that was, until recently, the
normal forecasting approach among distribution network operators. ``ClimatologyForecaster`` emits
the distribution of past power at the same calendar month, time of day, and day type. See the
package README for what each baseline emits.
"""

from baseline_forecasters.climatology import ClimatologyForecaster
from baseline_forecasters.manual_heuristic import ManualHeuristicForecaster

__all__ = ["ClimatologyForecaster", "ManualHeuristicForecaster"]
