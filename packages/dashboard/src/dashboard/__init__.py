"""Shared helpers for the marimo dashboard apps at ``packages/dashboard/``.

The marimo notebook scripts (``map_and_timeseries.py``, ``view_forecasts.py``) live at the
package root and import their shared, unit-testable logic from this package: the local/S3
data-source toggle (:mod:`dashboard.data_source`) and the forecast chart builder
(:mod:`dashboard.forecast_chart`).
"""
