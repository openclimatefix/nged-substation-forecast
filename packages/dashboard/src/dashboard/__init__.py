"""Shared helpers for the marimo dashboard apps at ``packages/dashboard/``.

The marimo notebook scripts (``map_and_timeseries.py``, ``view_forecasts.py``) live at the
package root and import their shared, unit-testable logic from this package: the local/S3
data-source toggle (`dashboard.data_source`) and the forecast chart builder
(`dashboard.forecast_chart`).

A marimo cell is a ``def _(...)`` function whose parameters are the names other cells export, and
marimo rebuilds a notebook from its cells rather than running the module, so no test can call a
cell the way a caller calls a function. Any logic worth a test is therefore pushed down into this
package, whose modules are ordinary library code. Of the two modules, ``forecast_chart`` has its
tests in ``packages/dashboard/tests/``; ``data_source`` has no tests today. What stays in a notebook
is the arrangement: which controls exist, which Delta queries run, and how the pieces stack on the
page.
"""
