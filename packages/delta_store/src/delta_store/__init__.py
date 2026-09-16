"""Physical storage policy for the project's Delta tables.

``contracts`` owns each table's *logical* shape and meaning; this package owns its *physical*
layout — parquet writer properties, compression-friendly sort orders, and significand-precision
rounding — plus the write helpers that apply them. Dagster assets stay thin by writing through
this package rather than calling ``write_deltalake`` with ad-hoc settings.

One module per table — ``power_forecasts``, ``nwp``, ``power_time_series``,
``eligible_time_series``, ``effective_capacity``, ``forecast_metrics`` — plus the shared precision
helper in ``precision``. Only ``power_forecasts`` and ``nwp`` carry writer-properties/sort-order/
precision tuning today, each chosen from measurements on real data and landing on different
choices — see
<https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#storage-formats-measured-not-assumed>
for the comparison. The other four modules exist so their writes go through ``delta_store`` like
every other table. They carry no tuning because there is no measurement yet to back any.
"""

from delta_store.effective_capacity import write_effective_capacity
from delta_store.eligible_time_series import write_eligible_time_series
from delta_store.forecast_metrics import write_forecast_metrics
from delta_store.nwp import write_nwp
from delta_store.power_forecasts import write_power_forecasts
from delta_store.power_time_series import write_power_time_series
from delta_store.precision import round_to_significand_bits

__all__ = [
    "round_to_significand_bits",
    "write_effective_capacity",
    "write_eligible_time_series",
    "write_forecast_metrics",
    "write_nwp",
    "write_power_forecasts",
    "write_power_time_series",
]
