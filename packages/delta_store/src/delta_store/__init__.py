"""Physical storage policy for the project's Delta tables.

``contracts`` owns each table's *logical* shape and meaning; this package owns its *physical*
layout — parquet writer properties, compression-friendly sort orders, and significand-precision
rounding — plus the write helpers that apply them. Dagster assets stay thin by writing through
this package rather than calling ``write_deltalake`` with ad-hoc settings.

One module per table (currently ``power_forecasts`` and ``nwp``), plus the shared precision
helper in ``precision``.
"""

from delta_store.nwp import write_nwp
from delta_store.power_forecasts import write_power_forecasts
from delta_store.precision import round_to_significand_bits

__all__ = ["round_to_significand_bits", "write_nwp", "write_power_forecasts"]
