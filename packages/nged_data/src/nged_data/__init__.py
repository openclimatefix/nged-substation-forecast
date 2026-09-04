"""Reads NGED's telemetry JSON from S3, parsing it into `PowerTimeSeries`/`TimeSeriesMetadata` rows.

``upsert_metadata`` — the only function re-exported here — writes the metadata to a Parquet
roster; the rest of the public surface lives in ``nged_data.storage``. This package reads the
``power_time_series`` Delta table, to work out which rows are already stored, but never writes to
it: the caller appends the parsed power observations to Delta itself. See the package
[README](https://openclimatefix.github.io/nged-substation-forecast/api/nged_data/) for the full
public surface.
"""

from .storage import upsert_metadata

__all__ = ["upsert_metadata"]
