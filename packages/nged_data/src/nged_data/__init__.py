"""Reading NGED's telemetry JSON from S3 into `PowerTimeSeries` and `TimeSeriesMetadata` rows.

``upsert_metadata`` — the only function re-exported here — writes the metadata to a Parquet
roster. The rest of the public surface lives in ``nged_data.storage``, whose module docstring says
which functions read the ``power_time_series`` Delta table and which write the roster. Package
[README](https://openclimatefix.github.io/nged-substation-forecast/api/nged_data/).
"""

from .storage import upsert_metadata

__all__ = ["upsert_metadata"]
