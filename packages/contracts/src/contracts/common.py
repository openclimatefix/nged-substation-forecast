from typing import Any

import patito as pt
import polars as pl

# Define our standard datetime type for all schemas
UTC_DATETIME_DTYPE = pl.Datetime(time_unit="us", time_zone="UTC")


def _get_time_series_id_dtype(**kwargs) -> Any:
    return pt.Field(
        dtype=pl.Int32,
        description=(
            "Provided by NGED. This is the primary key for identifying the time series."
            " There's _almost_ a one-to-one mapping between time_series_id and the"
            " asset ID, so you can think of time_series_id as the asset ID"
            " (where an 'asset' is a physical asset like a substation or PV farm)"
        ),
        **kwargs,
    )
