"""Shared data processing logic for ML models."""

from typing import cast

import patito as pt
import polars as pl
from contracts.data_schemas import (
    PowerTimeSeries,
)


def calculate_peak_capacity(flows: pt.LazyFrame[PowerTimeSeries]) -> pt.DataFrame:
    """Calculate the peak capacity for each time series.

    Args:
        flows: Historical power flow data.

    Returns:
        A Patito DataFrame containing the peak capacity for each time series.
    """
    return pt.DataFrame(
        flows.group_by("time_series_id")
        .agg(peak_capacity=pl.col("value").abs().max().fill_null(1.0).clip(lower_bound=1.0))
        .collect()
    )
