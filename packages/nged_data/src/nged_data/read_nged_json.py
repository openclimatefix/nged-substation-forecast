"""Extracts metadata and time series from NGED JSON data.

The parser expects two things of each JSON file. Metadata fields sit at the top level. A `data`
field holds an array of time series data points.
"""

import logging
import re
from typing import NamedTuple

import patito as pt
import polars as pl
import polars_h3
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import ECMWF_ENS_H3_RESOLUTION

log = logging.getLogger(__name__)

# TODO: When we move to using multiple NWPs (at different resolutions), we should use a high
# H3 resolution for TimeSeriesMetadata, say res 9, and then use `polars_h3.cell_to_parent` to
# dynamically convert `TimeSeriesMetadata.h3_res_9` to each NWP model's own resolution before
# joining. See https://github.com/openclimatefix/nged-substation-forecast/issues/114
#
# For now, to keep things simple, we're just fixing the H3 resolution to `ECMWF_ENS_H3_RESOLUTION`
# for both the ECMWF NWP and the NGED locations.


class ExtractedPowerTimeSeries(NamedTuple):
    """Result of parsing ``PowerTimeSeries`` rows out of one NGED JSON file.

    ``n_dropped`` counts rows dropped by ``PowerTimeSeries.drop_implausible_rows`` for a
    malformed ``time`` — see that method's docstring for why ingestion degrades rather than
    raising.
    """

    dataframe: pt.DataFrame[PowerTimeSeries]
    n_dropped: int


def _extract_time_series_metadata(df: pl.DataFrame) -> pt.DataFrame[TimeSeriesMetadata]:
    """Extract TimeSeriesMetadata from NGED's JSON data converted to DataFrame.

    The parser assumes every column other than `data` is metadata.
    """
    metadata_df = df.drop("data")

    metadata_df = metadata_df.unnest("Area", separator="_")
    metadata_df = metadata_df.rename({col: _camel_to_snake(col) for col in metadata_df.columns})

    # Compute H3 index
    metadata_df = metadata_df.with_columns(
        polars_h3.latlng_to_cell(
            pl.col("latitude"), pl.col("longitude"), ECMWF_ENS_H3_RESOLUTION
        ).alias(f"h3_res_{ECMWF_ENS_H3_RESOLUTION}")
    )

    metadata_df = metadata_df.sort("time_series_id")

    return pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata).drop().cast().validate()


def _extract_power_time_series(df: pl.DataFrame, time_series_id: int) -> ExtractedPowerTimeSeries:
    """Extract PowerTimeSeries from NGED's JSON data converted to DataFrame.

    If NGED's meter reported no values, the `data` field in the JSON will be Null. Less commonly
    the field is an empty array `[]`, which `pl.read_json` infers as `List(Null)`. In both cases
    this function raises the following exception: polars.exceptions.InvalidOperationError:
    invalid dtype: expected 'Struct', got 'Null' for 'data'
    """
    # Extract time series data: explode the 'data' column and unnest the struct. 'explode' expands
    # the list of structs into individual rows. 'unnest' expands the struct fields into individual
    # columns.
    #
    # empty_as_null=False matches the Polars 2.0 default and silences the deprecation warning; it
    # has no effect on output here. An empty 'data: []' array is read by pl.read_json as List(Null),
    # because a single file can't infer the struct fields of an empty array. After explode, under
    # *both* settings, the .unnest("data") below then raises the same InvalidOperationError that a
    # Null 'data' field raises: "expected 'Struct', got 'Null' for 'data'". storage.py's
    # download_and_parse_files catches that exact message, logs a warning, and skips the file. An
    # empty-data file is therefore handled identically to the documented null-data case, whichever
    # way empty_as_null is set. The two settings differ on one case only: an empty List(Struct) with
    # a known schema. That case can't arise from pl.read_json of a single file.
    time_series_df = df.select("data").explode("data", empty_as_null=False).unnest("data")

    time_series_df = time_series_df.rename({"endTime": "time", "value": "power"})

    # Parse ISO 8601 datetime strings directly to UTC. Polars natively infers ISO 8601 formats
    # (including 'T' separators and 'Z' timezones) when no explicit format string is provided.
    time_series_df = time_series_df.with_columns(
        time=pl.col("time").str.to_datetime(time_zone="UTC")
    )

    time_series_df = time_series_df.with_columns(time_series_id=pl.lit(time_series_id))
    time_series_df = pt.DataFrame(time_series_df).set_model(PowerTimeSeries).drop().cast()
    time_series_df = time_series_df.sort(by=PowerTimeSeries.columns_to_sort_by)

    time_series_df, n_dropped = PowerTimeSeries.drop_implausible_rows(time_series_df)
    if n_dropped > 0:
        log.warning(
            f"Dropped {n_dropped} row(s) with a malformed `time` for {time_series_id=}: outside"
            " the plausible datetime range or not aligned to :00/:30."
        )

    return ExtractedPowerTimeSeries(
        dataframe=PowerTimeSeries.validate(time_series_df), n_dropped=n_dropped
    )


def _camel_to_snake(camel_str: str) -> str:
    """Converts a CamelCase string to snake_case."""
    s1 = re.sub("([^_])([A-Z][a-z]+)", r"\1_\2", camel_str)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()
