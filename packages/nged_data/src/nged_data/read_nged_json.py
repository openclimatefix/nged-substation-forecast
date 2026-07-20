"""
Extracts metadata and time series from NGED JSON data.

The JSON is expected to have a structure where metadata fields are at the top level,
and a 'data' field contains an array of time series data points.
"""

import re
from typing import Final

import patito as pt
import polars as pl
import polars_h3
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata

# TODO: When we move to using multiple NWPs (at different resolutions), we should use a high
# H3 resolution for TimeSeriesMetadata, say res 9, and then use `polars_h3.cell_to_parent` to
# dynamically convert `TimeSeriesMetadata.h3_res_9` to `NwpMetaData.h3_resolution` before joining.
# See https://github.com/openclimatefix/nged-substation-forecast/issues/114
#
# For now, to keep things simple, we're just fixing the H3 resolution to 5 for the ECMWF NWP and for
# the NGED locations.
_H3_RESOLUTION: Final[int] = 5


def _extract_time_series_metadata(df: pl.DataFrame) -> pt.DataFrame[TimeSeriesMetadata]:
    """Extract TimeSeriesMetadata from NGED's JSON data converted to DataFrame.

    This assumes that all columns other than 'data' are metadata.
    """
    metadata_df = df.drop("data")

    metadata_df = metadata_df.unnest("Area", separator="_")
    metadata_df = metadata_df.rename({col: _camel_to_snake(col) for col in metadata_df.columns})

    # Compute H3 index
    metadata_df = metadata_df.with_columns(
        polars_h3.latlng_to_cell(pl.col("latitude"), pl.col("longitude"), _H3_RESOLUTION).alias(
            f"h3_res_{_H3_RESOLUTION}"
        )
    )

    metadata_df = metadata_df.sort("time_series_id")

    return pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata).drop().cast().validate()


def _extract_power_time_series(
    df: pl.DataFrame, time_series_id: int
) -> pt.DataFrame[PowerTimeSeries]:
    """Extract PowerTimeSeries from NGED's JSON data converted to DataFrame.

    If NGED's meter reported no values, then the `data` field in the JSON will be Null (or, less
    commonly, an empty array `[]`, which `pl.read_json` infers as `List(Null)`), and this function
    will raise the following exception:
        polars.exceptions.InvalidOperationError: invalid dtype: expected 'Struct', got 'Null' for 'data'
    """
    # Extract time series data: explode the 'data' column and unnest the struct.
    # 'explode' expands the list of structs into individual rows.
    # 'unnest' expands the struct fields into individual columns.
    #
    # empty_as_null=False matches the Polars 2.0 default and silences the deprecation warning; it
    # has no effect on output here. An empty 'data: []' array is read by pl.read_json as
    # List(Null) -- a single file can't infer the struct fields of an empty array -- so after
    # explode, under *both* settings, the .unnest("data") below raises the same
    # InvalidOperationError ("expected 'Struct', got 'Null' for 'data'") that a Null 'data' field
    # raises. storage.py's download_and_parse_files catches that exact message, logs a warning,
    # and skips the file, so an empty-data file is handled identically to the documented null-data
    # case regardless of empty_as_null. (The only case where the two settings differ -- an empty
    # List(Struct) with a known schema -- can't arise from pl.read_json of a single file.)
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

    return PowerTimeSeries.validate(time_series_df)


def _camel_to_snake(camel_str: str) -> str:
    """Converts a CamelCase string to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
