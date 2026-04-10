import polars as pl
import patito as pt
from pathlib import Path
import re
from contracts.data_schemas import TimeSeriesMetadata, PowerTimeSeries, UTC_DATETIME_DTYPE


def camel_to_snake(name: str) -> str:
    """Converts a CamelCase string to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def load_nged_json(
    file_path: Path,
) -> tuple[pt.DataFrame[TimeSeriesMetadata], pt.DataFrame[PowerTimeSeries]]:
    """
    Loads NGED JSON data from a file.

    The JSON file is expected to have a structure where metadata fields are at the top level,
    and a 'data' field contains an array of time series data points.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A tuple containing:
            - A DataFrame with the metadata.
            - A DataFrame with the time series data.
    """
    # Read the JSON file into a Polars DataFrame.
    # pl.read_json() is efficient for reading JSON files into a DataFrame.
    df = pl.read_json(file_path)

    # Extract metadata: all columns except 'data'.
    # This assumes that all columns other than 'data' are metadata.
    metadata_df = df.drop("data")

    # Convert metadata keys from CamelCase to snake_case
    metadata_df = metadata_df.rename({col: camel_to_snake(col) for col in metadata_df.columns})

    # Cast columns to match TimeSeriesMetadata contract
    if "time_series_id" in metadata_df.columns:
        metadata_df = metadata_df.with_columns(pl.col("time_series_id").cast(pl.Int32))
    if "substation_number" in metadata_df.columns:
        metadata_df = metadata_df.with_columns(pl.col("substation_number").cast(pl.Int32))
    if "substation_type" in metadata_df.columns:
        metadata_df = metadata_df.with_columns(pl.col("substation_type").cast(pl.Categorical))
    if "latitude" in metadata_df.columns:
        metadata_df = metadata_df.with_columns(pl.col("latitude").cast(pl.Float32))
    if "longitude" in metadata_df.columns:
        metadata_df = metadata_df.with_columns(pl.col("longitude").cast(pl.Float32))

    # Validate the metadata DataFrame against the new TimeSeriesMetadata contract.
    metadata_df = TimeSeriesMetadata.validate(metadata_df)

    # Extract time series data: explode the 'data' column and unnest the struct.
    # 'explode' expands the list of structs into individual rows.
    # 'unnest' expands the struct fields into individual columns.
    time_series_df = df.select("data").explode("data")
    if time_series_df["data"].dtype == pl.Null:
        empty_df = pl.DataFrame(
            schema={
                "time_series_id": pl.Int32,
                "period_end_time": UTC_DATETIME_DTYPE,
                "power": pl.Float32,
            }
        )
        return metadata_df, pt.DataFrame[PowerTimeSeries](empty_df)

    time_series_df = time_series_df.unnest("data")

    # Rename endTime to period_end_time and parse it as a UTC datetime.
    time_series_df = time_series_df.rename({"endTime": "period_end_time"})
    time_series_df = time_series_df.with_columns(
        pl.col("period_end_time")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ")
        .dt.replace_time_zone("UTC")
    )

    # Add the time_series_id from the metadata to the time_series_df.
    # Assuming metadata_df has only one row, which is the time_series_id.
    time_series_id = metadata_df["time_series_id"].item()
    time_series_df = time_series_df.with_columns(
        pl.lit(time_series_id).cast(pl.Int32).alias("time_series_id")
    )

    # Rename value to power
    time_series_df = time_series_df.rename({"value": "power"})

    # Select only the required columns: ["time_series_id", "period_end_time", "power"].
    time_series_df = time_series_df.select(["time_series_id", "period_end_time", "power"])

    # Cast power to Float32
    time_series_df = time_series_df.with_columns(pl.col("power").cast(pl.Float32))

    # Validate the time series against the PowerTimeSeries data contract.
    time_series_df = PowerTimeSeries.validate(time_series_df)

    return metadata_df, time_series_df
