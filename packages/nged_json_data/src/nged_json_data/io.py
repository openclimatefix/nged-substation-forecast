import polars as pl
from pathlib import Path


def load_nged_json(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
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

    # Extract time series data: explode the 'data' column and unnest the struct.
    # 'explode' expands the list of structs into individual rows.
    # 'unnest' expands the struct fields into individual columns.
    time_series_df = df.select("data").explode("data")
    if time_series_df["data"].dtype == pl.Null:
        return metadata_df, pl.DataFrame()

    time_series_df = time_series_df.unnest("data")

    return metadata_df, time_series_df
