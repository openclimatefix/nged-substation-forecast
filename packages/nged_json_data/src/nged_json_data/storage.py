import polars as pl
import patito as pt
from deltalake import write_deltalake, DeltaTable
from pathlib import Path
from contracts.data_schemas import PowerTimeSeries


def append_to_delta(df: pt.DataFrame[PowerTimeSeries], delta_path: Path):
    """
    Appends data to a Delta table, ensuring no duplicates based on (time_series_id, end_time).

    Args:
        df: The Patito DataFrame to append.
        delta_path: The path to the Delta table.
    """

    # Ensure the directory exists
    delta_path.parent.mkdir(parents=True, exist_ok=True)

    if not delta_path.exists():
        # If the table doesn't exist, just write it
        write_deltalake(
            delta_path, df.to_arrow(), mode="overwrite", partition_by=["time_series_id"]
        )
        return

    # If it exists, we need to merge.
    # To avoid duplicates, we filter out existing (time_series_id, end_time) pairs.

    # Load the existing Delta table
    dt = DeltaTable(str(delta_path))

    # Read existing keys to identify duplicates
    existing_keys = pl.from_arrow(dt.to_pyarrow_table(columns=["time_series_id", "end_time"]))

    # Ensure existing_keys is a DataFrame
    if isinstance(existing_keys, pl.Series):
        existing_keys = existing_keys.to_frame()

    # Filter new data to only include rows not already in the Delta table
    new_data = df.join(existing_keys, on=["time_series_id", "end_time"], how="anti")

    if len(new_data) > 0:
        write_deltalake(
            delta_path, new_data.to_arrow(), mode="append", partition_by=["time_series_id"]
        )
