import polars as pl
import patito as pt
from deltalake import write_deltalake, DeltaTable
from pathlib import Path
from contracts.data_schemas import NgedJsonPowerFlows


def append_to_delta(df: pt.DataFrame[NgedJsonPowerFlows], delta_path: Path):
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

    # If it exists, we need to merge
    # To avoid duplicates, we can use a merge operation.

    # Load the existing Delta table
    dt = DeltaTable(str(delta_path))

    # Read existing keys
    existing_keys = dt.to_pyarrow_table(columns=["time_series_id", "end_time"]).to_pandas()
    existing_keys = pl.from_pandas(existing_keys)

    # Filter new data
    new_data = df.join(existing_keys, on=["time_series_id", "end_time"], how="anti")

    if len(new_data) > 0:
        write_deltalake(
            delta_path, new_data.to_arrow(), mode="append", partition_by=["time_series_id"]
        )

        return

    # If it exists, we need to merge
    # To avoid duplicates, we can use a merge operation.

    # Load the existing Delta table
    dt = DeltaTable(delta_path)

    # Perform a merge to avoid duplicates
    # We want to insert rows from `df` where `(time_series_id, end_time)` does not exist in `dt`.

    # This is a common pattern in Delta Lake.
    # Since I cannot easily use `merge` without knowing the exact API,
    # I will use a simpler approach:
    # 1. Read the existing keys
    # 2. Filter the new data
    # 3. Append the new data

    # Load the existing Delta table
    dt = DeltaTable(str(delta_path))

    # Read existing keys
    existing_keys = dt.to_pyarrow_table(columns=["time_series_id", "end_time"]).to_pandas()
    existing_keys = pl.from_pandas(existing_keys)

    # Filter new data
    new_data = df.join(existing_keys, on=["time_series_id", "end_time"], how="anti")

    if len(new_data) > 0:
        write_deltalake(
            delta_path, new_data.to_arrow(), mode="append", partition_by=["time_series_id"]
        )
