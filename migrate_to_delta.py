import polars as pl
from pathlib import Path
from deltalake import write_deltalake


def migrate():
    parquet_dir = Path("data/NGED/parquet/live_primary_flows")
    delta_dir = Path("data/NGED/delta/live_primary_flows")

    if not parquet_dir.exists():
        print(f"Parquet directory {parquet_dir} does not exist.")
        return

    if delta_dir.exists():
        print(f"Delta directory {delta_dir} already exists. Please delete it first.")
        return

    parquet_files = list(parquet_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files to migrate.")

    if not parquet_files:
        return

    # We will read them all, add substation_name and ingested_at, and write to delta
    dfs = []
    for i, file_path in enumerate(parquet_files):
        substation_name = file_path.stem
        df = pl.read_parquet(file_path)

        # Add required columns
        df = df.with_columns(
            [
                pl.lit(substation_name).alias("substation_name"),
                pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("ingested_at"),
            ]
        )

        # Ensure all possible measurement columns exist so the schema is uniform
        for col in ["MW", "MVA", "MVAr"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float32).alias(col))

        # Reorder columns for consistency
        df = df.select(["timestamp", "substation_name", "MW", "MVA", "MVAr", "ingested_at"])
        dfs.append(df)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(parquet_files)} files...")

    print("Concatenating dataframes...")
    final_df = pl.concat(dfs)

    print(f"Writing to Delta Lake at {delta_dir}...")
    write_deltalake(str(delta_dir), final_df.to_arrow(), partition_by=["substation_name"])
    print("Migration complete!")


if __name__ == "__main__":
    migrate()
