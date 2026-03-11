import polars as pl
from pathlib import Path
from contracts.settings import Settings
import shutil


def migrate():
    settings = Settings()
    old_delta_path = settings.nged_data_path / "delta" / "live_primary_flows"
    new_delta_path = settings.nged_data_path / "delta" / "live_primary_flows_v2"
    metadata_path = settings.nged_data_path / "parquet" / "substation_metadata.parquet"

    if not old_delta_path.exists():
        print(f"Old Delta table not found at {old_delta_path}")
        return

    if not metadata_path.exists():
        print(f"Metadata not found at {metadata_path}")
        return

    print("Loading metadata...")
    metadata = pl.read_parquet(metadata_path)

    # We need a mapping from the old 'substation_name' (CSV stem) to 'substation_number'
    # In the old code, substation_name was PurePosixPath(url).stem
    mapping = metadata.filter(pl.col("url").is_not_null()).select(
        [
            pl.col("url")
            .map_elements(lambda x: Path(x).stem, return_dtype=pl.String)
            .alias("substation_name"),
            "substation_number",
        ]
    )

    print("Loading old Delta table...")
    df = pl.read_delta(str(old_delta_path))

    print("Joining with metadata...")
    df = df.join(mapping, on="substation_name", how="inner")

    print("Dropping old column and selecting new schema...")
    df = df.drop("substation_name")

    # Ensure correct column order and types
    df = df.select(["timestamp", "substation_number", "MW", "MVA", "MVAr", "ingested_at"])

    print(f"Writing new Delta table to {new_delta_path}...")
    if new_delta_path.exists():
        shutil.rmtree(new_delta_path)

    df.write_delta(str(new_delta_path), delta_write_options={"partition_by": ["substation_number"]})

    print("Migration complete. You can now swap the directories:")
    print(f"mv {old_delta_path} {old_delta_path}_backup")
    print(f"mv {new_delta_path} {old_delta_path}")


if __name__ == "__main__":
    migrate()
