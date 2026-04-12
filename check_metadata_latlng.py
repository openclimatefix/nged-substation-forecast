import polars as pl
from contracts.settings import Settings

settings = Settings()
metadata = pl.read_parquet(settings.nged_data_path / "parquet" / "time_series_metadata.parquet")

print("Metadata latitude values:", metadata["latitude"].unique())
print("Metadata longitude values:", metadata["longitude"].unique())
