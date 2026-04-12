import polars as pl
from contracts.settings import Settings

settings = Settings()
metadata = pl.read_parquet(settings.nged_data_path / "parquet" / "time_series_metadata.parquet")

print("Metadata columns:", metadata.columns)
print("Metadata h3_res_5 values:", metadata["h3_res_5"].unique())
