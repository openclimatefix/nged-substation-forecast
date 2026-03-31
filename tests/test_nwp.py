from typing import cast
import polars as pl
from contracts.settings import Settings

settings = Settings()
nwp_path = settings.nwp_data_path / "ECMWF" / "ENS" / "*.parquet"
df = pl.scan_parquet(nwp_path)
max_init_df = cast(pl.DataFrame, df.select(pl.col("init_time").max()).collect())
max_init = max_init_df.item()
print(f"Max init_time: {max_init}")

# Check valid times for this init_time
valid_times = (
    df.filter(pl.col("init_time") == max_init)
    .select(
        pl.col("valid_time").min().alias("min_valid"), pl.col("valid_time").max().alias("max_valid")
    )
    .collect()
)
print(f"Valid times: {valid_times}")
