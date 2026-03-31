from typing import cast
import polars as pl
from contracts.settings import Settings
from datetime import timedelta

settings = Settings()
actuals_path = settings.nged_data_path / "delta" / "live_primary_flows"
df = pl.read_delta(str(actuals_path)).lazy()
max_date_df = cast(pl.DataFrame, df.select(pl.col("timestamp").max()).collect())
max_date = max_date_df.item()
print(f"Max date: {max_date}")

test_end = max_date.date()
test_start = test_end - timedelta(days=14)
train_end = test_start - timedelta(days=1)
print(f"Test: {test_start} to {test_end}")
print(f"Train end: {train_end}")
