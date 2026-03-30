import polars as pl
from datetime import datetime

df = pl.DataFrame(
    {
        "valid_time": [datetime(2023, 1, 8, 2)],
        "init_time": [datetime(2023, 1, 1, 0)],
    }
)

df = df.with_columns(lead_time_days=(pl.col("valid_time") - pl.col("init_time")).dt.total_days())

print(f"Lead time days: {df['lead_time_days'][0]}")
print(f"Is <= 7: {df['lead_time_days'][0] <= 7}")
