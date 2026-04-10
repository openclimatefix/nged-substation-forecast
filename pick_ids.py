import polars as pl

df = pl.read_parquet("data/NGED/parquet/time_series_metadata.parquet")

# Pick 4 IDs: 1 primary substation (Disaggregated Demand), 1 BSP (Raw Flow?), 1 solar farm (PV), 1 wind farm (Wind)
# Let's just pick the first one of each type.
ids = []
for t in ["Disaggregated Demand", "Raw Flow", "PV", "Wind"]:
    id = df.filter(pl.col("time_series_type") == t)["time_series_id"][0]
    ids.append(id)

print(ids)
