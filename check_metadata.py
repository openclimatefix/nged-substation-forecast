import polars as pl

df = pl.read_parquet("data/NGED/parquet/time_series_metadata.parquet")
print(df.head())
print(df.shape)
