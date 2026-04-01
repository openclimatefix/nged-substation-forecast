import polars as pl

df = pl.DataFrame({"a": pl.Series([1, None, 3], dtype=pl.UInt8)})
print(df.with_columns(pl.col("a").interpolate()).schema)
