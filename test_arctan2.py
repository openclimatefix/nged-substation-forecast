import polars as pl

df = pl.DataFrame({"y": [1.0, 0.0, -1.0], "x": [0.0, 1.0, 0.0]})
print(df.with_columns(pl.arctan2("y", "x")))
