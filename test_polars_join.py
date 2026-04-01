import polars as pl

df1 = pl.DataFrame({"a": [1.0], "b": [2.0]}, schema={"a": pl.Float32, "b": pl.Float32})
df2 = pl.DataFrame({"a": [1.0], "c": [3.0]}, schema={"a": pl.Float64, "c": pl.Float64})
try:
    print(df1.join(df2, on="a"))
except Exception as e:
    print(f"Caught expected error: {e}")
