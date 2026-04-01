import polars as pl
import numpy as np

val = 55.9
df1 = pl.DataFrame({"a": [np.float32(val)], "b": [2.0]})
df2 = pl.DataFrame({"a": [np.float64(val)], "c": [3.0]})
print(f"df1 schema: {df1.schema}")
print(f"df2 schema: {df2.schema}")
print(f"Join result:\n{df1.join(df2, on='a')}")
