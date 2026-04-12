import polars as pl
import polars_h3 as plh3

df = pl.DataFrame({"latitude": [52.948212], "longitude": [-0.015023]})

df = df.with_columns(h3_res_5=plh3.latlng_to_cell(pl.col("latitude"), pl.col("longitude"), 5))
print(df)
print(df.schema)
