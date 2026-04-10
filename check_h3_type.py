import polars as pl
import polars_h3 as plh3

df = pl.DataFrame({"latitude": [52.948212], "longitude": [-0.015023]})

h3_cell = plh3.latlng_to_cell(df["latitude"], df["longitude"], 5)
print(f"Type: {type(h3_cell)}")
print(f"Value: {h3_cell}")
