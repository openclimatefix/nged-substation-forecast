import polars as pl
from typing import cast

h3_grid = pl.read_parquet("./data/NWP/reference/gb_h3_grid_weights.parquet")
lngs = h3_grid.get_column("nwp_lng").unique().sort()
diffs = lngs.diff().drop_nulls()
crosses_antimeridian = len(diffs) > 0 and cast(float, diffs.max()) > 180
print(f"crosses_antimeridian: {crosses_antimeridian}")
print(f"diffs.max(): {diffs.max()}")
