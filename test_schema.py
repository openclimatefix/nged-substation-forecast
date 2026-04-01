import polars as pl
import polars_h3 as plh3

GRID_SIZE = 0.25


def compute_h3_grid_weights(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(h3_res7=plh3.cell_to_children("h3_index", 7))
        .explode("h3_res7")
        .with_columns(
            nwp_lat=((plh3.cell_to_lat("h3_res7") + (GRID_SIZE / 2)) / GRID_SIZE).floor()
            * GRID_SIZE,
            nwp_lng=((plh3.cell_to_lng("h3_res7") + (GRID_SIZE / 2)) / GRID_SIZE).floor()
            * GRID_SIZE,
        )
        .group_by(["h3_index", "nwp_lat", "nwp_lng"])
        .len()
        .with_columns(total=pl.col("len").sum().over("h3_index"))
        .with_columns(proportion=pl.col("len") / pl.col("total"))
    )


df = pl.DataFrame({"h3_index": [599148110664433663]}, schema={"h3_index": pl.UInt64})
res = compute_h3_grid_weights(df)
print(res.schema)
