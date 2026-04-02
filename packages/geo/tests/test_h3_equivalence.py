import polars as pl
import polars.testing
import polars_h3 as plh3
import numpy as np
import h3.api.basic_int as h3_raw
from geo.h3 import compute_h3_grid_weights


def test_h3_equivalence():
    GRID_SIZE = 0.25
    H3_RES = 5
    CHILD_RES = 7

    # Synthetic input: a few H3 res 5 cells around London
    # Use h3_raw to get valid res 5 cells
    # London: 51.5074° N, 0.1278° W
    # Note: h3.geo_to_cells expects (lat, lng) or a shape.
    # Actually geo_to_cells(shape) is what processing.py uses.
    # For synthetic test, let's just get a few cells.
    cells = [h3_raw.latlng_to_cell(51.5, 0.1, H3_RES), h3_raw.latlng_to_cell(52.5, -0.1, H3_RES)]
    df = pl.DataFrame({"h3_index": cells}, schema={"h3_index": pl.UInt64})

    # Old Logic (nested struct/value_counts/explode)
    # This matches the description of what was presumably there before.
    old_result = (
        df.with_columns(h3_res7=plh3.cell_to_children("h3_index", CHILD_RES))
        .with_columns(
            grid=pl.col("h3_res7").map_elements(
                lambda cells: [
                    {
                        "nwp_lat": np.floor(
                            (h3_raw.cell_to_latlng(c)[0] + (GRID_SIZE / 2)) / GRID_SIZE
                        )
                        * GRID_SIZE,
                        "nwp_lng": np.floor(
                            (h3_raw.cell_to_latlng(c)[1] + (GRID_SIZE / 2)) / GRID_SIZE
                        )
                        * GRID_SIZE,
                    }
                    for c in cells
                ],
                return_dtype=pl.List(
                    pl.Struct([pl.Field("nwp_lat", pl.Float64), pl.Field("nwp_lng", pl.Float64)])
                ),
            )
        )
        .with_columns(counts=pl.col("grid").list.eval(pl.element().value_counts()))
        .explode("counts")
        .unnest("counts")
        .select([pl.col("h3_index"), pl.col("").alias("grid"), pl.col("count").alias("len")])
        .unnest("grid")
        .with_columns(total=pl.col("len").sum().over("h3_index"))
        .with_columns(proportion=pl.col("len") / pl.col("total"))
        .select(["h3_index", "nwp_lat", "nwp_lng", "len", "total", "proportion"])
        .sort(["h3_index", "nwp_lat", "nwp_lng"])
    )

    # New Logic (exploded group_by)
    new_result = (
        df.with_columns(h3_res7=plh3.cell_to_children("h3_index", CHILD_RES))
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
        .sort(["h3_index", "nwp_lat", "nwp_lng"])
    )

    polars.testing.assert_frame_equal(old_result, new_result)

    # Library implementation
    lib_result = compute_h3_grid_weights(df, grid_size=GRID_SIZE, child_res=CHILD_RES).sort(
        ["h3_index", "nwp_lat", "nwp_lng"]
    )

    # The library implementation returns UInt32 for len and total, so we cast for comparison
    new_result = new_result.with_columns(
        len=pl.col("len").cast(pl.UInt32), total=pl.col("total").cast(pl.UInt32)
    )

    polars.testing.assert_frame_equal(new_result, lib_result)
