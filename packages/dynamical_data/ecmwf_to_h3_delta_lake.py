import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl
    import h3.api.numpy_int as h3
    import polars_h3 as plh3
    import shapely.geometry
    import shapely.wkt
    from typing import Final
    from lonboard import Map, H3HexagonLayer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fetch geometry of GB
    """)
    return


@app.cell
def _():
    # TODO: Move all this geo code into a new `package/geo_utils` package.

    # Downloaded from https://onsdigital.github.io/uk-topojson/ for the year 2025.
    with open("england_scotland_wales.geojson") as f:
        file_contents = f.read()
    return (file_contents,)


@app.cell
def _(file_contents):
    shape = shapely.from_geojson(file_contents)

    # Buffer by 0.25 degrees (approx 20km) to catch islands/coasts.
    # This turns the "rough" map into a "safe" container.
    # This takes about 20 seconds to compute!
    shape = shape.buffer(0.25)

    shape
    return (shape,)


@app.cell
def _(shape):
    # Which H3 resolution to use?
    # ECMWF ENS has a res of 0.25°.
    # For GB, a 0.25° grid box is approx 28 km north-south (lat) x 16 km east-west (lon) ~= 450 km².
    # H3 average hexagon areas (from https://h3geo.org/docs/core-library/restable/#average-area-in-km2):
    #   res 4 = 1,770 km² (far too coarse).
    #   res 5 =   253 km² (the right choice).
    H3_RES: Final[int] = 5

    cells = h3.geo_to_cells(shape, res=H3_RES)

    df = pl.DataFrame({"h3_index": list(cells)}, schema={"h3_index": pl.UInt64}).sort("h3_index")
    df
    return H3_RES, df


@app.cell
def _(H3_RES: Final[int], df):
    # Verify we caught the Isles of Scilly:
    _scilly_hex = h3.latlng_to_cell(lat=49.9, lng=-6.3, res=H3_RES)
    assert _scilly_hex in df["h3_index"]
    return


@app.cell
def _(df):
    layer = H3HexagonLayer(
        df,
        get_hexagon=df["h3_index"],
        opacity=0.2,
    )
    Map(layer)
    return


@app.cell
def _(df):
    df_with_children = df.with_columns(h3_res7=plh3.cell_to_children("h3_index", 7)).explode(
        "h3_res7"
    )
    return (df_with_children,)


@app.cell
def _(df_with_children):
    # --- 3. The "Coordinate Lookup" Step ---
    # Instead of spatial join, we compute the NWP Key mathematically
    GRID_SIZE = 0.25

    # TODO: Need to think if ECMWF's grid boxes are centered on these coords, and how that interacts with our code below:

    df_with_grid_x_y = df_with_children.with_columns(
        # MATHEMATICAL MAPPING: Round down to nearest 0.25 degree
        # This effectively bins every child into an NWP box
        h3_res7_grid_cell=pl.struct(
            y=(plh3.cell_to_lat("h3_res7") / GRID_SIZE).floor().cast(pl.Int64),
            x=(plh3.cell_to_lng("h3_res7") / GRID_SIZE).floor().cast(pl.Int64),
        ),
    )

    df_with_grid_x_y
    return (df_with_grid_x_y,)


@app.cell
def _(df_with_grid_x_y):
    df_with_counts = (
        df_with_grid_x_y.group_by("h3_index")
        .agg(grid_cell_counts=pl.col("h3_res7_grid_cell").value_counts())
        .with_columns(
            total=pl.col("grid_cell_counts").list.agg(pl.element().struct.field("count").sum())
        )
        .explode("grid_cell_counts")
        .unnest("grid_cell_counts")
        .with_columns(propotion=pl.col.count / pl.col.total)
    )
    df_with_counts

    # TODO:
    # 1. Join h3_res7_grid_cell with the actual NWP data, to end up with a dataframe that has `proportion` and the raw NWP value
    # 2. Multiply `proportion` with each NWP value.
    # 3. Groupby h3_index, and sum each NWP value column
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
