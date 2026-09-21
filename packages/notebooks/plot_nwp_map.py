import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    from datetime import UTC, datetime
    from typing import cast

    import altair as alt
    import polars as pl
    from contracts.settings import Settings
    from contracts.weather_schemas import Nwp
    from lonboard import H3HexagonLayer, Map

    SETTINGS = Settings()


@app.cell
def _():
    # The scan is lazy: nothing is read until a cell below collects. nwp_model_id is dropped because
    # the table holds one NWP model today, so the column carries no information here.
    df = Nwp.scan_delta().drop("nwp_model_id")
    return (df,)


@app.cell
def _(df):
    df.head().collect()
    return


@app.cell
def _(df):
    # The chart below draws one run at one H3 cell, as a line per ensemble member. The plotted cell,
    # 599148110664433663, is a resolution-5 cell in Shetland, at roughly 60.6 N, 0.7 W. init_time is
    # a Delta partition column, so the scan prunes to a single daily partition. The table is
    # written sorted by (init_time, ensemble_member, valid_time, h3_index), so h3_index sorts
    # last, and a filter on a late sort key skips few row groups: the h3_index filter prunes
    # little further.
    NWP_INIT_TIME = datetime(2026, 5, 15, tzinfo=UTC)
    NWP_VAR_TO_PLOT = "temperature_2m"

    _df = (
        df.filter(
            pl.col("h3_index") == 599148110664433663,
            pl.col("init_time") == NWP_INIT_TIME,
        )
        .select(["valid_time", "ensemble_member", NWP_VAR_TO_PLOT])
        .collect()
    )

    _chart = (
        alt.Chart(_df)
        .mark_line(
            strokeWidth=1,
            opacity=0.3,
        )
        .encode(  # ty: ignore[unresolved-attribute]  # astral-sh/ty#2520
            x=alt.X(field="valid_time", type="temporal"),
            y=alt.Y(field=NWP_VAR_TO_PLOT, type="quantitative"),
            detail="ensemble_member:N",
        )
        .properties(height=290, width="container")
    )
    _chart
    return NWP_INIT_TIME, NWP_VAR_TO_PLOT


@app.cell
def _(NWP_INIT_TIME):
    VALID_TIME_TO_PLOT = datetime(2026, 5, 15, hour=12, tzinfo=UTC)
    ENS_MEMBER_TO_PLOT = 0

    assert VALID_TIME_TO_PLOT >= NWP_INIT_TIME
    return ENS_MEMBER_TO_PLOT, VALID_TIME_TO_PLOT


@app.cell
def _(ENS_MEMBER_TO_PLOT, NWP_INIT_TIME, VALID_TIME_TO_PLOT, df):
    # The filter leaves one value per H3 cell: one run, one member, one valid time, and every cell
    # in the grid. Those values are what the hexagon map below shades.
    filtered_df = cast(
        pl.DataFrame,
        df.filter(
            pl.col("init_time") == NWP_INIT_TIME,
            pl.col("ensemble_member") == ENS_MEMBER_TO_PLOT,
            pl.col("valid_time") == VALID_TIME_TO_PLOT,
        ).collect(),
    )

    filtered_df
    return (filtered_df,)


@app.cell
def _(NWP_VAR_TO_PLOT, filtered_df):
    from lonboard.colormap import apply_continuous_cmap
    from palettable.matplotlib import Viridis_20  # ty: ignore[unresolved-import]

    # apply_continuous_cmap wants values in 0-1, so rescale the chosen variable across the cells
    # drawn. Despite the name, max_bound holds the *range* rather than the maximum.
    values = filtered_df[NWP_VAR_TO_PLOT]
    min_bound = values.min()
    max_bound = values.max() - min_bound

    normalized = (values - min_bound) / max_bound
    return Viridis_20, apply_continuous_cmap, normalized


@app.cell
def _(Viridis_20, apply_continuous_cmap, filtered_df, normalized):
    Map(
        H3HexagonLayer(
            filtered_df.select("h3_index"),
            get_hexagon=filtered_df["h3_index"],
            get_fill_color=apply_continuous_cmap(normalized, Viridis_20, alpha=1),
            opacity=1,
        )
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
