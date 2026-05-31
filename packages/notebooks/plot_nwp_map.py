import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    from datetime import datetime, timezone
    from typing import cast

    import altair as alt
    import polars as pl
    from contracts.settings import Settings
    from contracts.weather_schemas import NwpOnDisk
    from lonboard import H3HexagonLayer, Map

    SETTINGS = Settings()


@app.cell
def _():
    df = NwpOnDisk.scan_delta().drop("nwp_model_id")
    return (df,)


@app.cell
def _(df):
    df.head().collect()
    return


@app.cell
def _(df):
    NWP_INIT_TIME = datetime(2026, 5, 15, tzinfo=timezone.utc)
    NWP_VAR_TO_PLOT = "temperature_2m"

    _df = (
        df.filter(
            pl.col("h3_index") == 599148110664433663,
            pl.col("init_time") == NWP_INIT_TIME,
        )
        .select(["valid_time", "ensemble_member"] + [NWP_VAR_TO_PLOT])
        .collect()
    )

    _chart = (
        alt.Chart(_df)
        .mark_line(
            strokeWidth=1,
            opacity=0.3,
        )
        .encode(
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
    VALID_TIME_TO_PLOT = datetime(2026, 5, 15, hour=12, tzinfo=timezone.utc)
    ENS_MEMBER_TO_PLOT = 0

    assert VALID_TIME_TO_PLOT >= NWP_INIT_TIME
    return ENS_MEMBER_TO_PLOT, VALID_TIME_TO_PLOT


@app.cell
def _(ENS_MEMBER_TO_PLOT, NWP_INIT_TIME, VALID_TIME_TO_PLOT, df):
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
