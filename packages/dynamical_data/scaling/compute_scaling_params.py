import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl
    from datetime import datetime


@app.cell
def _():
    df = pl.scan_parquet("data/*.parquet")
    return (df,)


@app.cell
def _(df):
    df.head().collect()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Compute the mins and maxes for each NWP variable
    """)
    return


@app.cell
def _(df):
    numeric_nwp_vars = [
        "pressure_surface",
        "pressure_reduced_to_mean_sea_level",
        "geopotential_height_500hpa",
        "downward_long_wave_radiation_flux_surface",
        "downward_short_wave_radiation_flux_surface",
        "precipitation_surface",
        "wind_u_10m",
        "wind_v_100m",
        "dew_point_temperature_2m",
        "temperature_2m",
        "wind_u_100m",
        "wind_v_10m",
    ]

    # Literal column to label our rows
    exprs = [pl.lit(["max", "min", "range"]).alias("statistic")]

    # Build the aggregation expressions for each column
    for col_name in numeric_nwp_vars:
        exprs.append(
            pl.concat_list(
                [
                    pl.col(col_name).max(),
                    pl.col(col_name).min(),
                    pl.col(col_name).max() - pl.col(col_name).min(),  # The range
                ]
            ).alias(col_name)
        )

    # Compute everything in one pass, then expand the lists into rows
    stats = df.select(exprs).explode(pl.all()).collect(engine="streaming")
    stats
    return numeric_nwp_vars, stats


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Create Polars expressions to re-scale; and save the stats to disk
    """)
    return


@app.cell
def _(numeric_nwp_vars, stats):
    _DTYPE = pl.UInt8
    BUFFER_PERCENT = 0.05  # Widen mins and maxes

    scaling_exprs = []
    scaling_params: list[dict] = []

    for _col_name in numeric_nwp_vars:
        raw_max, raw_min, raw_range = stats[_col_name]

        buffered_max = raw_max + (raw_range * BUFFER_PERCENT)
        buffered_min = raw_min - (raw_range * BUFFER_PERCENT)
        buffered_range = buffered_max - buffered_min

        scaling_params.append(
            {
                "col_name": _col_name,
                "raw_max": raw_max,
                "buffered_max": buffered_max,
                "raw_min": raw_min,
                "buffered_min": buffered_min,
                "raw_range": raw_range,
                "buffered_range": buffered_range,
                "buffer_percent": BUFFER_PERCENT,
            }
        )

        # Convert float NaNs to Polars nulls
        base_col = pl.col(_col_name).fill_nan(None)

        # TODO: Move this code into the other notebook, that will load the scaling params from disk.
        # Avoid division by zero if min == max
        if raw_range == 0.0:
            expr = pl.lit(0, dtype=_DTYPE).alias(_col_name)
        else:
            clipped_col = base_col.clip(lower_bound=buffered_min, upper_bound=buffered_max)
            expr = ((clipped_col - buffered_min) / buffered_range) * (2**8 - 1)
            expr = expr.round().cast(_DTYPE)

        scaling_exprs.append(expr)

    scaling_exprs.append(pl.col("categorical_precipitation_type_surface").round().cast(_DTYPE))
    return scaling_exprs, scaling_params


@app.cell
def _(scaling_params: list[dict]):
    pl.DataFrame(scaling_params).write_csv("ecmwf_scaling_params.csv")
    return


@app.cell
def _(df, scaling_exprs):
    rescaled = (
        df.filter(pl.col.init_time == datetime(2026, 2, 23, 0))
        .with_columns(scaling_exprs)
        .collect(engine="streaming")
    )
    return (rescaled,)


@app.cell
def _(rescaled):
    rescaled
    return


@app.cell
def _(rescaled):
    rescaled.sort(by=["init_time", "lead_time", "ensemble_member", "h3_index"]).write_parquet(
        "uint8_nwp.parquet", statistics="full", compression="zstd", compression_level=14
    )
    return


if __name__ == "__main__":
    app.run()
