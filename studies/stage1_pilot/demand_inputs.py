"""Series, paths, feature columns, and ERA5 weather for the demand experiment (00003).

The demand experiment repeats the stage 1 pilot for the 22 series of the fold A frames that are
not wind or solar farms. This module names those series (labelled D1 to D22), the weather columns
every arm shares, the variants that add calendar and lag columns, and builds the ERA5 inputs from
the stage 1 box on the same half-hourly convention as the ENS rows.
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import stage1_inputs as inputs
from studies.pv_dataset import nearest_era5_cell
from studies.sources import product_dir_for

OUTPUT_ROOT: Final[Path] = inputs.study_subfolder(name="demand")
"""Where this experiment writes every table, prediction file, and report."""


def demand_subfolder(*, name: str) -> Path:
    """Return one of the experiment's output folders, such as `inputs` or `scores`.

    Args:
        name: The folder's name under the experiment's output root.

    Returns:
        The folder's path.
    """
    return OUTPUT_ROOT / name


INPUTS_DIR: Final[Path] = demand_subfolder(name="inputs")
BOX_DIR: Final[Path] = product_dir_for(product="ERA5") / "stage1_box"
"""ERA5 hourly weather on the 121-cell stage 1 box, one file per variable."""

LABEL_SEED: Final[int] = 1013
"""Seeds the permutation that turns series identifiers into the labels D1 to D22."""

GROUPS: Final[dict[str, str]] = {
    "Disaggregated Demand": "primary demand",
    "Raw Flow": "grid flow",
    "BESS": "battery or biofuel",
    "Biofuel": "battery or biofuel",
}
"""The metadata's series type mapped to the three groups the results report."""

WEATHER: Final[tuple[str, ...]] = (
    "temperature_2m",
    "dew_point_temperature_2m",
    "shortwave",
    "longwave",
    "pressure_surface",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_speed_100m",
    "precipitation_type",
)
"""The weather columns of every arm."""

CALENDAR: Final[tuple[str, ...]] = ("hour_local", "day_of_week_local")
LAGS: Final[tuple[str, ...]] = (
    "power_lag_24h",
    "power_lag_48h",
    "power_lag_168h",
    "power_lag_336h",
)
LAG_HOURS: Final[dict[str, float]] = {
    "power_lag_24h": 24.0,
    "power_lag_48h": 48.0,
    "power_lag_168h": 168.0,
    "power_lag_336h": 336.0,
}
VARIANTS: Final[dict[str, tuple[str, ...]]] = {
    "V0": WEATHER,
    "V1": (*WEATHER, *CALENDAR),
    "V2": (*WEATHER, *CALENDAR, *LAGS),
}
"""Each variant's columns: weather only; plus calendar; plus calendar and power lags."""

ENS_COLUMNS: Final[dict[str, str]] = {
    "temperature_2m": "temperature_2m",
    "dew_point_temperature_2m": "dew_point_temperature_2m",
    "shortwave": "downward_short_wave_radiation_flux_surface",
    "longwave": "downward_long_wave_radiation_flux_surface",
    "pressure_surface": "pressure_surface",
    "pressure_msl": "pressure_reduced_to_mean_sea_level",
    "precipitation": "precipitation_surface",
    "wind_speed_10m": "wind_speed_10m",
    "wind_speed_100m": "wind_speed_100m",
    "precipitation_type": "categorical_precipitation_type_surface",
}
"""Which ENS frame column fills each weather column."""

MAX_LEAD_HOURS: Final[float] = 351.0
MIN_LEAD_HOURS: Final[float] = 0.5

BOX_VARIABLES: Final[tuple[str, ...]] = (
    "t2m",
    "d2m",
    "sp",
    "msl",
    "ssrd",
    "strd",
    "tp",
    "u10",
    "v10",
    "u100",
    "v100",
    "ptype",
)


def series_table() -> pl.DataFrame:
    """Return the 22 non-generator series with label, group, capacity, and coordinates.

    Returns:
        One row per series with `time_series_id`, `site` (D1 to D22), `group`, `latitude`,
        `longitude`, `h3_res_5`, and `effective_capacity_mw`. The coordinates never leave this
        process.
    """
    metadata = pl.read_parquet(inputs.FRAMES_DIR / "metadata.parquet").filter(
        ~pl.col("time_series_type").is_in(["PV", "Wind"])
    )
    capacity = pl.read_parquet(inputs.FRAMES_DIR / "capacity.parquet").select(
        "time_series_id", "effective_capacity_mw"
    )
    table = (
        metadata.select(
            "time_series_id",
            pl.col("time_series_type").cast(pl.String).replace_strict(GROUPS).alias("group"),
            "latitude",
            "longitude",
            "h3_res_5",
        )
        .join(capacity, on="time_series_id", how="left", validate="1:1")
        .sort("time_series_id")
    )
    if table.height != 22:
        msg = f"expected 22 series, found {table.height}"
        raise ValueError(msg)
    order = np.random.default_rng(LABEL_SEED).permutation(table.height)
    labels = [f"D{i + 1}" for i in range(table.height)]
    return table.with_columns(site=pl.Series([labels[i] for i in order])).sort(
        pl.col("site").str.slice(1).cast(pl.Int32)
    )


def _read_box_variable(*, name: str, cells: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.scan_parquet(BOX_DIR / f"{name}.parquet")
        .join(cells.lazy(), on=["latitude", "longitude"], how="inner")
        .select("latitude", "longitude", "time", pl.col("value").alias(name))
        .collect()
    )


def era5_hourly(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return ERA5 hourly weather at each series' nearest cell of the stage 1 box.

    Args:
        series: `series_table()`.

    Returns:
        One row per `(time_series_id, time)` with the box variables in ARCO's units, as
        `t2m` ... `ptype`. Accumulations are totals over the hour ending at `time`.
    """
    sample = pl.read_parquet(BOX_DIR / "t2m.parquet", n_rows=121 * 2)
    assigned = nearest_era5_cell(sites=series, era5=sample).select(
        "time_series_id",
        pl.col("cell_latitude").alias("latitude"),
        pl.col("cell_longitude").alias("longitude"),
    )
    cells = assigned.select("latitude", "longitude").unique()
    frame = _read_box_variable(name=BOX_VARIABLES[0], cells=cells)
    for name in BOX_VARIABLES[1:]:
        frame = frame.join(
            _read_box_variable(name=name, cells=cells),
            on=["latitude", "longitude", "time"],
            how="inner",
            validate="1:1",
        )
    return assigned.join(frame, on=["latitude", "longitude"], how="inner").sort(
        "time_series_id", "time"
    )


def era5_features(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the weather columns from ERA5 for every series at every half-hour.

    Radiation, precipitation, and the precipitation-type code are hour-ending values held over the
    two half-hours of the hour (radiation is spread by the clear-sky shape); the others are
    instants, interpolated linearly between hours. Shortwave is clipped at zero before use.

    Args:
        series: `series_table()`.

    Returns:
        One row per `(time_series_id, valid_time)` with `WEATHER` columns.
    """
    hourly = era5_hourly(series=series)
    pieces = []
    for row in series.iter_rows(named=True):
        one = (
            hourly.filter(pl.col("time_series_id") == row["time_series_id"])
            .select(
                "time",
                (pl.col("t2m") - 273.15).alias("temperature_2m"),
                (pl.col("d2m") - 273.15).alias("dew_point_temperature_2m"),
                pl.col("sp").alias("pressure_surface"),
                pl.col("msl").alias("pressure_msl"),
                "u10",
                "v10",
                "u100",
                "v100",
                (pl.col("ssrd").clip(lower_bound=0.0) / 3600.0).alias("shortwave"),
                (pl.col("strd") / 3600.0).alias("longwave"),
                (pl.col("tp").clip(lower_bound=0.0) * 1000.0 / 3600.0).alias("precipitation"),
                pl.col("ptype").round().alias("precipitation_type"),
            )
            .sort("time")
        )
        grid = inputs.half_hourly(
            hourly=one,
            instants=[
                "temperature_2m",
                "dew_point_temperature_2m",
                "pressure_surface",
                "pressure_msl",
                "u10",
                "v10",
                "u100",
                "v100",
            ],
            period=["longwave", "precipitation", "precipitation_type"],
            irradiance=["shortwave"],
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        pieces.append(
            grid.with_columns(
                wind_speed_10m=(pl.col("u10") ** 2 + pl.col("v10") ** 2).sqrt(),
                wind_speed_100m=(pl.col("u100") ** 2 + pl.col("v100") ** 2).sqrt(),
                time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32),
            ).drop("u10", "v10", "u100", "v100")
        )
    return pl.concat(pieces).sort("time_series_id", "valid_time")


def calendar_columns() -> list[pl.Expr]:
    """Return the local hour of day (fractional) and the local day of week (Monday is 1)."""
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    return [
        (local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60.0).alias(
            "hour_local"
        ),
        local.dt.weekday().cast(pl.Float64).alias("day_of_week_local"),
    ]


def ens_features(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the weather, calendar, and lag columns of a fold A frame's rows.

    Args:
        frame: A fold A frame (`train.parquet` or one validation run) for the demand series.

    Returns:
        The key columns, `power`, `WEATHER`, `CALENDAR`, and `LAGS`, with the lags as the frame
        has them (null where the lag is no longer than the lead).
    """
    return frame.select(
        *inputs.KEY_COLUMNS,
        "power",
        *[pl.col(source).alias(name) for name, source in ENS_COLUMNS.items()],
        *LAGS,
    ).with_columns(*calendar_columns())
