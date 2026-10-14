"""Series, ERA5 weather, and every candidate input column for experiment 00006.

Builds one table per series on the half-hourly, period-ending grid, holding the weather block `W`,
every candidate column the sweep adds, the 3 by 3 block-mean weather, and the target (power over
effective capacity). The table is written once under the experiment's private data folder and read
by `e00006_fit.py`.

    uv run python ml-experiments/00006-stage1a-demand-input-feature-sweep/e00006_inputs.py

The half-hourly resampling reproduces experiment 00003's convention (an experiment may not import
another study's folder, so the function is copied); `check_against_experiment_00003` compares the
result with that experiment's stored ERA5 features.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import holidays
import numpy as np
import polars as pl
from contracts.power_schemas import TimeSeriesMetadata
from studies.power import scan_power
from studies.pv_dataset import nearest_era5_cell
from studies.solar import azimuth, cos_zenith, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR, product_dir_for

EXPERIMENT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00006-stage1a-demand-input-feature-sweep"
"""The private data folder `reserve_experiment` created for this experiment."""

INPUTS_DIR: Final[Path] = EXPERIMENT_DIR / "inputs"
BOX_DIR: Final[Path] = product_dir_for(product="ERA5") / "stage1_box"
METADATA_PATH: Final[Path] = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet"
CAPACITY_PATH: Final[Path] = (
    PER_STUDY_DIR / "quick_wins_fixed_pipeline_1156" / "frames" / "combined" / "fold_a"
) / "capacity.parquet"
"""Effective capacity per series, the denominator experiment 00003 used."""
EXPERIMENT_00003_ERA5: Final[Path] = (
    PER_STUDY_DIR / "stage1_pilot" / "demand" / "inputs" / "era5_features.parquet"
)
"""Experiment 00003's stored ERA5 features, read only to check the resampling."""

TRAIN_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
LABEL_SEED: Final[int] = 6006
"""Seeds the permutation that turns series identifiers into the labels P1 to P16, B1, B2, G1, G2."""

LABEL_PREFIXES: Final[dict[str, str]] = {"Primary": "P", "BSP": "B", "GSP": "G"}
BOX_VARIABLES: Final[tuple[str, ...]] = (
    "t2m",
    "d2m",
    "sp",
    "ssrd",
    "strd",
    "tp",
    "u10",
    "v10",
)
MIN_HOUR_SHAPE: Final[float] = 1e-3
HALF_HOURS_PER_DAY: Final[int] = 48
LOCKDOWN_START: Final[datetime] = datetime(2020, 3, 23, tzinfo=UTC)
LOCKDOWN_END: Final[datetime] = datetime(2021, 7, 19, tzinfo=UTC)
"""The 0/1 lockdown indicator reads 1 from the first to the last of these days, inclusive."""
LOCKDOWN_LEVELS: Final[tuple[tuple[str, str, float], ...]] = (
    ("2020-03-23", "2020-05-12", 1.0),
    ("2020-05-13", "2020-11-04", 0.5),
    ("2020-11-05", "2020-12-01", 1.0),
    ("2020-12-02", "2021-01-05", 0.5),
    ("2021-01-06", "2021-03-07", 1.0),
    ("2021-03-08", "2021-07-18", 0.5),
)
"""First day, last day, and level of the graded lockdown indicator: 1 for a national lockdown, 0.5
for the restrictions between the lockdowns. The levels are approximate dates and judgement, not a
measured series."""
HEATING_BASE_C: Final[float] = 15.5
BLOCK_RADIUS_CELLS: Final[int] = 1
"""The block mean covers the nearest cell and this many cells on each side (3 by 3)."""

W: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
    "solar_zenith",
    "solar_azimuth",
)
"""The weather block, the baseline set every arm adds to."""

BLOCK_SOURCE: Final[tuple[str, ...]] = W[:7]
"""The weather columns the 3 by 3 block mean replaces."""

HOLIDAY_YEARS: Final[range] = range(2018, 2029)


def series_table() -> pl.DataFrame:
    """Return the Primary, BSP, and GSP series with an anonymous label and capacity.

    Returns:
        One row per series with `time_series_id`, `substation_type`, `site` (P1 to P16, B1, B2, G1,
        G2), `latitude`, `longitude`, and `effective_capacity_mw`. The coordinates never leave
        this process.
    """
    metadata = pl.read_parquet(METADATA_PATH).filter(
        pl.col("substation_type").cast(pl.String).is_in(TimeSeriesMetadata.SUBSTATION_TYPES)
    )
    capacity = pl.read_parquet(CAPACITY_PATH).select("time_series_id", "effective_capacity_mw")
    table = (
        metadata.select(
            "time_series_id",
            pl.col("substation_type").cast(pl.String),
            "latitude",
            "longitude",
        )
        .join(capacity, on="time_series_id", how="left", validate="1:1")
        .sort("time_series_id")
    )
    if table["effective_capacity_mw"].null_count():
        msg = "a series has no effective capacity"
        raise ValueError(msg)
    order = np.random.default_rng(LABEL_SEED).permutation(table.height)
    shuffled = table[order.tolist()]
    labelled = shuffled.with_columns(
        site=pl.col("substation_type").replace_strict(LABEL_PREFIXES)
        + (pl.int_range(pl.len()).over("substation_type") + 1).cast(pl.String)
    )
    return labelled.sort("time_series_id")


def _read_box_variable(*, name: str, cells: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.scan_parquet(BOX_DIR / f"{name}.parquet")
        .join(cells.lazy(), on=["latitude", "longitude"], how="inner")
        .select("latitude", "longitude", "time", pl.col("value").alias(name))
        .collect()
    )


def block_cells(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return, for each series, the 3 by 3 block of cells around its nearest cell.

    Args:
        series: `series_table()`.

    Returns:
        One row per `(time_series_id, latitude, longitude)` of a block cell, with `is_nearest`
        true for the nearest cell. A block at the edge of the 11 by 11 box holds fewer cells.
    """
    sample = pl.read_parquet(BOX_DIR / "t2m.parquet", n_rows=121 * 2)
    latitudes = np.sort(sample["latitude"].unique().to_numpy())
    longitudes = np.sort(sample["longitude"].unique().to_numpy())
    nearest = nearest_era5_cell(sites=series, era5=sample)
    rows = []
    for row in nearest.iter_rows(named=True):
        lat_index = int(np.argmin(np.abs(latitudes - row["cell_latitude"])))
        lon_index = int(np.argmin(np.abs(longitudes - row["cell_longitude"])))
        offsets = range(-BLOCK_RADIUS_CELLS, BLOCK_RADIUS_CELLS + 1)
        rows.extend(
            {
                "time_series_id": row["time_series_id"],
                "latitude": float(latitudes[lat_index + i]),
                "longitude": float(longitudes[lon_index + j]),
                "is_nearest": i == 0 and j == 0,
            }
            for i in offsets
            for j in offsets
            if 0 <= lat_index + i < len(latitudes) and 0 <= lon_index + j < len(longitudes)
        )
    return pl.DataFrame(rows).with_columns(pl.col("time_series_id").cast(pl.Int32))


def box_hourly(*, cells: pl.DataFrame) -> pl.DataFrame:
    """Return the box variables at the distinct cells of `cells`, joined on the cell and time."""
    distinct = cells.select("latitude", "longitude").unique()
    frame = _read_box_variable(name=BOX_VARIABLES[0], cells=distinct)
    for name in BOX_VARIABLES[1:]:
        frame = frame.join(
            _read_box_variable(name=name, cells=distinct),
            on=["latitude", "longitude", "time"],
            how="inner",
            validate="1:1",
        )
    return frame


def half_hourly(
    *,
    hourly: pl.DataFrame,
    instants: list[str],
    period: list[str],
    irradiance: list[str],
    latitude: float,
    longitude: float,
) -> pl.DataFrame:
    """Put one cell's hourly ERA5 table onto the half-hourly, period-ending grid.

    A period-ending value labelled `T` is the mean over `(T - 1 h, T]`. The half-hours ending at
    `T - 30 min` and at `T` both take it. An irradiance is spread over those two half-hours by the
    clear-sky shape, so the two values average back to the hour's value. An instant is
    interpolated linearly between its two neighbouring hours. This is experiment 00003's
    convention, copied because an experiment may not import another study's folder.

    Args:
        hourly: Rows sorted by `time` with one cell's hourly values.
        instants: Columns that are instants.
        period: Columns that are period-ending means, held constant within the hour.
        irradiance: Columns that are period-ending irradiances in W m-2.
        latitude: Degrees north of the series, which sets the clear-sky shape.
        longitude: Degrees east of the series.

    Returns:
        Rows at every `:00` and `:30` stamp from the first hour to the last, with `valid_time` and
        every column named above.
    """
    stamps = pl.datetime_range(hourly["time"].min(), hourly["time"].max(), "30m", eager=True)  # ty: ignore[no-matching-overload]
    on_hour = pl.col("valid_time").dt.minute() == 0
    grid = pl.DataFrame({"valid_time": stamps}).with_columns(
        hour_start=pl.col("valid_time").dt.truncate("1h"),
        hour_end=pl.when(on_hour)
        .then(pl.col("valid_time"))
        .otherwise(pl.col("valid_time") + pl.duration(minutes=30)),
        shape=pl.Series(
            cos_zenith(
                zenith_deg=zenith(
                    stamps=stamps.dt.offset_by("-15m"), latitude=latitude, longitude=longitude
                )
            )
        ),
    )
    grid = grid.join(
        grid.group_by("hour_end").agg(hour_shape=pl.col("shape").mean()), on="hour_end", how="left"
    )
    low = hourly.select(
        pl.col("time").alias("hour_start"), *[pl.col(c).alias(f"{c}_lo") for c in instants]
    )
    high = hourly.select(
        (pl.col("time") - pl.duration(hours=1)).alias("hour_start"),
        *[pl.col(c).alias(f"{c}_hi") for c in instants],
    )
    ending = hourly.select(pl.col("time").alias("hour_end"), *period, *irradiance)
    joined = (
        grid.join(low, on="hour_start", how="left")
        .join(high, on="hour_start", how="left")
        .join(ending, on="hour_end", how="left")
    )
    return joined.select(
        "valid_time",
        *[
            pl.when(on_hour)
            .then(pl.col(f"{c}_lo"))
            .otherwise((pl.col(f"{c}_lo") + pl.col(f"{c}_hi")) / 2.0)
            .alias(c)
            for c in instants
        ],
        *period,
        *[
            pl.when(pl.col("hour_shape") > MIN_HOUR_SHAPE)
            .then(pl.col(c) * pl.col("shape") / pl.col("hour_shape"))
            .otherwise(0.0)
            .alias(c)
            for c in irradiance
        ],
    )


def cell_weather(*, cell_hourly: pl.DataFrame, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the seven weather columns of `W` for one cell on the half-hourly grid.

    Args:
        cell_hourly: One cell's rows of `box_hourly`.
        latitude: Degrees north of the series the cell serves.
        longitude: Degrees east of the series the cell serves.

    Returns:
        `valid_time` and the columns `BLOCK_SOURCE`. Shortwave is clipped at zero before use.
    """
    one = cell_hourly.sort("time").select(
        "time",
        (pl.col("t2m") - 273.15).alias("temperature_2m"),
        (pl.col("d2m") - 273.15).alias("dew_point_temperature_2m"),
        pl.col("sp").alias("pressure_surface"),
        "u10",
        "v10",
        (pl.col("ssrd").clip(lower_bound=0.0) / 3600.0).alias("shortwave"),
        (pl.col("strd") / 3600.0).alias("longwave"),
        (pl.col("tp").clip(lower_bound=0.0) * 1000.0 / 3600.0).alias("precipitation"),
    )
    grid = half_hourly(
        hourly=one,
        instants=["temperature_2m", "dew_point_temperature_2m", "pressure_surface", "u10", "v10"],
        period=["longwave", "precipitation"],
        irradiance=["shortwave"],
        latitude=latitude,
        longitude=longitude,
    )
    return grid.with_columns(
        wind_speed_10m=(pl.col("u10") ** 2 + pl.col("v10") ** 2).sqrt()
    ).select("valid_time", *BLOCK_SOURCE)


def holiday_tables() -> tuple[pl.DataFrame, list[str]]:
    """Return one row per calendar date with the holiday columns, and the holiday-name list.

    Returns:
        A frame with `date`, `holiday_name_code` (0 for none, otherwise 1 plus the name's index),
        `is_holiday`, `holiday_day_before`, `holiday_day_after`, `days_to_nearest_holiday`
        (signed, clipped at 7), `days_to_christmas` (signed, clipped at 14), and `bridge_day`; and
        the sorted holiday names. Holidays are England's, from the `holidays` package.
    """
    calendar = holidays.country_holidays("GB", subdiv="ENG", years=HOLIDAY_YEARS)
    names = {d: str(n).removesuffix(" (observed)") for d, n in calendar.items()}
    name_list = sorted(set(names.values()))
    dates = pl.date_range(
        datetime(HOLIDAY_YEARS.start, 1, 1, tzinfo=UTC).date(),
        datetime(HOLIDAY_YEARS.stop - 1, 12, 31, tzinfo=UTC).date(),
        eager=True,
    )
    frame = pl.DataFrame({"date": dates}).with_columns(
        holiday_name_code=pl.col("date").map_elements(
            lambda d: name_list.index(names[d]) + 1 if d in names else 0, return_dtype=pl.Int32
        )
    )
    frame = frame.with_columns(is_holiday=pl.col("holiday_name_code") > 0)
    holiday_dates = frame.filter(pl.col("is_holiday"))["date"].to_numpy()
    days = frame["date"].to_numpy()
    gaps = (holiday_dates[None, :] - days[:, None]).astype("timedelta64[D]").astype(int)
    nearest = gaps[np.arange(len(days)), np.abs(gaps).argmin(axis=1)]
    christmas = np.array(
        [np.datetime64(f"{y}-12-25") for y in range(HOLIDAY_YEARS.start, HOLIDAY_YEARS.stop)]
    )
    to_christmas = (christmas[None, :] - days[:, None]).astype("timedelta64[D]").astype(int)
    nearest_christmas = to_christmas[np.arange(len(days)), np.abs(to_christmas).argmin(axis=1)]
    frame = frame.with_columns(
        days_to_nearest_holiday=pl.Series(np.clip(nearest, -7, 7)).cast(pl.Int32),
        days_to_christmas=pl.Series(np.clip(nearest_christmas, -14, 14)).cast(pl.Int32),
        weekday=pl.col("date").dt.weekday(),
    ).with_columns(
        holiday_day_before=pl.col("is_holiday").shift(-1, fill_value=False),
        holiday_day_after=pl.col("is_holiday").shift(1, fill_value=False),
    )
    frame = frame.with_columns(
        bridge_day=(
            ~pl.col("is_holiday")
            & (
                ((pl.col("weekday") == 1) & pl.col("is_holiday").shift(-1, fill_value=False))
                | ((pl.col("weekday") == 5) & pl.col("is_holiday").shift(1, fill_value=False))
            )
        )
    ).drop("weekday")
    return frame, name_list


def sun_columns(*, valid_time: pl.Series, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the sun-position columns at the midpoint of the half-hour ending at each stamp.

    Args:
        valid_time: Period-ending UTC stamps.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        `solar_zenith`, `solar_azimuth` (degrees), `azimuth_sin`, `azimuth_cos`, `day_length_h`
        (approximate, from the declination formula), and `hours_since_sunrise` (solar time less the
        sunrise time, negative before dawn, ignoring the equation of time).
    """
    midpoint = valid_time.dt.offset_by("-15m")
    zen = zenith(stamps=midpoint, latitude=latitude, longitude=longitude)
    azi = azimuth(stamps=midpoint, latitude=latitude, longitude=longitude)
    day = midpoint.dt.ordinal_day().to_numpy().astype(np.float64)
    declination = np.radians(23.44) * np.sin(2.0 * np.pi * (284.0 + day) / 365.0)
    cos_hour_angle = np.clip(-np.tan(np.radians(latitude)) * np.tan(declination), -1.0, 1.0)
    length = 24.0 / np.pi * np.arccos(cos_hour_angle)
    utc_hours = (midpoint.dt.hour().to_numpy() + midpoint.dt.minute().to_numpy() / 60.0).astype(
        np.float64
    )
    solar_time = (utc_hours + longitude / 15.0) % 24.0
    return pl.DataFrame(
        {
            "solar_zenith": zen,
            "solar_azimuth": azi,
            "azimuth_sin": np.sin(np.radians(azi)),
            "azimuth_cos": np.cos(np.radians(azi)),
            "day_length_h": length,
            "hours_since_sunrise": solar_time - (12.0 - length / 2.0),
        }
    )


def windchill_celsius(*, temperature_c: pl.Expr, wind_m_s: pl.Expr) -> pl.Expr:
    """Return the Environment Canada windchill, equal to the temperature where it is undefined.

    The formula applies at or below 10 °C and above 4.8 km/h of wind. Experiment 00006 implements it
    here only.
    """
    speed = wind_m_s * 3.6
    factor = speed.pow(0.16)
    chill = 13.12 + 0.6215 * temperature_c - 11.37 * factor + 0.3965 * temperature_c * factor
    return pl.when((temperature_c <= 10.0) & (speed > 4.8)).then(chill).otherwise(temperature_c)


def _graded_lockdown(*, date: pl.Expr) -> pl.Expr:
    """Return the graded lockdown level of each local date: 0 outside the listed periods."""
    level = pl.lit(0.0)
    for first, last, value in LOCKDOWN_LEVELS:
        level = (
            pl.when(date.is_between(pl.lit(first).str.to_date(), pl.lit(last).str.to_date()))
            .then(value)
            .otherwise(level)
        )
    return level


def _trailing_columns() -> list[pl.Expr]:
    """Return the trailing-temperature columns, computed per series in time order.

    The window ends at the valid time and covers earlier half-hours only.
    """
    half_hours = {"24h": 48, "72h": 144, "7d": 336}
    degree_hours = (HEATING_BASE_C - pl.col("temperature_2m")).clip(lower_bound=0.0)
    columns = [
        pl.col("temperature_2m")
        .rolling_mean(window_size=size, min_samples=size)
        .alias(f"temperature_mean_{label}")
        for label, size in half_hours.items()
    ]
    columns += [
        pl.col("temperature_2m")
        .ewm_mean(half_life=hours * 2.0, adjust=False)
        .alias(f"ewm_t_{name}")
        for name, hours in {"12h": 12, "72h": 72, "7d": 168}.items()
    ]
    columns += [
        degree_hours.rolling_mean(window_size=size, min_samples=size).alias(f"heating_dh_{label}")
        for label, size in {"24h": 48, "72h": 144}.items()
    ]
    return columns


def candidate_columns(*, frame: pl.DataFrame, latitude: float, longitude: float) -> pl.DataFrame:
    """Add every candidate input column to one series' weather frame.

    Args:
        frame: One series' half-hourly rows sorted by `valid_time`, with `W` less the sun columns.
        latitude: Degrees north of the series.
        longitude: Degrees east of the series.

    Returns:
        `frame` with the sun, clock, time-of-year, holiday, trailing-temperature, wind, and
        lockdown columns added. Each is a function of the calendar and of weather at or before the
        valid time, so each is computable at forecast time.
    """
    sun = sun_columns(valid_time=frame["valid_time"], latitude=latitude, longitude=longitude)
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    holiday_frame, _ = holiday_tables()
    day_of_year = local.dt.ordinal_day().cast(pl.Float64)
    angle = 2.0 * np.pi * day_of_year / 365.25
    out = (
        pl.concat([frame, sun], how="horizontal")
        .with_columns(
            (local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60.0).alias(
                "hour_local"
            ),
            local.dt.weekday().cast(pl.Float64).alias("day_of_week_local"),
            (local.dt.dst_offset().dt.total_hours().cast(pl.Float64)).alias("utc_offset_h"),
            angle.sin().alias("doy_sin_1"),
            angle.cos().alias("doy_cos_1"),
            (2.0 * angle).sin().alias("doy_sin_2"),
            (2.0 * angle).cos().alias("doy_cos_2"),
            local.dt.date().alias("date"),
            (
                (pl.col("valid_time") >= LOCKDOWN_START)
                & (pl.col("valid_time") < LOCKDOWN_END + timedelta(days=1))
            )
            .cast(pl.Float64)
            .alias("covid_indicator"),
            _graded_lockdown(date=local.dt.date()).alias("covid_graded"),
        )
        .with_columns(*_trailing_columns())
        .join(holiday_frame, on="date", how="left")
        .drop("date")
    )
    return out.with_columns(
        windchill_c=windchill_celsius(
            temperature_c=pl.col("temperature_2m"), wind_m_s=pl.col("wind_speed_10m")
        ),
        wind_heat_loss=pl.col("wind_speed_10m")
        * (HEATING_BASE_C - pl.col("temperature_2m")).clip(lower_bound=0.0),
    )


def build_series_features(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return every series' candidate columns, block-mean weather, and target.

    Args:
        series: `series_table()`.

    Returns:
        One row per `(time_series_id, valid_time)` from the first ERA5 hour to the last, with `W`,
        the candidates, `<column>_block` for the seven `BLOCK_SOURCE` columns, `power_mw`, and
        `target` (power over effective capacity), null where power was not observed.
    """
    cells = block_cells(series=series)
    hourly = box_hourly(cells=cells)
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(series["time_series_id"].to_list()),
            pl.col("time") >= TRAIN_START,
        )
        .select(
            "time_series_id", pl.col("time").alias("valid_time"), pl.col("power").alias("power_mw")
        )
        .collect()
        .filter(pl.col("power_mw").is_finite())
    )
    pieces = []
    for row in series.iter_rows(named=True):
        members = cells.filter(pl.col("time_series_id") == row["time_series_id"])
        per_cell = []
        for member in members.iter_rows(named=True):
            one = hourly.filter(
                (pl.col("latitude") == member["latitude"])
                & (pl.col("longitude") == member["longitude"])
            )
            weather = cell_weather(
                cell_hourly=one, latitude=row["latitude"], longitude=row["longitude"]
            )
            per_cell.append((member["is_nearest"], weather))
        nearest = next(w for is_nearest, w in per_cell if is_nearest)
        block = (
            pl.concat([w for _, w in per_cell])
            .group_by("valid_time")
            .agg(*[pl.col(c).mean().alias(f"{c}_block") for c in BLOCK_SOURCE])
        )
        frame = nearest.join(block, on="valid_time", how="left", validate="1:1").sort("valid_time")
        frame = candidate_columns(frame=frame, latitude=row["latitude"], longitude=row["longitude"])
        frame = frame.with_columns(time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32))
        pieces.append(frame)
    features = pl.concat(pieces)
    return (
        features.join(
            power.with_columns(pl.col("time_series_id").cast(pl.Int32)),
            on=["time_series_id", "valid_time"],
            how="left",
            validate="1:1",
        )
        .join(
            series.select("time_series_id", "effective_capacity_mw"),
            on="time_series_id",
            how="left",
        )
        .with_columns(target=pl.col("power_mw") / pl.col("effective_capacity_mw"))
        .sort("time_series_id", "valid_time")
    )


def check_against_experiment_00003(*, features: pl.DataFrame, series: pl.DataFrame) -> list[str]:
    """Compare the nearest-cell weather with experiment 00003's stored features.

    Args:
        features: `build_series_features` output.
        series: `series_table()`.

    Returns:
        Report lines giving, for each shared column, the largest absolute difference over the
        series that experiment 00003 also holds.

    Raises:
        ValueError: If no series is shared or a column differs by more than 1e-3.
    """
    if not EXPERIMENT_00003_ERA5.exists():
        return ["Experiment 00003's stored features are not on disk, so no comparison was made."]
    stored = pl.read_parquet(EXPERIMENT_00003_ERA5).rename(
        {
            "temperature_2m": "temperature_2m_00003",
            "dew_point_temperature_2m": "dew_point_temperature_2m_00003",
            "shortwave": "shortwave_00003",
            "longwave": "longwave_00003",
            "pressure_surface": "pressure_surface_00003",
            "precipitation": "precipitation_00003",
            "wind_speed_10m": "wind_speed_10m_00003",
        }
    )
    joined = features.join(stored, on=["time_series_id", "valid_time"], how="inner")
    if joined.is_empty():
        msg = "no series is shared with experiment 00003"
        raise ValueError(msg)
    lines = [f"Compared {joined.height:,} rows of {joined['time_series_id'].n_unique()} series."]
    for column in BLOCK_SOURCE:
        gap = float((joined[column] - joined[f"{column}_00003"]).abs().max())  # ty: ignore[invalid-argument-type]
        lines.append(f"- `{column}`: largest absolute difference {gap:.3g}.")
        if gap > 1e-3:
            msg = f"{column} differs from experiment 00003 by {gap}"
            raise ValueError(msg)
    del series
    return lines


def main() -> None:
    """Build the feature table and run the checks, skipping the build if the file exists."""
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = INPUTS_DIR / "features.parquet"
    series = series_table()
    series.write_parquet(INPUTS_DIR / "series.parquet")
    if not path.exists():
        features = build_series_features(series=series)
        features.write_parquet(path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
    features = pl.read_parquet(path)
    print("\n".join(check_against_experiment_00003(features=features, series=series)))
    print(features.height, "rows;", features["target"].drop_nulls().len(), "with power")


if __name__ == "__main__":
    main()
