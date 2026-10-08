"""Load NGED battery A without ever naming it.

NGED battery A is the one series in NGED's telemetry whose name marks it as a battery. NGED's list
of time series holds exactly one series whose name matches battery, battery energy storage system
(BESS), or storage. The case study is not on the battery study page: its figures and report wait
for the follow-up studies on forecasting embedded batteries and on sizing an unmetered battery.

The lookup runs here, in code, each time a script needs the series, and nothing the lookup finds
(name, series ID, substation number, coordinates, capacity) is printed, logged, or written. A script
reports only
aggregates under the alias "NGED battery A", and shows output as a fraction of the series' own 99th
percentile absolute output, so no megawatt value identifies the site.

NGED stamps each reading with the end of its half-hour (`PowerTimeSeries.correct_late_timestamps`
already repaired the stamps at ingest, so this module takes them at face value). The price files
stamp each period at its start, so `battery_a_output` moves the stamp back 30 minutes.
"""

from datetime import timedelta
from typing import Final

import numpy as np
import polars as pl

from studies.battery_market import (
    FOLD_MONTHS,
    HALF_HOURS_PER_DAY,
    MINUTES_PER_HALF_HOUR,
    WINDOW_START,
)
from studies.power import scan_power
from studies.sources import MARKET_DOWNLOADS_DIR, REANALYSIS_DOWNLOADS_DIR, REPO_DATA_DIR

ALIAS: Final[str] = "NGED battery A"
NAME_PATTERN: Final[str] = "(?i)battery|bess|storage"
"""The case-insensitive pattern the search of series names uses."""
WINDOW_DAYS: Final[int] = 365
"""The window is the public batteries' year, so the numbers compare."""
CAMS_PUBLIC_POINTS_PATH: Final = (
    REANALYSIS_DOWNLOADS_DIR / "CAMS_public_points" / ("cams_public_points.parquet")
)


def _battery_row() -> dict[str, object]:
    """Return the row of NGED's list of time series for the one matching series.

    Returns:
        The row of NGED's list of time series, as a dictionary.

    Raises:
        ValueError: If the search does not match exactly one series. The message states the count
            only.
    """
    metadata = pl.read_parquet(REPO_DATA_DIR / "NGED" / "metadata.parquet")
    hits = metadata.filter(pl.col("time_series_name").str.contains(NAME_PATTERN))
    if hits.height != 1:
        msg = f"The metadata search matched {hits.height} series, not 1."
        raise ValueError(msg)
    return hits.row(0, named=True)


def battery_a_site() -> dict[str, object]:
    """Return NGED battery A's name, position and unit, for matching it against public registers.

    The caller must keep every value out of anything printed, logged, or written. A script reports
    only whether a match was found.

    Returns:
        The keys `name`, `latitude`, `longitude`, and `units` of the series' metadata row.
    """
    row = _battery_row()
    return {
        "name": str(row["time_series_name"]),
        "latitude": float(row["latitude"]),  # ty: ignore[invalid-argument-type]
        "longitude": float(row["longitude"]),  # ty: ignore[invalid-argument-type]
        "units": str(row["units"]),
    }


def battery_a_units() -> str:
    """Return the series' unit, `MW` or `MVA`."""
    return str(_battery_row()["units"])


def battery_a_raw() -> pl.DataFrame:
    """Return every cleaned reading of the series on period starts.

    Returns:
        Columns `time` (the period start, UTC) and `power` (as metered, Float64).
    """
    time_series_id = _battery_row()["time_series_id"]
    return (
        scan_power()
        .filter(pl.col("time_series_id") == time_series_id)
        .select(
            time=pl.col("time").dt.cast_time_unit("us") - timedelta(minutes=MINUTES_PER_HALF_HOUR),
            power=pl.col("power").cast(pl.Float64),
        )
        .sort("time")
        .collect()
    )


def window_filter() -> pl.Expr:
    """Return the expression that keeps the study's 365-day window."""
    return (pl.col("time") >= WINDOW_START) & (
        pl.col("time") < WINDOW_START + timedelta(days=WINDOW_DAYS)
    )


def market_full() -> pl.DataFrame:
    """Return the day-ahead price and the system price on one half-hourly grid, unfiltered.

    Returns:
        Columns `time` (period start), `day_ahead_gbp_per_mwh` and `system_price_gbp_per_mwh`.
    """
    base = MARKET_DOWNLOADS_DIR
    n2ex = pl.read_parquet(base / "neso_n2ex_day_ahead" / "neso_n2ex_day_ahead.parquet").select(
        hour=pl.col("time"), day_ahead_gbp_per_mwh=pl.col("price_gbp_per_mwh")
    )
    system = pl.read_parquet(base / "elexon_system_prices" / "elexon_system_prices.parquet").select(
        "time", system_price_gbp_per_mwh=pl.col("system_sell_price_gbp_per_mwh")
    )
    return (
        system.with_columns(hour=pl.col("time").dt.truncate("1h"))
        .join(n2ex, on="hour", how="left")
        .drop("hour")
        .sort("time")
    )


def battery_a_frame(*, sign: float = 1.0) -> tuple[pl.DataFrame, float]:
    """Return the series in the shape the public batteries' rung 2 and 3 code expects.

    Args:
        sign: Multiplies the metered power, so -1 flips a series metered as import-positive.

    Returns:
        The frame, and the 99th percentile absolute output in the metered unit. `output_mw` and
        `output_mwh` are expressed as fractions of that percentile, so the frame carries no
        capacity. The columns are the ones `battery_frame` of the battery study returns: `time`, the
        outputs, the day-ahead price, `date`, `tod`, `month`, `rank_pct`, `fold`, and `tercile`.
    """
    raw = battery_a_raw().filter(window_filter())
    scale = float(np.quantile(raw["power"].abs().to_numpy(), 0.99))
    frame = (
        raw.select(
            "time",
            output_mw=pl.col("power") * sign / scale,
            output_mwh=pl.col("power") * sign / scale / 2.0,
        )
        .join(market_full(), on="time", how="inner")
        .drop_nulls("day_ahead_gbp_per_mwh")
        .with_columns(
            date=pl.col("time").dt.date(),
            tod=pl.col("time").dt.hour().cast(pl.Int32) * 2 + pl.col("time").dt.minute() // 30,
            month=pl.col("time").dt.strftime("%Y-%m"),
            calendar_month=pl.col("time").dt.month(),
        )
        .with_columns(
            rank_pct=(
                (pl.col("day_ahead_gbp_per_mwh").rank("average").over("date") - 0.5)
                / HALF_HOURS_PER_DAY
            ),
            fold=pl.col("calendar_month").replace_strict(FOLD_MONTHS, return_dtype=pl.Int64),
        )
        .with_columns(tercile=(pl.col("rank_pct") * 3).floor().cast(pl.Int64).clip(0, 2))
        .drop("calendar_month")
        .sort("time")
    )
    return frame, scale


def nearest_cams_hourly() -> pl.DataFrame:
    """Return the CAMS irradiance at the public grid point nearest the series.

    Returns:
        Columns `hour_start` (UTC), `ghi_w_m2`, and `clear_sky_ghi_w_m2`.
    """
    row = _battery_row()
    cams = pl.read_parquet(CAMS_PUBLIC_POINTS_PATH)
    grid = (
        cams.filter(pl.col("point_id").str.starts_with("gb_"))
        .select("point_id", "latitude", "longitude")
        .unique()
    )
    distance = (
        (grid["latitude"] - float(row["latitude"])) ** 2  # ty: ignore[invalid-argument-type]
        + ((grid["longitude"] - float(row["longitude"])) * 0.6) ** 2  # ty: ignore[invalid-argument-type]
    )
    nearest = grid["point_id"][int(distance.arg_min())]  # ty: ignore[invalid-argument-type]
    return (
        cams.filter(pl.col("point_id") == nearest)
        .select(
            hour_start=pl.col("time").dt.cast_time_unit("us") - timedelta(hours=1),
            ghi_w_m2=pl.col("ghi_w_m2").cast(pl.Float64),
            clear_sky_ghi_w_m2=pl.col("clear_sky_ghi_w_m2").cast(pl.Float64),
        )
        .sort("hour_start")
    )
