"""Helpers shared by the battery studies: a calendar replica and a battery's smallest energy.

`calendar_replica` builds a noise-free copy of a series from its own monthly profile, which is the
negative control of the battery studies. `cell_energy_path` and `smallest_capacity` integrate a
battery's settled output into the energy in its cells and find the smallest energy capacity that
holds the path.
"""

import numpy as np
import polars as pl

LOCAL_TIME_ZONE = "Europe/London"
"""The clock that demand follows, so the replica's half-hour of day tracks daylight saving."""


def calendar_replica(*, output: np.ndarray, half_hour_end_time: pl.Series) -> np.ndarray:
    """Return the mean output of each month, half-hour of the day, and day type, on the same grid.

    Months, half-hours, and day types follow UK local time, because demand follows the clock.

    Args:
        output: The output on the grid, one value per half-hour. NaN marks a missing half-hour.
        half_hour_end_time: The UTC end time of each half-hour of `output`.

    Returns:
        A series on the same grid. Each half-hour holds the mean of the finite outputs that share
        its month, local half-hour of the day, and day type. Half-hours with no output stay NaN.
    """
    local = half_hour_end_time.dt.offset_by("-15m").dt.convert_time_zone(LOCAL_TIME_ZONE)
    frame = pl.DataFrame(
        {
            "month": local.dt.month(),
            "half_hour": local.dt.hour() * 2 + local.dt.minute() // 30,
            "weekday": local.dt.weekday(),
            "output": output,
        },
        nan_to_null=True,
    )
    frame = frame.with_columns(
        day_type=pl.when(pl.col("weekday") >= 6).then(pl.col("weekday")).otherwise(0)
    )
    mean = pl.col("output").mean().over("month", "half_hour", "day_type")
    replica = frame.select(pl.when(pl.col("output").is_not_null()).then(mean))["output"]
    return replica.fill_null(float("nan")).to_numpy()


def cell_energy_path(*, output_mwh: np.ndarray, eta: float) -> np.ndarray:
    """Return the cumulative energy added to the cells, in megawatt-hours.

    Args:
        output_mwh: The half-hourly output, positive for export.
        eta: The one-way efficiency.

    Returns:
        The running sum of `-x / eta` for export and `-x * eta` for import.
    """
    step = np.where(output_mwh > 0, -output_mwh / eta, -output_mwh * eta)
    return np.cumsum(step)


def smallest_capacity(*, output_mwh: np.ndarray, eta: float) -> tuple[float, np.ndarray]:
    """Return the smallest capacity that holds the path, and the path that starts it at zero.

    Args:
        output_mwh: The half-hourly output, positive for export.
        eta: The one-way efficiency.

    Returns:
        The capacity `max(c) - min(c)` and the SoC path `c - min(c)`.
    """
    path = cell_energy_path(output_mwh=output_mwh, eta=eta)
    return float(path.max() - path.min()), path - path.min()
