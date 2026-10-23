"""Build the network's design matrices, splits, and row groups for experiment 201.

Rows are the half-hours of arm D's daily forecasts at six substations, outside the hand-filtered
switching events and with a reading. The pool for training and early stopping is every row before
the scored year (past-weather rows and ensemble-era tuning and replicate rows); the scored year
(`period == "scored"`) is the test set. Early stopping holds out whole calendar months of the pool
(every fourth month, by the month of the run's origin).

Arms 1 to 4 add inputs cumulatively (arm 0 is the blind control described at `ARMS`):

- Arm 1: arm D's mean, filter level and anomaly, 5-95% width, and lead; the convex model's
  heating hinges at 15.5 C, solar term, darkness, sun elevation, half-hour harmonics, and day type.
- Arm 2: arm 1 plus weekday and day-of-year harmonics and bank-holiday, Christmas, and Economy 7
  flags.
- Arm 3: arm 2 plus the driver temperature, the driver shortwave, and the driver temperature 24
  hours earlier.
- Arm 4: arm 3 plus the power lagged by a whole number of days beyond the lead, and a site
  indicator.
"""

from dataclasses import dataclass
from typing import Final

import e00278_h_common as common
import numpy as np
import polars as pl
from studies.dp_calendar import bank_holiday_names

LONDON: Final[str] = "Europe/London"
BASE_TEMPERATURE_C: Final[float] = common.BASE_THRESHOLD_C
ARMS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
ABLATION_ARMS: Final[tuple[int, ...]] = (5, 6)
"""Post-hoc ablations of arm 1: 5 drops the filter's level and anomaly; 6 drops the driver weather
terms (hinges, solar term, darkness, sun elevation)."""
"""Arm 0 is the blind control (post hoc): lead, calendar, and site only, no forecast or weather."""
DAY_TYPES: Final[tuple[int, ...]] = (0, 1, 1, 1, 2, 3, 4)
"""Day type of Monday (0) to Sunday (6): Monday, Tuesday-Thursday, Friday, Saturday, Sunday."""
VAL_MONTH_STRIDE: Final[int] = 4
COLD_C: Final[float] = 2.0


@dataclass(frozen=True)
class Data:
    """All rows of all substations, with the matrices of each arm.

    Attributes:
        frame: The rows' keys, quantiles, actual, and group flags.
        matrices: The float32 design matrix of each arm, row-aligned with `frame`.
        split: `train`, `val`, or `test` per row.
    """

    frame: pl.DataFrame
    matrices: dict[int, np.ndarray]
    split: np.ndarray


def load_rows() -> pl.DataFrame:
    """Return every site's feature frame with the calendar columns, in one frame."""
    holidays = bank_holiday_names()
    holiday_days = pl.Series(sorted(holidays)).cast(pl.Date)
    frame = pl.concat(
        [
            pl.read_parquet(common.E201_FEATURES / f"{site}.parquet").with_columns(
                site=pl.lit(site)
            )
            for site in common.SCORED_SITES
        ]
    )
    local = pl.col("valid_time").dt.convert_time_zone(LONDON)
    local_date = local.dt.date()
    return (
        frame.filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
        .with_columns(
            local_hh=(local.dt.hour().cast(pl.Int32) * 2 + local.dt.minute().cast(pl.Int32) // 30),
            dow=local.dt.weekday().cast(pl.Int32) - 1,
            doy=local.dt.ordinal_day().cast(pl.Int32),
            bank_holiday=local_date.is_in(holiday_days),
            month_key=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        )
        .with_columns(
            christmas=(local.dt.month() == 12) & (local.dt.day() >= 24)
            | (local.dt.month() == 1) & (local.dt.day() <= 1),
            e7=pl.col("local_hh").is_between(1, 14),
        )
        .sort(["site", "power_fcst_init_time", "valid_time"])
    )


def _harmonics(*, x: np.ndarray, period: float, n: int) -> list[np.ndarray]:
    return [f(2 * np.pi * k * x / period) for k in range(1, n + 1) for f in (np.sin, np.cos)]


def design_columns(*, frame: pl.DataFrame, arm: int) -> np.ndarray:
    """Return the unstandardised float32 matrix of an arm."""
    f = frame
    mean = f["mean"].to_numpy().astype(np.float64)
    temp = f["drv_temp_c"].to_numpy().astype(np.float64)
    sw = f["drv_sw_w_m2"].to_numpy().astype(np.float64)
    elev = f["elevation_deg"].to_numpy().astype(np.float64)
    hh = f["local_hh"].to_numpy().astype(np.float64)
    dow = f["dow"].to_numpy()
    day_type = np.array(DAY_TYPES)[dow]
    if arm == 0:
        return np.column_stack(
            [
                f["lead_hours"].to_numpy(),
                *_harmonics(x=hh, period=48.0, n=2),
                *[(day_type == k).astype(np.float64) for k in range(5)],
                *_harmonics(x=dow.astype(np.float64), period=7.0, n=1),
                *_harmonics(x=f["doy"].to_numpy().astype(np.float64), period=365.25, n=2),
                f["bank_holiday"].to_numpy().astype(np.float64),
                f["christmas"].to_numpy().astype(np.float64),
                f["e7"].to_numpy().astype(np.float64),
                *[(f["site"] == s).to_numpy().astype(np.float64) for s in common.SCORED_SITES],
            ]
        ).astype(np.float32)
    cols = [
        mean,
        f["level"].to_numpy(),
        f["anomaly"].to_numpy(),
        (f["q95"] - f["q05"]).to_numpy(),
        f["lead_hours"].to_numpy(),
        np.maximum(BASE_TEMPERATURE_C - temp, 0.0),
        np.maximum(temp - BASE_TEMPERATURE_C, 0.0),
        sw * np.sin(np.deg2rad(np.maximum(elev, 0.0))) / 1000.0,
        f["dark"].to_numpy().astype(np.float64),
        elev,
        *_harmonics(x=hh, period=48.0, n=2),
        *[(day_type == k).astype(np.float64) for k in range(5)],
    ]
    if arm == 5:
        del cols[1:3]
    if arm == 6:
        del cols[5:10]
    if arm in (2, 3, 4):
        cols += [
            *_harmonics(x=dow.astype(np.float64), period=7.0, n=1),
            *_harmonics(x=f["doy"].to_numpy().astype(np.float64), period=365.25, n=2),
            f["bank_holiday"].to_numpy().astype(np.float64),
            f["christmas"].to_numpy().astype(np.float64),
            f["e7"].to_numpy().astype(np.float64),
        ]
    if arm in (3, 4):
        prev = f["drv_temp_prev_c"].to_numpy().astype(np.float64)
        missing = ~np.isfinite(prev)
        cols += [temp, sw, np.where(missing, temp, prev), missing.astype(np.float64)]
    if arm == 4:
        lag = f["lag_power"].to_numpy().astype(np.float64)
        lag_missing = ~np.isfinite(lag)
        cols += [
            np.where(lag_missing, 0.0, lag),
            lag_missing.astype(np.float64),
            f["lag_hours"].to_numpy().astype(np.float64),
            *[(f["site"] == s).to_numpy().astype(np.float64) for s in common.SCORED_SITES],
        ]
    return np.column_stack(cols).astype(np.float32)


def splits(*, frame: pl.DataFrame) -> np.ndarray:
    """Return `train`, `val`, or `test` per row (see the module docstring)."""
    months = sorted(frame.filter(pl.col("period") != "scored")["month_key"].unique().to_list())
    val_months = {m for i, m in enumerate(months) if i % VAL_MONTH_STRIDE == VAL_MONTH_STRIDE - 1}
    out = np.where(frame["month_key"].is_in(list(val_months)).to_numpy(), "val", "train")
    return np.where((frame["period"] == "scored").to_numpy(), "test", out)


def load_data() -> Data:
    """Return the rows, every arm's matrix, and the split."""
    frame = load_rows()
    return Data(
        frame=frame,
        matrices={arm: design_columns(frame=frame, arm=arm) for arm in (*ARMS, *ABLATION_ARMS)},
        split=splits(frame=frame),
    )
