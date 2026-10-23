"""The winter night offset of experiment 155, applied to any base forecast frame.

The offset is a shift added to the base frame's mean and quantiles, then the spread is refitted.
`add_calendar` adds the columns that the offset reads. `winter_offset_shift` is the night term of
experiment 137 with a winter indicator in place of the cold depth. The offset reads nothing dated
on or after the origin of the run it corrects.
"""

from typing import Final

import e00281_c155 as common
import numpy as np
import polars as pl
from studies.dp_ens import LeadHourSpread

LONDON: Final[str] = "Europe/London"
N_COLUMNS: Final[int] = common.N_NIGHT_SLOTS * 3
VARIANTS: Final[tuple[str, ...]] = ("winter", "const", "placebo")
"""`winter`: constant, winter indicator, weekend. `const`: no winter indicator. `placebo`: the
winter indicator moved six months (May to September)."""


def add_calendar(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the columns the corrections read.

    Args:
        frame: A forecast frame with `valid_time` and `power_fcst_init_time`.

    Returns:
        The frame with `origin_day` (days since 1970-01-01 of the origin's UTC date), `local_hh`
        (the local half-hour of day, 0 to 47, from the valid time in Europe/London), `slot` (0 to
        23 for the night half-hours that end between 21:30 and 09:00 UTC, else -1), `winter`,
        `placebo_winter`, and `weekend` (flags of the evening's date).
    """
    minute = pl.col("valid_time").dt.hour().cast(pl.Int32) * 60 + pl.col(
        "valid_time"
    ).dt.minute().cast(pl.Int32)
    slot = ((minute - common.NIGHT_START_MINUTE) % 1440) // 30
    evening = (pl.col("valid_time") - pl.duration(hours=21, minutes=30)).dt.date()
    local = pl.col("valid_time").dt.convert_time_zone(LONDON)
    return frame.with_columns(
        origin_day=pl.col("power_fcst_init_time").dt.date().cast(pl.Int32),
        local_hh=(local.dt.hour().cast(pl.Int32) * 2 + local.dt.minute().cast(pl.Int32) // 30),
        slot=pl.when(slot < common.N_NIGHT_SLOTS).then(slot).otherwise(-1).cast(pl.Int32),
        evening=evening,
    ).with_columns(
        winter=evening.dt.month().is_in(list(common.WINTER_MONTHS)).cast(pl.Float64),
        placebo_winter=((evening.dt.month() + 5) % 12 + 1)
        .is_in(list(common.WINTER_MONTHS))
        .cast(pl.Float64),
        weekend=(evening.dt.weekday() >= 6).cast(pl.Float64),
    )


def night_design(*, frame: pl.DataFrame, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the offset's design matrix on the night rows, and the mask of those rows.

    Each night slot has three columns: a constant, a second flag (the winter indicator, zero for
    `const`, or the placebo indicator), and a weekend flag.
    """
    slot = frame["slot"].to_numpy()
    keep = slot >= 0
    second = {
        "winter": frame["winter"].to_numpy(),
        "const": np.zeros(frame.height),
        "placebo": frame["placebo_winter"].to_numpy(),
    }[variant]
    weekend = frame["weekend"].to_numpy()
    rows = np.flatnonzero(keep)
    x = np.zeros((len(rows), N_COLUMNS))
    s = slot[rows]
    x[np.arange(len(rows)), 3 * s] = 1.0
    x[np.arange(len(rows)), 3 * s + 1] = second[rows]
    x[np.arange(len(rows)), 3 * s + 2] = weekend[rows]
    return x, keep


def winter_offset_shift(
    *, frame: pl.DataFrame, variant: str = "winter", ridge: float = common.OFFSET_RIDGE
) -> tuple[np.ndarray, dict[object, np.ndarray]]:
    """Return the offset's shift at every row, and its coefficients at each monthly cut.

    The coefficients of the cut at the first of a month are a ridge regression of the forecast's
    own errors (actual minus mean) on the night rows whose run origin and valid time are both
    before that cut (an expanding window). The shift applies to the runs whose origin is in the
    month. A month with no training rows gets a shift of zero.

    Args:
        frame: The forecast frame, with the columns of `add_calendar`.
        variant: `winter`, `const`, or `placebo`.
        ridge: The ridge penalty.

    Returns:
        The shift per row, and the coefficients per cut.
    """
    x, keep = night_design(frame=frame, variant=variant)
    night = frame.filter(pl.Series(keep))
    error = (night["actual"] - night["mean"]).to_numpy().astype(np.float64)
    finite = np.isfinite(error)
    origin = night["power_fcst_init_time"].to_numpy()
    valid = night["valid_time"].to_numpy()
    month = night["power_fcst_init_time"].dt.truncate("1mo").to_numpy()
    shift_night = np.zeros(night.height)
    betas: dict[object, np.ndarray] = {}
    for cut in np.unique(month):
        train = finite & (origin < cut) & (valid < cut)
        if not train.any():
            continue
        beta = np.linalg.solve(
            x[train].T @ x[train] + ridge * np.eye(N_COLUMNS), x[train].T @ error[train]
        )
        betas[cut] = beta
        this = month == cut
        shift_night[this] = x[this] @ beta
    shift = np.zeros(frame.height)
    shift[np.flatnonzero(keep)] = shift_night
    return shift, betas


def apply_shift_and_refit_spread(*, frame: pl.DataFrame, shift: np.ndarray) -> pl.DataFrame:
    """Return the frame with `shift` added to the mean and the spread refitted on the tuning runs.

    The quantiles are the corrected mean plus the centred residual table (hour band, calendar
    quarter, and lead band) of the corrected forecasts of the tuning runs outside the hand-filtered
    switching events, every spread factor 1, as experiment 136 builds them.

    Args:
        frame: The base forecast frame, with the `period` column.
        shift: The shift per row, in points.

    Returns:
        The corrected frame, with the same columns as `frame`.
    """
    mean = frame["mean"].to_numpy().astype(np.float64) + shift
    tuning = (
        (frame["period"] == "tuning").to_numpy()
        & ~frame["switching_mask"].to_numpy()
        & np.isfinite(frame["actual"].to_numpy())
    )
    band = frame["hour_band"].to_numpy().astype(np.int64)
    quarter = frame["quarter"].to_numpy().astype(np.int64)
    lead = frame["lead_hours"].to_numpy()
    actual = frame["actual"].to_numpy().astype(np.float64)
    spread = LeadHourSpread.fit(
        residuals=(actual - mean)[tuning],
        band=band[tuning],
        quarter=quarter[tuning],
        lead_hours=lead[tuning],
    )
    quantiles = spread.quantiles(mean=mean, band=band, quarter=quarter, lead_hours=lead)
    return frame.with_columns(
        pl.Series("mean", mean.astype(np.float32)),
        *[
            pl.Series(name, quantiles[:, position].astype(np.float32))
            for position, name in enumerate(common.QUANTILES)
        ],
    )
