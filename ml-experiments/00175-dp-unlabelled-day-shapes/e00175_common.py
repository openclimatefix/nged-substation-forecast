"""Paths, the half-hour grid, and the residual series shared by the scripts of experiment 175.

The residual of a series is observed minus forecast over the D+1 band (lead 15 to 39 hours), one
value per half-hour, on a continuous UTC grid with `NaN` where a row is missing. A half-hour is
dated by its start (`valid_time` is the half-hour's end), in Europe/London local time.
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

EXPERIMENT_ID: Final[int] = 175
SLUG: Final[str] = "dp-unlabelled-day-shapes"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
E174_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00174-dp-unlabelled-worst-days-weather-shared-or-single-site"
    / "work"
)
E172_WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00172-dp-error-budget-worst-days" / "work"
E147_INTERVALS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00147-causal-live-end-rule-for-ro-nc" / "run" / "intervals.parquet"
)
E156_PAIR_ALARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00156-switching-faster-onset-alarm-from-opposite-steps"
    / "detect"
    / "real_W8_alarms.parquet"
)
BG2_PREDICTIONS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)
HIERARCHY_DIR: Final[Path] = REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15"

LOCAL_TZ: Final[str] = "Europe/London"
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)
D1_LEAD_HOURS: Final[tuple[float, float]] = (15.0, 39.0)
DETECTOR_GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
"""Stamp 0 of the detectors' integer half-hour index (experiments 147 and 156)."""
GRID_FIRST: Final[datetime] = datetime(2025, 6, 1, tzinfo=UTC)
GRID_LAST: Final[datetime] = datetime(2026, 7, 31, tzinfo=UTC)
MIN_ROWS_PER_DAY: Final[int] = 24
SEED: Final[int] = 175


class ResidualGrid:
    """A continuous half-hour grid and the residual series of several sites on it.

    The grid holds `end_times` (the UTC end of each half-hour), `local_dates` and `local_hours`
    (the Europe/London date and hour of day of each half-hour's start), and `residual` (each
    site's observed minus forecast, `NaN` where the row is missing).
    """

    def __init__(self, *, residual: dict[str, np.ndarray]) -> None:
        """Build the grid and attach the residual series."""
        index = pl.datetime_range(
            GRID_FIRST + HALF_HOUR, GRID_LAST, interval="30m", time_zone="UTC", eager=True
        )
        starts = (index - HALF_HOUR).dt.convert_time_zone(LOCAL_TZ)
        self.end_times: pl.Series = index
        self.local_dates: np.ndarray = starts.dt.date().to_numpy()
        self.local_hours: np.ndarray = (
            starts.dt.hour().to_numpy() + starts.dt.minute().to_numpy() / 60.0
        )
        self.residual: dict[str, np.ndarray] = residual

    def positions_of(self, *, day: date) -> np.ndarray:
        """Return the grid positions of a local day's half-hours."""
        return np.flatnonzero(self.local_dates == np.datetime64(day))

    def time_of(self, *, position: int) -> datetime:
        """Return the UTC end of the half-hour at a grid position."""
        return self.end_times[position]

    def position_of_stamp(self, *, stamp: int) -> int:
        """Return the grid position of a detector stamp (a half-hour index from 2022-01-01)."""
        end = DETECTOR_GRID_START + (stamp + 1) * HALF_HOUR
        return round((end - self.end_times[0]).total_seconds() / 1800)


def _on_grid(*, valid_time: pl.Series, value: pl.Series) -> np.ndarray:
    """Return `value` placed at its half-hour on the grid, `NaN` elsewhere."""
    first = GRID_FIRST + HALF_HOUR
    position = ((valid_time - first).dt.total_seconds() // 1800).cast(pl.Int64).to_numpy()
    length = int((GRID_LAST - first).total_seconds() // 1800) + 1
    out = np.full(length, np.nan)
    keep = (position >= 0) & (position < length)
    out[position[keep]] = value.to_numpy()[keep]
    return out


def arm_d_residuals(*, sites: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Return arm D's residual (actual minus mean) per site on the grid.

    Rows outside the D+1 band, and the hand-filtered switching rows, are left as `NaN`, as in
    experiments 172 and 174.
    """
    out: dict[str, np.ndarray] = {}
    for site in sites:
        rows = pl.read_parquet(E174_WORK_DIR / "preds" / f"ens_d_scored_{site}.parquet").filter(
            pl.col("actual").is_finite(),
            pl.col("lead_hours") >= D1_LEAD_HOURS[0],
            pl.col("lead_hours") < D1_LEAD_HOURS[1],
            ~pl.col("switching_mask"),
        )
        out[site] = _on_grid(valid_time=rows["valid_time"], value=rows["actual"] - rows["mean"])
    return out


def arm_d_masked_times(*, site: str) -> pl.Series:
    """Return the `valid_time` of arm D's hand-filtered switching rows in the D+1 band."""
    return (
        pl.read_parquet(E174_WORK_DIR / "preds" / f"ens_d_scored_{site}.parquet")
        .filter(
            pl.col("lead_hours") >= D1_LEAD_HOURS[0],
            pl.col("lead_hours") < D1_LEAD_HOURS[1],
            pl.col("switching_mask"),
        )["valid_time"]
        .sort()
    )


def bg2_residuals() -> dict[str, np.ndarray]:
    """Return BG2's residual per D-label on the grid, in points of effective capacity.

    BG2 stores output as a fraction of capacity, so the residual (actual minus median forecast
    `q50`) is multiplied by 100 to match arm D's units.
    """
    rows = pl.read_parquet(BG2_PREDICTIONS).filter(
        pl.col("actual").is_finite(),
        pl.col("lead_hours") >= D1_LEAD_HOURS[0],
        pl.col("lead_hours") < D1_LEAD_HOURS[1],
    )
    return {
        site: _on_grid(valid_time=part["valid_time"], value=100.0 * (part["actual"] - part["q50"]))
        for (site,), part in rows.group_by("site")
    }


def series_ids() -> dict[str, int]:
    """Return each BG2 D-label mapped to its `time_series_id` (the detectors' series number)."""
    pairs = pl.read_parquet(BG2_PREDICTIONS, columns=["site", "time_series_id"]).unique()
    return {row["site"]: int(row["time_series_id"]) for row in pairs.iter_rows(named=True)}


def neighbours() -> dict[str, set[str]]:
    """Return, for each BG2 D-label, the D-labels of the Primaries interconnected with its Primary.

    A label that shares its Primary with the key (the other transformer) is not a neighbour.
    """
    series = pl.read_parquet(BG2_PREDICTIONS, columns=["site", "time_series_id"]).unique()
    monitor = pl.read_csv(HIERARCHY_DIR / "Monitor Direction.csv").select(
        "TimeSeriesID", "SubstationNumber"
    )
    joined = series.join(monitor, left_on="time_series_id", right_on="TimeSeriesID", how="left")
    if joined["SubstationNumber"].null_count() or joined["site"].n_unique() != joined.height:
        msg = "a BG2 series has no unique substation number"
        raise ValueError(msg)
    number_of = {row["site"]: int(row["SubstationNumber"]) for row in joined.iter_rows(named=True)}
    labels_of: dict[int, set[str]] = {}
    for label, number in number_of.items():
        labels_of.setdefault(number, set()).add(label)
    pairs = pl.read_csv(HIERARCHY_DIR / "Primary Substation Interconnections.csv")
    out: dict[str, set[str]] = {label: set() for label in number_of}
    for first, second in pairs.iter_rows():
        for here, there in ((first, second), (second, first)):
            for label in labels_of.get(here, set()):
                out[label] |= labels_of.get(there, set())
    return out


def unlabelled_days() -> pl.DataFrame:
    """Return the 37 unlabelled worst (site, local_date) days of experiment 174, sorted."""
    return (
        pl.read_parquet(E174_WORK_DIR / "worst_days_split.parquet")
        .filter(~pl.col("labelled"))
        .select("site", "local_date", "excess_ens")
        .sort(["site", "local_date"])
    )


def day_excess(*, site: str) -> pl.DataFrame:
    """Return each scored local day of a site with its excess loss, from experiment 172's days."""
    days = pl.read_parquet(E172_WORK_DIR / "days.parquet")
    return days.filter(pl.col("site") == site, pl.col("n_rows") >= MIN_ROWS_PER_DAY).select(
        "site", "local_date", "excess"
    )
