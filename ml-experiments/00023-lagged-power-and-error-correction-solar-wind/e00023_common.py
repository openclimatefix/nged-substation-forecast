"""Paths, series, and helpers shared by the scripts of experiment 00023.

The experiment asks whether recent observed power, or the 14-day error correction, helps the 6 solar
farms and 3 wind farms (`docs/ml-experiments/00023-lagged-power-and-error-correction-solar-wind/
result.md`). The mixture, loss, and lead-band arithmetic copy experiment 00020's, and the lag and
summary columns copy experiments 00019 and 00008; an experiment may not import another experiment.
The inputs it reads were built by the stage 1 and stage 2 pilots and are read only.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.power import scan_power
from studies.pv_dataset import pv_sites, wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR

EXPERIMENT_ID: Final[int] = 23
SLUG: Final[str] = "lagged-power-and-error-correction-solar-wind"
ROW_SET_ID: Final[str] = "fold_a_tuesdays_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MEMBERS: Final[int] = 51
KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
CASE_ORDER: Final[tuple[str, ...]] = ("power_fcst_init_time", "time_series_id", "valid_time")

TechnologyType = Literal["solar", "wind"]
TECHNOLOGIES: Final[tuple[TechnologyType, ...]] = ("solar", "wind")
ArmType = Literal[
    "mixture",
    "cal0",
    "cal_lags",
    "cal_sums",
    "cal_both",
    "mix_c14",
    "cal_both_c14",
    "mix_c14_median",
]
CALIBRATOR_ARMS: Final[tuple[str, ...]] = ("cal0", "cal_lags", "cal_sums", "cal_both")
ARMS: Final[tuple[ArmType, ...]] = (
    "mixture",
    "cal0",
    "cal_lags",
    "cal_sums",
    "cal_both",
    "mix_c14",
    "cal_both_c14",
    "mix_c14_median",
)

PILOT_DIR: Final[Path] = PER_STUDY_DIR / "stage1_pilot"
STAGE2_DIR: Final[Path] = PER_STUDY_DIR / "stage2_pilot"
STAGE1A_DIR: Final[Path] = STAGE2_DIR / "stage1a_q" / "full"
SCORES_DIR: Final[Path] = STAGE2_DIR / "scores"
FRAMES_DIR: Final[Path] = (
    PER_STUDY_DIR / "quick_wins_fixed_pipeline_1156" / "frames" / "combined" / "fold_a"
)
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""

GRID_START: Final[datetime] = datetime(2024, 1, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
STEPS_PER_DAY: Final[int] = 48
STEPS_PER_WEEK: Final[int] = 7 * STEPS_PER_DAY
TELEMETRY_DELAY_STEPS: Final[int] = 6
"""A value is usable once it is 3 hours old, plus the half-hour it spans (experiment 00019)."""
MIN_OBSERVED: Final[int] = 24
"""A 48-half-hour summary window needs at least this many observed half-hours."""
TRAINING_VALID_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""The calibrator trains on rows whose valid time is before this."""

CORRECTION_WINDOW_DAYS: Final[int] = 14
CORRECTION_DELAY_DAYS: Final[int] = 2
MIN_CORRECTION_CASES: Final[int] = 500
TRIVIAL_LEVEL: Final[float] = 0.01
"""A case is trivial when the uncorrected 95% quantile is at most this fraction of capacity."""

CALIBRATOR_PARAMETERS: Final[dict[str, object]] = {
    "objective": "reg:quantileerror",
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 12,
}
"""Experiment 00005's calibrator settings with the thread count of this experiment's slot; the
quantile levels are added at fit time."""
CALIBRATOR_ROUNDS: Final[int] = 300

SETTLING_LEVEL_FRACTION: Final[float] = 0.05
SETTLING_RUN_HALF_HOURS: Final[int] = 48
SETTLING_WEEKS: Final[int] = 6
"""The `new_meter_settling` rule, applied as the stage 1 pilot applied it."""

LAG_COLUMNS: Final[tuple[str, ...]] = ("lag_newest", "lag_day", "lag_week")
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "recent_min",
    "recent_mean",
    "recent_max",
    "weekday_min",
    "weekday_mean",
    "weekday_max",
)

LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
)
MEDIAN_INDEX: Final[int] = 4
LOWER_INDEX: Final[int] = 0
UPPER_INDEX: Final[int] = 8
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass each level carries: the interval between the mid-points to its neighbours."""
FINE_LEVELS: Final[np.ndarray] = np.arange(1, 200) / 200
"""The 199 levels on which each mixture's quantile function is stored."""


LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0 to 24 h", 0.0, 24.0),
    ("1 to 2 days", 24.0, 72.0),
    ("3 to 6 days", 72.0, 168.0),
    ("7 to 10 days", 168.0, 264.0),
    ("11 to 15 days", 264.0, 10_000.0),
)


def band_names(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the lead band name of each lead in hours."""
    names = np.full(lead_hours.shape, "none", dtype=object)
    for name, low, high in LEAD_BANDS:
        names[(lead_hours >= low) & (lead_hours < high)] = name
    return names


def mixture_quantiles(*, member_quantiles: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Return each case's mixture quantile function on `FINE_LEVELS`.

    Args:
        member_quantiles: Shape (cases, members, 9) member quantiles, each member's levels sorted.
        chunk: Cases sorted at a time.

    Returns:
        Shape (cases, 199) float32 quantiles.
    """
    cases, members, levels = member_quantiles.shape
    weights = np.tile(ATOM_MASSES / members, members)
    out = np.empty((cases, FINE_LEVELS.size), dtype=np.float32)
    for start in range(0, cases, chunk):
        flat = member_quantiles[start : start + chunk].reshape(-1, members * levels)
        order = np.argsort(flat, axis=1, kind="stable")
        values = np.take_along_axis(flat, order, axis=1)
        mass = weights[order]
        midpoint = np.cumsum(mass, axis=1) - mass / 2
        for row in range(flat.shape[0]):
            out[start + row] = np.interp(FINE_LEVELS, midpoint[row], values[row])
    return out


def read_levels(*, fine: np.ndarray) -> np.ndarray:
    """Return the 9 output quantiles of each row of `fine`, by linear interpolation."""
    position = np.clip(LEVELS * 200 - 1, 0, FINE_LEVELS.size - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, FINE_LEVELS.size - 1)
    fraction = (position - low).astype(np.float32)
    return fine[:, low] * (1 - fraction) + fine[:, high] * fraction


def crps_estimate(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the 9 levels, per case (a fraction)."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def covered(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return whether the observed power lies in the 5% to 95% band, per case."""
    return (actual >= quantiles[:, LOWER_INDEX]) & (actual <= quantiles[:, UPPER_INDEX])


def scale_about_median(*, quantiles: np.ndarray, shift: float, factor: float) -> np.ndarray:
    """Return `median + shift + factor * (q - median)`, clipped to 0 to 1 and sorted."""
    median = quantiles[:, MEDIAN_INDEX : MEDIAN_INDEX + 1]
    scaled = median + shift + factor * (quantiles - median)
    return np.sort(np.clip(scaled, 0.0, 1.0), axis=1)


def grid_times() -> pl.Series:
    """Return the dense half-hourly grid every per-series power array is indexed on."""
    return pl.datetime_range(
        GRID_START, GRID_END, interval="30m", time_unit="us", time_zone="UTC", eager=True
    )


def grid_index(*, times: pl.Series) -> np.ndarray:
    """Return the grid index of each UTC timestamp."""
    return ((times.dt.epoch("us") - int(GRID_START.timestamp() * 1e6)) // 1_800_000_000).to_numpy()


def take(*, values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Return `values[index]`, NaN where the index falls off either end of the grid."""
    inside = (index >= 0) & (index < values.size)
    out = np.full(index.shape, np.nan, dtype=np.float32)
    out[inside] = values[index[inside]]
    return out


def day_summaries(*, power: np.ndarray) -> np.ndarray:
    """Return the minimum, mean, and maximum of the 48 half-hours ending at each grid index.

    Missing half-hours are ignored, and a window with fewer than `MIN_OBSERVED` observed half-hours
    gives NaN. Windows that start before the grid are padded with missing half-hours.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.

    Returns:
        A float32 array of shape (grid size, 3): minimum, mean, and maximum.
    """
    padded = np.concatenate([np.full(STEPS_PER_DAY - 1, np.nan, dtype=np.float32), power])
    windows = sliding_window_view(padded, STEPS_PER_DAY)
    seen = np.isfinite(windows)
    count = seen.sum(axis=1)
    total = np.where(seen, windows, 0.0).sum(axis=1, dtype=np.float64)
    low = np.where(seen, windows, np.inf).min(axis=1)
    high = np.where(seen, windows, -np.inf).max(axis=1)
    empty = count < MIN_OBSERVED
    return np.stack(
        [
            np.where(empty, np.nan, low),
            np.where(empty, np.nan, total / np.maximum(count, 1)),
            np.where(empty, np.nan, high),
        ],
        axis=1,
    ).astype(np.float32)


def recent_power_columns(
    *, power: np.ndarray, summaries: np.ndarray, target: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """Return the lag and summary columns of each case, from observations usable at the origin.

    The newest usable value is at grid index `origin - TELEMETRY_DELAY_STEPS - 1`. The lags are
    that value, the value at the target's half-hour of day on the latest earlier day, and the value
    at the target's weekday and half-hour on the latest earlier week. The summaries cover the 48
    half-hours ending at the newest usable value, and the 48 half-hours of the latest UTC day on the
    target's weekday that ends at or before it.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        summaries: `day_summaries(power=power)`.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.

    Returns:
        A float32 array of shape (cases, 9): `LAG_COLUMNS` then `SUMMARY_COLUMNS`.

    Raises:
        ValueError: If a case's valid time is not after its origin, or a window ends after the
            newest usable value.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    newest = origin - TELEMETRY_DELAY_STEPS - 1
    gap = target - newest
    day = target - STEPS_PER_DAY * np.ceil(gap / STEPS_PER_DAY).astype(int)
    week = target - STEPS_PER_WEEK * np.ceil(gap / STEPS_PER_WEEK).astype(int)
    # The grid starts at 00:00 UTC, so a multiple of 48 is a UTC midnight.
    day_start = (target // STEPS_PER_DAY) * STEPS_PER_DAY
    weeks_back = np.maximum(1, np.ceil((day_start + STEPS_PER_DAY - 1 - newest) / STEPS_PER_WEEK))
    weekday_end = day_start - STEPS_PER_WEEK * weeks_back.astype(int) + STEPS_PER_DAY - 1
    if (weekday_end > newest).any() or (np.maximum(day, week) > newest).any():
        msg = "a lag or summary window must end at or before the newest usable value"
        raise ValueError(msg)
    blocks = []
    for end in (newest, weekday_end):
        block = np.full((end.size, 3), np.nan, dtype=np.float32)
        inside = (end >= 0) & (end < power.size)
        block[inside] = summaries[end[inside]]
        blocks.append(block)
    return np.concatenate(
        [
            np.stack(
                [
                    take(values=power, index=newest),
                    take(values=power, index=day),
                    take(values=power, index=week),
                ],
                axis=1,
            ),
            *blocks,
        ],
        axis=1,
    )


def site_table() -> pl.DataFrame:
    """Return the nine generator series with anonymous label, technology, and capacity.

    Returns:
        One row per series with `time_series_id`, `site`, `technology`, and `effective_capacity_mw`.

    Raises:
        ValueError: If the anonymisation helper's labels have changed.
    """
    capacity = pl.read_parquet(FRAMES_DIR / "capacity.parquet").select(
        "time_series_id", "effective_capacity_mw"
    )
    solar = pv_sites().select("time_series_id", "site")
    wind = wind_sites().select("time_series_id", "site")
    if set(solar["site"].to_list()) != set(SITE_LABELS) or LABEL_PERMUTATION_SEED != 784:
        msg = "the anonymisation helper's solar labels have changed"
        raise ValueError(msg)
    return (
        pl.concat(
            [
                solar.with_columns(technology=pl.lit("solar")),
                wind.with_columns(technology=pl.lit("wind")),
            ]
        )
        .join(capacity, on="time_series_id", how="left", validate="1:1")
        .sort("technology", "site")
    )


def power_grids(*, sites: pl.DataFrame) -> dict[int, np.ndarray]:
    """Return each series' cleaned power over capacity on the grid, NaN where not usable.

    Rows before each series' settling cutoff are dropped, as the stage 1 pilot did.

    Args:
        sites: `site_table()`.

    Returns:
        Each `time_series_id`'s float32 array over `grid_times()`.
    """
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(sites["time_series_id"].to_list()),
            pl.col("time") >= GRID_START,
            pl.col("time") < GRID_END,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .sort("time_series_id", "valid_time")
    )
    live = (
        pl.col("power").abs() > SETTLING_LEVEL_FRACTION * pl.col("effective_capacity_mw")
    ).fill_null(value=False)
    run_end = (
        live.cast(pl.Int32).rolling_sum(window_size=SETTLING_RUN_HALF_HOURS).over("time_series_id")
        == SETTLING_RUN_HALF_HOURS
    )
    cutoffs = (
        power.filter(run_end)
        .group_by("time_series_id")
        .agg(
            cutoff=pl.col("valid_time")
            .min()
            .dt.offset_by(f"-{SETTLING_RUN_HALF_HOURS * 30 - 30}m")
            .dt.offset_by(f"{SETTLING_WEEKS}w")
        )
    )
    kept = (
        power.join(cutoffs, on="time_series_id", how="left")
        .filter(pl.col("cutoff").is_null() | (pl.col("valid_time") >= pl.col("cutoff")))
        .filter(pl.col("power").is_finite())
    )
    size = grid_times().len()
    out: dict[int, np.ndarray] = {}
    for series_id, part in kept.partition_by("time_series_id", as_dict=True).items():
        grid = np.full(size, np.nan, dtype=np.float32)
        capacity = float(part["effective_capacity_mw"][0])
        grid[grid_index(times=part["valid_time"])] = (part["power"].to_numpy() / capacity).astype(
            np.float32
        )
        out[int(series_id[0])] = grid
    return out


def technology_table() -> pl.DataFrame:
    """Return `time_series_id` and `technology` for the 6 solar farms and 3 wind farms."""
    return pl.concat(
        [
            pv_sites().select("time_series_id", technology=pl.lit("solar")),
            wind_sites().select("time_series_id", technology=pl.lit("wind")),
        ]
    )


def validation_cases() -> pl.DataFrame:
    """Return experiment 00002's scored cases with technology and lead band, in `CASE_ORDER`.

    Columns: the key columns, `lead_hours`, `actual_norm` (a fraction of capacity), `technology`,
    and `band`.
    """
    cases = (
        pl.read_parquet(SCORES_DIR / "rows_seed0.parquet")
        .filter(pl.col("arm") == "b_mixture_full")
        .select(*KEY_COLUMNS, "lead_hours", "actual_norm")
        .join(technology_table(), on="time_series_id", how="left", validate="m:1")
        .sort(*CASE_ORDER)
    )
    return cases.with_columns(band=pl.Series(band_names(lead_hours=cases["lead_hours"].to_numpy())))
