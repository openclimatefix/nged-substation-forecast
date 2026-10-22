"""Paths, constants, and the day table shared by the scripts of experiment 174.

A day is a local (Europe/London) calendar day dated by the half-hour's start (`valid_time` is the
half-hour's end). A row is one half-hour of the D+1 band (lead 15 to 39 hours), outside the
hand-filtered switching masks, so the day table matches experiment 172's.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00174_c155 as c155
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 174
SLUG: Final[str] = "dp-unlabelled-worst-days-weather-shared-or-single-site"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
PREDS_DIR: Final[Path] = WORK_DIR / "preds"
E136_WORK_DIR: Final[Path] = c155.E136_WORK_DIR
E155_PREDS_DIR: Final[Path] = c155.OUT_DIR
E172_WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00172-dp-error-budget-worst-days" / "work"
BG2_PREDICTIONS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)

DECISION_SITES: Final[tuple[str, ...]] = c155.DECISION_SITES
SITES: Final[tuple[str, ...]] = c155.SCORED_SITES
LEVELS: Final[dict[str, float]] = {
    "q01": 0.01, "q05": 0.05, "q10": 0.10, "q25": 0.25, "q40": 0.40, "q50": 0.50,
    "q60": 0.60, "q75": 0.75, "q90": 0.90, "q95": 0.95, "q99": 0.99,
}  # fmt: skip
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)
LOCAL_TZ: Final[str] = "Europe/London"
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 2, tzinfo=UTC)
SCORED_LAST: Final[datetime] = datetime(2026, 6, 30, 23, 59, tzinfo=UTC)
D1_LEAD_HOURS: Final[tuple[float, float]] = (15.0, 39.0)
MIN_ROWS_PER_DAY: Final[int] = 24
WORST_FRACTION: Final[float] = 0.05
N_RESAMPLES: Final[int] = 20_000
N_SHIFT_DRAWS: Final[int] = 1_000
INJECTED_POINTS: Final[float] = 10.0
INJECTED_DAYS: Final[int] = 10
SEED: Final[int] = 174


def preds_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one site."""
    return PREDS_DIR / f"{arm}_{period}_{site}.parquet"


def pinball_score_expr() -> pl.Expr:
    """Return twice the mean pinball loss over the 11 quantiles, as a Polars expression."""
    losses = []
    for column, level in LEVELS.items():
        error = pl.col("actual") - pl.col(column)
        losses.append(pl.max_horizontal(level * error, (level - 1.0) * error))
    return 2.0 * pl.mean_horizontal(*losses)


def local_date_expr() -> pl.Expr:
    """Return the local date of the half-hour that ends at `valid_time`."""
    return (pl.col("valid_time") - HALF_HOUR).dt.convert_time_zone(LOCAL_TZ).dt.date()


def read_arm_rows(*, arm: str, sites: tuple[str, ...] = SITES) -> pl.DataFrame:
    """Return an arm's scored-year D+1 rows outside the switching masks, with the row score."""
    frames = [
        pl.read_parquet(common_path(arm=arm, site=site)).with_columns(site=pl.lit(site))
        for site in sites
    ]
    return (
        pl.concat(frames)
        .filter(
            pl.col("actual").is_finite(),
            pl.col("lead_hours") >= D1_LEAD_HOURS[0],
            pl.col("lead_hours") < D1_LEAD_HOURS[1],
            pl.col("valid_time") >= SCORED_FIRST,
            pl.col("valid_time") <= SCORED_LAST,
            ~pl.col("switching_mask"),
        )
        .with_columns(score=pinball_score_expr(), local_date=local_date_expr())
    )


def common_path(*, arm: str, site: str) -> Path:
    """Return the scored-period file of an arm: experiment 155's `d`, or this folder's arms."""
    if arm == "e155_d":
        return E155_PREDS_DIR / f"d_scored_{site}.parquet"
    return preds_path(arm=arm, period="scored", site=site)


def day_table(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return one row per (site, local day) with the score, the excess loss, and the mean error."""
    days = rows.group_by(["site", "local_date"]).agg(
        n_rows=pl.len(),
        score=pl.col("score").mean(),
        mean_abs_error=(pl.col("actual") - pl.col("mean")).abs().mean(),
    )
    days = days.filter(pl.col("n_rows") >= MIN_ROWS_PER_DAY)
    return days.with_columns(
        excess=pl.col("score") - pl.col("score").median().over("site"),
        excess_mae=pl.col("mean_abs_error") - pl.col("mean_abs_error").median().over("site"),
    ).sort(["site", "local_date"])


def total(series: pl.Series) -> float:
    """Return the sum of a numeric series as a float."""
    return float(series.cast(pl.Float64).to_numpy().sum())


def mean_of(series: pl.Series) -> float:
    """Return the mean of a numeric or boolean series as a float."""
    return float(series.cast(pl.Float64).to_numpy().mean())
