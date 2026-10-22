"""Paths, constants, and the day table shared by the scripts of experiment 172.

The forecasts are experiment 155's arm D, read from its private folder. This experiment writes only
under its own `WORK_DIR`. A day is a local (Europe/London) calendar day, and a row is one half-hour
of the D+1 band (lead 15 to 39 hours), which holds each valid half-hour once.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.dp_calendar import HOLIDAY_PERIODS
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

EXPERIMENT_ID: Final[int] = 172
SLUG: Final[str] = "dp-error-budget-worst-days"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
PREDS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
RO_NC_INTERVALS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00147-causal-live-end-rule-for-ro-nc" / "run" / "intervals.parquet"
)
PAIR_ALARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00156-switching-faster-onset-alarm-from-opposite-steps"
    / "detect"
    / "real_W8_alarms.parquet"
)
CLEANED_POWER: Final[str] = str(REPO_DATA_DIR / "NGED" / "cleaned_power_time_series.delta")

DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
SITES: Final[tuple[str, ...]] = (*DECISION_SITES, "D9")
LEVELS: Final[dict[str, float]] = {
    "q01": 0.01, "q05": 0.05, "q10": 0.10, "q25": 0.25, "q40": 0.40, "q50": 0.50,
    "q60": 0.60, "q75": 0.75, "q90": 0.90, "q95": 0.95, "q99": 0.99,
}  # fmt: skip
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)
LOCAL_TZ: Final[str] = "Europe/London"
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 2, tzinfo=UTC)
SCORED_LAST: Final[datetime] = datetime(2026, 6, 30, 23, 59, tzinfo=UTC)
D1_LEAD_HOURS: Final[tuple[float, float]] = (15.0, 39.0)
MIN_ROWS_PER_DAY: Final[int] = 24
WORST_FRACTION: Final[float] = 0.05
COLD_DAY_C: Final[float] = 2.0
MIN_READINGS: Final[int] = 42
E7_SHARE: Final[float] = 0.35
E7_UTC_HALF_HOURS: Final[frozenset[tuple[int, int]]] = frozenset({(0, 0), (0, 30), (1, 0), (1, 30)})
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
LABELS: Final[tuple[str, ...]] = ("switching", "flagged", "cold", "economy7", "holiday")
N_RESAMPLES: Final[int] = 20_000
N_RANDOM_DRAWS: Final[int] = 1_000
INJECTED_POINTS: Final[float] = 5.0
INJECTED_DAYS: Final[int] = 10
SEED: Final[int] = 172


def pinball_score_expr() -> pl.Expr:
    """Return twice the mean pinball loss over the 11 quantiles, as a Polars expression."""
    losses = []
    for column, level in LEVELS.items():
        error = pl.col("actual") - pl.col(column)
        losses.append(pl.max_horizontal(level * error, (level - 1.0) * error))
    return 2.0 * pl.mean_horizontal(*losses)


def read_rows(*, lead_band: tuple[float, float], drop_masked: bool) -> pl.DataFrame:
    """Return arm D's scored-year rows in a lead band, with the row score and the local date."""
    frames = [
        pl.read_parquet(PREDS_DIR / f"d_scored_{site}.parquet").with_columns(site=pl.lit(site))
        for site in SITES
    ]
    rows = (
        pl.concat(frames)
        .filter(
            pl.col("actual").is_finite(),
            pl.col("lead_hours") >= lead_band[0],
            pl.col("lead_hours") < lead_band[1],
            pl.col("valid_time") >= SCORED_FIRST,
            pl.col("valid_time") <= SCORED_LAST,
        )
        .with_columns(score=pinball_score_expr())
    )
    if drop_masked:
        rows = rows.filter(~pl.col("switching_mask"))
    local = pl.col("valid_time").dt.convert_time_zone(LOCAL_TZ)
    return rows.with_columns(
        local_date=local.dt.date(),
        local_hour=local.dt.hour(),
        month=local.dt.month(),
        utc_hour=pl.col("valid_time").dt.hour(),
        utc_minute=pl.col("valid_time").dt.minute(),
    )


def load_cache() -> FeatureCache:
    """Return the feature cache that experiment 155 used."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def weather_and_calendar(*, rows: pl.DataFrame, cache: FeatureCache) -> pl.DataFrame:
    """Add each row's holiday flag, and each local day's ERA5 mean temperature, to the rows."""
    pieces, day_means = [], []
    end_times = cache.end_times
    local_date_of_grid = (
        pl.Series(end_times.astype("datetime64[us]"))
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(LOCAL_TZ)
        .dt.date()
    )
    for site in SITES:
        part = rows.filter(pl.col("site") == site)
        target = np.searchsorted(
            end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        holiday = cache.calendar.holiday_period[target] >= 0
        pieces.append(part.with_columns(holiday=pl.Series(holiday)))
        temperature = pl.DataFrame(
            {
                "site": site,
                "local_date": local_date_of_grid,
                "temperature_c": cache.sites[site].temperature_c,
            }
        )
        day_means.append(
            temperature.filter(pl.col("temperature_c").is_finite())
            .group_by(["site", "local_date"])
            .agg(day_temperature_c=pl.col("temperature_c").mean())
        )
    return pl.concat(pieces).join(pl.concat(day_means), on=["site", "local_date"], how="left")


def switching_days(*, series_of_site: dict[str, int]) -> pl.DataFrame:
    """Return the (site, local_date) pairs that a detector's interval or alarm covers."""

    def local_dates(*, first: int, last: int) -> list:
        starts = [GRID_START + HALF_HOUR * i for i in (first, last)]
        days = pl.datetime_range(starts[0], starts[1], interval="30m", time_zone="UTC", eager=True)
        return days.dt.convert_time_zone(LOCAL_TZ).dt.date().unique().to_list()

    intervals = pl.read_parquet(RO_NC_INTERVALS).filter(pl.col("setting") == "R0")
    alarms = pl.read_parquet(PAIR_ALARMS)
    out = []
    for site, series in series_of_site.items():
        for row in intervals.filter(pl.col("series") == series).iter_rows(named=True):
            out += [
                (site, day, "ro_nc") for day in local_dates(first=row["onset"], last=row["end"])
            ]
        for row in alarms.filter(pl.col("series") == series).iter_rows(named=True):
            out += [
                (site, day, "pair_alarm")
                for day in local_dates(first=row["stamp"], last=row["stamp"])
            ]
    return pl.DataFrame(out, schema=["site", "local_date", "detector"], orient="row")


def flagged_days(*, series_of_site: dict[str, int]) -> pl.DataFrame:
    """Return, per site and local day, the flagged readings and the readings kept."""
    ids = list(series_of_site.values())
    local = pl.col("time").dt.convert_time_zone(LOCAL_TZ)
    cleaned = (
        pl.scan_delta(CLEANED_POWER)
        .filter(
            pl.col("time_series_id").is_in(ids),
            pl.col("time") >= SCORED_FIRST - timedelta(days=1),
            pl.col("time") <= SCORED_LAST + timedelta(days=1),
        )
        .group_by(["time_series_id", local.dt.date().alias("local_date")])
        .agg(
            n_flagged=pl.col("drop_reason").is_not_null().sum(),
            n_readings=pl.col("drop_reason").is_null().sum(),
        )
        .collect()
    )
    names = pl.DataFrame({"site": list(series_of_site), "time_series_id": ids})
    return cleaned.join(names, on="time_series_id", how="inner").drop("time_series_id")


def series_ids() -> dict[str, int]:
    """Return each site's NGED `time_series_id`, read from the stored forecasts."""
    out = {}
    for site in SITES:
        frame = pl.read_parquet(PREDS_DIR / f"d_scored_{site}.parquet", columns=["time_series_id"])
        out[site] = int(frame["time_series_id"][0])
    return out


def write_json(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2, default=str))
    temporary.replace(path)


def holiday_names() -> tuple[str, ...]:
    """Return the holiday-period names of `studies.dp_calendar`."""
    return tuple(HOLIDAY_PERIODS)
