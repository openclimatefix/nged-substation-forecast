"""Score every arm of experiment 00087: quantile score, error of the median, tails, and contrasts.

Reads `work/arms/<name>.parquet` (from `e00087_assemble.py`) and writes tidy tables under
`work/results/` plus `work/results/report.txt`, which holds every number the page quotes. All
scores are in points of capacity (fraction of capacity times 100). Sources, each copied here (an
experiment may not import another): the per-case metrics and the month-resampled intervals of
experiment 00048's `e00048_stats.py` and `e00048_common.py` (`case_metrics`, `month_bounds`), and
the threshold-weighted CRPS of experiment 00033's `e00033_common.py` and `e00033_stats.py`.

**Cases.** A class (`Primary`, `BSP/GSP`, `Solar farms`, `Wind farms`) is scored on the cases that
every model arm of the class holds (`e00087_arms` staircase and ablation arms), for valid times
before 2026-07-01 with a finite observation and finite quantiles. A reference
(`ref_climatology`, `ref_manual_heuristic`, `ref_xgboost_baseline`) may lack some of those cases;
its absolute score is the mean over the cases it holds, and a contrast against it uses the cases
both arms hold. The headline is the 292 runs of `fold_a_daily_gap_v1`; the 73 runs of
`fold_a_every5d_v1` are scored once more as a check.

**Metrics.** `qs` is twice the mass-weighted pinball loss of the 11 levels (`ATOM_MASSES`),
`mae` is the absolute error of the median, and `tw_p99`, `tw_p95`, and `tw_p05` are the quantile
score of forecast and observation both floored (upper tail) or capped (lower tail) at the series'
99th, 95th, and 5th percentile of its history before 2025-07-01 (threshold-weighted CRPS). The
tails table reports coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99 through
`studies.tail_calibration`; the reliability table reports the observed share at or below each of
the 11 quantiles. The scope tables split `qs` and `mae` by lead day 0 to 14, lead band, UK hour of
day, midday (11:00 to 15:00 UK) by lead day, origin month, season, sunny or cloudy day, and series.
Sunny and cloudy are the top and bottom third of the CAMS daily clear-sky index within each series
and calendar month (CAMS and its clear-sky irradiance from experiment 00039's frame, as
experiment 00046 read it); a wind farm takes the class of the mean index of the six solar farms,
because the CAMS frame has no wind farm.

**Intervals.** Every interval resamples whole origin months with the random stream of
`studies.bootstrap`, summing each month once instead of redrawing its rows
(`month_intervals`; `--verify` compares it with `bootstrap_difference_at_level` and
`bootstrap_absolute`). Absolute scores, verdict contrasts, and every other contrast use 20,000
resamples; the scope tables use 2,000. The four verdict contrasts (BG2 minus the class's previous
best on the same cases) are read at 98.75% (5% over four, Bonferroni). For `Primary` and
`BSP/GSP` BG2 wins when the interval lies wholly below zero; for solar and wind it is
non-inferior when the upper bound is below the margin of +0.10 (solar) or +0.30 (wind). Every
other contrast is read at 95% and decides nothing.

    uv run python ml-experiments/00087-*/e00087_score.py [--classes Primary ...]
    uv run python ml-experiments/00087-*/e00087_score.py --repro
    uv run python ml-experiments/00087-*/e00087_score.py --check-scorer
    uv run python ml-experiments/00087-*/e00087_score.py --verify
"""

import argparse
import functools
import itertools
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00087_arms as arms
import e00087_assemble as assemble
import e00087_common as common
import e00087_gen_common as gen
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from studies.tail_calibration import tail_calibration_table

from studies import bootstrap

N_RESAMPLES: Final[int] = 20_000
N_SCOPE_RESAMPLES: Final[int] = 2_000
LEVEL_95: Final[float] = 95.0
POINTS: Final[float] = 100.0
"""Scores are reported in points of capacity: the fraction of capacity times this."""
METRICS: Final[tuple[str, ...]] = ("qs", "mae", "tw_p99", "tw_p95", "tw_p05")
HEADLINE_METRICS: Final[tuple[str, ...]] = ("qs", "mae")
SEASON_OF_MONTH: Final[dict[int, str]] = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}
"""Meteorological seasons, from the UK calendar month of the half-hour."""
SEASONS: Final[tuple[str, ...]] = ("Winter", "Spring", "Summer", "Autumn")
CSI_CLASSES: Final[tuple[str, ...]] = ("cloudy", "middle", "sunny")
MIDDAY_HOURS: Final[tuple[int, ...]] = (11, 12, 13, 14)
"""UK hours of day (the hour in which the half-hour starts) that make up 11:00 to 15:00."""
LAST_LEAD_DAY: Final[int] = 14
LEAD_BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _, _ in common.LEAD_BANDS)
CAMS_FRAME_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
"""Experiment 00039's CAMS frame (`ghi_u` and `ghi_u_clear` per site and half-hour), read only."""
EXPERIMENT_00048_ARMS_DIR: Final[Path] = common.EXPERIMENT_00048_WORK_DIR / "arms"
READ_COLUMNS: Final[tuple[str, ...]] = (
    *assemble.KEY_COLUMNS,
    "actual",
    *common.LEVEL_COLUMNS,
    "group",
)
CLASS_GROUPS: Final[dict[str, tuple[str, ...]]] = arms.GROUPS_OF_CLASS
DESIGN_PREDICTION: Final[str] = (
    "Design prediction (before any result): Primary BG2 about 5.30 against the stack's 5.64 and "
    "manual_heuristic 6.26 (9 levels)."
)
REPRO_TARGETS: Final[tuple[tuple[str, str, float], ...]] = (
    ("weekly Primary", "Primary", 5.44),
    ("weekly BSP", "BSP", 7.84),
    ("weekly GSP", "GSP", 18.66),
)
"""Experiment 00048's weekly-refit cells: the arm group and its published quantile score."""
REPRO_TOLERANCE: Final[float] = 0.02
CHECK_SCORER_TOLERANCE: Final[float] = 0.006
"""Experiment 00048 published two decimals, so the scorer must agree to within half of the last."""


# --- month-resampled intervals -----------------------------------------------------------------


@functools.cache
def draw_counts(*, n_months: int, n_resamples: int) -> np.ndarray:
    """Return, per resample, how many times each of `n_months` months is drawn.

    The stream is that of `studies.bootstrap`: for each resample one draw of the seed (there is one
    seed here, so `integers(0, 1)`), then one draw of the months with replacement.

    Args:
        n_months: How many distinct months the rows span.
        n_resamples: How many resamples.

    Returns:
        Shape (n_resamples, n_months), float64.
    """
    generator = np.random.default_rng(bootstrap.BOOTSTRAP_SEED)
    counts = np.empty((n_resamples, n_months))
    for resample in range(n_resamples):
        generator.integers(0, 1)
        drawn = generator.integers(0, n_months, size=n_months)
        counts[resample] = np.bincount(drawn, minlength=n_months)
    return counts


def month_intervals(
    *, sums: np.ndarray, counts: np.ndarray, level: float, n_resamples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the interval of each row's mean, resampling whole months.

    The mean of a resample is the sum of the drawn months' sums over the sum of their row counts,
    which equals the mean of the drawn months' rows. A row of the inputs holds one scope, and a
    month in which the scope has no rows is not drawn, as in `studies.bootstrap`.

    Args:
        sums: Shape (scopes, months): each scope's sum of the metric in each month.
        counts: Shape (scopes, months): each scope's row count in each month.
        level: The interval's coverage in percent.
        n_resamples: The number of resamples.

    Returns:
        The lower and upper bounds, each of shape (scopes,), NaN for a scope with no rows.

    Raises:
        ValueError: If `level` is not above 1 and below 100.
    """
    if not 1.0 < level < 100.0:
        msg = f"level is a coverage in percent, above 1 and below 100: {level}"
        raise ValueError(msg)
    lower = np.full(sums.shape[0], np.nan)
    upper = np.full(sums.shape[0], np.nan)
    tail = (100.0 - level) / 2.0
    present = counts > 0
    patterns: dict[tuple[bool, ...], list[int]] = {}
    for row in range(sums.shape[0]):
        if present[row].any():
            patterns.setdefault(tuple(present[row].tolist()), []).append(row)
    for pattern, rows in patterns.items():
        columns = np.flatnonzero(np.array(pattern))
        draws = draw_counts(n_months=len(columns), n_resamples=n_resamples)
        numerator = sums[np.ix_(rows, columns)] @ draws.T
        denominator = counts[np.ix_(rows, columns)] @ draws.T
        resampled = numerator / denominator
        bounds = np.percentile(resampled, [tail, 100.0 - tail], axis=1)
        lower[rows], upper[rows] = bounds[0], bounds[1]
    return lower, upper


def month_totals(
    *,
    values: np.ndarray,
    valid: np.ndarray,
    codes: np.ndarray,
    n_codes: int,
    months: np.ndarray,
    n_months: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sum and row count of `values` in each (scope code, month) cell.

    Args:
        values: One metric value per case.
        valid: Whether a case counts (finite value and in the run subset).
        codes: Each case's scope code, from 0 to `n_codes - 1`, or -1 for a case in no scope.
        n_codes: The number of scopes.
        months: Each case's origin-month index.
        n_months: The number of months.

    Returns:
        Two arrays of shape (n_codes, n_months).
    """
    use = valid & (codes >= 0)
    cell = codes[use].astype(np.int64) * n_months + months[use].astype(np.int64)
    size = n_codes * n_months
    sums = np.bincount(cell, weights=values[use].astype(np.float64), minlength=size)
    counts = np.bincount(cell, minlength=size).astype(np.float64)
    return sums.reshape(n_codes, n_months), counts.reshape(n_codes, n_months)


# --- per-case metrics --------------------------------------------------------------------------


def case_metrics(
    *, quantiles: np.ndarray, actual: np.ndarray, thresholds: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return every per-case metric in points of capacity.

    Args:
        quantiles: Shape (cases, 11), sorted.
        actual: Observed power over capacity, shape (cases,).
        thresholds: Each case's series' `p99`, `p95`, and `p05` of history, each shape (cases,).

    Returns:
        `qs`, `mae` (absolute error of the median), and the threshold-weighted `tw_p99`, `tw_p95`
        (floored at the threshold) and `tw_p05` (capped at it).
    """
    out = {
        "qs": common.pinball_quantile_score(quantiles=quantiles, actual=actual) * POINTS,
        "mae": np.abs(quantiles[:, common.MEDIAN] - actual) * POINTS,
    }
    for name in ("p99", "p95"):
        threshold = thresholds[name]
        out[f"tw_{name}"] = (
            common.pinball_quantile_score(
                quantiles=np.maximum(quantiles, threshold[:, None]),
                actual=np.maximum(actual, threshold),
            )
            * POINTS
        )
    threshold = thresholds["p05"]
    out["tw_p05"] = (
        common.pinball_quantile_score(
            quantiles=np.minimum(quantiles, threshold[:, None]),
            actual=np.minimum(actual, threshold),
        )
        * POINTS
    )
    return out


def history_thresholds(*, power: np.ndarray) -> dict[str, float]:
    """Return a series' 99th, 95th, and 5th percentile of observed power before 2025-07-01.

    Copied from experiment 00033's `historical_thresholds` (valid times up to
    `common.LAST_TRAINING_VALID`), with the 5th percentile added for the lower tail.
    """
    values = power[: int(common.grid_index(times=common.LAST_TRAINING_VALID)) + 1]
    values = values[np.isfinite(values)]
    return {
        "p99": float(np.quantile(values, 0.99)),
        "p95": float(np.quantile(values, 0.95)),
        "p05": float(np.quantile(values, 0.05)),
    }


def all_thresholds() -> dict[int, dict[str, float]]:
    """Return the thresholds of every demand and generator series, by `time_series_id`."""
    series = common.series_table()
    sites = gen.site_table()
    arrays = common.load_power_arrays(series=series)
    out = {
        int(r["time_series_id"]): history_thresholds(power=arrays[r["site"]].power)
        for r in series.iter_rows(named=True)
    }
    out.update(
        {
            int(i): history_thresholds(power=power)
            for i, power in gen.full_power(sites=sites).items()
        }
    )
    return out


# --- descriptors of the cases ------------------------------------------------------------------


def daily_clear_sky_classes() -> pl.DataFrame:
    """Return each site's sunny, middle, or cloudy class for every UK calendar day.

    The daily clear-sky index of a site is the sum of CAMS global irradiance over the sum of the
    clear-sky irradiance, over the half-hours of the UK day with a positive clear-sky value. Within
    each site and calendar month, the top third of days is `sunny` and the bottom third `cloudy`.
    A wind farm has no CAMS series, so each of the three takes the class of the mean index of the
    six solar farms.

    Returns:
        One row per site and day with `site`, `date`, and `csi_class`.
    """
    frame = pl.read_parquet(
        CAMS_FRAME_PATH, columns=["site", "kind", "time", "ghi_u", "ghi_u_clear"]
    )
    daily = (
        frame.filter(pl.col("ghi_u_clear") > 0)
        .with_columns(
            date=(pl.col("time") - timedelta(minutes=30))
            .dt.convert_time_zone("Europe/London")
            .dt.date()
        )
        .group_by("site", "kind", "date")
        .agg(csi=pl.col("ghi_u").sum() / pl.col("ghi_u_clear").sum())
    )
    trial = (
        daily.filter(pl.col("kind") == "farm")
        .group_by("date")
        .agg(csi=pl.col("csi").mean())
        .select(site=pl.lit("TRIAL_AREA"), kind=pl.lit("farm"), date="date", csi="csi")
    )
    pooled = pl.concat([daily.select("site", "kind", "date", "csi"), trial])
    classed = pooled.with_columns(month=pl.col("date").dt.strftime("%Y-%m")).with_columns(
        low=pl.col("csi").quantile(1 / 3).over("site", "month"),
        high=pl.col("csi").quantile(2 / 3).over("site", "month"),
    )
    classed = classed.with_columns(
        csi_class=pl.when(pl.col("csi") >= pl.col("high"))
        .then(pl.lit("sunny"))
        .when(pl.col("csi") <= pl.col("low"))
        .then(pl.lit("cloudy"))
        .otherwise(pl.lit("middle"))
    ).select("site", "date", "csi_class")
    wind = pl.DataFrame({"site": ["W1", "W2", "W3"]}).join(
        classed.filter(pl.col("site") == "TRIAL_AREA").drop("site"), how="cross"
    )
    return pl.concat([classed.filter(pl.col("site") != "TRIAL_AREA"), wind])


@dataclass
class ClassData:
    """One class's cases, their descriptors, and every arm's metrics aligned to the cases.

    Attributes:
        name: The class name.
        cases: One row per case: the keys, `site`, `lead_hours`, `origin_month`, and descriptors.
        months: The sorted origin-month labels.
        month_index: Each case's index into `months`.
        gap: Whether a case belongs to the 292 runs of `fold_a_daily_gap_v1`.
        every5d: Whether a case belongs to the 73 runs of `fold_a_every5d_v1`.
        dimensions: Each scope dimension's case codes (-1 for a case outside the dimension) and
            labels.
        metrics: Each arm's per-case metrics, NaN where the arm lacks the case.
        reliability: Each arm's count, per quantile level and month, of cases at or below the
            quantile, shape (11, months), counting only the arm's own cases.
        reliability_counts: Each arm's case count per month.
        tails: Each arm's tail-calibration table.
        missing: Arms of the class that have no file.
    """

    name: str
    cases: pl.DataFrame
    months: list[str]
    month_index: np.ndarray
    gap: np.ndarray
    every5d: np.ndarray
    dimensions: dict[str, tuple[np.ndarray, list[str]]]
    metrics: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    reliability: dict[str, np.ndarray] = field(default_factory=dict)
    reliability_counts: dict[str, np.ndarray] = field(default_factory=dict)
    tails: dict[str, pl.DataFrame] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def arm_metrics(self, *, name: str) -> dict[str, np.ndarray]:
        """Return an arm's metrics, following an alias."""
        return self.metrics[arms.resolve(name=name)]

    def has(self, *, name: str) -> bool:
        """Return whether an arm (or the arm an alias stands for) was loaded."""
        return arms.resolve(name=name) in self.metrics

    def subset(self, *, runs: str) -> np.ndarray:
        """Return the mask of the cases in a run subset: `gap` or `every5d`."""
        return self.gap if runs == "gap" else self.every5d


def read_arm(*, arms_dir: Path, name: str, groups: Sequence[str]) -> pl.DataFrame | None:
    """Read an arm's usable rows for some groups, sorted by the keys, or `None` if it has none.

    A row is usable when its valid time is before the holdout and its observation and 11
    quantiles are finite.
    """
    path = arms_dir / f"{arms.resolve(name=name)}.parquet"
    if not path.exists():
        return None
    frame = pl.read_parquet(path, columns=list(READ_COLUMNS)).filter(
        pl.col("group").is_in(list(groups))
        & (pl.col("valid_time") < common.HOLDOUT)
        & pl.all_horizontal(pl.col("actual", *common.LEVEL_COLUMNS).is_finite())
    )
    return None if frame.is_empty() else frame.sort(*assemble.KEY_COLUMNS)


def case_descriptors(*, keys: pl.DataFrame, csi: pl.DataFrame) -> pl.DataFrame:
    """Add the lead, UK hour, season, origin month, and sun class to the cases' keys.

    Args:
        keys: The keys plus `site`.
        csi: `daily_clear_sky_classes()`.

    Returns:
        The keys with `lead_hours`, `lead_day`, `band`, `hour_uk`, `season`, `origin_month`, and
        `csi_class`.
    """
    lead = (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    local = (pl.col("valid_time") - timedelta(minutes=30)).dt.convert_time_zone("Europe/London")
    edges = np.array([low for _, low, _ in common.LEAD_BANDS[1:]])
    described = keys.with_columns(
        lead_hours=lead,
        hour_uk=local.dt.hour(),
        month_uk=local.dt.month(),
        date=local.dt.date(),
        origin_month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
    )
    band = np.searchsorted(edges, described["lead_hours"].to_numpy(), side="right")
    return (
        described.with_columns(
            lead_day=(pl.col("lead_hours") // 24).clip(upper_bound=LAST_LEAD_DAY).cast(pl.Int32),
            band=pl.Series(band),
            season=pl.col("month_uk").replace_strict(SEASON_OF_MONTH, return_dtype=pl.String),
        )
        .join(csi, on=["site", "date"], how="left", maintain_order="left")
        .with_columns(csi_class=pl.col("csi_class").fill_null("middle"))
        .drop("month_uk", "date")
    )


def codes_of(*, values: pl.Series, labels: Sequence[str]) -> np.ndarray:
    """Return each value's index in `labels`."""
    return values.replace_strict(
        {label: i for i, label in enumerate(labels)}, return_dtype=pl.Int64
    ).to_numpy()


def dimensions_of(*, cases: pl.DataFrame) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Return the scope dimensions: case codes and labels for each way of splitting the cases."""
    sites = sorted(cases["site"].unique().to_list(), key=lambda s: (len(s), s))
    midday = cases["hour_uk"].is_in(list(MIDDAY_HOURS)).to_numpy()
    lead_day = cases["lead_day"].to_numpy()
    months = sorted(cases["origin_month"].unique().to_list())
    day_labels = [str(d) for d in range(LAST_LEAD_DAY + 1)]
    return {
        "lead_day": (lead_day, day_labels),
        "lead_day_midday": (np.where(midday, lead_day, -1), day_labels),
        "lead_band": (cases["band"].to_numpy(), list(LEAD_BAND_NAMES)),
        "hour_uk": (cases["hour_uk"].to_numpy(), [f"{h:02d}" for h in range(24)]),
        "origin_month": (codes_of(values=cases["origin_month"], labels=months), months),
        "season": (codes_of(values=cases["season"], labels=SEASONS), list(SEASONS)),
        "sun": (codes_of(values=cases["csi_class"], labels=CSI_CLASSES), list(CSI_CLASSES)),
        "series": (codes_of(values=cases["site"], labels=sites), sites),
    }


def reliability_counts(
    *, quantiles: np.ndarray, actual: np.ndarray, months: np.ndarray, n_months: int
) -> np.ndarray:
    """Return, per level and month, the cases with the observation at or below the quantile."""
    out = np.empty((quantiles.shape[1], n_months))
    for level in range(quantiles.shape[1]):
        out[level] = np.bincount(months[actual <= quantiles[:, level]], minlength=n_months).astype(
            np.float64
        )
    return out


def add_arm(
    *,
    data: ClassData,
    arm: str,
    aligned: pl.DataFrame,
    case_thresholds: Mapping[str, np.ndarray],
) -> None:
    """Score one arm on its rows among the class's cases and store the results in `data`.

    Args:
        data: The class data, which receives the arm's metrics, reliability counts, and tails.
        arm: The arm's file name.
        aligned: The arm's usable rows that are among the class's cases.
        case_thresholds: Each case's `p99`, `p95`, and `p05` of its series' history.
    """
    quantiles = np.sort(aligned.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1)
    actual = aligned["actual"].to_numpy().astype(np.float64)
    own = (
        aligned.select(*assemble.KEY_COLUMNS)
        .join(
            data.cases.select(*assemble.KEY_COLUMNS).with_row_index("case"),
            on=list(assemble.KEY_COLUMNS),
            how="left",
            maintain_order="left",
        )["case"]
        .to_numpy()
    )
    metrics = case_metrics(
        quantiles=quantiles,
        actual=actual,
        thresholds={key: value[own] for key, value in case_thresholds.items()},
    )
    full = {}
    for metric, values in metrics.items():
        column = np.full(data.cases.height, np.nan)
        column[own] = values
        full[metric] = column
    data.metrics[arm] = full
    n_months = len(data.months)
    data.reliability[arm] = reliability_counts(
        quantiles=quantiles, actual=actual, months=data.month_index[own], n_months=n_months
    )
    data.reliability_counts[arm] = np.bincount(data.month_index[own], minlength=n_months).astype(
        np.float64
    )
    scaled = aligned.with_columns(
        *[(pl.col(c) * POINTS).alias(c) for c in ("actual", "q01", "q05", "q95", "q99")]
    )
    data.tails[arm] = tail_calibration_table(frame=scaled)


def load_class(
    *,
    name: str,
    arms_dir: Path,
    thresholds: Mapping[int, Mapping[str, float]],
    csi: pl.DataFrame,
    site_of: Mapping[int, str],
) -> ClassData:
    """Read a class's arms and compute every arm's per-case metrics on the shared cases.

    Args:
        name: A class name from `arms.CLASSES`.
        arms_dir: The folder of arm tables.
        thresholds: Each series' history thresholds.
        csi: `daily_clear_sky_classes()`.
        site_of: Each series' site label.

    Returns:
        The class's data.

    Raises:
        FileNotFoundError: If no model arm of the class exists.
    """
    groups = CLASS_GROUPS[name]
    model_arms = list(
        dict.fromkeys(arms.resolve(name=a) for a in (*arms.STAIRCASE[name], *arms.ABLATIONS[name]))
    )
    reference_arms = list(arms.REFERENCES)
    shared: pl.DataFrame | None = None
    present_models, missing = [], []
    for arm in model_arms:
        frame = read_arm(arms_dir=arms_dir, name=arm, groups=groups)
        if frame is None:
            missing.append(arm)
            continue
        present_models.append(arm)
        keys = frame.select(*assemble.KEY_COLUMNS)
        shared = (
            keys
            if shared is None
            else shared.join(keys, on=list(assemble.KEY_COLUMNS), how="inner")
        )
        del frame
    if shared is None:
        msg = f"no model arm of class {name!r} found under {arms_dir}"
        raise FileNotFoundError(msg)
    shared = shared.sort(*assemble.KEY_COLUMNS).with_columns(
        site=pl.col("time_series_id").replace_strict(dict(site_of), return_dtype=pl.String)
    )
    cases = case_descriptors(keys=shared, csi=csi)
    months = sorted(cases["origin_month"].unique().to_list())
    month_index = np.searchsorted(months, cases["origin_month"].to_numpy())
    gap_runs = common.origins_of_subset(subset="gap")
    every5d_runs = common.origins_of_subset(subset="every5d")
    data = ClassData(
        name=name,
        cases=cases,
        months=months,
        month_index=month_index,
        gap=cases["power_fcst_init_time"].is_in(gap_runs).to_numpy(),
        every5d=cases["power_fcst_init_time"].is_in(every5d_runs).to_numpy(),
        dimensions=dimensions_of(cases=cases),
        missing=missing,
    )
    ids = cases["time_series_id"].to_numpy()
    case_thresholds = {
        key: np.array([thresholds[int(i)][key] for i in ids]) for key in ("p99", "p95", "p05")
    }
    keys_only = shared.select(*assemble.KEY_COLUMNS)
    for arm in [*present_models, *reference_arms]:
        frame = read_arm(arms_dir=arms_dir, name=arm, groups=groups)
        aligned = (
            None
            if frame is None
            else frame.join(keys_only, on=list(assemble.KEY_COLUMNS), how="semi")
        )
        if aligned is None or aligned.is_empty():
            if arm in reference_arms or aligned is not None:
                data.missing.append(arm)
            continue
        add_arm(data=data, arm=arm, aligned=aligned, case_thresholds=case_thresholds)
        print(f"{name}: scored {arm} on {aligned.height:,} of {cases.height:,} cases", flush=True)
    return data


# --- tables ------------------------------------------------------------------------------------


def interval_of_mean(
    *, data: ClassData, values: np.ndarray, runs: str, level: float, n_resamples: int
) -> dict[str, float]:
    """Return the mean of a per-case metric with its month-resampled interval.

    Args:
        data: The class data.
        values: One value per case, NaN where an arm lacks the case.
        runs: `gap` or `every5d`.
        level: The interval's coverage in percent.
        n_resamples: The number of resamples.

    Returns:
        `value`, `lower`, `upper`, the case count `cases`, and `months`.
    """
    valid = np.isfinite(values) & data.subset(runs=runs)
    sums, counts = month_totals(
        values=values,
        valid=valid,
        codes=np.zeros(values.size, dtype=np.int64),
        n_codes=1,
        months=data.month_index,
        n_months=len(data.months),
    )
    lower, upper = month_intervals(sums=sums, counts=counts, level=level, n_resamples=n_resamples)
    cases = float(counts.sum())
    return {
        "value": float(sums.sum() / cases) if cases else float("nan"),
        "lower": float(lower[0]),
        "upper": float(upper[0]),
        "cases": int(cases),
        "months": int((counts[0] > 0).sum()),
    }


def absolute_table(*, data: ClassData, runs: str) -> pl.DataFrame:
    """Return every arm's mean metrics with 95% intervals, on the run subset."""
    rows = []
    for arm in data.metrics:
        row: dict[str, object] = {"class": data.name, "arm": arm, "runs": runs}
        for metric in METRICS:
            result = interval_of_mean(
                data=data,
                values=data.metrics[arm][metric],
                runs=runs,
                level=LEVEL_95,
                n_resamples=N_RESAMPLES,
            )
            row[metric] = result["value"]
            row[f"{metric}_lower"] = result["lower"]
            row[f"{metric}_upper"] = result["upper"]
            row["cases"] = result["cases"]
        if row.get("cases"):
            rows.append(row)
    return pl.DataFrame(rows)


def scope_table(*, data: ClassData, runs: str = "gap") -> pl.DataFrame:
    """Return each arm's `qs` and `mae` by every scope dimension, with 95% intervals.

    The origin-month dimension has no interval, because each of its scopes is one month.
    """
    subset = data.subset(runs=runs)
    rows = []
    for arm, metrics in data.metrics.items():
        for dimension, (codes, labels) in data.dimensions.items():
            for metric in HEADLINE_METRICS:
                values = metrics[metric]
                sums, counts = month_totals(
                    values=values,
                    valid=np.isfinite(values) & subset,
                    codes=codes,
                    n_codes=len(labels),
                    months=data.month_index,
                    n_months=len(data.months),
                )
                if dimension == "origin_month":
                    lower = upper = np.full(len(labels), np.nan)
                else:
                    lower, upper = month_intervals(
                        sums=sums, counts=counts, level=LEVEL_95, n_resamples=N_SCOPE_RESAMPLES
                    )
                total = counts.sum(axis=1)
                for index, label in enumerate(labels):
                    if total[index] == 0:
                        continue
                    rows.append(
                        {
                            "class": data.name,
                            "arm": arm,
                            "dimension": dimension,
                            "scope": label,
                            "metric": metric,
                            "value": float(sums[index].sum() / total[index]),
                            "lower": float(lower[index]),
                            "upper": float(upper[index]),
                            "cases": int(total[index]),
                        }
                    )
    return pl.DataFrame(rows)


def contrast_row(
    *,
    data: ClassData,
    treatment: str,
    reference: str,
    metric: str,
    level: float,
    runs: str,
    n_resamples: int = N_RESAMPLES,
) -> dict[str, object]:
    """Return treatment minus reference in one metric on the cases both arms hold.

    Args:
        data: The class data.
        treatment: The arm whose metric is compared.
        reference: The arm it is compared against.
        metric: A key of `METRICS`.
        level: The interval's coverage in percent.
        runs: `gap` or `every5d`.
        n_resamples: The number of resamples.

    Returns:
        The difference (negative means the treatment scores lower, which is better), its interval,
        the case count, and the number of months.
    """
    difference = data.arm_metrics(name=treatment)[metric] - data.arm_metrics(name=reference)[metric]
    result = interval_of_mean(
        data=data, values=difference, runs=runs, level=level, n_resamples=n_resamples
    )
    return {
        "class": data.name,
        "treatment": treatment,
        "reference": reference,
        "metric": metric,
        "level": level,
        "runs": runs,
        "difference": result["value"],
        "lower": result["lower"],
        "upper": result["upper"],
        "cases": result["cases"],
        "months": result["months"],
    }


def verdict_text(*, row: Mapping[str, object], margin: float) -> str:
    """Return the verdict of a pre-registered contrast in plain words.

    Args:
        row: The contrast row at the verdict level.
        margin: The non-inferiority margin in points (0 for a superiority test).

    Returns:
        The verdict sentence.
    """
    lower, upper = float(row["lower"]), float(row["upper"])  # ty: ignore[invalid-argument-type]
    if margin == 0.0:
        if upper < 0:
            return (
                "BG2 scores lower (better), statistically significant at the 5% level (family-wise)"
            )
        if lower > 0:
            return (
                "BG2 scores higher (worse), statistically significant at the 5% level (family-wise)"
            )
        return "not resolved: the interval contains zero"
    if upper < 0:
        return f"BG2 is better, and so non-inferior within +{margin:.2f} (family-wise 5% level)"
    if upper < margin:
        return f"non-inferior: the upper bound is below the +{margin:.2f} margin"
    return f"not shown non-inferior: the upper bound is not below the +{margin:.2f} margin"


def contrast_tables(*, data: ClassData, runs: str = "gap") -> pl.DataFrame:
    """Return the verdict, reference, staircase-step, and ablation contrasts of one class."""
    bg2 = arms.BG2_OF_CLASS[data.name]
    previous = arms.PREVIOUS_BEST_OF_CLASS[data.name]
    rows: list[dict[str, object]] = []

    def add(
        *,
        family: str,
        label: str,
        treatment: str,
        reference: str,
        metrics: Sequence[str],
        level: float,
    ) -> None:
        if not (data.has(name=treatment) and data.has(name=reference)):
            return
        for metric in metrics:
            row = contrast_row(
                data=data,
                treatment=treatment,
                reference=reference,
                metric=metric,
                level=level,
                runs=runs,
            )
            if row["cases"]:
                rows.append({"family": family, "id": label, **row})

    add(
        family="verdict",
        label=f"verdict {data.name}",
        treatment=bg2,
        reference=previous,
        metrics=("qs",),
        level=arms.VERDICT_LEVEL,
    )
    add(
        family="previous best",
        label="BG2 minus previous best",
        treatment=bg2,
        reference=previous,
        metrics=METRICS,
        level=LEVEL_95,
    )
    for reference in arms.REFERENCES:
        add(
            family="reference",
            label=f"BG2 minus {reference}",
            treatment=bg2,
            reference=reference,
            metrics=METRICS,
            level=LEVEL_95,
        )
    staircase = arms.STAIRCASE[data.name]
    for earlier, later in itertools.pairwise(staircase):
        add(
            family="step",
            label=f"step {later} minus {earlier}",
            treatment=later,
            reference=earlier,
            metrics=HEADLINE_METRICS,
            level=LEVEL_95,
        )
    for ablation in arms.ABLATIONS[data.name]:
        add(
            family="ablation",
            label=f"{ablation} minus BG2",
            treatment=ablation,
            reference=bg2,
            metrics=HEADLINE_METRICS,
            level=LEVEL_95,
        )
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def staircase_table(*, data: ClassData, runs: str = "gap") -> pl.DataFrame:
    """Return the class's staircase: each rung's score, interval, and step from the rung before."""
    rows = []
    previous = None
    for position, arm in enumerate(arms.STAIRCASE[data.name]):
        if not data.has(name=arm):
            continue
        absolute = interval_of_mean(
            data=data,
            values=data.arm_metrics(name=arm)["qs"],
            runs=runs,
            level=LEVEL_95,
            n_resamples=N_RESAMPLES,
        )
        mae = interval_of_mean(
            data=data,
            values=data.arm_metrics(name=arm)["mae"],
            runs=runs,
            level=LEVEL_95,
            n_resamples=N_RESAMPLES,
        )
        row: dict[str, object] = {
            "class": data.name,
            "position": position,
            "arm": arm,
            "label": arms.arm_label(name=arm),
            "qs": absolute["value"],
            "qs_lower": absolute["lower"],
            "qs_upper": absolute["upper"],
            "mae": mae["value"],
            "mae_lower": mae["lower"],
            "mae_upper": mae["upper"],
            "cases": absolute["cases"],
        }
        if previous is not None:
            step = contrast_row(
                data=data, treatment=arm, reference=previous, metric="qs", level=LEVEL_95, runs=runs
            )
            row["step_from"] = previous
            row["step"] = step["difference"]
            row["step_lower"] = step["lower"]
            row["step_upper"] = step["upper"]
        rows.append(row)
        if arm != "g_a_wind_margin":
            previous = arm
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def reliability_table(*, data: ClassData) -> pl.DataFrame:
    """Return, per arm and level, the observed share at or below the quantile, with intervals."""
    rows = []
    for arm, counts in data.reliability.items():
        totals = data.reliability_counts[arm]
        lower, upper = month_intervals(
            sums=counts,
            counts=np.tile(totals, (counts.shape[0], 1)),
            level=LEVEL_95,
            n_resamples=N_SCOPE_RESAMPLES,
        )
        for index, level in enumerate(common.LEVELS):
            rows.append(
                {
                    "class": data.name,
                    "arm": arm,
                    "nominal": float(level),
                    "observed": float(counts[index].sum() / totals.sum()),
                    "lower": float(lower[index]),
                    "upper": float(upper[index]),
                    "cases": int(totals.sum()),
                }
            )
    return pl.DataFrame(rows)


def tails_table(*, data: ClassData) -> pl.DataFrame:
    """Return every arm's tail-calibration table (library output, pinball in points of capacity)."""
    return pl.concat(
        [
            table.with_columns(**{"class": pl.lit(data.name), "arm": pl.lit(arm)})
            for arm, table in data.tails.items()
        ]
    )


def noise_floor_table(
    *, arms_dir: Path, thresholds: Mapping[int, Mapping[str, float]]
) -> pl.DataFrame:
    """Return the repeat fit of the weekly model against the first fit, at the series it covers.

    The repeat is `p_noise_repeat_weekly` (a second fit of the weekly model; `e00087_assemble`
    writes it when `work/anchor_preds_repeat` exists). The table holds the paired difference in
    `qs` and `mae`, repeat minus first fit, with 95% month-resampled intervals; the absolute value
    of the difference is the noise floor of the fitting seed.

    Args:
        arms_dir: The folder of arm tables.
        thresholds: Each series' history thresholds.

    Returns:
        One row per metric, or an empty frame if the repeat or the first fit is absent.
    """
    repeat = read_arm(arms_dir=arms_dir, name="p_noise_repeat_weekly", groups=("Primary",))
    first = read_arm(arms_dir=arms_dir, name="p_s4_weekly", groups=("Primary",))
    if repeat is None or first is None:
        return pl.DataFrame()
    joined = first.join(
        repeat.select(*assemble.KEY_COLUMNS, *common.LEVEL_COLUMNS),
        on=list(assemble.KEY_COLUMNS),
        how="inner",
        suffix="_repeat",
        validate="1:1",
    ).sort(*assemble.KEY_COLUMNS)
    ids = joined["time_series_id"].to_numpy()
    actual = joined["actual"].to_numpy().astype(np.float64)
    th = {key: np.array([thresholds[int(i)][key] for i in ids]) for key in ("p99", "p95", "p05")}
    first_q = np.sort(joined.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1)
    repeat_q = np.sort(
        joined.select([f"{c}_repeat" for c in common.LEVEL_COLUMNS]).to_numpy().astype(np.float64),
        axis=1,
    )
    metrics_first = case_metrics(quantiles=first_q, actual=actual, thresholds=th)
    metrics_repeat = case_metrics(quantiles=repeat_q, actual=actual, thresholds=th)
    months = sorted(joined["power_fcst_init_time"].dt.strftime("%Y-%m").unique().to_list())
    month_index = np.searchsorted(
        months, joined["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    )
    rows = []
    for metric in HEADLINE_METRICS:
        difference = metrics_repeat[metric] - metrics_first[metric]
        sums, counts = month_totals(
            values=difference,
            valid=np.ones(difference.size, dtype=bool),
            codes=np.zeros(difference.size, dtype=np.int64),
            n_codes=1,
            months=month_index,
            n_months=len(months),
        )
        lower, upper = month_intervals(
            sums=sums, counts=counts, level=LEVEL_95, n_resamples=N_RESAMPLES
        )
        rows.append(
            {
                "metric": metric,
                "series": joined["site"].n_unique(),
                "cases": joined.height,
                "first_fit": float(metrics_first[metric].mean()),
                "repeat_fit": float(metrics_repeat[metric].mean()),
                "difference": float(difference.mean()),
                "lower": float(lower[0]),
                "upper": float(upper[0]),
            }
        )
    return pl.DataFrame(rows)


# --- reproduction and scorer checks ------------------------------------------------------------


def inner_score_by_group(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return the nine-level quantile score of a frame's rows, per group, in points of capacity.

    Args:
        frame: Rows with `group`, `actual`, and the columns `q05` to `q95`.

    Returns:
        Each group's mean score on the nine inner levels with experiment 00048's masses.
    """
    out = {}
    for (group,), part in frame.partition_by("group", as_dict=True).items():
        quantiles = np.sort(
            part.select(common.INNER_LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1
        )
        score = common.pinball_quantile_score(
            quantiles=quantiles,
            actual=part["actual"].to_numpy().astype(np.float64),
            levels=common.INNER_LEVELS,
            masses=common.INNER_ATOM_MASSES,
        )
        out[str(group)] = float(score.mean() * POINTS)
    return out


def repro_check(*, arms_dir: Path) -> bool:
    """Score the weekly arms on the nine inner levels and compare them with experiment 00048.

    The weekly Primary arm (`p_s4_weekly`) must score within 0.02 points of 5.44, and the weekly
    BSP and GSP arm (`b_weekly`) within 0.02 of 7.84 and 18.66, on the 292 runs of
    `fold_a_daily_gap_v1`. The arms already carry the spread factor 1.15 and no correction.

    Args:
        arms_dir: The folder of arm tables.

    Returns:
        Whether every number passed. A number whose arm is missing counts as a failure.
    """
    gap = common.origins_of_subset(subset="gap")
    scores: dict[str, float] = {}
    for name in ("p_s4_weekly", "b_weekly"):
        frame = read_arm(arms_dir=arms_dir, name=name, groups=common.GROUPS)
        if frame is not None:
            scores.update(
                inner_score_by_group(frame=frame.filter(pl.col("power_fcst_init_time").is_in(gap)))
            )
    ok = True
    for label, group, target in REPRO_TARGETS:
        value = scores.get(group)
        passed = value is not None and abs(value - target) <= REPRO_TOLERANCE
        ok = ok and passed
        shown = "missing" if value is None else f"{value:.3f}"
        verdict = "PASS" if passed else "FAIL"
        print(f"{verdict}: {label}: {shown} against {target} (tolerance {REPRO_TOLERANCE})")
    return ok


def check_scorer() -> bool:
    """Score experiment 00048's stored weekly arm and check the scorer reproduces its table.

    Experiment 00048's `weekly_s` arm holds nine levels. Scoring it with `inner_score_by_group` on
    the 292 gap runs must give 5.44, 7.84, and 18.66 to within the two decimals that experiment
    00048 published. This tests the scorer without any fit of this experiment.

    Returns:
        Whether every number matched.
    """
    path = EXPERIMENT_00048_ARMS_DIR / "weekly_s.parquet"
    if not path.exists():
        print(f"FAIL: {path} is missing")
        return False
    gap = common.origins_of_subset(subset="gap")
    frame = pl.read_parquet(path).filter(
        pl.col("power_fcst_init_time").is_in(gap)
        & (pl.col("valid_time") < common.HOLDOUT)
        & pl.all_horizontal(pl.col("actual", *common.INNER_LEVEL_COLUMNS).is_finite())
    )
    scores = inner_score_by_group(frame=frame)
    ok = True
    for label, group, target in REPRO_TARGETS:
        value = scores.get(group)
        passed = value is not None and abs(value - target) <= CHECK_SCORER_TOLERANCE
        ok = ok and passed
        shown = "missing" if value is None else f"{value:.4f}"
        verdict = "PASS" if passed else "FAIL"
        print(f"{verdict}: {label} (experiment 00048's stored arm): {shown} against {target}")
    return ok


# --- verification on synthetic data ------------------------------------------------------------


def verify_intervals() -> list[str]:
    """Compare `month_intervals` with `studies.bootstrap` on synthetic paired losses.

    Returns:
        Report lines with both intervals at two levels; they agree to float precision.

    Raises:
        AssertionError: If the two disagree by more than 1e-9.
    """
    rng = np.random.default_rng(1)
    count = 6000
    months = np.array([f"2025-{m:02d}" for m in rng.integers(1, 13, size=count)])
    base = rng.normal(5.0, 1.0, size=count)
    treatment = base - 0.1 + rng.normal(0.0, 0.3, size=count)
    frame = pl.DataFrame(
        {
            "site": ["s"] * count,
            "time": [f"t{i:06d}" for i in range(count)],
            "seed": [0] * count,
            "month": months,
        }
    )
    losses = pl.concat(
        [
            frame.with_columns(qs=pl.Series(values), arm=pl.lit(arm))
            for arm, values in (("treatment", treatment), ("reference", base))
        ]
    )
    unique_months, index = np.unique(months, return_inverse=True)
    sums, counts = month_totals(
        values=treatment - base,
        valid=np.ones(count, dtype=bool),
        codes=np.zeros(count, dtype=np.int64),
        n_codes=1,
        months=index,
        n_months=len(unique_months),
    )
    lines = []
    for level in (95.0, arms.VERDICT_LEVEL):
        library = bootstrap.bootstrap_difference_at_level(
            losses=losses,
            treatment="treatment",
            reference="reference",
            metric="qs",
            level=level,
            n_resamples=2000,
        )
        lower, upper = month_intervals(sums=sums, counts=counts, level=level, n_resamples=2000)
        assert abs(library[0] - lower[0]) < 1e-9, (library, lower)
        assert abs(library[1] - upper[0]) < 1e-9, (library, upper)
        lines.append(
            f"level {level:.2f}: library ({library[0]:.6f}, {library[1]:.6f}), "
            f"month totals ({lower[0]:.6f}, {upper[0]:.6f})"
        )
    absolute = bootstrap.bootstrap_absolute(losses=losses, arm="treatment", metric="qs")
    sums_abs, counts_abs = month_totals(
        values=treatment,
        valid=np.ones(count, dtype=bool),
        codes=np.zeros(count, dtype=np.int64),
        n_codes=1,
        months=index,
        n_months=len(unique_months),
    )
    lower, upper = month_intervals(sums=sums_abs, counts=counts_abs, level=95.0, n_resamples=2000)
    assert abs(absolute["lower_95"] - lower[0]) < 1e-9, (absolute, lower)
    assert abs(absolute["upper_95"] - upper[0]) < 1e-9, (absolute, upper)
    lines.append("absolute interval of one arm agrees with bootstrap_absolute")
    return lines


def verify_metrics() -> list[str]:
    """Check the per-case metrics on cases whose answers are known by hand.

    Returns:
        One report line.

    Raises:
        AssertionError: If a metric differs from its hand value.
    """
    levels = common.LEVELS
    perfect = np.tile(np.array([0.5]), (3, 11))
    actual = np.array([0.5, 0.5, 0.5])
    th = {"p99": np.full(3, 0.9), "p95": np.full(3, 0.8), "p05": np.full(3, 0.1)}
    metrics = case_metrics(quantiles=perfect, actual=actual, thresholds=th)
    assert np.allclose(metrics["qs"], 0.0)
    assert np.allclose(metrics["mae"], 0.0)
    # A point forecast 0.1 below the observation: pinball is level * 0.1 at every level.
    shifted = np.tile(np.array([0.4]), (1, 11))
    expected = 2.0 * (levels * 0.1 * common.ATOM_MASSES).sum() * POINTS
    metrics = case_metrics(
        quantiles=shifted, actual=np.array([0.5]), thresholds={k: v[:1] for k, v in th.items()}
    )
    assert np.isclose(metrics["qs"][0], expected), (metrics["qs"][0], expected)
    assert np.isclose(metrics["mae"][0], 10.0)
    # Both below the p99 floor: the upper-tail score is zero. Both above the p05 cap: lower is zero.
    assert np.isclose(metrics["tw_p99"][0], 0.0)
    assert np.isclose(metrics["tw_p05"][0], 0.0)
    # The observation 0.2 above a p99 floor of 0.9, forecast at the floor, scores in the tail.
    tail = case_metrics(
        quantiles=np.tile(np.array([0.9]), (1, 11)),
        actual=np.array([1.1]),
        thresholds={k: v[:1] for k, v in th.items()},
    )
    assert tail["tw_p99"][0] > 0.0
    # A forecast of the observation on a low-tail case scores in the lower tail only.
    low = case_metrics(
        quantiles=np.tile(np.array([0.2]), (1, 11)),
        actual=np.array([0.0]),
        thresholds={k: v[:1] for k, v in th.items()},
    )
    assert low["tw_p05"][0] > 0.0
    assert np.isclose(low["tw_p99"][0], 0.0)
    return ["metric check passed: perfect, shifted, and tail cases match their hand values"]


# --- synthetic end-to-end test -----------------------------------------------------------------


def write_synthetic_arms(*, folder: Path) -> None:
    """Write fake arm tables for the Primary class and one reference, on real keys and series.

    Four Primary series and 37 gap runs spread over the year. Each arm's median is the observation
    plus
    a bias; the bias order is S1 > S2 > ... > S6 so the scored order is known, and the reference
    has a wide band that over-covers.

    Args:
        folder: Where the arm tables go.
    """
    series = common.series_table()
    primary = series.filter(pl.col("substation_type") == "Primary").head(4)
    context = assemble.make_context(
        smoke=False, store=False, origins=common.all_origins()[1::10], root=folder
    )
    biases = {
        "p_s1_frozen": 0.060,
        "p_s2_stack": 0.050,
        "p_s3_monthly": 0.040,
        "p_s4_weekly": 0.030,
        "p_s5_blend_no_offset": 0.020,
        "p_s6_bg2": 0.010,
        "p_a2_no_blend": 0.030,
        "p_a3_monthly_in_weekly_slot": 0.012,
        "p_a4_correction_all": 0.015,
        "p_a5_spread_off": 0.011,
        "p_a6_capacity_held": 0.010,
        "p_a7_three_members": 0.013,
        "ref_manual_heuristic": 0.080,
        "ref_climatology": 0.100,
        "ref_xgboost_baseline": 0.070,
    }
    rng = np.random.default_rng(5)
    keys = []
    for row in primary.iter_rows(named=True):
        for origin in context.origins:
            valid = [origin + (step + 1) * common.STEP for step in range(48)]
            keys.append(
                pl.DataFrame(
                    {
                        "time_series_id": [row["time_series_id"]] * 48,
                        "power_fcst_init_time": [origin] * 48,
                        "valid_time": valid,
                    },
                    schema={
                        "time_series_id": pl.Int32,
                        "power_fcst_init_time": pl.Datetime("us", "UTC"),
                        "valid_time": pl.Datetime("us", "UTC"),
                    },
                )
            )
    table = pl.concat(keys).with_columns(seed=pl.lit(0).cast(pl.Int8))
    actual = assemble.actual_array(frame=table, power_of=context.power_of)
    noise = rng.normal(0.0, 0.05, size=table.height)
    for name, bias in biases.items():
        width = 0.20 if name == "ref_climatology" else 0.05
        centre = np.nan_to_num(actual, nan=0.3) + bias + noise
        z = np.array(
            [-2.326, -1.645, -1.282, -0.674, -0.253, 0.0, 0.253, 0.674, 1.282, 1.645, 2.326]
        )
        quantiles = centre[:, None] + width * z
        frame = table.hstack(
            pl.DataFrame(quantiles.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
        )
        assemble.write_arm(
            ctx=context, name=name, frame=assemble.finalise(ctx=context, frame=frame)
        )


def self_test() -> None:
    """Run the metric and interval checks, then score synthetic arms end to end."""
    for line in verify_metrics():
        print(line)
    for line in verify_intervals():
        print(line)
    with tempfile.TemporaryDirectory(dir=common.WORK_DIR.parent, prefix="selftest_") as name:
        folder = Path(name)
        write_synthetic_arms(folder=folder)
        results = folder / "results"
        analyse(
            arms_dir=folder / "arms",
            results_dir=results,
            classes=("Primary",),
        )
        staircase = pl.read_parquet(results / "staircase.parquet").sort("position")
        scores = staircase["qs"].to_list()
        assert scores == sorted(scores, reverse=True), scores
        contrasts = pl.read_parquet(results / "contrasts.parquet")
        verdict = contrasts.filter(pl.col("family") == "verdict")
        assert verdict.height == 1, verdict
        assert verdict["upper"][0] < 0, verdict
        assert abs(verdict["level"][0] - arms.VERDICT_LEVEL) < 1e-9
        a1 = contrasts.filter(
            (pl.col("treatment") == "p_a1_no_offset") & (pl.col("metric") == "qs")
        )
        assert a1.height, a1
        assert abs(a1["difference"][0] - (staircase["qs"][4] - staircase["qs"][5])) < 1e-9
        reliability = pl.read_parquet(results / "reliability.parquet")
        well = reliability.filter(pl.col("arm") == "p_s6_bg2")
        worst_gap = float(np.abs(well["observed"].to_numpy() - well["nominal"].to_numpy()).max())
        assert worst_gap < 0.2, worst_gap
        print("self-test passed: the staircase order, the verdict, aliases, and reliability agree")


# --- the full analysis -------------------------------------------------------------------------


def markdown(*, frame: pl.DataFrame, digits: int = 3) -> str:
    """Render a frame as a pipe table, floats to `digits` decimals."""
    if frame.is_empty():
        return "(none)"
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in frame.iter_rows()
    ]
    return "\n".join([header, rule, *body])


def write_table(*, frame: pl.DataFrame, results_dir: Path, name: str) -> None:
    """Write a results table, skipping an empty one."""
    if not frame.is_empty():
        common.atomic_write(frame=frame, target=results_dir / f"{name}.parquet")


def analyse(*, arms_dir: Path, results_dir: Path, classes: Sequence[str]) -> None:
    """Score the classes and write the tables and `report.txt`.

    Args:
        arms_dir: The folder of arm tables.
        results_dir: Where the tables and the report go.
        classes: The classes to score.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    thresholds = all_thresholds()
    csi = daily_clear_sky_classes()
    site_of = {
        **{
            int(r["time_series_id"]): r["site"] for r in common.series_table().iter_rows(named=True)
        },
        **{int(r["time_series_id"]): r["site"] for r in gen.site_table().iter_rows(named=True)},
    }
    lines = ["# Experiment 00087 report", "", DESIGN_PREDICTION, ""]
    collected: dict[str, list[pl.DataFrame]] = {
        key: []
        for key in (
            "absolute",
            "absolute_every5d",
            "scopes",
            "contrasts",
            "contrasts_every5d",
            "staircase",
            "reliability",
            "tails",
        )
    }
    for name in classes:
        try:
            data = load_class(
                name=name, arms_dir=arms_dir, thresholds=thresholds, csi=csi, site_of=site_of
            )
        except FileNotFoundError as error:
            lines += [f"## {name}", "", f"not scored: {error}", ""]
            continue
        n_gap = int(data.gap.sum())
        lines += [
            f"## {name}",
            "",
            (
                f"Cases: {data.cases.height:,} (292-run subset {n_gap:,}, 73-run subset "
                f"{int(data.every5d.sum()):,}); series {data.cases['site'].n_unique()}; "
                f"arms without a file: {sorted(set(data.missing)) or 'none'}."
            ),
            "",
        ]
        outputs = {
            "absolute": absolute_table(data=data, runs="gap"),
            "absolute_every5d": absolute_table(data=data, runs="every5d"),
            "scopes": scope_table(data=data),
            "contrasts": contrast_tables(data=data),
            "contrasts_every5d": contrast_tables(data=data, runs="every5d"),
            "staircase": staircase_table(data=data),
            "reliability": reliability_table(data=data),
            "tails": tails_table(data=data),
        }
        for key, frame in outputs.items():
            if not frame.is_empty():
                collected[key].append(frame)
        lines += _class_report(name=name, outputs=outputs)
    for key, frames in collected.items():
        if frames:
            write_table(
                frame=pl.concat(frames, how="diagonal_relaxed"), results_dir=results_dir, name=key
            )
    noise = noise_floor_table(arms_dir=arms_dir, thresholds=thresholds)
    write_table(frame=noise, results_dir=results_dir, name="noise_floor")
    lines += ["## Noise floor (repeat fit of the weekly model)", "", markdown(frame=noise), ""]
    report = "\n".join(lines)
    (results_dir / "report.txt").write_text(report)
    print(report)


def _class_report(*, name: str, outputs: Mapping[str, pl.DataFrame]) -> list[str]:
    """Return the report lines of one class from its tables."""
    lines: list[str] = []
    absolute = outputs["absolute"]
    if not absolute.is_empty():
        lines += [
            "Absolute scores on the 292 runs (points of capacity; 95% month-resampled intervals):",
            "",
            markdown(
                frame=absolute.select(
                    "arm",
                    "qs",
                    "qs_lower",
                    "qs_upper",
                    "mae",
                    "mae_lower",
                    "mae_upper",
                    "tw_p99",
                    "tw_p95",
                    "tw_p05",
                    "cases",
                ).sort("qs")
            ),
            "",
        ]
    contrasts = outputs["contrasts"]
    verdict = (
        contrasts.filter(pl.col("family") == "verdict") if not contrasts.is_empty() else contrasts
    )
    if not verdict.is_empty():
        row = verdict.row(0, named=True)
        margin = arms.NON_INFERIORITY_MARGIN.get(name, 0.0)
        lines += [
            (
                f"**Verdict contrast** ({row['treatment']} minus {row['reference']}, "
                f"{row['level']:.2f}% interval, {N_RESAMPLES:,} resamples): "
                f"{row['difference']:+.3f} ({row['lower']:+.3f} to {row['upper']:+.3f}); "
                f"{verdict_text(row=row, margin=margin)}."
            ),
            "",
        ]
    for family, title in (
        ("previous best", "BG2 minus the previous best (95%)"),
        (
            "reference",
            "BG2 minus each reference (95%; negative means BG2 scores lower, which is better)",
        ),
        ("step", "Staircase steps (95%)"),
        ("ablation", "Ablation minus BG2 (95%; positive means dropping the idea scores worse)"),
    ):
        if contrasts.is_empty():
            break
        part = contrasts.filter(pl.col("family") == family)
        if not part.is_empty():
            lines += [
                f"{title}:",
                "",
                markdown(
                    frame=part.select(
                        "treatment", "reference", "metric", "difference", "lower", "upper", "cases"
                    )
                ),
                "",
            ]
    staircase = outputs["staircase"]
    if not staircase.is_empty():
        lines += [
            "Staircase (quantile score; step is the change from the rung before, 95% interval):",
            "",
            markdown(
                frame=staircase.select(
                    "position",
                    "arm",
                    "qs",
                    "qs_lower",
                    "qs_upper",
                    "mae",
                    "step",
                    "step_lower",
                    "step_upper",
                )
            ),
            "",
        ]
        lines += [f"Rung not run: {label} ({note})." for label, note in arms.NOT_RUN_STEPS]
        lines.append("")
    if not contrasts.is_empty():
        ablations = contrasts.filter((pl.col("family") == "ablation") & (pl.col("metric") == "qs"))
        if not ablations.is_empty():
            worst = ablations.sort("difference", descending=True).row(0, named=True)
            lines += [
                (
                    f"Largest ablation: {worst['treatment']} scores {worst['difference']:+.3f} "
                    f"points ({worst['lower']:+.3f} to {worst['upper']:+.3f}) against BG2."
                ),
                "",
            ]
    every5d = outputs["absolute_every5d"]
    if not every5d.is_empty():
        lines += [
            "Check on the 73 runs of fold_a_every5d_v1 (95% intervals):",
            "",
            markdown(
                frame=every5d.select("arm", "qs", "qs_lower", "qs_upper", "mae", "cases").sort("qs")
            ),
            "",
        ]
        previous = outputs["contrasts_every5d"]
        previous = previous.filter(
            (pl.col("family") == "previous best") & (pl.col("metric") == "qs")
        )
        if not previous.is_empty():
            row = previous.row(0, named=True)
            lines += [
                (
                    f"On the 73 runs BG2 minus the previous best is {row['difference']:+.3f} "
                    f"({row['lower']:+.3f} to {row['upper']:+.3f}, 95%)."
                ),
                "",
            ]
    tails = outputs["tails"]
    if not tails.is_empty():
        lines += [
            "Tail calibration (coverage target equals the level; pinball in points of capacity):",
            "",
            markdown(
                frame=tails.filter(pl.col("n_rows") > 0).select(
                    "arm", "level", "coverage", "coverage_lower_95", "coverage_upper_95", "pinball"
                )
            ),
            "",
        ]
    return lines


def main() -> None:
    """Run the requested checks or the full analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="+", choices=arms.CLASSES, default=list(arms.CLASSES))
    parser.add_argument("--arms-dir", type=Path, default=common.ARMS_DIR)
    parser.add_argument("--results-dir", type=Path, default=common.RESULTS_DIR)
    parser.add_argument(
        "--repro", action="store_true", help="the reproduction check; exits 1 on FAIL"
    )
    parser.add_argument("--check-scorer", action="store_true", help="score experiment 00048's arm")
    parser.add_argument("--verify", action="store_true", help="check the intervals and metrics")
    parser.add_argument("--self-test", action="store_true", help="score synthetic arms end to end")
    args = parser.parse_args()
    if args.repro:
        sys.exit(0 if repro_check(arms_dir=args.arms_dir) else 1)
    if args.check_scorer:
        sys.exit(0 if check_scorer() else 1)
    if args.verify:
        print("\n".join([*verify_metrics(), *verify_intervals()]))
        return
    if args.self_test:
        self_test()
        return
    analyse(arms_dir=args.arms_dir, results_dir=args.results_dir, classes=args.classes)


if __name__ == "__main__":
    main()
