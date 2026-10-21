"""Load experiment 135's arms on common rows and score their contrasts.

Every arm is put on one frame shape: keys (`time_series_id`, `power_fcst_init_time`, `valid_time`),
`site`, `lead_hours`, the 11 quantiles, `mean`, and `actual` in points of effective capacity, and
`seed` 0. A row is scored when it belongs to a run of the period, has a reading, lies outside the
hand-filtered switching events at its site, and lies at a lead of at most 168 hours.
`common_rows` keeps the rows every arm in a contrast holds. The quantile score of a row is twice the
mean pinball loss over the 11 levels, in points. A contrast's interval resamples whole calendar
months of run initialisations, paired across arms (`studies.bootstrap`).
"""

import json
from datetime import UTC, datetime
from typing import Final

import e00135_common as common
import e00135_run as run_module
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap

LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("12-36 h", 12.0, 36.0),
    ("0-24 h", 0.0, 24.0),
    ("24-72 h (days 2-3)", 24.0, 72.0),
    ("72-168 h (days 4-7)", 72.0, 168.0),
    ("all leads", 0.0, 168.0),
)
CONTRAST_LEVEL: Final[float] = 97.5
"""Coverage in percent of the two-sided interval: the Bonferroni level for two contrasts."""
FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "time_series_id",
    "power_fcst_init_time",
    "valid_time",
    "lead_hours",
    "seed",
    *LEVEL_COLUMNS,
    "mean",
    "actual",
)


def chosen_ridge_arm(*, site: str) -> str:
    """Return the J2 arm with the lowest tuning-month quantile score at a site.

    The score of an arm is the lowest, over the factor grid, of its summed quantile score on the
    tuning months, divided by the number of rows (`work/tuning_scores.json`).
    """
    stored = json.loads((common.WORK_DIR / "tuning_scores.json").read_text())["per_site"]
    result = next(item for item in stored if item["site"] == site)
    best, best_name = np.inf, ""
    for name in common.J2_ARM_NAMES:
        for scores in result["factor_scores"][name].values():
            total = np.array(scores)
            mean = total[:, 0].sum() / total[:, 1].sum()
            if mean < best:
                best, best_name = mean, name
    return best_name


def site_meta(*, sites: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Return each site's switching mask on the grid."""
    cache = common.load_cache()
    return {site: cache.sites[site].switching_mask for site in sites}


def in_period(*, frame: pl.DataFrame, periods: set[str]) -> pl.DataFrame:
    """Keep the rows whose run (by `power_fcst_init_time`) belongs to the periods."""
    runs = {
        common.origin_of(run=run): common.period_of_run(run=run)
        for run in common.run_dates()
        if common.period_of_run(run=run) in periods
    }
    return frame.filter(pl.col("power_fcst_init_time").is_in(list(runs)))


def drop_unscorable(*, frame: pl.DataFrame, sites: tuple[str, ...]) -> pl.DataFrame:
    """Drop rows with no reading, inside a switching event, or beyond 168 hours."""
    cache = common.load_cache()
    last = cache.end_times[-1].astype("datetime64[us]").astype(datetime).replace(tzinfo=UTC)
    frame = frame.filter(pl.col("valid_time") <= last)
    kept = []
    for site in sites:
        part = frame.filter(pl.col("site") == site)
        position = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        kept.append(part.filter(pl.Series(~cache.sites[site].switching_mask[position])))
    return pl.concat(kept).filter(pl.col("actual").is_finite() & (pl.col("lead_hours") <= 168.0))


def load_arm(*, arm: str, periods: set[str], sites: tuple[str, ...]) -> pl.DataFrame:
    """Return an arm's scorable rows for the periods; `j2` is each site's chosen J2 arm."""
    pieces = []
    for period in sorted(periods):
        for site in sites:
            name = chosen_ridge_arm(site=site) if arm == "j2" else arm
            path = run_module.out_path(arm=name, period=period, site=site)
            if path.exists():
                pieces.append(pl.read_parquet(path).with_columns(site=pl.lit(site)))
    frame = pl.concat(pieces).select(FRAME_COLUMNS)
    return drop_unscorable(frame=in_period(frame=frame, periods=periods), sites=sites)


def quantile_score(*, frame: pl.DataFrame) -> pl.Series:
    """Return twice the mean pinball loss over the 11 levels for each row, in points."""
    actual = frame["actual"].to_numpy().astype(np.float64)[:, None]
    quantiles = frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    error = actual - quantiles
    score = 2.0 * np.mean(
        np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error), axis=1
    )
    return pl.Series("qs", score)


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score and the absolute errors of the median and the mean forecast."""
    return frame.with_columns(
        quantile_score(frame=frame),
        abs_median=(pl.col("actual") - pl.col("q50")).abs(),
        abs_mean=(pl.col("actual") - pl.col("mean")).abs(),
    )


def common_rows(*, arms_by_name: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the rows all arms hold, keyed by series, origin, and valid time."""
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    shared = None
    for frame in arms_by_name.values():
        current = frame.select(keys).unique()
        shared = current if shared is None else shared.join(current, on=keys, how="inner")
    assert shared is not None
    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        "time_series_id": pl.Int32,
        "lead_hours": pl.Float64,
        "seed": pl.Int8,
        "actual": pl.Float32,
        "mean": pl.Float32,
        **dict.fromkeys(LEVEL_COLUMNS, pl.Float32),
    }
    return {
        name: frame.join(shared, on=keys, how="inner")
        .sort(keys)
        .with_columns(pl.col(column).cast(dtype) for column, dtype in schema.items())
        for name, frame in arms_by_name.items()
    }


def lead_filter(*, frame: pl.DataFrame, band: tuple[str, float, float]) -> pl.DataFrame:
    """Keep the rows whose lead lies in a band: lower bound open, upper bound closed."""
    _, low, high = band
    return frame.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))


def contrast(
    *,
    losses: pl.DataFrame,
    treatment: str,
    reference: str,
    metric: str = "qs",
    level: float = CONTRAST_LEVEL,
) -> dict:
    """Return a paired contrast: the estimate, its interval, and the sites it favours.

    Args:
        losses: `losses_for_bootstrap`'s output restricted to the rows wanted.
        treatment: The first-named arm.
        reference: The second-named arm.
        metric: The loss column.
        level: The interval's coverage in percent.

    Returns:
        Keys `estimate` (treatment minus reference, negative favours the treatment), `lower`,
        `upper`, `sites_better`, `site_differences`, and `n_sites`.
    """
    t = losses.filter(pl.col("arm") == treatment)
    r = losses.filter(pl.col("arm") == reference)
    estimate = float(t[metric].to_numpy().mean() - r[metric].to_numpy().mean())
    lower, upper = bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric=metric,
        level=level,
        n_resamples=20_000,
    )
    by_site = (
        t.group_by("site")
        .agg(t=pl.col(metric).mean())
        .join(r.group_by("site").agg(r=pl.col(metric).mean()), on="site")
        .sort("site")
    )
    differences = (by_site["t"] - by_site["r"]).to_list()
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "sites_better": int(sum(d < 0 for d in differences)),
        "site_differences": dict(zip(by_site["site"].to_list(), differences, strict=True)),
        "n_sites": by_site.height,
    }


def stack_losses(*, arms_by_name: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return `losses_for_bootstrap`'s frame for arms already on common rows.

    The string `site` label is dropped because the stacked frame builds its own `site` column from
    the series identifier.
    """
    return losses_for_bootstrap(
        arm_losses={
            name: with_losses(frame=frame).drop("site") for name, frame in arms_by_name.items()
        }
    )
