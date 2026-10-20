"""Load experiment 118's arms and the references on common rows and score their contrasts.

Every arm is put on one frame shape: keys (`time_series_id`, `power_fcst_init_time`, `valid_time`),
`site`, `lead_hours`, the 11 quantiles and `actual` in points of effective capacity, and `seed` 0.
A row is scored when it belongs to a replicate or scored run, has a reading, lies outside the
hand-filtered switching events at its site, and lies at a lead of at most 168 hours.
`common_rows` keeps the rows every arm in a contrast holds.

The quantile score of a row is twice the mean pinball loss over the 11 levels, in points. A
contrast's interval resamples whole calendar months of run initialisations, paired across arms
(`studies.bootstrap`), or whole local days where a subset of days is scored.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00118_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap

LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("D+1", 15.0, 39.0),
    ("0-24 h", 0.0, 24.0),
    ("24-48 h", 24.0, 48.0),
    ("48-144 h", 48.0, 144.0),
    ("144-168 h", 144.0, 168.0),
    ("12-36 h", 12.0, 36.0),
    ("all leads", 0.0, 168.0),
)
"""Each band's label and its lead range in hours: lower bound open, upper bound closed.

D+1 is the next UTC day's 48 half-hours from a 09:00 UTC origin (valid-time ends 15.5 to 39 hours
after the origin).
"""
E87_PREDICTIONS: Final[Path] = (
    common.DATA_DIR.parent
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
)
E87_NAMES: Final[dict[str, str]] = {
    "bg2": "p_s6_bg2",
    "climatology": "ref_climatology",
    "manual_heuristic": "ref_manual_heuristic",
}
PREDS_DIR: Final[Path] = common.WORK_DIR / "preds"
FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "time_series_id",
    "power_fcst_init_time",
    "valid_time",
    "lead_hours",
    "seed",
    *QUANTILE_COLUMNS,
    "actual",
)
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
EVENING_PEAK: Final[tuple[float, float]] = (16.0, 20.0)
"""Local hours of a half-hour's end that count as the evening peak: after 16:00 to 20:00."""
COLD_SNAP_PERCENTILE: Final[float] = 10.0
COLD_SNAP_BASELINE: Final[tuple[datetime, datetime]] = (
    datetime(2019, 9, 1, tzinfo=UTC),
    datetime(2024, 7, 1, tzinfo=UTC),
)
MIN_DAYS_FOR_INTERVAL: Final[int] = 15
LOCAL_ZONE: Final[str] = "Europe/London"
ARM_PERIOD_ORDER: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")


def site_meta(*, sites: tuple[str, ...]) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Return each site's series identifier and switching mask on the grid."""
    cache = common.load_cache()
    return (
        {site: cache.sites[site].table.time_series_id for site in sites},
        {site: cache.sites[site].switching_mask for site in sites},
    )


def in_period(*, frame: pl.DataFrame, periods: set[str]) -> pl.DataFrame:
    """Keep the rows whose run (by `power_fcst_init_time`) belongs to the periods."""
    origins = [
        common.origin_of(run=run)
        for run in common.run_dates()
        if common.period_of_run(run=run) in periods
    ]
    return frame.filter(pl.col("power_fcst_init_time").is_in(origins))


def drop_unscorable(*, frame: pl.DataFrame, sites: tuple[str, ...]) -> pl.DataFrame:
    """Drop rows with no reading, inside a switching event, or beyond 168 hours."""
    cache = common.load_cache()
    last = cache.end_times[-1].astype("datetime64[us]").astype(datetime).replace(tzinfo=UTC)
    frame = frame.filter(pl.col("valid_time") <= last)
    kept = []
    for site in sites:
        features = cache.sites[site]
        part = frame.filter(pl.col("site") == site)
        position = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        kept.append(part.filter(pl.Series(~features.switching_mask[position])))
    return pl.concat(kept).filter(pl.col("actual").is_finite() & (pl.col("lead_hours") <= 168.0))


def load_mine(*, arm: str, periods: set[str], sites: tuple[str, ...]) -> pl.DataFrame:
    """Return an arm of this experiment for the periods, in the common frame shape."""
    pieces = []
    for period in sorted(periods):
        for site in sites:
            path = PREDS_DIR / f"{arm}_{period}_{site}.parquet"
            if path.exists():
                pieces.append(pl.read_parquet(path).with_columns(site=pl.lit(site)))
    frame = pl.concat(pieces).select(*FRAME_COLUMNS, "mean")
    return drop_unscorable(frame=in_period(frame=frame, periods=periods), sites=sites)


def load_gam0(*, periods: set[str], sites: tuple[str, ...]) -> pl.DataFrame:
    """Return experiment 111's GAM-0 for the periods, in the common frame shape."""
    pieces = []
    for period in sorted(periods):
        for site in sites:
            path = common.E111_WORK / "dp_preds" / f"gam0_{period}_{site}.parquet"
            if path.exists():
                pieces.append(pl.read_parquet(path).with_columns(site=pl.lit(site)))
    frame = pl.concat(pieces).select(FRAME_COLUMNS)
    return drop_unscorable(frame=in_period(frame=frame, periods=periods), sites=sites)


def load_xnl_k(*, periods: set[str], sites: tuple[str, ...]) -> pl.DataFrame:
    """Return experiment 111's lagless XGBoost with the Kalman correction (X-nl+K)."""
    pieces = [
        pl.read_parquet(common.E111_WORK / "xnl_k" / f"{site}.parquet").select(FRAME_COLUMNS)
        for site in sites
        if (common.E111_WORK / "xnl_k" / f"{site}.parquet").exists()
    ]
    frame = pl.concat(pieces)
    return drop_unscorable(frame=in_period(frame=frame, periods=periods), sites=sites)


def load_xlag_d(*, periods: set[str], sites: tuple[str, ...]) -> pl.DataFrame:
    """Return experiment 111's lagged XGBoost with its default start (Xlag-d), in points."""
    cache = common.load_cache()
    pieces = []
    for site in sites:
        files = sorted((common.E111_WORK / "xgb_preds" / "xlag_d").glob(f"{site}_*.parquet"))
        if not files:
            continue
        frame = pl.concat([pl.read_parquet(f) for f in files])
        features = cache.sites[site]
        last = cache.end_times[-1].astype("datetime64[us]").astype(datetime).replace(tzinfo=UTC)
        frame = frame.filter(pl.col("valid_time") <= last)
        position = np.searchsorted(
            cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        pieces.append(
            frame.with_columns(
                site=pl.lit(site),
                actual=pl.Series(features.y[position].astype(np.float64)),
                seed=pl.lit(0).cast(pl.Int8),
            ).select(FRAME_COLUMNS)
        )
    frame = pl.concat(pieces)
    return drop_unscorable(frame=in_period(frame=frame, periods=periods), sites=sites)


def load_e87(*, name: str, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return an experiment 87 prediction set (BG2, climatology, ...) at the sites, in points."""
    ids, _ = site_meta(sites=sites)
    label_of = {series: site for site, series in ids.items()}
    frame = pl.read_parquet(E87_PREDICTIONS / E87_NAMES[name] / "predictions.parquet").filter(
        pl.col("time_series_id").is_in(list(label_of))
    )
    frame = frame.with_columns(
        site=pl.col("time_series_id").replace_strict(label_of, return_dtype=pl.String),
        actual=100.0 * pl.col("actual"),
        **{q: 100.0 * pl.col(q) for q in QUANTILE_COLUMNS},
    ).select(FRAME_COLUMNS)
    return drop_unscorable(frame=in_period(frame=frame, periods={"scored"}), sites=sites)


def quantile_score(*, frame: pl.DataFrame) -> pl.Series:
    """Return twice the mean pinball loss over the 11 levels for each row, in points."""
    actual = frame["actual"].to_numpy().astype(np.float64)[:, None]
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    error = actual - quantiles
    score = 2.0 * np.mean(
        np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error), axis=1
    )
    return pl.Series("qs", score)


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score and the median-forecast absolute error to a frame."""
    return frame.with_columns(
        quantile_score(frame=frame),
        abs_median=(pl.col("actual") - pl.col("q50")).abs(),
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
        **dict.fromkeys(QUANTILE_COLUMNS, pl.Float32),
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


def stack_losses(*, arms_by_name: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return `losses_for_bootstrap`'s frame for arms already on common rows, with local fields.

    The extra columns are `local_date` (the local date of the half-hour's start), `local_month`,
    and `local_hour` (the local hour at the half-hour's end), for subsets of days and hours.
    """
    stacked = losses_for_bootstrap(
        arm_losses={
            name: with_losses(frame=frame).drop("site") for name, frame in arms_by_name.items()
        }
    )
    local = pl.col("valid_time").dt.convert_time_zone(LOCAL_ZONE)
    return stacked.with_columns(
        local_date=(local - pl.duration(minutes=30)).dt.date().cast(pl.String),
        local_month=(local - pl.duration(minutes=30)).dt.month(),
        local_hour=local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60.0,
    )


def contrast(
    *,
    losses: pl.DataFrame,
    treatment: str,
    reference: str,
    metric: str = "qs",
    level: float = 95.0,
    by_day: bool = False,
    interval: bool = True,
) -> dict:
    """Return a paired contrast: the estimate, its interval, and the sites it favours.

    Args:
        losses: `stack_losses`'s output restricted to the rows wanted.
        treatment: The first-named arm.
        reference: The second-named arm.
        metric: The loss column.
        level: The interval's coverage in percent.
        interval: Whether to compute the interval; a secondary read of the estimate alone skips it.
        by_day: Resample whole local days instead of whole calendar months. With fewer than
            `MIN_DAYS_FOR_INTERVAL` days the interval is `None` and `too_few_days` is true.

    Returns:
        Keys `estimate` (treatment minus reference, negative favours the treatment), `lower`,
        `upper`, `sites_better`, `n_sites`, `n_rows`, `n_days`, and `too_few_days`.
    """
    t = losses.filter(pl.col("arm") == treatment)
    r = losses.filter(pl.col("arm") == reference)
    n_days = int(t["local_date"].n_unique()) if t.height else 0
    estimate = (
        float(t[metric].to_numpy().mean() - r[metric].to_numpy().mean()) if t.height else np.nan
    )
    too_few = by_day and n_days < MIN_DAYS_FOR_INTERVAL
    lower = upper = None
    if interval and t.height and not too_few and (by_day or t["month"].n_unique() > 1):
        scope = losses.with_columns(month=pl.col("local_date")) if by_day else losses
        lower, upper = bootstrap_difference_at_level(
            losses=scope, treatment=treatment, reference=reference, metric=metric, level=level
        )
    by_site = (
        t.group_by("site")
        .agg(t=pl.col(metric).mean())
        .join(r.group_by("site").agg(r=pl.col(metric).mean()), on="site")
    )
    return {
        "estimate": estimate,
        "lower": None if lower is None else float(lower),
        "upper": None if upper is None else float(upper),
        "sites_better": int((by_site["t"] < by_site["r"]).sum()),
        "n_sites": by_site.height,
        "n_rows": t.height,
        "n_days": n_days,
        "too_few_days": bool(too_few),
    }


def day_labels(
    *, losses: pl.DataFrame, kind: str, sites: dict[str, np.ndarray] | None = None
) -> pl.DataFrame:
    """Return the rows of one subset of days or hours: `winter`, `evening_peak`, or `winter_night`.

    Args:
        losses: `stack_losses`'s output.
        kind: `winter` (local November to March), `evening_peak` (winter half-hours ending after
            16:00 up to 20:00 local), or `winter_night` (winter half-hours ending after 00:00 up to
            06:00 local).
        sites: Unused; present for a uniform signature with `cold_snap_days`.

    Returns:
        The rows of the subset.

    Raises:
        ValueError: If `kind` is unknown.
    """
    winter = losses.filter(pl.col("local_month").is_in(list(WINTER_MONTHS)))
    if kind == "winter":
        return winter
    if kind == "evening_peak":
        return winter.filter(
            (pl.col("local_hour") > EVENING_PEAK[0]) & (pl.col("local_hour") <= EVENING_PEAK[1])
        )
    if kind == "winter_night":
        return winter.filter((pl.col("local_hour") > 0.0) & (pl.col("local_hour") <= 6.0))
    msg = f"unknown subset {kind!r}"
    raise ValueError(msg)


def cold_snap_dates(*, site: str) -> tuple[float, set[str]]:
    """Return a site's cold-snap threshold (degrees Celsius) and the local dates below it.

    A cold-snap day is a local November-to-March day whose mean ERA5 temperature is below the 10th
    percentile of such days from 2019-09 to 2024-06. The threshold uses only data before the tuning
    year.

    Args:
        site: The substation's label.

    Returns:
        The threshold and the set of cold-snap local dates (as `YYYY-MM-DD`) over the whole grid.
    """
    cache = common.load_cache()
    features = cache.sites[site]
    frame = pl.DataFrame(
        {
            "valid_time": cache.end_times,
            "temperature": features.temperature_c,
        }
    ).with_columns(pl.col("valid_time").dt.replace_time_zone("UTC"))
    local = pl.col("valid_time").dt.convert_time_zone(LOCAL_ZONE) - pl.duration(minutes=30)
    daily = (
        frame.with_columns(local_date=local.dt.date(), month=local.dt.month())
        .filter(pl.col("temperature").is_finite() & pl.col("month").is_in(list(WINTER_MONTHS)))
        .group_by("local_date")
        .agg(mean=pl.col("temperature").mean(), n=pl.len())
        .filter(pl.col("n") >= 40)
    )
    baseline = daily.filter(
        (pl.col("local_date") >= COLD_SNAP_BASELINE[0].date())
        & (pl.col("local_date") < COLD_SNAP_BASELINE[1].date())
    )
    threshold = float(np.percentile(baseline["mean"].to_numpy(), COLD_SNAP_PERCENTILE))
    cold = daily.filter(pl.col("mean") < threshold)
    return threshold, {str(d) for d in cold["local_date"].to_list()}
