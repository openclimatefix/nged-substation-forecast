"""Load experiment 111's arms on common rows and score their contrasts.

Every arm is put on one frame shape: keys (`time_series_id`, `power_fcst_init_time`,
`valid_time`), `site`, `lead_hours`, the 11 quantiles and `actual` in points of effective
capacity, and `seed` 0. A row is scored when it belongs to a scored run (or a replicate run), has a
reading, lies outside the hand-filtered switching events at its site, and lies at a lead of at most
168 hours. `common_rows` keeps the rows every arm in a contrast holds.

The quantile score of a row is twice the mean pinball loss over the 11 levels, in points. A
contrast's interval resamples whole calendar months of run initialisations, paired across arms
(`studies.bootstrap`).
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00111_common as common
import e00111_dp_run as dp_run
import e00111_xgb as xgb_module
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap

LEVEL_COLUMNS: Final[tuple[str, ...]] = xgb_module.LEVEL_COLUMNS
LEVELS: Final[np.ndarray] = xgb_module.LEVELS
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("24-48 h", 24.0, 48.0),
    ("48-144 h", 48.0, 144.0),
    ("144-168 h", 144.0, 168.0),
    ("12-36 h", 12.0, 36.0),
)
SIMP1_DIR: Final[Path] = (
    common.WORK_DIR.parents[1]
    / "00102-simp1-simplifying-the-bg2-best-guess-before-production"
    / "work"
    / "anchor_preds"
    / "n1h_nolag"
)
E87_PREDICTIONS: Final[Path] = (
    common.WORK_DIR.parents[1]
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
)
BONFERRONI_LEVEL: Final[float] = 98.75
"""98.75% intervals for the four screening contrasts C1 to C4."""
FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "time_series_id",
    "power_fcst_init_time",
    "valid_time",
    "lead_hours",
    "seed",
    *LEVEL_COLUMNS,
    "actual",
)


def site_meta(
    *, sites: tuple[str, ...]
) -> tuple[dict[str, int], dict[str, np.ndarray], dict[str, float]]:
    """Return each site's series identifier, switching mask on the grid, and capacity."""
    cache = common.load_cache()
    ids, masks, capacity = {}, {}, {}
    for site in sites:
        features = cache.sites[site]
        ids[site] = features.table.time_series_id
        masks[site] = features.switching_mask
        capacity[site] = features.table.effective_capacity_mw
    return ids, masks, capacity


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
        features = cache.sites[site]
        part = frame.filter(pl.col("site") == site)
        position = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        masked = features.switching_mask[position]
        kept.append(part.filter(pl.Series(~masked)))
    return pl.concat(kept).filter(pl.col("actual").is_finite() & (pl.col("lead_hours") <= 168.0))


def load_dp(*, arm: str, periods: set[str], sites: tuple[str, ...]) -> pl.DataFrame:
    """Return a DP arm's forecast rows for the periods, in the common frame shape."""
    pieces = []
    for period in sorted(periods):
        for site in sites:
            path = dp_run.out_path(arm=arm, period=period, site=site)
            if path.exists():
                pieces.append(pl.read_parquet(path).with_columns(site=pl.lit(site)))
    frame = pl.concat(pieces).select(FRAME_COLUMNS)
    return drop_unscorable(frame=in_period(frame=frame, periods=periods), sites=sites)


def load_xgb(*, arm: str, periods: set[str], sites: tuple[str, ...]) -> pl.DataFrame:
    """Return an XGBoost arm's scorable rows for the periods, in the common frame shape."""
    frame = load_xgb_raw(arm=arm, sites=sites)
    return drop_unscorable(frame=in_period(frame=frame, periods=periods), sites=sites)


def load_xgb_raw(*, arm: str, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return an XGBoost arm's rows, with `actual` from the observed power grid, in points."""
    import e00111_xgb_data as data

    ids, _, capacity = site_meta(sites=sites)
    pieces = []
    for site in sites:
        files = sorted((common.WORK_DIR / "xgb_preds" / arm).glob(f"{site}_*.parquet"))
        if not files:
            continue
        frame = pl.concat([pl.read_parquet(f) for f in files])
        power = data.power_array(time_series_id=ids[site], effective_capacity_mw=capacity[site])
        actual = 100.0 * power[data.grid_index(times=frame["valid_time"])]
        pieces.append(
            frame.with_columns(
                site=pl.lit(site),
                actual=pl.Series(actual),
                seed=pl.lit(0).cast(pl.Int8),
            )
        )
    return pl.concat(pieces).select(FRAME_COLUMNS)


def load_simp1_xnl(*, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return SIMP1's stored no-lag monthly refits, scorable rows of the scored year."""
    frame = load_simp1_raw(sites=sites)
    return drop_unscorable(frame=in_period(frame=frame, periods={"scored"}), sites=sites)


def load_simp1_raw(*, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return SIMP1's stored no-lag monthly refits of every run of the scored year, in points.

    Each monthly refit's file holds the runs the refit serves in either SIMP1 regime; the rows kept
    here are those of the monthly regime (origin from the refit to the next monthly refit).
    """
    import e00111_xgb_data as data

    ids, _, capacity = site_meta(sites=sites)
    refits = [
        t
        for t in common.monthly_cuts(last=xgb_module.LAST_MONTHLY_REFIT)
        if t >= common.SCORED_FIRST_RUN
    ]
    pieces = []
    for site in sites:
        power = data.power_array(time_series_id=ids[site], effective_capacity_mw=capacity[site])
        for index, refit in enumerate(refits):
            path = SIMP1_DIR / f"{site}_s0_{refit:%Y%m%dT%H%M}.parquet"
            if not path.exists():
                continue
            following = (
                refits[index + 1] if index + 1 < len(refits) else datetime(2026, 8, 1, tzinfo=UTC)
            )
            frame = pl.read_parquet(path).filter(
                (pl.col("power_fcst_init_time") >= refit)
                & (pl.col("power_fcst_init_time") < following)
            )
            actual = 100.0 * power[data.grid_index(times=frame["valid_time"])]
            quantiles = {q: 100.0 * frame[q] for q in LEVEL_COLUMNS}
            pieces.append(
                frame.with_columns(
                    site=pl.lit(site),
                    actual=pl.Series(actual),
                    seed=pl.lit(0).cast(pl.Int8),
                    lead_hours=(
                        pl.col("valid_time") - pl.col("power_fcst_init_time")
                    ).dt.total_minutes()
                    / 60.0,
                    **quantiles,
                ).select(FRAME_COLUMNS)
            )
    return pl.concat(pieces).filter(pl.col("lead_hours") <= 168.0)


def load_e87(*, name: str, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return an experiment 87 prediction set (BG2, climatology, ...) at the sites, in points."""
    ids, _, _ = site_meta(sites=sites)
    label_of = {series: site for site, series in ids.items()}
    frame = pl.read_parquet(E87_PREDICTIONS / name / "predictions.parquet").filter(
        pl.col("time_series_id").is_in(list(label_of))
    )
    frame = frame.with_columns(
        site=pl.col("time_series_id").replace_strict(label_of, return_dtype=pl.String),
        actual=100.0 * pl.col("actual"),
        **{q: 100.0 * pl.col(q) for q in LEVEL_COLUMNS},
    ).select(FRAME_COLUMNS)
    return drop_unscorable(frame=in_period(frame=frame, periods={"scored"}), sites=sites)


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
    level: float = 95.0,
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
        `upper`, `sites_better`, and `n_sites`.
    """
    t = losses.filter(pl.col("arm") == treatment)
    r = losses.filter(pl.col("arm") == reference)
    estimate = float(t[metric].to_numpy().mean() - r[metric].to_numpy().mean())
    lower, upper = bootstrap_difference_at_level(
        losses=losses, treatment=treatment, reference=reference, metric=metric, level=level
    )
    by_site = (
        t.group_by("site")
        .agg(t=pl.col(metric).mean())
        .join(r.group_by("site").agg(r=pl.col(metric).mean()), on="site")
    )
    better = int((by_site["t"] < by_site["r"]).sum())
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "sites_better": better,
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
