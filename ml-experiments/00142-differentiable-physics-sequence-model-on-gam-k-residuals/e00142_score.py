"""Load experiment 115's arms and the stored references on common rows and score their contrasts.

A row is scored when it belongs to the period's row rule (`row_period`: the origin and the valid
time both in the period's months for the tuning and replicate periods, any row of a scored run for
the scored year), has a reading, lies outside the hand-filtered switching events at its site, and
lies at a lead of at most 168 hours. `common_rows` keeps the rows every arm in a contrast holds.

The quantile score of a row is twice the mean pinball loss over the 11 levels, in points of
effective capacity. A contrast's interval resamples whole calendar months of run initialisations,
paired across arms (`studies.bootstrap`).
"""

from pathlib import Path
from typing import Final

import e00142_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap, read_predictions

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
    ("D+1", 15.0, 39.0),
    ("0-24 h", 0.0, 24.0),
    ("24-48 h", 24.0, 48.0),
    ("48-144 h", 48.0, 144.0),
    ("144-168 h", 144.0, 168.0),
    ("12-36 h", 12.0, 36.0),
    ("0.5-168 h", 0.0, 168.0),
)
BONFERRONI_LEVEL: Final[float] = 98.75
E87_PREDICTIONS: Final[Path] = (
    common.DATA_DIR.parent
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
)
REPLICATE_ROW_SET: Final[str] = "dp_replicate_v1"
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


def site_of_series() -> dict[int, str]:
    """Return the site label of each series identifier."""
    cache = common.load_cache()
    return {cache.sites[s].table.time_series_id: s for s in common.SCORED_SITES}


def load_arm(
    *, arm: str, period: str, sites: tuple[str, ...] = common.SCORED_SITES
) -> pl.DataFrame:
    """Return an arm's scorable rows for a period, in the common frame shape.

    Extra columns `mean`, `level`, `anomaly`, `shape`, `hour_band`, and `run` are kept.
    """
    pieces = []
    for site in sites:
        path = common.frame_path(arm=arm, site=site)
        frame = pl.read_parquet(path).filter(
            (pl.col("row_period") == period)
            & ~pl.col("switching_mask")
            & (pl.col("lead_hours") <= 168.0)
        )
        pieces.append(frame.with_columns(site=pl.lit(site)))
    return pl.concat(pieces)


def stored_reference(*, name: str, period: str, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return a stored experiment 111 prediction set as scorable rows in the common frame shape."""
    row_set = common.ROW_SET_ID if period == "scored" else REPLICATE_ROW_SET
    frame = read_predictions(experiment_id=111, name=f"{name}_{period}", row_set_id=row_set)
    labels = site_of_series()
    frame = frame.filter(pl.col("time_series_id").is_in(list(labels)))
    frame = frame.with_columns(
        site=pl.col("time_series_id").replace_strict(labels, return_dtype=pl.String)
    )
    frame = frame.filter(
        pl.col("site").is_in(list(sites))
        & ~pl.col("switching_mask")
        & (pl.col("lead_hours") <= 168.0)
    )
    return frame.select(FRAME_COLUMNS)


def e87_reference(*, folder: str, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return an experiment 87 prediction set (BG2, climatology, manual heuristic) in points."""
    labels = site_of_series()
    frame = pl.read_parquet(E87_PREDICTIONS / folder / "predictions.parquet")
    frame = frame.filter(pl.col("time_series_id").is_in(list(labels)))
    frame = frame.with_columns(
        site=pl.col("time_series_id").replace_strict(labels, return_dtype=pl.String),
        actual=100.0 * pl.col("actual"),
        **{q: 100.0 * pl.col(q) for q in LEVEL_COLUMNS if q in frame.columns},
    )
    cache = common.load_cache()
    kept = []
    for site in sites:
        part = frame.filter(pl.col("site") == site)
        position = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        masked = cache.sites[site].switching_mask[np.minimum(position, len(cache.end_times) - 1)]
        kept.append(part.filter(pl.Series(~masked)))
    runs = {
        common.origin_of(run=r)
        for r in common.run_dates()
        if common.period_of_run(run=r) == "scored"
    }
    return (
        pl.concat(kept)
        .filter(
            pl.col("actual").is_finite()
            & (pl.col("lead_hours") <= 168.0)
            & pl.col("power_fcst_init_time").is_in(list(runs))
        )
        .select(FRAME_COLUMNS)
    )


def quantile_score(*, frame: pl.DataFrame) -> pl.Series:
    """Return twice the mean pinball loss over the 11 levels for each row, in points."""
    actual = frame["actual"].to_numpy().astype(np.float64)[:, None]
    quantiles = frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    error = actual - quantiles
    score = 2.0 * np.mean(
        np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error), axis=1
    )
    return pl.Series("qs", score)


def mean_of(series: pl.Series) -> float:
    """Return the mean of a polars series as a float."""
    return float(np.mean(series.to_numpy()))


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score and the absolute errors of the median and of the mean forecast."""
    extra = {"abs_median": (pl.col("actual") - pl.col("q50")).abs()}
    if "mean" in frame.columns:
        extra["abs_mean"] = (pl.col("actual") - pl.col("mean")).abs()
    return frame.with_columns(quantile_score(frame=frame), **extra)


def common_rows(*, arms_by_name: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the rows all arms hold, keyed by series, origin, and valid time."""
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    shared = None
    for frame in arms_by_name.values():
        current = frame.select(keys).unique()
        shared = current if shared is None else shared.join(current, on=keys, how="inner")
    assert shared is not None
    return {
        name: frame.join(shared, on=keys, how="inner")
        .sort(keys)
        .with_columns(pl.col("time_series_id").cast(pl.Int32), pl.col("seed").cast(pl.Int8))
        for name, frame in arms_by_name.items()
    }


def lead_filter(*, frame: pl.DataFrame, band: tuple[str, float, float]) -> pl.DataFrame:
    """Keep the rows whose lead lies in a band: lower bound open, upper bound closed."""
    _, low, high = band
    return frame.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))


def stack_losses(*, arms_by_name: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return `losses_for_bootstrap`'s frame for arms already on common rows.

    Loss columns are kept only when every arm has them: the quantile score and the median's
    absolute error always, the mean's absolute error and the hour band when all arms hold a mean.
    """
    keys = ("time_series_id", "power_fcst_init_time", "valid_time", "seed")
    losses = {name: with_losses(frame=frame) for name, frame in arms_by_name.items()}
    wanted = ["qs", "abs_median", "abs_mean", "hour_band"]
    shared = [c for c in wanted if all(c in frame.columns for frame in losses.values())]
    return losses_for_bootstrap(
        arm_losses={
            name: frame.select([*keys, *shared]).with_columns(
                pl.col(c).cast(pl.Float64) for c in shared if c != "hour_band"
            )
            for name, frame in losses.items()
        }
    )


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
        `upper`, `sites_better`, `n_sites`, and `rows`.
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
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "sites_better": int((by_site["t"] < by_site["r"]).sum()),
        "n_sites": by_site.height,
        "rows": t.height,
    }
