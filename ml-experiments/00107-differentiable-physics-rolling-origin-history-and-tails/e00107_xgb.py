"""Fit the lagless XGBoost arms of experiment 107 at every cut and substation.

The recipe is experiment 104's `x_12`: experiment 87's Primary hyperparameters without the power
lags (eta 0.05, depth 6, subsample 0.8, 500 rounds, 11 quantile levels), one thread per fit, on the
same inputs as the convex model. Arms, all on the 12 months before the cut:

| Arm | Inputs and start |
|---|---|
| `x_12` | experiment 104's inputs, XGBoost's default start |
| `x_12_nodoy` | `x_12` without the day-of-year column: calendar-memorisation control |
| `x_12amq` | each level started at its training quantile plus experiment 64's solar offset |
| `x_12_seed1` | `x_12` with a second seed, at the burn-in cuts only: the seed floor |

`x_12amq` needs a median-only replica trained on the 12 months ending 12 months before the cut,
which overlaps the lockdown window before the 2023-10-01 cut, so it runs from that cut on, at the
five decision sites. The solar weight `w` is stored for each site and cut.

Each job (substation, cut, variant) writes `work/xgb/<variant>/<site>_<cut>.parquet` atomically and
is skipped if the file exists. `--variant d10_masked` refits D10, at the cuts whose training window
holds one of its two unlabelled jumps, with those windows out of training and scoring.

    uv run python ml-experiments/00107-*/e00107_xgb.py [--workers 8] [--variant base|d10_masked]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Final

import e00107_common as common
import numpy as np
import polars as pl
import xgboost as xgb
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    QUANTILE_COLUMNS,
    QUANTILE_LEVELS,
    TrainingWindow,
    replica_cut,
    training_rows,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

VINTAGES_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00045-pvdis-4-weekly-capacity-asymmetric-l1"
    / "capacity_weekly_vintages.parquet"
)
VINTAGE_VARIANT: Final[str] = "frozen"
N_ROUNDS: Final[int] = 500
XGB_PARAMETERS: Final[dict[str, object]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "objective": "reg:quantileerror",
    "nthread": 1,
}
"""Experiment 87's N1 parameters, one thread per fit; the seed and levels are added per fit."""
OFFSET_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "temperature_c",
    "temperature_c_k6",
    "temperature_c_k24",
    "temperature_c_k72",
    "temperature_c_k168",
    "ghi_w_m2",
    "elevation_deg",
    "half_hour",
    "day_type",
    "holiday_period",
    "tent",
    "signed_tent",
    "day_of_year",
)
NO_DAY_OF_YEAR: Final[tuple[str, ...]] = ("day_of_year",)
AMQ_FIRST_CUT: Final[datetime] = datetime(2023, 10, 1, tzinfo=common.UTC)
"""The first cut whose replica window avoids the lockdown window."""
D10_MASKED_CUTS: Final[tuple[str, ...]] = (
    "2025-01-01",
    "2025-04-01",
    "2025-07-01",
    "2025-10-01",
    "2026-01-01",
    "2026-04-01",
)
"""The cuts whose 12-month window holds a D10 jump window, so a refit without it differs."""


def feature_matrix(
    *, cache: FeatureCache, site: str, index: np.ndarray, drop: tuple[str, ...] = ()
) -> np.ndarray:
    """Return the lagless inputs of the rows `index`, minus the columns named in `drop`."""
    features = cache.sites[site]
    calendar = cache.calendar
    columns = {
        "temperature_c": features.temperature_c[index],
        "temperature_c_k6": features.smoothed_temperature_c[index, 0],
        "temperature_c_k24": features.smoothed_temperature_c[index, 1],
        "temperature_c_k72": features.smoothed_temperature_c[index, 2],
        "temperature_c_k168": features.smoothed_temperature_c[index, 3],
        "ghi_w_m2": features.ghi_w_m2[index],
        "elevation_deg": features.elevation_deg[index],
        "half_hour": calendar.half_hour[index],
        "day_type": calendar.day_type[index],
        "holiday_period": calendar.holiday_period[index],
        "tent": calendar.tent[index],
        "signed_tent": calendar.signed_tent[index],
        "day_of_year": calendar.day_of_year[index],
    }
    if set(columns) != set(FEATURE_COLUMNS):
        msg = "the feature dictionary and FEATURE_COLUMNS disagree"
        raise ValueError(msg)
    return np.column_stack([columns[name] for name in FEATURE_COLUMNS if name not in drop]).astype(
        np.float32
    )


@dataclass(frozen=True)
class VintageCapacity:
    """A substation's weekly capacity from one real-time vintage, in points of effective capacity.

    Attributes:
        origin: The vintage's origin Monday.
        weeks: Week-start dates as `datetime64[D]`, ascending.
        points: The capacity of each week in points; unobserved weeks hold the last observed value.
    """

    origin: np.datetime64
    weeks: np.ndarray
    points: np.ndarray


def vintage_capacity(*, cache: FeatureCache, site: str, cut: datetime) -> VintageCapacity:
    """Read the latest `frozen` vintage whose origin is on or before the cut.

    Args:
        cache: The feature cache.
        site: The substation's label.
        cut: The fit's cut.

    Returns:
        The vintage's weekly capacity in points of effective capacity.

    Raises:
        ValueError: If no origin is on or before the cut.
    """
    table = cache.sites[site].table
    vintages = pl.read_parquet(VINTAGES_PATH).filter(
        pl.col("variant") == VINTAGE_VARIANT,
        pl.col("time_series_id") == table.time_series_id,
        pl.col("origin_monday") <= cut.date(),
    )
    if vintages.is_empty():
        msg = f"{site} has no capacity vintage with an origin on or before {cut.date()}"
        raise ValueError(msg)
    origin = vintages["origin_monday"].max()
    if not isinstance(origin, date):
        msg = "the vintage origin is not a date"
        raise TypeError(msg)
    weekly = vintages.filter(pl.col("origin_monday") == origin).sort("week_start")
    capacity = weekly["capacity_mw"].to_numpy().astype(np.float64)
    unobserved = weekly["unobserved"].to_numpy()
    held = capacity.copy()
    last = np.nan
    for position in range(len(held)):
        if unobserved[position] and np.isfinite(last):
            held[position] = last
        else:
            last = held[position]
    return VintageCapacity(
        origin=np.datetime64(origin.isoformat(), "D"),
        weeks=weekly["week_start"].to_numpy().astype("datetime64[D]"),
        points=100.0 * held / table.effective_capacity_mw,
    )


def capacity_points_at(
    *, cache: FeatureCache, vintage: VintageCapacity, index: np.ndarray
) -> np.ndarray:
    """Return the vintage's capacity at each row, holding the first and last weeks beyond range."""
    day = (cache.end_times[index] - np.timedelta64(30, "m")).astype("datetime64[D]")
    monday = day - ((day.astype(np.int64) + 3) % 7).astype("timedelta64[D]")
    position = np.searchsorted(vintage.weeks, monday, side="right") - 1
    return vintage.points[np.clip(position, 0, len(vintage.weeks) - 1)]


def solar_shape_anomaly(
    *, cache: FeatureCache, site: str, index: np.ndarray, train_index: np.ndarray
) -> np.ndarray:
    """Return the unit PV shape minus its month-by-half-hour mean over the training rows."""
    shape = cache.sites[site].pv_unit_shape
    month = cache.calendar.local_date.astype("datetime64[M]").astype(np.int64) % 12
    cell = month * 48 + cache.calendar.half_hour
    sums = np.bincount(cell[train_index], weights=shape[train_index], minlength=12 * 48)
    counts = np.bincount(cell[train_index], minlength=12 * 48)
    climatology = np.divide(sums, counts, out=np.zeros(12 * 48), where=counts > 0)
    return np.where(counts[cell[index]] > 0, shape[index] - climatology[cell[index]], 0.0)


def offset_per_unit_weight(
    *,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    train_index: np.ndarray,
    vintage: VintageCapacity,
) -> np.ndarray:
    """Return `s * a(t)` in points for April to September rows and zero elsewhere."""
    anomaly = solar_shape_anomaly(cache=cache, site=site, index=index, train_index=train_index)
    capacity = capacity_points_at(cache=cache, vintage=vintage, index=index)
    month_number = (
        cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64) % 12 + 1
    )
    return np.where(np.isin(month_number, OFFSET_MONTHS), capacity * anomaly, 0.0)


def fit_booster(
    *,
    x: np.ndarray,
    y: np.ndarray,
    levels: tuple[float, ...],
    margin: np.ndarray | None,
    seed: int,
) -> xgb.Booster:
    """Fit one quantile model with experiment 87's parameters; a margin overrides the start."""
    parameters: dict[str, object] = {**XGB_PARAMETERS, "quantile_alpha": list(levels), "seed": seed}
    if margin is not None:
        parameters["base_score"] = 0.0
    return xgb.train(parameters, xgb.DMatrix(x, label=y, base_margin=margin), N_ROUNDS)


def predict(
    *, booster: xgb.Booster, x: np.ndarray, margin: np.ndarray | None, n_levels: int
) -> np.ndarray:
    """Return a booster's quantiles, sorted along the level axis, shape (rows, n_levels)."""
    raw = booster.predict(xgb.DMatrix(x, base_margin=margin))
    return np.sort(raw.reshape(len(x), n_levels), axis=1)


def solar_weight(
    *,
    cache: FeatureCache,
    site: str,
    cut: datetime,
    window: TrainingWindow,
    final_train: np.ndarray,
    vintage: VintageCapacity,
) -> tuple[float, int]:
    """Fit the solar weight `w` on the median residuals of a replica of the matching forecast.

    A median-only model with the default start is trained on the window that ends 12 months before
    the cut and predicts the following 12 months, so its residuals are as stale as the scored
    forecast. `w` is the least-squares slope of those residuals on the offset, over the April to
    September rows outside the switching masks.

    Args:
        cache: The feature cache.
        site: The substation's label.
        cut: The final fit's cut.
        window: The arm's window rule.
        final_train: The final fit's training rows, which set the anomaly's climatology.
        vintage: The capacity vintage.

    Returns:
        `w` and the number of April to September residual rows it was fitted on.
    """
    features = cache.sites[site]
    fit_cut = replica_cut(cut=cut)
    train = training_rows(features=features, cache=cache, cut=fit_cut, window=window)
    booster = fit_booster(
        x=feature_matrix(cache=cache, site=site, index=train),
        y=features.y[train],
        levels=(0.5,),
        margin=None,
        seed=0,
    )
    rows = common.rows_between(cache=cache, site=site, after=fit_cut, through=cut)
    rows = rows[~features.switching_mask[rows]]
    median = predict(
        booster=booster,
        x=feature_matrix(cache=cache, site=site, index=rows),
        margin=None,
        n_levels=1,
    )[:, 0]
    shape = offset_per_unit_weight(
        cache=cache, site=site, index=rows, train_index=final_train, vintage=vintage
    )
    residual = features.y[rows] - median
    return float((shape * residual).sum() / (shape * shape).sum()), int((shape != 0.0).sum())


def quantile_columns(*, quantiles: np.ndarray) -> dict[str, np.ndarray]:
    """Return an (n, 11) array as the 11 named quantile columns plus `mean` (the median)."""
    columns = {name: quantiles[:, position] for position, name in enumerate(QUANTILE_COLUMNS)}
    return {**columns, "mean": quantiles[:, QUANTILE_LEVELS.index(0.5)]}


def fit_job(
    *, cache: FeatureCache, site: str, cut: datetime, variant: str
) -> tuple[pl.DataFrame, list[dict]]:
    """Fit the XGBoost arms of one substation at one cut.

    Args:
        cache: The feature cache (the masked variant for `d10_masked`).
        site: The substation's label.
        cut: The cut.
        variant: `base` or `d10_masked`.

    Returns:
        A long frame with one row per arm and scored half-hour, and one record per fitted weight.
    """
    features = cache.sites[site]
    window = TrainingWindow(months=12)
    train = training_rows(features=features, cache=cache, cut=cut, window=window)
    if cache.end_times[train].max() > common.to_64(moment=cut):
        msg = "a training row ends after the cut"
        raise ValueError(msg)
    index = common.scoring_index(cache=cache, site=site, cut=cut)
    levels = tuple(QUANTILE_LEVELS)
    arms: dict[str, np.ndarray] = {}
    records: list[dict] = []

    def plain(*, drop: tuple[str, ...], seed: int) -> np.ndarray:
        booster = fit_booster(
            x=feature_matrix(cache=cache, site=site, index=train, drop=drop),
            y=features.y[train],
            levels=levels,
            margin=None,
            seed=seed,
        )
        return predict(
            booster=booster,
            x=feature_matrix(cache=cache, site=site, index=index, drop=drop),
            margin=None,
            n_levels=len(levels),
        )

    arms["x_12"] = plain(drop=(), seed=0)
    if variant == "base":
        arms["x_12_nodoy"] = plain(drop=NO_DAY_OF_YEAR, seed=0)
    if variant == "base" and cut in common.BURN_IN_CUTS:
        arms["x_12_seed1"] = plain(drop=(), seed=1)
    if variant == "d10_masked":
        arms["x_12_nodoy"] = plain(drop=NO_DAY_OF_YEAR, seed=0)
    if variant == "base" and cut >= AMQ_FIRST_CUT and site in common.DECISION_SITES:
        vintage = vintage_capacity(cache=cache, site=site, cut=cut)
        weight, n_fit = solar_weight(
            cache=cache, site=site, cut=cut, window=window, final_train=train, vintage=vintage
        )
        margins = [
            (
                np.quantile(features.y[train], np.array(QUANTILE_LEVELS))[None, :]
                + weight
                * offset_per_unit_weight(
                    cache=cache, site=site, index=rows, train_index=train, vintage=vintage
                )[:, None]
            ).astype(np.float32)
            for rows in (train, index)
        ]
        booster = fit_booster(
            x=feature_matrix(cache=cache, site=site, index=train),
            y=features.y[train],
            levels=levels,
            margin=margins[0],
            seed=0,
        )
        arms["x_12amq"] = predict(
            booster=booster,
            x=feature_matrix(cache=cache, site=site, index=index),
            margin=margins[1],
            n_levels=len(levels),
        )
        records.append(
            {
                "site": site,
                "origin": common.cut_label(cut=cut),
                "arm": "x_12amq",
                "w": weight,
                "rows": n_fit,
                "vintage_origin": str(vintage.origin),
            }
        )
    frames = [
        common.base_frame(
            cache=cache,
            site=site,
            cut=cut,
            index=index,
            extra={"arm": np.full(len(index), name), **quantile_columns(quantiles=quantiles)},
        )
        for name, quantiles in arms.items()
    ]
    return pl.concat(frames), records


_STATE: dict[str, Any] = {}


def _worker(job: tuple[str, datetime, str]) -> str:
    """Fit one job in a worker process and write its file."""
    site, cut, variant = job
    folder = common.WORK_DIR / "xgb" / variant
    target = folder / f"{site}_{common.cut_label(cut=cut)}.parquet"
    if target.exists():
        return f"skip {target.name}"
    if "cache" not in _STATE:
        cache = common.load_cache()
        _STATE["cache"] = common.d10_masked_cache(cache=cache) if variant == "d10_masked" else cache
    frame, records = fit_job(cache=_STATE["cache"], site=site, cut=cut, variant=variant)
    if records:
        common.write_atomic(
            frame=pl.DataFrame(records),
            path=folder / "records" / f"{site}_{common.cut_label(cut=cut)}.parquet",
        )
    common.write_atomic(frame=frame, path=target)
    return f"done {target.name}: {frame.height} rows"


def main() -> None:
    """Fit every job of a variant in parallel worker processes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--variant", choices=("base", "d10_masked"), default="base")
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    if arguments.variant == "base":
        jobs = [
            (site, cut, "base") for cut in common.CUTS for site in common.sites_for_cut(cut=cut)
        ]
    else:
        jobs = [
            ("D10", cut, "d10_masked")
            for cut in common.CUTS
            if common.cut_label(cut=cut) in D10_MASKED_CUTS
        ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for message in pool.map(_worker, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
