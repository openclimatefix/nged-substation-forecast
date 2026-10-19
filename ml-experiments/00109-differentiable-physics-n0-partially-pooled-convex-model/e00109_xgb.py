"""Fit the lagless XGBoost benchmark arms of experiment 109, per substation and cut.

Every arm is experiment 87's Primary XGBoost recipe with the three power lags removed (eta 0.05,
depth 6, subsample 0.8, 500 rounds, 11 quantile levels, uniform row weights inside the window), on
the same inputs as the convex model: ERA5 temperature and its four trailing means, CAMS
irradiance, the sun's elevation, half-hour of day, day type, holiday period, the Christmas tent,
and day of year. Inputs are in points of effective capacity, so no scale is rebuilt for a short
history.

| Arm | Training rows | Start |
|---|---|---|
| `x_2y` | the substation's 2 years before the cut | XGBoost's estimated base score |
| `x_2yamq` | the same | training quantiles plus experiment 64's solar offset |
| `x_4w`, `x_3m` | the substation's last 4 weeks, or 3 months | estimated base score |
| `x_pool_4w` | every other pool substation's 2 years, plus the held-out substation's last 4 weeks, with no site input | estimated base score |

The C4 bar is whichever of `x_2y` and `x_2yamq` scores better on the tuning year, chosen by
`e00109_report.py` before any scored number is read. The solar offset is the base margin
`w * s * a(t)` of experiment 64: `s` is the real-time vintage capacity (`studies.dp_pinned_capacity`
rules apply to the convex model only; here the vintage is read as experiment 104 read it), `a(t)`
is the unit PV shape minus its month-by-half-hour mean over the training window, and `w` is fitted
per substation on the median residuals of a replica of the matching plain arm.

    uv run python ml-experiments/00109-*/e00109_xgb.py --cuts tuning replicate scored winter \\
        [--workers 6]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Final, Literal

import e00109_common as common
import numpy as np
import polars as pl
import xgboost as xgb
from studies.dp_features import FeatureCache
from studies.dp_gam import (
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
"""Experiment 45's real-time capacity vintages, one block of weekly capacities per origin."""
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
    "seed": 0,
}
"""Experiment 87's N1 parameters; the quantile levels are added per fit."""
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
LAG_COLUMNS: Final[tuple[str, ...]] = (
    "lag_origin",
    "lag_same_time_latest_day",
    "lag_same_time_latest_week",
    "lead_hours",
)


def feature_matrix(
    *, cache: FeatureCache, site: str, index: np.ndarray, drop: tuple[str, ...] = ()
) -> np.ndarray:
    """Return the lagless inputs of the rows `index`.

    Args:
        cache: The feature cache.
        site: The substation's label.
        index: Grid rows.
        drop: Names in `FEATURE_COLUMNS` to leave out.

    Returns:
        A float32 array of shape (len(index), len(FEATURE_COLUMNS) - len(drop)).
    """
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
        n_unobserved: How many weeks were unobserved and took the held value.
    """

    origin: np.datetime64
    weeks: np.ndarray
    points: np.ndarray
    n_unobserved: int


def vintage_capacity(*, cache: FeatureCache, site: str, cut: datetime) -> VintageCapacity:
    """Read the latest `frozen` vintage whose origin is on or before the cut.

    Args:
        cache: The feature cache.
        site: The substation's label.
        cut: The fit's cut.

    Returns:
        The weekly capacity in points.

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
        n_unobserved=int(unobserved.sum()),
    )


def capacity_points_at(
    *, cache: FeatureCache, vintage: VintageCapacity, index: np.ndarray
) -> tuple[np.ndarray, int]:
    """Return the vintage's capacity at each row, holding the first and last weeks beyond range.

    Args:
        cache: The feature cache.
        vintage: The vintage.
        index: Grid rows.

    Returns:
        The capacity in points of each row, and the number of rows that fell before the vintage's
        first week and took its first value.
    """
    day = (cache.end_times[index] - np.timedelta64(30, "m")).astype("datetime64[D]")
    monday = day - ((day.astype("datetime64[D]").astype(np.int64) + 3) % 7).astype("timedelta64[D]")
    position = np.searchsorted(vintage.weeks, monday, side="right") - 1
    before = int((position < 0).sum())
    return vintage.points[np.clip(position, 0, len(vintage.weeks) - 1)], before


def solar_shape_anomaly(
    *, cache: FeatureCache, site: str, index: np.ndarray, train_index: np.ndarray
) -> np.ndarray:
    """Return the unit PV shape minus its month-by-half-hour mean over the training rows.

    Args:
        cache: The feature cache.
        site: The substation's label.
        index: Grid rows to evaluate.
        train_index: The training rows that set the climatology.

    Returns:
        The anomaly of each row; zero in a month-by-half-hour cell the training rows do not cover.
    """
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
    """Return `s * a(t)` in points for April to September rows and zero elsewhere.

    Args:
        cache: The feature cache.
        site: The substation's label.
        index: Grid rows.
        train_index: The training rows that set the climatology.
        vintage: The capacity vintage.

    Returns:
        The offset of each row per unit of the fitted weight `w`.
    """
    anomaly = solar_shape_anomaly(cache=cache, site=site, index=index, train_index=train_index)
    capacity, _ = capacity_points_at(cache=cache, vintage=vintage, index=index)
    month_number = (
        cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64) % 12 + 1
    )
    return np.where(np.isin(month_number, OFFSET_MONTHS), capacity * anomaly, 0.0)


StartType = Literal["estimated", "zero"]
"""XGBoost's own base score, or a base score of zero."""
NO_IRRADIANCE: Final[tuple[str, ...]] = ("ghi_w_m2",)
"""The input the `ng` arms leave out."""


def fit_booster(
    *,
    x: np.ndarray,
    y: np.ndarray,
    levels: tuple[float, ...],
    start: StartType,
    margin: np.ndarray | None,
) -> xgb.Booster:
    """Fit one quantile model with experiment 87's parameters.

    Args:
        x: Inputs.
        y: Targets in points.
        levels: The quantile levels to fit.
        start: The base score. A margin overrides it, so `margin` is used when given.
        margin: A base margin of shape (rows, len(levels)), or `None`.

    Returns:
        The fitted booster.
    """
    parameters: dict[str, object] = {
        **XGB_PARAMETERS,
        "quantile_alpha": list(levels),
        "nthread": 1,
    }
    if start == "zero" and margin is None:
        parameters["base_score"] = 0.0
    matrix = xgb.DMatrix(x, label=y, base_margin=margin)
    return xgb.train(parameters, matrix, num_boost_round=N_ROUNDS)


def predict(
    *, booster: xgb.Booster, x: np.ndarray, margin: np.ndarray | None, n_levels: int
) -> np.ndarray:
    """Return a booster's quantiles, sorted along the level axis, shape (rows, n_levels)."""
    raw = booster.predict(xgb.DMatrix(x, base_margin=margin))
    return np.sort(raw.reshape(len(x), n_levels), axis=1)


def tile_margin(
    *, offset: np.ndarray, weight: float, n_levels: int, base: np.ndarray | None = None
) -> np.ndarray:
    """Return `base + weight * offset` for every quantile level, shape (rows, n_levels).

    Args:
        offset: The solar offset per unit weight, one per row.
        weight: The fitted weight `w`.
        n_levels: The number of quantile levels.
        base: The start of each level (zero if `None`).

    Returns:
        A float32 margin, the same offset for every level on top of the level's start.
    """
    start = np.zeros(n_levels) if base is None else base
    return (start[None, :] + (weight * offset)[:, None]).astype(np.float32)


def replica_weight(
    *,
    cache: FeatureCache,
    site: str,
    cut: datetime,
    window: TrainingWindow,
    final_train: np.ndarray,
    vintage: VintageCapacity,
    start: StartType,
    drop: tuple[str, ...] = (),
) -> tuple[float, int]:
    """Fit the solar weight `w` on the median residuals of a replica of the matching forecast.

    A median-only model with the arm's parameters is trained on the window that ends 12 months
    before the cut, and predicts the following 12 months, so its residuals are as stale as the
    scored forecast. The weight is the least-squares slope of those residuals on the offset, over
    the April to September rows outside the switching masks.

    Args:
        cache: The feature cache.
        site: The substation's label.
        cut: The final fit's cut.
        window: The arm's window rule.
        final_train: The final fit's training rows, which set the climatology of the anomaly.
        vintage: The capacity vintage.
        start: The matching arm's start (`zero` or `estimated`).
        drop: Inputs the matching arm leaves out.

    Returns:
        `w` and the number of April to September residual rows it was fitted on.
    """
    features = cache.sites[site]
    fit_cut = replica_cut(cut=cut)
    train = training_rows(features=features, cache=cache, cut=fit_cut, window=window)
    booster = fit_booster(
        x=feature_matrix(cache=cache, site=site, index=train, drop=drop),
        y=features.y[train],
        levels=(0.5,),
        start=start,
        margin=None,
    )
    rows = common.scoring_index(cache=cache, site=site, after=fit_cut, through=cut)
    rows = rows[~features.switching_mask[rows]]
    median = predict(
        booster=booster,
        x=feature_matrix(cache=cache, site=site, index=rows, drop=drop),
        margin=None,
        n_levels=1,
    )[:, 0]
    shape = offset_per_unit_weight(
        cache=cache, site=site, index=rows, train_index=final_train, vintage=vintage
    )
    residual = features.y[rows] - median
    return float((shape * residual).sum() / (shape * shape).sum()), int((shape != 0.0).sum())


def check_no_lookahead(*, cache: FeatureCache, train: np.ndarray, cut: datetime) -> None:
    """Raise if a training row ends after the cut."""
    if cache.end_times[train].max() > common.to_64(moment=cut):
        msg = "a training row ends after the cut"
        raise ValueError(msg)


@dataclass(frozen=True)
class SiteJob:
    """Everything the arms of one substation share.

    Attributes:
        cache: The feature cache.
        site: The substation's label.
        cut: The cut.
        index: The rows to score.
    """

    cache: FeatureCache
    site: str
    cut: datetime
    index: np.ndarray

    def x_score(self, *, drop: tuple[str, ...] = ()) -> np.ndarray:
        """Return the lagless inputs of the rows to score."""
        return feature_matrix(cache=self.cache, site=self.site, index=self.index, drop=drop)

    def frame_of(self, *, quantiles: np.ndarray, arm: str) -> pl.DataFrame:
        """Return the predictions of the rows to score as a prediction frame."""
        return common.prediction_frame(
            cache=self.cache,
            site=self.site,
            index=self.index,
            quantiles=quantiles,
            mean=np.sort(quantiles, axis=1)[:, 5],
            arm=arm,
        )


def plain_arm(
    *, job: SiteJob, train: np.ndarray, start: StartType, drop: tuple[str, ...] = ()
) -> np.ndarray:
    """Fit an arm with no solar offset and return its quantiles on the rows to score."""
    features = job.cache.sites[job.site]
    booster = fit_booster(
        x=feature_matrix(cache=job.cache, site=job.site, index=train, drop=drop),
        y=features.y[train],
        levels=tuple(QUANTILE_LEVELS),
        start=start,
        margin=None,
    )
    return predict(
        booster=booster, x=job.x_score(drop=drop), margin=None, n_levels=len(QUANTILE_LEVELS)
    )


def offset_arm_quantiles(
    *, job: SiteJob, train: np.ndarray, window: TrainingWindow, vintage: VintageCapacity
) -> tuple[np.ndarray, float]:
    """Fit the quantile-start arm with the solar offset; return its quantiles and the weight `w`.

    Args:
        job: The substation's shared inputs.
        train: The window's training rows.
        window: The window rule.
        vintage: The capacity vintage.

    Returns:
        The quantiles on the rows to score, and the fitted weight.
    """
    cache, site = job.cache, job.site
    features = cache.sites[site]
    levels = tuple(QUANTILE_LEVELS)
    weight, _ = replica_weight(
        cache=cache,
        site=site,
        cut=job.cut,
        window=window,
        final_train=train,
        vintage=vintage,
        start="estimated",
    )
    base = np.quantile(features.y[train], np.array(levels)).astype(np.float64)
    margins = [
        tile_margin(
            offset=offset_per_unit_weight(
                cache=cache, site=site, index=rows, train_index=train, vintage=vintage
            ),
            weight=weight,
            n_levels=len(levels),
            base=base,
        )
        for rows in (train, job.index)
    ]
    booster = fit_booster(
        x=feature_matrix(cache=cache, site=site, index=train),
        y=features.y[train],
        levels=levels,
        start="zero",
        margin=margins[0],
    )
    return predict(
        booster=booster, x=job.x_score(), margin=margins[1], n_levels=len(levels)
    ), weight


def pooled_short_arm(*, job: SiteJob, short_window: TrainingWindow) -> np.ndarray:
    """Fit one site-agnostic XGBoost model on every other site plus the held-out site's short window.

    Args:
        job: The held-out substation's shared inputs.
        short_window: The held-out substation's window.

    Returns:
        The held-out substation's quantiles on the rows to score.
    """
    cache = job.cache
    parts_x, parts_y = [], []
    for other in common.POOL_SITES:
        window = short_window if other == job.site else common.FULL_WINDOW
        rows = training_rows(features=cache.sites[other], cache=cache, cut=job.cut, window=window)
        if len(rows) < common.MIN_TRAINING_ROWS and other != job.site:
            continue
        parts_x.append(feature_matrix(cache=cache, site=other, index=rows))
        parts_y.append(cache.sites[other].y[rows])
    booster = fit_booster(
        x=np.concatenate(parts_x),
        y=np.concatenate(parts_y),
        levels=tuple(QUANTILE_LEVELS),
        start="estimated",
        margin=None,
    )
    return predict(booster=booster, x=job.x_score(), margin=None, n_levels=len(QUANTILE_LEVELS))


def fit_site(*, site: str, cut_name: str) -> dict[str, pl.DataFrame]:
    """Fit every XGBoost arm of one substation at one cut.

    Args:
        site: The substation's label.
        cut_name: A key of `common.CUTS`.

    Returns:
        Arm name to predictions. The winter cut runs the short-history arms only.
    """
    cache = common.load_cache()
    cut = common.CUTS[cut_name]
    features = cache.sites[site]
    job = SiteJob(
        cache=cache,
        site=site,
        cut=cut,
        index=common.scoring_index(
            cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
        ),
    )
    frames: dict[str, pl.DataFrame] = {}
    full = training_rows(features=features, cache=cache, cut=cut, window=common.FULL_WINDOW)
    if cut_name != "winter" and len(full) >= common.MIN_TRAINING_ROWS:
        check_no_lookahead(cache=cache, train=full, cut=cut)
        frames[f"{cut_name}_x_2y"] = job.frame_of(
            quantiles=plain_arm(job=job, train=full, start="estimated"), arm="x_2y"
        )
        try:
            vintage = vintage_capacity(cache=cache, site=site, cut=cut)
        except ValueError:
            vintage = None
        if vintage is not None:
            quantiles, _ = offset_arm_quantiles(
                job=job, train=full, window=common.FULL_WINDOW, vintage=vintage
            )
            frames[f"{cut_name}_x_2yamq"] = job.frame_of(quantiles=quantiles, arm="x_2yamq")
    if site not in common.HELD_OUT_SITES or cut_name == "tuning":
        return frames
    for history in (common.FOUR_WEEKS, common.THREE_MONTHS):
        if cut_name == "winter" and history is not common.FOUR_WEEKS:
            continue
        window = history.window(cut=cut)
        rows = training_rows(features=features, cache=cache, cut=cut, window=window)
        if len(rows) < common.MIN_TRAINING_ROWS:
            continue
        check_no_lookahead(cache=cache, train=rows, cut=cut)
        frames[f"{cut_name}_x_{history.label}"] = job.frame_of(
            quantiles=plain_arm(job=job, train=rows, start="estimated"), arm=f"x_{history.label}"
        )
    frames[f"{cut_name}_x_pool_4w"] = job.frame_of(
        quantiles=pooled_short_arm(job=job, short_window=common.FOUR_WEEKS.window(cut=cut)),
        arm="x_pool_4w",
    )
    return frames


def _worker(job: tuple[str, str]) -> dict[str, pl.DataFrame]:
    """Run `fit_site` in a worker process."""
    site, cut_name = job
    return fit_site(site=site, cut_name=cut_name)


def main() -> None:
    """Fit all substations at the requested cuts and write the prediction sets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts", nargs="+", default=["tuning", "replicate", "scored", "winter"])
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.load_cache()
    jobs = [(site, cut) for cut in arguments.cuts for site in common.HELD_OUT_SITES]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(_worker, jobs))
    names = [name for result in results for name in result]
    for name in dict.fromkeys(names):
        frame = pl.concat([result[name] for result in results if name in result])
        common.store_arm(name=name, frame=frame)


if __name__ == "__main__":
    main()
