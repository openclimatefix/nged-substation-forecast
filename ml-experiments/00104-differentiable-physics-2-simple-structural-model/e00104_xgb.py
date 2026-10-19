"""Fit the lagless XGBoost benchmark arms, with and without the solar offset, per substation.

Every arm is experiment 87's Primary XGBoost recipe with the three power lags removed (eta 0.05,
depth 6, subsample 0.8, 500 rounds, 11 quantile levels, uniform row weights inside the window), on
the same inputs as the convex model: ERA5 temperature and its four trailing means, CAMS
irradiance, the sun's elevation, half-hour of day, day type, holiday period, the Christmas tent,
and day of year.

| Arm | Window | Start |
|---|---|---|
| `x_12` | 12 months | XGBoost's estimated base score |
| `x_12z` | 12 months | zero |
| `x_12am` | 12 months | zero plus the solar offset |
| `x_allz` | all years | zero |
| `x_allam` | all years | zero plus the solar offset |
| `l_12am_d1`, `l_12am_d6` | 12 months | `x_12am` plus the three power lags, descriptive |

**The solar offset** (experiment 64's `am`) is a base margin `w * s * a(t)` for every quantile,
in April to September and zero outside. `s` is the capacity over the effective capacity, in
points per unit of PV shape: the real-time vintage of experiment 45, variant `frozen`, from the
latest origin on or before the cut, held at its last week afterwards. `a(t)` is the unit PV shape
minus its month-by-half-hour mean over the training window. `w` is fitted per substation by least
squares on the forward cross-fitted median residuals of the matching zero-start arm over April to
September rows. The cross-fit trains a median-only model with the arm's parameters, because only
the median residual is used.

**The lagged reference arms** add `lag_origin`, `lag_same_time_latest_day`,
`lag_same_time_latest_week`, and `lead_hours` as experiment 87 defines them, with a telemetry
delay of 3.5 hours, for an initialisation at 00:00 UTC on the day before the valid day (`d1`,
leads of 24.5 to 48 hours) or six days before it (`d6`, leads of 144.5 to 168 hours).

    uv run python ml-experiments/00104-*/e00104_xgb.py [--cold] [--workers 6]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import e00104_common as common
import numpy as np
import polars as pl
import xgboost as xgb
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    DERATING_PER_KELVIN,
    QUANTILE_LEVELS,
    TrainingWindow,
    crossfit_cuts,
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


def feature_matrix(*, cache: FeatureCache, site: str, index: np.ndarray) -> np.ndarray:
    """Return the lagless inputs of the rows `index`.

    Args:
        cache: The feature cache.
        site: The substation's label.
        index: Grid rows.

    Returns:
        A float32 array of shape (len(index), len(FEATURE_COLUMNS)).
    """
    features = cache.sites[site]
    calendar = cache.calendar
    return np.column_stack(
        [
            features.temperature_c[index],
            features.smoothed_temperature_c[index],
            features.ghi_w_m2[index],
            features.elevation_deg[index],
            calendar.half_hour[index],
            calendar.day_type[index],
            calendar.holiday_period[index],
            calendar.tent[index],
            calendar.signed_tent[index],
            calendar.day_of_year[index],
        ]
    ).astype(np.float32)


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
        origin=np.datetime64(origin, "D"),
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


def fit_booster(
    *,
    x: np.ndarray,
    y: np.ndarray,
    levels: tuple[float, ...],
    start: str,
    margin: np.ndarray | None,
) -> xgb.Booster:
    """Fit one quantile model with experiment 87's parameters.

    Args:
        x: Inputs.
        y: Targets in points.
        levels: The quantile levels to fit.
        start: `estimated` for XGBoost's own base score, `zero` for a base score of zero. A
            margin overrides the start, so `margin` is used when given.
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


def predict(*, booster: xgb.Booster, x: np.ndarray, margin: np.ndarray | None, n_levels: int) -> np.ndarray:
    """Return a booster's quantiles, sorted along the level axis, shape (rows, n_levels)."""
    raw = booster.predict(xgb.DMatrix(x, base_margin=margin))
    return np.sort(raw.reshape(len(x), n_levels), axis=1)


def tile_margin(*, offset: np.ndarray, weight: float, n_levels: int) -> np.ndarray:
    """Repeat the offset `weight * offset` for every quantile level."""
    return np.tile((weight * offset)[:, None], (1, n_levels)).astype(np.float32)


def crossfit_weight(
    *,
    cache: FeatureCache,
    site: str,
    cut: datetime,
    window: TrainingWindow,
    final_train: np.ndarray,
    vintage: VintageCapacity,
) -> tuple[float, int]:
    """Fit the solar weight `w` on forward cross-fitted median residuals of a zero-start model.

    Only the quarters of the last 12 months that hold April to September rows need a fit. Each
    quarter is predicted by a median model trained on the window before the quarter's start.

    Args:
        cache: The feature cache.
        site: The substation's label.
        cut: The final fit's cut.
        window: The arm's window rule.
        final_train: The final fit's training rows, which set the climatology of the anomaly.
        vintage: The capacity vintage.

    Returns:
        `w` and the number of residual rows it was fitted on.
    """
    features = cache.sites[site]
    numerator = 0.0
    denominator = 0.0
    n_rows = 0
    for start, end in crossfit_cuts(cut=cut):
        quarter_months = {(start + timedelta(days=45 + 30 * i)).month for i in range(3)}
        if not quarter_months & set(OFFSET_MONTHS):
            continue
        train = training_rows(features=features, cache=cache, cut=start, window=window)
        booster = fit_booster(
            x=feature_matrix(cache=cache, site=site, index=train),
            y=features.y[train],
            levels=(0.5,),
            start="zero",
            margin=None,
        )
        rows = common.scoring_index(cache=cache, site=site, after=start, through=end)
        rows = rows[~features.switching_mask[rows]]
        x = feature_matrix(cache=cache, site=site, index=rows)
        median = predict(booster=booster, x=x, margin=None, n_levels=1)[:, 0]
        shape = offset_per_unit_weight(
            cache=cache, site=site, index=rows, train_index=final_train, vintage=vintage
        )
        residual = features.y[rows] - median
        numerator += float((shape * residual).sum())
        denominator += float((shape * shape).sum())
        n_rows += int((shape != 0.0).sum())
    return numerator / denominator, n_rows


def lag_columns(
    *, cache: FeatureCache, site: str, index: np.ndarray, scheme: common.SchemeType
) -> np.ndarray:
    """Return the three power lags and the lead in hours for an initialisation scheme.

    The newest usable reading is 3.5 hours before the initialisation. The latest-day and
    latest-week lags are the same time of day, or the same time of day and weekday, as the target,
    taken from the newest day or week that is no newer than that reading.

    Args:
        cache: The feature cache.
        site: The substation's label.
        index: Grid rows (valid times).
        scheme: The initialisation scheme (`d1` or `d6`).

    Returns:
        A float32 array of shape (len(index), 4): `lag_origin`, `lag_same_time_latest_day`,
        `lag_same_time_latest_week`, and `lead_hours`; lags are `NaN` where unobserved.
    """
    features = cache.sites[site]
    observed = np.where(features.present, features.y, np.nan)
    valid = cache.end_times[index]
    init = common.initialisation_times(end_times=valid, scheme=scheme)
    origin = np.searchsorted(cache.end_times, init)
    if not np.array_equal(cache.end_times[origin], init):
        msg = "an initialisation time is not on the half-hourly grid"
        raise ValueError(msg)
    newest = origin - common.TELEMETRY_DELAY_STEPS
    columns = [observed[newest]]
    for period in (common.STEPS_PER_DAY, common.STEPS_PER_WEEK):
        back = np.ceil((index - newest) / period).astype(np.int64)
        columns.append(observed[index - back * period])
    columns.append(((valid - init) / np.timedelta64(1, "h")).astype(np.float64))
    if (np.column_stack(columns)[:, 1:3] == np.inf).any() or (newest >= index).any():
        msg = "a lag would read a value at or after its target"
        raise ValueError(msg)
    return np.column_stack(columns).astype(np.float32)


def check_no_lookahead(*, cache: FeatureCache, train: np.ndarray, cut: datetime) -> None:
    """Raise if a training row ends after the cut."""
    if cache.end_times[train].max() > common.to_64(moment=cut):
        msg = "a training row ends after the cut"
        raise ValueError(msg)


def fit_site(*, site: str, cold: bool) -> dict[str, object]:
    """Fit every XGBoost arm of one substation.

    Args:
        site: The substation's label.
        cold: True for the cold block, which runs `x_12` only.

    Returns:
        `frames` (arm name to predictions), `weights` rows, and `capacity` diagnostics.
    """
    cache = common.load_cache()
    features = cache.sites[site]
    cut = common.COLD_CUT if cold else common.FINAL_CUT
    through = common.COLD_END if cold else common.SCORING_END
    index = common.scoring_index(cache=cache, site=site, after=cut, through=through)
    x_score = feature_matrix(cache=cache, site=site, index=index)
    n_levels = len(QUANTILE_LEVELS)
    levels = tuple(QUANTILE_LEVELS)
    frames: dict[str, pl.DataFrame] = {}
    weights: list[dict] = []
    windows = {"12": TrainingWindow(months=12), "all": TrainingWindow()}

    def frame_of(*, quantiles: np.ndarray, scheme: common.SchemeType = "day_ahead") -> pl.DataFrame:
        return common.prediction_frame(
            cache=cache, site=site, index=index, quantiles=quantiles, scheme=scheme
        )

    vintage = None if cold else vintage_capacity(cache=cache, site=site, cut=cut)
    for label, window in windows.items():
        if cold and label == "all":
            continue
        train = training_rows(features=features, cache=cache, cut=cut, window=window)
        check_no_lookahead(cache=cache, train=train, cut=cut)
        x_train = feature_matrix(cache=cache, site=site, index=train)
        y_train = features.y[train]
        starts = ("estimated", "zero") if label == "12" else ("zero",)
        zero_quantiles: np.ndarray | None = None
        for start in starts:
            booster = fit_booster(x=x_train, y=y_train, levels=levels, start=start, margin=None)
            quantiles = predict(booster=booster, x=x_score, margin=None, n_levels=n_levels)
            name = {"estimated": f"x_{label}", "zero": f"x_{label}z"}[start]
            frames[f"cold_{name}" if cold else name] = frame_of(quantiles=quantiles)
            if start == "zero":
                zero_quantiles = quantiles
        if cold or vintage is None:
            continue
        weight, n_fit = crossfit_weight(
            cache=cache, site=site, cut=cut, window=window, final_train=train, vintage=vintage
        )
        weights.append({"site": site, "window": label, "w": weight, "rows": n_fit})
        _, before = capacity_points_at(cache=cache, vintage=vintage, index=train)
        weights[-1]["train_rows_before_vintage"] = before
        offset_train = offset_per_unit_weight(
            cache=cache, site=site, index=train, train_index=train, vintage=vintage
        )
        offset_score = offset_per_unit_weight(
            cache=cache, site=site, index=index, train_index=train, vintage=vintage
        )
        margin_train = tile_margin(offset=offset_train, weight=weight, n_levels=n_levels)
        margin_score = tile_margin(offset=offset_score, weight=weight, n_levels=n_levels)
        booster = fit_booster(
            x=x_train, y=y_train, levels=levels, start="zero", margin=margin_train
        )
        quantiles = predict(booster=booster, x=x_score, margin=margin_score, n_levels=n_levels)
        frames[f"x_{label}am"] = frame_of(quantiles=quantiles)
        if label != "12":
            continue
        for scheme in common.LAGGED_SCHEMES:
            # The lagless arm, re-keyed to the scheme's initialisation so the contrast pairs rows.
            frames[f"x_12am_{scheme}"] = frame_of(quantiles=quantiles, scheme=scheme)
            lag_train = lag_columns(cache=cache, site=site, index=train, scheme=scheme)
            lag_score = lag_columns(cache=cache, site=site, index=index, scheme=scheme)
            lagged = fit_booster(
                x=np.column_stack([x_train, lag_train]),
                y=y_train,
                levels=levels,
                start="zero",
                margin=margin_train,
            )
            lagged_quantiles = predict(
                booster=lagged,
                x=np.column_stack([x_score, lag_score]),
                margin=margin_score,
                n_levels=n_levels,
            )
            frames[f"l_12am_{scheme}"] = frame_of(quantiles=lagged_quantiles, scheme=scheme)
        del zero_quantiles
    print(f"{site}: {sorted(frames)}", flush=True)
    return {"frames": frames, "weights": weights}


def _worker(job: tuple[str, bool]) -> dict[str, object]:
    """Run `fit_site` in a worker process."""
    site, cold = job
    return fit_site(site=site, cold=cold)


def main() -> None:
    """Fit all substations and write the prediction sets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    sites = common.COLD_SITES if arguments.cold else common.SCORED_SITES
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.load_cache()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(_worker, [(site, arguments.cold) for site in sites]))
    names = [name for result in results for name in result["frames"]]
    for name in dict.fromkeys(names):
        frame = pl.concat([result["frames"][name] for result in results if name in result["frames"]])
        common.store_arm(name=name, frame=frame)
    if not arguments.cold:
        pl.DataFrame([row for result in results for row in result["weights"]]).write_parquet(
            common.WORK_DIR / "xgb_weights.parquet"
        )


if __name__ == "__main__":
    main()
