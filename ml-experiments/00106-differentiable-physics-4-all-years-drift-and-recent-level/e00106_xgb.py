"""Fit experiment 106's lagless XGBoost bar, with and without the solar offset, per substation.

Every arm is experiment 87's Primary XGBoost recipe with the three power lags removed (eta 0.05,
depth 6, subsample 0.8, 500 rounds, 11 quantile levels, uniform row weights inside the window), on
the same inputs as the convex model: ERA5 temperature and its four trailing means, CAMS
irradiance, the sun's elevation, half-hour of day, day type, holiday period, the Christmas tent,
and day of year.

**The solar offset** (experiment 64's `am`) is a base margin `w * s(t) * a(t)` for every quantile,
in April to September and zero outside. `s(t)` is the pinned capacity of
`studies.dp_pinned_capacity`, the same capacity the convex model uses, and `a(t)` is the unit PV shape minus its
month-by-half-hour mean over the training window. `w` is fitted per substation by least squares on
the median residuals of a replica of the matching offset-free arm over April to September rows.

**Two stages.**

- `--stage tune` fits, at the tuning cut 2023-07-01, the plain default-start arm and the
  default-quantile-start-plus-offset arm (`amq`) on three windows (12 months, 24 months, all
  years), scores them on 2023-24 with the arm's own quantiles, and writes `work/tuned_xgb.json`:
  the window `X-best` takes (lowest plain score), whether the C6 bar is `x_best` or `x_bestamq`,
  and the Kalman filter's hyperparameters for `x_best`'s median residuals.
- `--stage score` fits the chosen window at the replicate and scored cuts, in four arms
  (`x_best`, `x_bestz`, `x_bestamq`, `x_bestam`), the descriptive lagged arms at leads of 24.5 to 48
  and 144.5 to 168 hours, and `x_best_k` (`x_best` with the Kalman filter on its median residuals,
  every quantile shifted by the correction).

    uv run python ml-experiments/00106-*/e00106_xgb.py --stage tune|score [--workers 6]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Literal

import e00106_common as common
import numpy as np
import polars as pl
import xgboost as xgb
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    FilterHyperparameters,
    correction_at,
    day_ahead_initialisation,
    fit_filter_hyperparameters,
    run_kalman,
    stitch_out_of_sample_residuals,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    QUANTILE_COLUMNS,
    QUANTILE_LEVELS,
    TrainingWindow,
    replica_cut,
    training_rows,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity, read_vintages

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
STEPS_PER_DAY: Final[int] = 48
STEPS_PER_WEEK: Final[int] = 7 * STEPS_PER_DAY
TELEMETRY_DELAY_STEPS: Final[int] = 7
"""A reading is usable once it is 3.5 hours old: 7 half-hours."""
WINDOWS: Final[dict[str, TrainingWindow]] = {
    "12m": TrainingWindow(months=12),
    "24m": TrainingWindow(months=24),
    "all": TrainingWindow(),
}
"""The windows the tuning stage tries for `X-best`."""
LAGGED_SCHEMES: Final[dict[str, tuple[int, int]]] = {"d1": (1, 0), "d6": (6, 0)}
"""Days before the valid day and the UTC hour of the initialisation of the lagged arms."""
HYPER_ARM: Final[str] = "x_best"

StartType = Literal["estimated", "zero"]
"""XGBoost's own base score, or a base score of zero."""


def feature_matrix(*, cache: FeatureCache, site: str, index: np.ndarray) -> np.ndarray:
    """Return the lagless inputs of the rows `index`, float32, one column per feature."""
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
    return np.column_stack([columns[name] for name in FEATURE_COLUMNS]).astype(np.float32)


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
    pinned: PinnedCapacity,
) -> np.ndarray:
    """Return `s * a(t)` in points for April to September rows and zero elsewhere."""
    anomaly = solar_shape_anomaly(cache=cache, site=site, index=index, train_index=train_index)
    capacity = pinned.at_rows(cache=cache, index=index)
    month_number = (
        cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64) % 12 + 1
    )
    return np.where(np.isin(month_number, OFFSET_MONTHS), capacity * anomaly, 0.0)


def fit_booster(
    *,
    x: np.ndarray,
    y: np.ndarray,
    levels: tuple[float, ...],
    start: StartType,
    margin: np.ndarray | None,
) -> xgb.Booster:
    """Fit one quantile model with experiment 87's parameters on one thread."""
    parameters: dict[str, object] = {**XGB_PARAMETERS, "quantile_alpha": list(levels), "nthread": 1}
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
    """Return `base + weight * offset` for every quantile level, shape (rows, n_levels)."""
    start = np.zeros(n_levels) if base is None else base
    return (start[None, :] + (weight * offset)[:, None]).astype(np.float32)


def replica_weight(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    site: str,
    cut: datetime,
    window: TrainingWindow,
    final_train: np.ndarray,
    start: StartType,
) -> float:
    """Fit the solar weight `w` on the median residuals of a replica of the matching arm.

    A median-only model with the arm's parameters is trained on the window that ends 12 months
    before the cut, and predicts the following 12 months, so its residuals are as stale as the
    scored forecast. The weight is the least-squares slope of those residuals on the offset, over
    the April to September rows outside the switching masks. The offset's capacity is the pinned
    capacity built at the replica's own cut.
    """
    features = cache.sites[site]
    fit_cut = replica_cut(cut=cut)
    pinned = pinned_capacity(vintages=vintages, table=features.table, cut=fit_cut)
    train = training_rows(features=features, cache=cache, cut=fit_cut, window=window)
    booster = fit_booster(
        x=feature_matrix(cache=cache, site=site, index=train),
        y=features.y[train],
        levels=(0.5,),
        start=start,
        margin=None,
    )
    rows = common.scoring_index(cache=cache, site=site, after=fit_cut, through=cut)
    rows = rows[~features.switching_mask[rows]]
    median = predict(
        booster=booster,
        x=feature_matrix(cache=cache, site=site, index=rows),
        margin=None,
        n_levels=1,
    )[:, 0]
    shape = offset_per_unit_weight(
        cache=cache, site=site, index=rows, train_index=final_train, pinned=pinned
    )
    residual = features.y[rows] - median
    return float((shape * residual).sum() / (shape * shape).sum())


def lag_columns(
    *, cache: FeatureCache, site: str, index: np.ndarray, scheme: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the three power lags with the lead in hours, and the initialisation times.

    The newest usable reading is 3.5 hours before the initialisation. The latest-day and
    latest-week lags are the same time of day, or the same time of day and weekday, as the target,
    taken from the newest day or week that is no newer than that reading.

    Args:
        cache: The feature cache.
        site: The substation's label.
        index: Grid rows (valid times).
        scheme: `d1` or `d6`.

    Returns:
        A float32 array of shape (len(index), 4): `lag_origin`, `lag_same_time_latest_day`,
        `lag_same_time_latest_week`, and `lead_hours`, and the initialisation times.

    Raises:
        ValueError: If an initialisation is off the grid or a lag would read its own target.
    """
    features = cache.sites[site]
    observed = np.where(features.present, features.y, np.nan)
    valid = cache.end_times[index]
    days_before, hour = LAGGED_SCHEMES[scheme]
    day = (valid - np.timedelta64(30, "m")).astype("datetime64[D]")
    init = (day - np.timedelta64(days_before, "D")).astype("datetime64[us]") + np.timedelta64(
        hour, "h"
    )
    origin = np.searchsorted(cache.end_times, init)
    if not np.array_equal(cache.end_times[origin], init):
        msg = "an initialisation time is not on the half-hourly grid"
        raise ValueError(msg)
    newest = origin - TELEMETRY_DELAY_STEPS
    columns = [observed[newest]]
    for period in (STEPS_PER_DAY, STEPS_PER_WEEK):
        back = np.ceil((index - newest) / period).astype(np.int64)
        columns.append(observed[index - back * period])
    columns.append(((valid - init) / np.timedelta64(1, "h")).astype(np.float64))
    if (newest >= index).any():
        msg = "a lag would read a value at or after its target"
        raise ValueError(msg)
    return np.column_stack(columns).astype(np.float32), init


def _frame(
    *,
    cache: FeatureCache,
    site: str,
    index: np.ndarray,
    quantiles: np.ndarray,
    arm: str,
    init: np.ndarray | None = None,
) -> pl.DataFrame:
    """Return an arm's predictions with the median as its central forecast."""
    return common.prediction_frame(
        cache=cache,
        site=site,
        index=index,
        quantiles=quantiles,
        mean=quantiles[:, QUANTILE_LEVELS.index(0.5)],
        arm=arm,
        init=init,
    )


@dataclass(frozen=True)
class OffsetFit:
    """A fitted offset arm.

    Attributes:
        quantiles: The quantiles on the rows to score.
        weight: The fitted solar weight `w`.
    """

    quantiles: np.ndarray
    weight: float


def offset_arm(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    site: str,
    cut: datetime,
    window: TrainingWindow,
    train: np.ndarray,
    index: np.ndarray,
    start: StartType,
) -> OffsetFit:
    """Fit the solar weight and an arm whose base margin is a start plus the offset.

    Args:
        cache: The feature cache.
        vintages: The real-time capacity vintages.
        site: The substation's label.
        cut: The cut.
        window: The window rule.
        train: The window's training rows.
        index: The rows to score.
        start: `zero` for a margin of the offset alone, `estimated` for each level's training
            quantile plus the offset (the weight is then fitted on the estimated-start replica).

    Returns:
        The fitted arm.
    """
    features = cache.sites[site]
    levels = tuple(QUANTILE_LEVELS)
    pinned = pinned_capacity(vintages=vintages, table=features.table, cut=cut)
    weight = replica_weight(
        cache=cache,
        vintages=vintages,
        site=site,
        cut=cut,
        window=window,
        final_train=train,
        start=start,
    )
    base = (
        None
        if start == "zero"
        else np.quantile(features.y[train], np.array(QUANTILE_LEVELS)).astype(np.float64)
    )
    margins = [
        tile_margin(
            offset=offset_per_unit_weight(
                cache=cache, site=site, index=rows, train_index=train, pinned=pinned
            ),
            weight=weight,
            n_levels=len(levels),
            base=base,
        )
        for rows in (train, index)
    ]
    booster = fit_booster(
        x=feature_matrix(cache=cache, site=site, index=train),
        y=features.y[train],
        levels=levels,
        start="zero",
        margin=margins[0],
    )
    quantiles = predict(
        booster=booster,
        x=feature_matrix(cache=cache, site=site, index=index),
        margin=margins[1],
        n_levels=len(levels),
    )
    return OffsetFit(quantiles=quantiles, weight=weight)


def plain_arm(
    *,
    cache: FeatureCache,
    site: str,
    train: np.ndarray,
    index: np.ndarray,
    start: StartType,
) -> np.ndarray:
    """Fit an arm with no solar offset and return its quantiles on the rows to score."""
    booster = fit_booster(
        x=feature_matrix(cache=cache, site=site, index=train),
        y=cache.sites[site].y[train],
        levels=tuple(QUANTILE_LEVELS),
        start=start,
        margin=None,
    )
    return predict(
        booster=booster,
        x=feature_matrix(cache=cache, site=site, index=index),
        margin=None,
        n_levels=len(QUANTILE_LEVELS),
    )


def tune_site(*, site: str) -> dict[str, Any]:
    """Fit the tuning-stage arms of one substation at the tuning cut and score them.

    Args:
        site: The substation's label.

    Returns:
        Per window, the mean quantile scores of the plain and offset arms on rows outside the
        switching masks, and the plain arm's median residual stream on the grid.
    """
    cache = common.load_cache()
    vintages = read_vintages()
    features = cache.sites[site]
    cut = common.TUNING_CUT
    index = common.scoring_index(
        cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    keep = ~features.switching_mask[index]
    result: dict[str, Any] = {}
    for name, window in WINDOWS.items():
        train = training_rows(features=features, cache=cache, cut=cut, window=window)
        plain = plain_arm(cache=cache, site=site, train=train, index=index, start="estimated")
        offset = offset_arm(
            cache=cache,
            vintages=vintages,
            site=site,
            cut=cut,
            window=window,
            train=train,
            index=index,
            start="estimated",
        )
        scores = {
            kind: float(
                common.quantile_score(actual=features.y[index][keep], quantiles=q[keep]).mean()
            )
            for kind, q in (("plain", plain), ("amq", offset.quantiles))
        }
        stream = np.full(len(cache.end_times), np.nan)
        stream[index] = features.y[index] - plain[:, QUANTILE_LEVELS.index(0.5)]
        result[name] = {"scores": scores, "weight": offset.weight, "stream": stream}
        print(f"{site} {name}: plain {scores['plain']:.3f}, amq {scores['amq']:.3f}", flush=True)
    return result


def score_site(*, site: str, window_name: str) -> dict[str, Any]:
    """Fit the scoring-stage arms of one substation at the replicate and scored cuts.

    Args:
        site: The substation's label.
        window_name: The window `X-best` takes, a key of `WINDOWS`.

    Returns:
        `frames` (set name to predictions), the median residual streams of `x_best` by cut label,
        and the solar weights.
    """
    cache = common.load_cache()
    vintages = read_vintages()
    window = WINDOWS[window_name]
    frames: dict[str, pl.DataFrame] = {}
    streams: dict[str, np.ndarray] = {}
    weights: list[dict] = []
    median_position = QUANTILE_LEVELS.index(0.5)
    for label in common.CUTS:
        if label == "tuning" and site == "D9":
            continue
        cut = common.CUTS[label]
        features = cache.sites[site]
        index = common.scoring_index(
            cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
        )
        train = training_rows(features=features, cache=cache, cut=cut, window=window)
        best = plain_arm(cache=cache, site=site, train=train, index=index, start="estimated")
        stream = np.full(len(cache.end_times), np.nan)
        stream[index] = features.y[index] - best[:, median_position]
        streams[label] = stream
        if label == "tuning" or (site == "D9" and label == "replicate"):
            continue
        frames[f"{label}_x_best"] = _frame(
            cache=cache, site=site, index=index, quantiles=best, arm="x_best"
        )
        frames[f"{label}_x_bestz"] = _frame(
            cache=cache,
            site=site,
            index=index,
            quantiles=plain_arm(cache=cache, site=site, train=train, index=index, start="zero"),
            arm="x_bestz",
        )
        for name, start in (("x_bestamq", "estimated"), ("x_bestam", "zero")):
            fitted = offset_arm(
                cache=cache,
                vintages=vintages,
                site=site,
                cut=cut,
                window=window,
                train=train,
                index=index,
                start=start,
            )
            frames[f"{label}_{name}"] = _frame(
                cache=cache, site=site, index=index, quantiles=fitted.quantiles, arm=name
            )
            weights.append({"site": site, "cut": label, "arm": name, "w": fitted.weight})
        x_train = feature_matrix(cache=cache, site=site, index=train)
        for scheme in LAGGED_SCHEMES:
            lag_train, _ = lag_columns(cache=cache, site=site, index=train, scheme=scheme)
            lag_score, init = lag_columns(cache=cache, site=site, index=index, scheme=scheme)
            frames[f"{label}_x_best_{scheme}"] = _frame(
                cache=cache, site=site, index=index, quantiles=best, arm="x_best", init=init
            )
            lagged = fit_booster(
                x=np.column_stack([x_train, lag_train]),
                y=features.y[train],
                levels=tuple(QUANTILE_LEVELS),
                start="estimated",
                margin=None,
            )
            quantiles = predict(
                booster=lagged,
                x=np.column_stack([feature_matrix(cache=cache, site=site, index=index), lag_score]),
                margin=None,
                n_levels=len(QUANTILE_LEVELS),
            )
            frames[f"{label}_l_best_{scheme}"] = _frame(
                cache=cache, site=site, index=index, quantiles=quantiles, arm="l_best", init=init
            )
        print(f"{site} {label}: x_best window {window_name} done", flush=True)
    return {"frames": frames, "streams": streams, "weights": weights}


def filtered_xgb_frames(
    *,
    cache: FeatureCache,
    results: dict[str, dict[str, Any]],
    hyper_by_site: dict[str, list[float]],
) -> dict[str, pl.DataFrame]:
    """Build `x_best_k`: `x_best` with the Kalman filter on its median residuals.

    Every quantile of `x_best` is shifted by the correction, so the spread is XGBoost's own.

    Args:
        cache: The feature cache.
        results: Each site's `score_site` result.
        hyper_by_site: The filter's hyperparameters per site.

    Returns:
        The sets `<cut>_x_best_k`.
    """
    frames: dict[str, list[pl.DataFrame]] = {}
    for site, result in results.items():
        hyper = FilterHyperparameters.from_array(values=np.array(hyper_by_site[site]))
        labels = [label for label in common.CUTS if label in result["streams"]]
        for label in ("replicate", "scored"):
            if f"{label}_x_best" not in result["frames"]:
                continue
            chain = labels[: labels.index(label) + 1]
            positions = [
                int(
                    np.searchsorted(
                        cache.end_times, common.to_64(moment=common.CUTS[name]), side="right"
                    )
                )
                - 1
                for name in chain
            ]
            stream = stitch_out_of_sample_residuals(
                cut_positions=positions, residuals=[result["streams"][name] for name in chain]
            )
            trace = run_kalman(residual=stream, band=cache.calendar.band, hyper=hyper)
            base = result["frames"][f"{label}_x_best"]
            index = common.scoring_index(
                cache=cache,
                site=site,
                after=common.CUTS[label],
                through=common.year_end(cut=common.CUTS[label]),
            )
            valid = cache.end_times[index]
            cutoff = usable_cutoff_positions(
                end_times=cache.end_times,
                initialisation=day_ahead_initialisation(end_times=valid, days_ahead=1),
                delay_hours=DEFAULT_DELAY_HOURS,
            )
            correction = correction_at(
                trace=trace, cutoff=cutoff, target=index, band=cache.calendar.band[index]
            )
            shifted = base.with_columns(
                [
                    (pl.col(column) + pl.Series(correction.total.astype(np.float32))).alias(column)
                    for column in QUANTILE_COLUMNS
                ]
            ).with_columns(
                arm=pl.lit("x_best_k"),
                level=pl.Series(correction.level.astype(np.float32)),
                anomaly=pl.Series(correction.anomaly.astype(np.float32)),
                mean=pl.col("mean") + pl.Series(correction.total.astype(np.float32)),
            )
            frames.setdefault(f"{label}_x_best_k", []).append(shifted)
    return {name: pl.concat(parts) for name, parts in frames.items()}


def _tune_worker(site: str) -> dict[str, Any]:
    """Run `tune_site` in a worker process."""
    return tune_site(site=site)


def _score_worker(job: tuple[str, str]) -> dict[str, Any]:
    """Run `score_site` in a worker process."""
    site, window_name = job
    return score_site(site=site, window_name=window_name)


def run_tune(*, workers: int) -> None:
    """Fit the tuning arms, choose the window and the bar, and fit the filter's hyperparameters."""
    cache = common.load_cache()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = dict(
            zip(
                common.DECISION_SITES,
                pool.map(_tune_worker, common.DECISION_SITES),
                strict=True,
            )
        )
    mean_scores = {
        name: {
            kind: float(np.mean([results[s][name]["scores"][kind] for s in common.DECISION_SITES]))
            for kind in ("plain", "amq")
        }
        for name in WINDOWS
    }
    window = min(mean_scores, key=lambda name: mean_scores[name]["plain"])
    bar = "x_bestamq" if mean_scores[window]["amq"] < mean_scores[window]["plain"] else "x_best"
    streams = {site: results[site][window]["stream"] for site in common.DECISION_SITES}
    fitted = fit_filter_hyperparameters(
        residuals=streams,
        band=cache.calendar.band,
        end_times=cache.end_times,
        window=(
            common.to_64(moment=common.TUNING_CUT),
            common.to_64(moment=common.year_end(cut=common.TUNING_CUT)),
        ),
    )
    values = {site: hyper.as_array() for site, hyper in fitted.items()}
    values["D9"] = np.mean(list(values.values()), axis=0)
    tuned = {
        "window": window,
        "bar": bar,
        "mean_scores": mean_scores,
        "scores_by_site": {
            name: {s: results[s][name]["scores"] for s in common.DECISION_SITES} for name in WINDOWS
        },
        "filter_hyper": {site: array.tolist() for site, array in values.items()},
    }
    (common.WORK_DIR / "tuned_xgb.json").write_text(json.dumps(tuned, indent=2) + "\n")
    print(json.dumps({k: v for k, v in tuned.items() if k != "filter_hyper"}, indent=2))


def run_score(*, workers: int) -> None:
    """Fit the scoring arms at the two cuts and write the prediction sets."""
    cache = common.load_cache()
    tuned = json.loads((common.WORK_DIR / "tuned_xgb.json").read_text())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = list(
            pool.map(_score_worker, [(site, tuned["window"]) for site in common.SCORED_SITES])
        )
    results = dict(zip(common.SCORED_SITES, outcomes, strict=True))
    names = dict.fromkeys(name for result in outcomes for name in result["frames"])
    for name in names:
        frame = pl.concat([r["frames"][name] for r in outcomes if name in r["frames"]])
        common.store_arm(name=name, frame=frame)
    for name, frame in filtered_xgb_frames(
        cache=cache, results=results, hyper_by_site=tuned["filter_hyper"]
    ).items():
        common.store_arm(name=name, frame=frame)
    pl.DataFrame([w for r in outcomes for w in r["weights"]]).write_parquet(
        common.WORK_DIR / "xgb_weights.parquet"
    )
    print(f"wrote {len(names) + 2} prediction sets")


def main() -> None:
    """Run the chosen stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("tune", "score"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.load_cache()
    if arguments.stage == "tune":
        run_tune(workers=arguments.workers)
    else:
        run_score(workers=arguments.workers)


if __name__ == "__main__":
    main()
