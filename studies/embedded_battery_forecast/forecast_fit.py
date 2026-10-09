"""Fit and score the forecast arms of the embedded-battery study, out of fold.

One arm is a method (`clim`, `persistence_conformal`, `rank_conformal`, or `xgb_quantile`) and, for
the last two, a feature recipe from `forecast_inputs`. `run_arm` returns the out-of-fold quantiles
of the 13 delivery levels for every scored half-hour of one battery at one issue time, with the
per-row losses beside them. Nothing here reads a file except `run_job`, which resumes by skipping an
arm whose output file exists.

Each battery's 3 seeds and 4 month-block folds are fitted one after another. The methods other than
`xgb_quantile` hold no random draw, so their result is stored once under every seed label.
"""

import multiprocessing
import os
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
import xgboost as xgb
from contracts.common import DELIVERY_QUANTILES
from forecast_inputs import (
    FEATURE_COLUMNS,
    ArmSpec,
    arm_frame,
    is_scored,
    price_columns,
)
from studies.battery_dispatch import rank_rule_schedule
from studies.battery_forecast import (
    conformal_quantiles,
    output_bounds,
    pinball_losses,
    repair_quantiles,
    residual_quantile_table,
    weighted_crps,
)
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
    HyperParameters,
    booster_parameters,
)
from studies.sources import EMBEDDED_BATTERY_FORECAST_DIR

MethodType = Literal["clim", "persistence_conformal", "rank_conformal", "xgb_quantile"]
SettingType = Literal["primary", "sensitivity"]

VariantType = Literal["pre_review", "as_written", "idle_dropped"]
"""Which set of fits a process reads or writes.

- `pre_review`: the fits made before the first science review, kept unchanged. Their price
  features treated the whole UTC day as published, and they are read only to report the size of
  that error.
- `as_written`: the plan's rule, every half-hour with its inputs, with price features that use
  only the prices public at each issue time.
- `idle_dropped`: `as_written` with each battery's idle lead-in (`in_service` false) dropped from
  training and scoring alike, for every arm. It refits only the batteries that have a lead-in.
"""

SETTINGS: Final[dict[SettingType, HyperParameters]] = {
    "primary": PRIMARY_HYPER_PARAMETERS,
    "sensitivity": SENSITIVITY_HYPER_PARAMETERS,
}
FIT_VARIANT_NAME: Final[str] = os.environ.get("FIT_VARIANT", "as_written")
if FIT_VARIANT_NAME not in ("pre_review", "as_written", "idle_dropped"):
    _MESSAGE = (
        f"FIT_VARIANT must be pre_review, as_written, or idle_dropped, not {FIT_VARIANT_NAME!r}."
    )
    raise ValueError(_MESSAGE)
FIT_VARIANT: Final[VariantType] = FIT_VARIANT_NAME
"""The variant this process fits and reads, set by the `FIT_VARIANT` environment variable. An
unknown name raises, so a typo cannot write fits into the `as_written` tree."""
FITS_DIRS: Final[dict[VariantType, Path]] = {
    "pre_review": EMBEDDED_BATTERY_FORECAST_DIR / "fits",
    "as_written": EMBEDDED_BATTERY_FORECAST_DIR / "fits_as_written",
    "idle_dropped": EMBEDDED_BATTERY_FORECAST_DIR / "fits_idle_dropped",
}
FITS_DIR: Final[Path] = FITS_DIRS[FIT_VARIANT]
"""One parquet file per battery, issue time, hyperparameter setting, and arm. A tree other than
`pre_review` holds real files for the fits it made and symbolic links to the earlier tree's files
for every fit that its changes leave as they were."""
LEVELS: Final[tuple[float, ...]] = DELIVERY_QUANTILES
Q_COLUMNS: Final[tuple[str, ...]] = tuple(f"q{level}" for level in LEVELS)
FIT_THREADS: Final[int] = int(os.environ.get("OMP_NUM_THREADS", "2"))
"""Threads per XGBoost fit; the study runs four fits at once, so this is 2 rather than 4."""
OVERWRITE_ARMS: Final[frozenset[str]] = frozenset(
    a for a in os.environ.get("FIT_OVERWRITE_ARMS", "").split(",") if a
)
"""Arm names that `run_job` refits even when a file exists, set by `FIT_OVERWRITE_ARMS` as a
comma-separated list, for a change that touches those arms alone."""
DEFAULT_WORKERS: Final[int] = int(os.environ.get("FIT_WORKERS", "4"))
"""Processes fitting at once, set by `FIT_WORKERS`; the cores used are this times `FIT_THREADS`."""
RANK_RULE_DURATIONS: Final[tuple[int, ...]] = (2, 4, 6, 8)
"""The durations (in half-hours) the rank rule's duration is fitted over, per battery and fold."""
N_FOLDS: Final[int] = 4
FIT_DEVICE: Final[Literal["cpu", "cuda"]] = (
    "cuda" if os.environ.get("FIT_DEVICE") == "cuda" else "cpu"
)
"""The XGBoost device, set by `FIT_DEVICE`. The study's reported fits ran on the CPU: four CPU
workers finish an arm in less time than a GPU shared by four processes. One device is used for
every arm."""


@dataclass(frozen=True)
class ArmDefinition:
    """One arm: a method and the feature recipe it reads.

    Attributes:
        name: The arm's name in the saved files, such as `xgb_quantile__price_actual`.
        method: How the arm produces quantiles.
        spec: The feature recipe, or None for `clim` and `persistence_conformal`.
    """

    name: str
    method: MethodType
    spec: ArmSpec | None


def arm_file(*, setting: SettingType, issue: str, battery_id: str, arm: str) -> Path:
    """Return the path of one arm's saved forecasts and losses."""
    return FITS_DIR / setting / issue / f"{battery_id}__{arm}.parquet"


def with_model_price(*, base: pl.DataFrame, price_model: pl.DataFrame) -> pl.DataFrame:
    """Add the price model's forecast and its derived columns to a wide input frame.

    Args:
        base: A frame `forecast_inputs.build_issue_frame` returns.
        price_model: Columns `time` and `price_model`.

    Returns:
        The frame with `price_model`, `price_rank_model`, `price_day_mean_model`,
        `price_day_range_model`, and `rank_rule_model`.
    """
    return price_columns(
        frame=base.drop([c for c in base.columns if c.endswith("_model")], strict=False).join(
            price_model.select("time", "price_model"), on="time", how="left"
        ),
        source="model",
    )


def _quantiles_xgb(
    *,
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    hyper_parameters: HyperParameters,
    seed: int,
) -> np.ndarray:
    """Fit the 13-level multi-quantile XGBoost model on the training rows; predict the test rows."""
    parameters = booster_parameters(hyper_parameters=hyper_parameters, seed=seed)
    parameters["nthread"] = FIT_THREADS
    parameters["device"] = FIT_DEVICE
    parameters["objective"] = "reg:quantileerror"
    parameters["quantile_alpha"] = np.asarray(LEVELS)
    train_matrix = xgb.DMatrix(features[train], label=target[train])
    model = xgb.train(parameters, train_matrix, num_boost_round=hyper_parameters["num_boost_round"])
    return np.atleast_2d(model.predict(xgb.DMatrix(features[test])))


def _fill_missing_groups(
    *, table: dict[object, np.ndarray], groups: np.ndarray, residuals: np.ndarray
) -> dict[object, np.ndarray]:
    """Give a group that no training row occupies the pooled residual quantiles."""
    pooled = np.quantile(residuals[~np.isnan(residuals)], LEVELS)
    return {**{g.item(): pooled for g in np.unique(groups)}, **table}


def _quantiles_persistence(
    *, frame: pl.DataFrame, train: np.ndarray, test: np.ndarray
) -> np.ndarray:
    """Persistence plus the training residuals' quantiles, by half-hour of day."""
    centre = frame["persistence_mw"].to_numpy()
    residual = frame["output_mw"].to_numpy() - centre
    groups = frame["tod"].to_numpy()
    table = residual_quantile_table(residuals=residual[train], groups=groups[train], levels=LEVELS)
    table = _fill_missing_groups(table=table, groups=groups, residuals=residual[train])
    return conformal_quantiles(centre=centre[test], groups=groups[test], table=table)


def schedule_at_issue(
    *, prices: np.ndarray, hidden: np.ndarray, unpublished: np.ndarray, duration: int
) -> np.ndarray:
    """Return the rank-rule schedule that a forecaster could build at each row's issue time.

    Args:
        prices: The price of each half-hour.
        hidden: The same prices with the hour that is not yet public replaced.
        unpublished: Whether each row's issue time precedes that hour's publication.
        duration: The schedule's duration in half-hours.

    Returns:
        For each row, the schedule built from `hidden` if `unpublished`, else from `prices`.
    """
    return np.where(
        unpublished,
        rank_rule_schedule(prices=hidden, duration_half_hours=duration),
        rank_rule_schedule(prices=prices, duration_half_hours=duration),
    )


def _quantiles_rank_rule(
    *, frame: pl.DataFrame, price_column: str, train: np.ndarray, test: np.ndarray
) -> np.ndarray:
    """The scaled rank-rule schedule plus the training residuals' quantiles, by schedule state.

    The schedule's duration is the one of `RANK_RULE_DURATIONS` with the lowest mean absolute
    error on the training rows once the schedule is scaled; the scale is the least-squares
    coefficient through the origin on the training rows. A row issued before the next UK day's
    price is public takes its schedule from the price with that hour replaced.
    """
    prices = frame[price_column].to_numpy()
    published_view = price_column == "price_actual" and "price_actual_unpublished" in frame.columns
    hidden = frame["price_actual_hidden"].to_numpy() if published_view else prices
    unpublished = (
        frame["price_actual_unpublished"].to_numpy()
        if published_view
        else np.zeros(len(prices), bool)
    )
    output = frame["output_mw"].to_numpy()
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for duration in RANK_RULE_DURATIONS:
        schedule = schedule_at_issue(
            prices=prices, hidden=hidden, unpublished=unpublished, duration=duration
        )
        squared = float(np.sum(schedule[train] ** 2))
        scale = float(np.sum(output[train] * schedule[train]) / squared) if squared > 0 else 0.0
        centre = scale * schedule
        error = float(np.mean(np.abs(output[train] - centre[train])))
        if best is None or error < best[0]:
            best = (error, centre, schedule)
    assert best is not None
    _, centre, schedule = best
    groups = np.sign(schedule).astype(np.int64)
    residual = output - centre
    table = residual_quantile_table(residuals=residual[train], groups=groups[train], levels=LEVELS)
    table = _fill_missing_groups(table=table, groups=groups, residuals=residual[train])
    return conformal_quantiles(centre=centre[test], groups=groups[test], table=table)


def run_arm(
    *,
    base: pl.DataFrame,
    arm: ArmDefinition,
    scored: np.ndarray,
    hyper_parameters: HyperParameters,
    model_price_by_fold: dict[int, pl.DataFrame] | None = None,
    seeds: Sequence[int] = SEEDS,
) -> pl.DataFrame:
    """Return one arm's out-of-fold quantiles and per-row losses on the scored half-hours.

    Args:
        base: The wide input frame `forecast_inputs.build_issue_frame` returns (with the plain
            out-of-fold `price_model` columns when the arm uses that price).
        arm: The arm.
        scored: Which rows of `base` are scored, shape (n_rows,), from `is_scored`. The training
            rows of every fold are the scored rows outside the fold.
        hyper_parameters: The XGBoost settings; unused by the other methods.
        model_price_by_fold: For an arm given the `model` price, the price model's forecast as it
            stood when its held-out fold was `f`: trained without fold `f`, with out-of-fold
            forecasts on the other folds. The quantile model of fold `f` reads fold `f`'s frame, so
            no training row's price forecast was built from the test fold's prices.
        seeds: The XGBoost seeds.

    Returns:
        One row per scored half-hour and seed: `time`, `month`, `fold`, `seed`, `truth_mw`,
        `p99_mw`, the 13 `q<level>` columns in megawatts (Float32), and the losses `crps_pct`,
        `pinball_pct` (mean over the levels), and `median_abs_error_pct`, each as a percentage of
        the battery's 99th-percentile absolute output.
    """
    output_all = base["output_mw"].drop_nulls().to_numpy()
    p99 = float(np.quantile(np.abs(output_all), 0.99))
    fold = base["fold"].to_numpy()
    quantiles_by_seed: dict[int, np.ndarray] = {}
    stochastic = arm.method == "xgb_quantile"
    for seed in seeds if stochastic else seeds[:1]:
        quantiles = np.full((base.height, len(LEVELS)), np.nan)
        for held_out in range(N_FOLDS):
            train = scored & (fold != held_out)
            test = scored & (fold == held_out)
            if not test.any():
                continue
            frame = base
            if arm.spec is not None and arm.spec.price_source == "model":
                assert model_price_by_fold is not None
                frame = with_model_price(base=base, price_model=model_price_by_fold[held_out])
            match arm.method:
                case "clim":
                    raw = base.select(Q_COLUMNS).to_numpy()[test]
                case "persistence_conformal":
                    raw = _quantiles_persistence(frame=base, train=train, test=test)
                case "rank_conformal":
                    assert arm.spec is not None
                    raw = _quantiles_rank_rule(
                        frame=frame,
                        price_column=f"price_{arm.spec.price_source}",
                        train=train,
                        test=test,
                    )
                case "xgb_quantile":
                    assert arm.spec is not None
                    columns = arm_frame(base=frame, arm=arm.spec)
                    raw = _quantiles_xgb(
                        features=columns.select(FEATURE_COLUMNS).to_numpy(),
                        target=columns["output_mw"].fill_null(0.0).to_numpy(),
                        train=train,
                        test=test,
                        hyper_parameters=hyper_parameters,
                        seed=seed,
                    )
            lower, upper = output_bounds(training_output=base["output_mw"].to_numpy()[train])
            quantiles[test] = repair_quantiles(quantiles=raw, lower=lower, upper=upper)
        quantiles_by_seed[seed] = quantiles
    frames = []
    truth = base["output_mw"].to_numpy()
    for seed in seeds:
        quantiles = quantiles_by_seed[seed if stochastic else seeds[0]][scored]
        actual = truth[scored]
        pinball = pinball_losses(truth=actual, quantiles=quantiles, levels=LEVELS)
        median = quantiles[:, LEVELS.index(0.5)]
        frames.append(
            pl.DataFrame(
                {
                    "time": base["time"].filter(pl.Series(scored)),
                    "month": base["month"].filter(pl.Series(scored)),
                    "fold": base["fold"].filter(pl.Series(scored)),
                    "seed": np.full(actual.size, seed, dtype=np.int64),
                    "truth_mw": actual,
                    "p99_mw": np.full(actual.size, p99),
                    **{
                        name: quantiles[:, i].astype(np.float32) for i, name in enumerate(Q_COLUMNS)
                    },
                    "crps_pct": 100.0
                    * weighted_crps(truth=actual, quantiles=quantiles, levels=LEVELS)
                    / p99,
                    "pinball_pct": 100.0 * pinball.mean(axis=1) / p99,
                    "median_abs_error_pct": 100.0 * np.abs(actual - median) / p99,
                }
            )
        )
    return pl.concat(frames)


def scored_mask(*, base: pl.DataFrame, arms: dict[str, ArmSpec]) -> np.ndarray:
    """Return `forecast_inputs.is_scored` as a NumPy Boolean array.

    In the `idle_dropped` variant the mask also excludes the half-hours before the end of the
    battery's idle lead-in, so those rows are neither trained on nor scored.
    """
    mask = is_scored(base=base, arms=arms)
    if FIT_VARIANT == "idle_dropped":
        mask = mask & base["in_service"]
    return mask.to_numpy()


def uses_the_actual_price(*, issue: str, arm: str) -> bool:
    """Return whether an arm's features include the N2EX price of the target day as published.

    The `as_written` fits refit these arms, because their price features changed. `DA-early`'s
    `price_actual` arm is the perfect-price upper bound and keeps every hour, so it is not one.

    Args:
        issue: The saved issue folder name (`A0_DA-late` for rung A0).
        arm: The arm name.

    Returns:
        True for an arm that the `as_written` fits refit.
    """
    if issue == "DA-early":
        return arm.endswith("price_model")
    if issue in ("DA-late", "A0_DA-late"):
        return arm.endswith("price_actual")
    return arm.startswith("xgb_quantile__") or arm == "rank_conformal__no_neighbour"


def link_fits(
    *, source: Path, target: Path, skip: Callable[[Path], bool] = lambda path: False
) -> int:
    """Link each fit file of `source` into `target`, except those `skip` names.

    Args:
        source: The tree to link from.
        target: The tree to link into.
        skip: Returns true for a file that `target` refits and so must not link.

    Returns:
        How many links were created.
    """
    created = 0
    for path in sorted(source.rglob("*.parquet")):
        destination = target / path.relative_to(source)
        if skip(path) or destination.is_symlink() or destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(path.resolve())
        created += 1
    return created


def link_unchanged_fits(*, affected: Collection[str] = ()) -> int:
    """Link into this process's tree the fits that its variant leaves unchanged.

    `as_written` links the `pre_review` fits except the arms that use the actual price.
    `idle_dropped` links the `as_written` fits of every battery not in `affected`.

    Args:
        affected: For `idle_dropped`, the batteries whose fits are real files.

    Returns:
        How many links were created.
    """
    if FIT_VARIANT == "as_written":
        return link_fits(
            source=FITS_DIRS["pre_review"],
            target=FITS_DIR,
            skip=lambda path: uses_the_actual_price(
                issue=path.parent.name, arm=path.stem.split("__", 1)[1]
            ),
        )
    if FIT_VARIANT == "idle_dropped":
        return link_fits(
            source=FITS_DIRS["as_written"],
            target=FITS_DIR,
            skip=lambda path: path.name.split("__")[0] in affected,
        )
    return 0


def run_job(
    *,
    battery_id: str,
    issue: str,
    setting: SettingType,
    base: pl.DataFrame,
    arms: Sequence[ArmDefinition],
    scoring_arms: dict[str, ArmSpec],
    model_price_by_fold: dict[int, pl.DataFrame] | None = None,
) -> list[str]:
    """Run and save every arm of one battery, issue time, and setting that has no file yet.

    Args:
        battery_id: The battery's identifier, as used in the file names.
        issue: The issue type.
        setting: The hyperparameter setting.
        base: The wide input frame, with the plain out-of-fold `price_model` columns if any arm
            or scoring arm uses that price.
        arms: The arms to run.
        scoring_arms: The arms whose inputs decide the scored half-hours.
        model_price_by_fold: See `run_arm`.

    Returns:
        The names of the arms that were run (not skipped).
    """
    scored = scored_mask(base=base, arms=scoring_arms)
    ran = []
    for arm in arms:
        path = arm_file(setting=setting, issue=issue, battery_id=battery_id, arm=arm.name)
        if FIT_VARIANT == "pre_review":
            msg = "The pre_review fits are read-only."
            raise ValueError(msg)
        if path.exists() and arm.name not in OVERWRITE_ARMS:
            continue
        result = run_arm(
            base=base,
            arm=arm,
            scored=scored,
            hyper_parameters=SETTINGS[setting],
            model_price_by_fold=model_price_by_fold,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
        ran.append(arm.name)
    return ran


def run_in_pool[T](
    *, function: Callable[[T], object], tasks: Sequence[T], workers: int = DEFAULT_WORKERS
) -> None:
    """Run `function` on every task in a pool of spawned processes, printing each as it finishes.

    A spawned process starts clean, which a CUDA context needs. A task that raises stops the pool
    and re-raises, because an R&D fit fails fast.

    Args:
        function: A module-level function of one argument.
        tasks: The arguments.
        workers: How many processes run at once.
    """
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(function, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            print(f"[{done}/{len(tasks)}] {futures[future]}: {future.result()}", flush=True)
