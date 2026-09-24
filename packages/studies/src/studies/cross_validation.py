"""Fit XGBoost per site, out of fold, and score every held-out row.

**Every model is fitted on one site's own history and scored on a fold it never saw.** The folds are
contiguous blocks of whole months, the training side drops the hours the network operator had
curtailed, and each fold is fitted once per seed. Every study that compares feature sets on the
metered generators runs this loop, so a fault in it would move every comparison at once.

**Every error is divided by its own row's capacity before it is averaged or differenced.** A table
of mean errors and a table of differences between those means then subtract exactly, which they do
not if one divides per row and the other divides a pooled megawatt difference by a pooled capacity.
"""

from collections.abc import Mapping, Sequence
from typing import Final, TypedDict

import numpy as np
import polars as pl
import xgboost as xgb

N_FOLDS: Final[int] = 5
"""Contiguous time blocks, each used once as the test fold.

Blocks are contiguous rather than random because neighbouring hours share a weather system, so a
random split would put near-copies of a test row in the training set. They are cut inside each
site's own span rather than across the whole roster's, because the roster's span is seven years and
the newest site has two and a half: global blocks would leave that site with three empty test folds
and train the remaining two on its own future.
"""

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
"""Each (arm, site, fold) is fitted once per seed.

Seed-to-seed variation is the noise floor any arm-to-arm difference has to clear, so it is measured
rather than assumed, and the bootstrap draws a seed as well as a set of months so that the noise
reaches the interval instead of being averaged out of it.
"""

THREADS_PER_FIT: Final[int] = 4
"""How many cores each XGBoost fit uses."""

QUANTILE_LEVELS: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
"""Quantiles for the continuous ranked probability score.

The score is the integral of twice the pinball loss over every level in (0, 1), approximated here by
a Riemann sum: the level spacing times twice the summed loss. The same levels are used for every
arm, so the approximation cannot favour one.
"""

QUANTILE_LEVEL_SPACING: Final[float] = 0.1
"""The gap between neighbouring `QUANTILE_LEVELS`, which weights each level in the Riemann sum."""


class HyperParameters(TypedDict):
    """The XGBoost settings held fixed across every arm, site, fold and seed."""

    max_depth: int
    learning_rate: float
    subsample: float
    min_child_weight: float
    reg_lambda: float
    num_boost_round: int


PRIMARY_HYPER_PARAMETERS: Final[HyperParameters] = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_child_weight": 20.0,
    "reg_lambda": 1.0,
    "num_boost_round": 500,
}
"""Fixed, not tuned.

Tuning per arm would let the tuner's own noise decide which arm wins, and early stopping would give
each arm a different number of rounds for reasons unrelated to the irradiance features. Both are
channels through which an arm could win without carrying more information, so neither is used.

**There is no `colsample_bytree` here, and its absence is load-bearing.** Column subsampling below 1
hands every arm with more columns a free win over an arm with fewer: the one-irradiance-feature arm
loses its only irradiance column in a large share of trees, while a three-column arm always keeps
one. Measured on this data with a feature set padded out by duplicate columns carrying no
information at all, `colsample_bytree=0.8` produced a 10% improvement in mean absolute error out of
pure redundancy — comfortably larger than the effect being measured. Row subsampling is kept,
because it does not depend on how many columns an arm has, and it is what gives the seeds something
to vary.

**Row subsampling samples rows by position**, so the order the training rows arrive in is part of
the model. Reordering them moves every prediction, which is why the loop below filters one
site-sorted frame rather than assembling its training rows any other way.
"""

SENSITIVITY_HYPER_PARAMETERS: Final[HyperParameters] = {
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "min_child_weight": 50.0,
    "reg_lambda": 5.0,
    "num_boost_round": 1200,
}
"""A shallower, more heavily regularised alternative.

A single hyperparameter setting can favour an arm by accident — a wider feature set changes how much
a fixed number of rounds overfits. Running a second, deliberately different setting says whether the
arm ordering is a property of the features or of the settings.
"""


def assign_folds(*, dataset: pl.DataFrame, by: Sequence[str] = ("site",)) -> pl.DataFrame:
    """Cut each group's own span into `N_FOLDS` contiguous blocks of whole months.

    The default cuts each site's span. Grouping by site and by an era label as well cuts each era
    separately, so every fold holds a slice of every era: a product that changed mid-record is then
    scored by models that trained on the version being scored, rather than by a last fold whose
    model saw only the earlier version.

    Args:
        dataset: Rows carrying the `by` columns and the `month` label.
        by: The columns each fold count is taken within.

    Returns:
        `dataset` with an integer `fold` column.
    """
    group = list(by)
    month_rank = pl.col("month").rank(method="dense").over(group)
    month_count = pl.col("month").n_unique().over(group)
    fold = (month_rank - 1) * N_FOLDS // month_count
    return dataset.with_columns(fold=fold.cast(pl.Int32))


def rotate_folds(*, frame: pl.DataFrame, fold_offsets: Mapping[int, int]) -> pl.DataFrame:
    """Rotate each era's fold numbers by an offset, modulo `N_FOLDS`.

    Rotating leaves the folds contiguous within an era. It changes which fold each era's months fall
    in, so that a calendar month held out in one era is training data in another.

    Args:
        frame: Rows carrying `era_code` and `fold`.
        fold_offsets: Each `era_code` to how far its fold numbers are rotated.

    Returns:
        The frame with the rotated `fold`.
    """
    rotation = pl.col("era_code").replace_strict(dict(fold_offsets), return_dtype=pl.Int32)
    return frame.with_columns(fold=(pl.col("fold") + rotation) % N_FOLDS)


def cut_eras(
    *, frame: pl.DataFrame, first_months: Sequence[str], fold_offsets: Mapping[int, int]
) -> pl.DataFrame:
    """Label eras that begin at the given months, cut folds inside each, and rotate them.

    An era is a stretch of months over which an input kept one version. Each era is cut into
    `N_FOLDS` blocks per site by `assign_folds`, and then each era's fold numbers are rotated by
    `fold_offsets`.

    Args:
        frame: Rows carrying `site` and `month`, where `month` is a `%Y-%m` string.
        first_months: The first month of every era after the first, in ascending order.
        fold_offsets: Each `era_code` (0 for the first era, counting up) to how far its fold numbers
            are rotated.

    Returns:
        The frame with `era_code`, `era` (`era_code` as a string, for `assign_folds`) and `fold`.
    """
    era_code = sum((pl.col("month") >= month).cast(pl.Int8) for month in first_months)
    labelled = frame.with_columns(era_code=era_code).with_columns(
        era=pl.col("era_code").cast(pl.String)
    )
    return rotate_folds(
        frame=assign_folds(dataset=labelled, by=("site", "era")), fold_offsets=fold_offsets
    )


def calendar_month_coverage(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Count, for each held-out calendar month, the training rows that carry it.

    For every (site, fold, calendar month) with rows in the fold, counts the rows of the same site
    and calendar month that lie in the other folds, from any era or year. A count of 0 means the
    fitted model has seen no row of that season.

    Args:
        frame: The row set carrying `site`, `fold` and `time`.

    Returns:
        One row per (site, fold, calendar_month) with `n_scored`, `n_train`, `n_years` (how many
        distinct years of the site's rows carry that calendar month) and `covered`.
    """
    rows = frame.select(
        "site",
        "fold",
        calendar_month=pl.col("time").dt.month(),
        year=pl.col("time").dt.year(),
    )
    totals = rows.group_by("site", "calendar_month").agg(
        n_total=pl.len(), n_years=pl.col("year").n_unique()
    )
    return (
        rows.group_by("site", "fold", "calendar_month")
        .agg(n_scored=pl.len())
        .join(totals, on=["site", "calendar_month"])
        .with_columns(n_train=pl.col("n_total") - pl.col("n_scored"))
        .with_columns(covered=pl.col("n_train") > 0)
        .drop("n_total")
        .sort("site", "fold", "calendar_month")
    )


def uncovered_months(*, coverage: pl.DataFrame) -> pl.DataFrame:
    """Return the coverage rows with no training row for a calendar month that a design could cover.

    Args:
        coverage: `calendar_month_coverage`'s result.

    Returns:
        The rows with no training row for a calendar month that occurs in more than one year: the
        failures the fold design could have avoided.
    """
    return coverage.filter(~pl.col("covered"), pl.col("n_years") > 1)


def raise_on_uncovered_months(*, coverage: pl.DataFrame) -> None:
    """Raise if a held-out calendar month that occurs in two years has no training row.

    Args:
        coverage: `calendar_month_coverage`'s result.

    Raises:
        ValueError: Naming the first failing (site, fold, calendar month) rows.
    """
    failures = uncovered_months(coverage=coverage)
    if failures.height:
        msg = (
            f"{failures.height} (site, fold, calendar month) cells hold out a calendar month that "
            f"occurs in two years and leave no training row for it: {failures.head(5).to_dicts()}"
        )
        raise ValueError(msg)


def crps(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return the per-row continuous ranked probability score, approximated from the quantiles.

    Args:
        actual: Observed power, shape (n_rows,).
        quantiles: Predicted quantiles, shape (n_rows, n_levels), in the order of `QUANTILE_LEVELS`.

    Returns:
        The per-row score, shape (n_rows,).
    """
    # XGBoost's multi-quantile head can return crossing quantiles; sorting each row repairs that
    # without changing any individual level's calibration much, and is applied to every arm alike.
    sorted_quantiles = np.sort(quantiles, axis=1)
    levels = np.asarray(QUANTILE_LEVELS)[None, :]
    difference = actual[:, None] - sorted_quantiles
    pinball = np.where(difference >= 0, difference * levels, -difference * (1.0 - levels))
    return 2.0 * QUANTILE_LEVEL_SPACING * pinball.sum(axis=1)


def booster_parameters(*, hyper_parameters: HyperParameters, seed: int) -> dict[str, object]:
    """Translate the settings above into XGBoost's own parameter names.

    Args:
        hyper_parameters: The settings to translate.
        seed: The XGBoost random seed.

    Returns:
        The parameter dictionary `xgb.train` takes, with no objective set.
    """
    return {
        "max_depth": hyper_parameters["max_depth"],
        "eta": hyper_parameters["learning_rate"],
        "subsample": hyper_parameters["subsample"],
        "min_child_weight": hyper_parameters["min_child_weight"],
        "lambda": hyper_parameters["reg_lambda"],
        "tree_method": "hist",
        "seed": seed,
        "nthread": THREADS_PER_FIT,
    }


def clamp_to_cap(*, prediction: np.ndarray, cap_mw: pl.Series) -> np.ndarray:
    """Hold a prediction at or below the export cap that was in force.

    Args:
        prediction: Either one prediction per row, or one row per prediction and one column per
            quantile level.
        cap_mw: The cap for each row, null where no setpoint record covers it.

    Returns:
        `prediction`, with every element that exceeded its row's cap replaced by that cap. Rows
        with no cap are returned unchanged.
    """
    ceiling = cap_mw.fill_null(np.inf).to_numpy()
    if prediction.ndim > 1:
        ceiling = ceiling[:, np.newaxis]
    return np.minimum(prediction, ceiling)


def fit_one_fold(
    *,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: list[str],
    target: str,
    hyper_parameters: HyperParameters,
    seed: int,
    with_quantiles: bool,
    weight: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit one point model, optionally one quantile model, and predict the test fold.

    Args:
        train: The training rows.
        test: The test rows.
        features: The feature columns to show the model.
        target: The column to predict.
        hyper_parameters: Settings shared by every arm.
        seed: The XGBoost random seed.
        with_quantiles: Whether to fit the quantile model as well.
        weight: A column of training-row weights, or `None` to weigh every row alike.

    Returns:
        The point predictions, and the quantile predictions or `None`.
    """
    train_matrix = xgb.DMatrix(
        train.select(features).to_numpy(),
        label=train[target].to_numpy(),
        weight=None if weight is None else train[weight].to_numpy(),
    )
    test_matrix = xgb.DMatrix(test.select(features).to_numpy())
    shared = booster_parameters(hyper_parameters=hyper_parameters, seed=seed)
    rounds = hyper_parameters["num_boost_round"]

    point_model = xgb.train(
        {**shared, "objective": "reg:absoluteerror"}, train_matrix, num_boost_round=rounds
    )
    point = point_model.predict(test_matrix)
    if not with_quantiles:
        return point, None

    quantile_model = xgb.train(
        {**shared, "objective": "reg:quantileerror", "quantile_alpha": np.asarray(QUANTILE_LEVELS)},
        train_matrix,
        num_boost_round=rounds,
    )
    return point, np.atleast_2d(quantile_model.predict(test_matrix))


def out_of_fold_losses(
    *,
    site_rows: pl.DataFrame,
    features: Sequence[str],
    target: str,
    hyper_parameters: HyperParameters,
    with_quantiles: bool,
    weight: str | None = None,
) -> pl.DataFrame:
    """Produce out-of-fold losses for one feature set at one site, one row per (test row, seed).

    A feature name carrying `{fold}` is resolved against the fold being scored, which is how a
    feature built by a model of its own keeps that model's training rows out of the scored fold.
    Every other name is left alone, because formatting a string with no placeholder in it returns
    the string.

    Args:
        site_rows: Every row for one site, already carrying `fold`, `month`, `cap_mw`,
            `constrained` and `effective_capacity_mw`.
        features: The feature columns to show the model.
        target: The column to predict.
        hyper_parameters: The setting to fit at.
        with_quantiles: Whether to score the continuous ranked probability score too.
        weight: A column of training-row weights, or `None` to weigh every row alike.

    Returns:
        One row per (time, seed) with the losses in megawatts and as a fraction of the row's own
        capacity.
    """
    outputs: list[pl.DataFrame] = []
    for fold in range(N_FOLDS):
        test = site_rows.filter(pl.col("fold") == fold)
        # A constrained hour is one the network operator turned down, so no irradiance product
        # could have predicted it and a model that trains on it learns to read network
        # instructions out of the sky. Scoring keeps those hours; only training drops them.
        train = site_rows.filter((pl.col("fold") != fold) & ~pl.col("constrained"))
        if test.is_empty() or train.is_empty():
            continue
        fold_features = [name.format(fold=fold) for name in features]
        actual = test[target].to_numpy()
        for seed in SEEDS:
            point, quantiles = fit_one_fold(
                train=train,
                test=test,
                features=fold_features,
                target=target,
                hyper_parameters=hyper_parameters,
                seed=seed,
                with_quantiles=with_quantiles,
                weight=weight,
            )
            outputs.append(
                _losses(test=test, actual=actual, point=point, quantiles=quantiles, seed=seed)
            )
    return pl.concat(outputs)


def _losses(
    *,
    test: pl.DataFrame,
    actual: np.ndarray,
    point: np.ndarray,
    quantiles: np.ndarray | None,
    seed: int,
) -> pl.DataFrame:
    """Score one fold's predictions, uncapped and held to the export cap.

    Every loss column is pinned to Float64. The real target is Float32 and a synthetic target is
    Float64, so without the cast two runs produce frames that cannot be stacked. The uncapped errors
    are formed in the target's own precision and only then widened, while the capped ones are formed
    in Float64, because the cap's null-filling ceiling is Float64.

    Args:
        test: The scored rows.
        actual: The target on those rows.
        point: The point prediction.
        quantiles: The quantile predictions, or `None`.
        seed: The seed the predictions were fitted at.

    Returns:
        One row per scored row.
    """
    crps_mw = (
        pl.Series(crps(actual=actual, quantiles=quantiles), dtype=pl.Float64)
        if quantiles is not None
        else pl.lit(None, dtype=pl.Float64)
    )
    capped_point = clamp_to_cap(prediction=point, cap_mw=test["cap_mw"])
    crps_capped_mw = (
        pl.Series(
            crps(
                actual=actual, quantiles=clamp_to_cap(prediction=quantiles, cap_mw=test["cap_mw"])
            ),
            dtype=pl.Float64,
        )
        if quantiles is not None
        else pl.lit(None, dtype=pl.Float64)
    )
    return (
        test.select("site", "time", "month", "fold", "effective_capacity_mw", "constrained")
        .cast({"effective_capacity_mw": pl.Float64})
        .with_columns(
            seed=pl.lit(seed, dtype=pl.Int32),
            absolute_error_mw=pl.Series(np.abs(actual - point), dtype=pl.Float64),
            signed_error_mw=pl.Series(point - actual, dtype=pl.Float64),
            crps_mw=crps_mw,
            absolute_error_capped_mw=pl.Series(np.abs(actual - capped_point), dtype=pl.Float64),
            signed_error_capped_mw=pl.Series(capped_point - actual, dtype=pl.Float64),
            crps_capped_mw=crps_capped_mw,
        )
        .with_columns(
            absolute_error_fraction_of_capacity=pl.col("absolute_error_mw")
            / pl.col("effective_capacity_mw"),
            absolute_error_capped_fraction_of_capacity=pl.col("absolute_error_capped_mw")
            / pl.col("effective_capacity_mw"),
        )
    )


def out_of_fold_member_forecasts(
    *,
    site_rows: pl.DataFrame,
    features: Sequence[str],
    target: str,
    hyper_parameters: HyperParameters,
) -> pl.DataFrame:
    """Fit one model on every ensemble member's rows at one site, and forecast each member.

    `site_rows` holds one row per (time, member), each member's weather beside the same measured
    target, so one model learns one mapping from a member's weather to power and is then applied to
    every member. The fold label belongs to the time, so every member of a scored month is left out
    of training together, and the fit loop is `out_of_fold_losses`, which drops curtailed hours
    from training as it does for every other arm.

    **Each member's row is weighted by one over the number of members, so each hour carries the
    weight one row carries in a model fitted on one input.** Without the weights an hour counts
    once per member towards `min_child_weight`, so the same hyperparameters grow finer leaves on the
    stacked rows than on one row per hour: on one generator over five months, 51 identical copies of
    the ensemble mean, unweighted, raised the mean absolute error by 0.7 points of capacity over the
    model fitted on the ensemble mean once.

    Args:
        site_rows: One site's rows, one per (time, member), carrying `time`, `member`, `fold`,
            `month`, `cap_mw`, `constrained`, `effective_capacity_mw`, the features, and the
            target.
        features: The feature columns to show the model.
        target: The column to predict.
        hyper_parameters: The setting to fit at.

    Returns:
        One row per (site, time, seed), with `forecasts`, the list of the members' uncapped
        forecasts in member order.

    Raises:
        ValueError: If the times do not all hold the same number of members.
    """
    members = site_rows.group_by("time").len()["len"]
    if members.n_unique() != 1:
        msg = "every time must hold the same number of members"
        raise ValueError(msg)
    losses = out_of_fold_losses(
        site_rows=site_rows.sort("time", "member").with_columns(
            member_weight=pl.lit(1.0 / members[0])
        ),
        features=features,
        target=target,
        hyper_parameters=hyper_parameters,
        with_quantiles=False,
        weight="member_weight",
    )
    actual = site_rows.group_by("site", "time").agg(actual=pl.col(target).first().cast(pl.Float64))
    return (
        losses.select("site", "time", "seed", "signed_error_mw")
        .join(actual, on=["site", "time"])
        .group_by("site", "time", "seed", maintain_order=True)
        .agg(forecasts=pl.col("signed_error_mw") + pl.col("actual"))
    )


def out_of_fold_forecasts_for_members(
    *,
    site_rows: pl.DataFrame,
    member_rows: pl.DataFrame,
    features: Sequence[str],
    target: str,
    hyper_parameters: HyperParameters,
) -> pl.DataFrame:
    """Fit on one input at one site, out of fold, and apply each fold's model to every member.

    The model is trained on `site_rows`, one row per time, such as the ensemble mean, and each
    fold's model forecasts every member's row at the fold's times. The input a model is scored on
    therefore differs from the one it was trained on.

    Args:
        site_rows: One site's training input, one row per time, carrying `time`, `fold`,
            `constrained`, the features and the target.
        member_rows: The same site's rows, one per (time, member), carrying `time`, `member`,
            `fold`, and the same feature columns.
        features: The feature columns.
        target: The column to predict.
        hyper_parameters: The setting to fit at.

    Returns:
        One row per (site, time, seed), with `forecasts`, the members' uncapped forecasts in
        member order.
    """
    outputs = []
    for fold in range(N_FOLDS):
        train = site_rows.filter((pl.col("fold") != fold) & ~pl.col("constrained"))
        test = member_rows.filter(pl.col("fold") == fold).sort("time", "member")
        if train.is_empty() or test.is_empty():
            continue
        for seed in SEEDS:
            point, _ = fit_one_fold(
                train=train,
                test=test,
                features=list(features),
                target=target,
                hyper_parameters=hyper_parameters,
                seed=seed,
                with_quantiles=False,
            )
            outputs.append(
                test.select("site", "time")
                .with_columns(
                    seed=pl.lit(seed, dtype=pl.Int32), forecast=pl.Series(point, dtype=pl.Float64)
                )
                .group_by("site", "time", "seed", maintain_order=True)
                .agg(forecasts=pl.col("forecast"))
            )
    return pl.concat(outputs)


def summarise_member_forecasts(*, forecasts: pl.DataFrame) -> pl.DataFrame:
    """Reduce each row's member forecasts to their mean, median, spread, and 10th and 90th centiles.

    Args:
        forecasts: One row per (site, time, seed), with `forecasts`, a list of member forecasts.

    Returns:
        One row per (site, time, seed), with `mean`, `median`, `spread` (the standard deviation),
        `p10`, `p90`, and `members`, the list's length.
    """
    members = pl.col("forecasts")
    return forecasts.select(
        "site",
        "time",
        "seed",
        mean=members.list.mean(),
        median=members.list.median(),
        spread=members.list.std(),
        p10=members.list.eval(pl.element().quantile(0.1)).list.first(),
        p90=members.list.eval(pl.element().quantile(0.9)).list.first(),
        members=members.list.len(),
    )


def score_prediction(*, rows: pl.DataFrame, prediction: pl.DataFrame, target: str) -> pl.DataFrame:
    """Score a prediction per (site, time, seed) as `out_of_fold_losses` scores its own.

    The prediction is held to the export cap in force and its error divided by the row's own
    generator's capacity, so a forecast made outside the fit loop, such as a reduction of member
    forecasts or a baseline, is scored exactly as a fitted arm is.

    Args:
        rows: The scored rows, carrying `site`, `time`, `month`, `fold`, `constrained`,
            `effective_capacity_mw`, `cap_mw`, and the target.
        prediction: `site`, `time`, `seed`, and `prediction`, in the target's unit.
        target: The target column.

    Returns:
        One row per (site, time, seed) in `prediction`, with `signed_error_capped_mw`,
        `absolute_error_capped_mw`, and `absolute_error_capped_fraction_of_capacity`.
    """
    scored = rows.select(
        "site", "time", "month", "fold", "constrained", "effective_capacity_mw", "cap_mw", target
    ).join(prediction, on=["site", "time"])
    capped = clamp_to_cap(prediction=scored["prediction"].to_numpy(), cap_mw=scored["cap_mw"])
    actual = scored[target].cast(pl.Float64).to_numpy()
    return scored.select(
        "site",
        "time",
        "month",
        "fold",
        "constrained",
        "seed",
        effective_capacity_mw=pl.col("effective_capacity_mw").cast(pl.Float64),
        signed_error_capped_mw=pl.Series(capped - actual, dtype=pl.Float64),
        absolute_error_capped_mw=pl.Series(np.abs(capped - actual), dtype=pl.Float64),
    ).with_columns(
        absolute_error_capped_fraction_of_capacity=pl.col("absolute_error_capped_mw")
        / pl.col("effective_capacity_mw")
    )
