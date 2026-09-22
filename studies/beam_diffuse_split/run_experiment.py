"""Train one XGBoost PV forecast per arm and measure what the beam/diffuse split is worth.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>. It reads the frame
`build_dataset.py` wrote and writes the per-row losses, the per-site metrics, the bootstrap
intervals and a JSON summary.

Every arm sees identical rows, identical folds, identical seeds, identical hyperparameters and
identical non-irradiance features. The only thing that changes between arms is which irradiance
columns the model is shown, and every one of those columns is a flux onto a horizontal plane, so no
arm is handed a different encoding of the same quantity:

- **A — global only**: ERA5 `ssrd` as global horizontal irradiance.
- **B — separation model**: `ssrd`, plus the beam and diffuse horizontal fluxes the Erbs separation
  model derives from `ssrd` alone.
- **C — the model's own split**: `ssrd`, ERA5's own `fdir` beam flux, and the diffuse remainder.
- **D — direct fraction**: `ssrd` and `fdir / ssrd`.
- **B-DISC — separation-model sensitivity**: arm B with the DISC separation model instead of Erbs.
- **B-LEARNED — the discriminator**: arm B with a fitted separation model instead of Erbs, so that
  arm C's advantage can be read as information rather than as a better-published correlation.

**The headline is arm C minus arm B, and arm B is a negative control the experiment gets for
free.** Erbs reads global irradiance and solar geometry and nothing else, all of which arm A already
holds, so arm B is mathematically incapable of carrying information arm A lacks. Whatever B−A comes
out as is therefore this pipeline's reading on a feature set known to be uninformative — the band
any real effect has to clear. Arm C carries a quantity the radiation scheme computed and `ssrd`
alone does not, and it has the same number of columns as arm B, which is why C−B is the contrast
that answers the question.

A second control runs against a synthetic target built by transposing the true split onto a tilted
plane, where the split must help by construction. A pipeline that cannot find the effect there has
not earned the right to report a null on the real meters.

Because ERA5 is a reanalysis rather than a forecast, what this measures is the *information content*
of the split, not forecast skill.

Run it with `uv run --no-project` plus `--with polars --with numpy --with xgboost
--with scipy --with pvlib --with xarray --with netcdf4 --with pandas --with deltalake`.
The long dependency list is `export_cap.py` reaching into `build_dataset.py` for the site
roster, which is what maps NGED's `time_series_id` to an anonymous label.
"""

import argparse
import concurrent.futures
import json
import logging
import sys
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import polars as pl
import xgboost as xgb
from commissioning import drop_commissioning_ramp
from export_cap import clamp_to_cap, with_export_cap
from sources import REPO_DATA_DIR, SOURCE_CHOICES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("run_experiment")

DEFAULT_SOURCE: Final[str] = "open-meteo"
"""The reanalysis route every run uses, the Copernicus archive being too slow to iterate on.

`verify_era5_sources.py` is what establishes that the mirror carries ERA5's own fields, and `cams`
is the other instrument rather than another route to this one.
"""


def dataset_path_for(*, source: str) -> Path:
    """Return the frame `build_dataset.py` wrote for one ERA5 source."""
    return REPO_DATA_DIR / "ERA5" / f"beam_diffuse_dataset_{source}.parquet"


def results_dir_for(*, source: str) -> Path:
    """Return where one run's results are written."""
    return REPO_DATA_DIR / "ERA5" / f"beam_diffuse_results_{source}"


LEARNED_BEAM_TEMPLATE: Final[str] = "learned_bhi_w_m2_fold{fold}"
"""Column holding the learned separation model's beam, for rows scored on one named fold."""

LEARNED_DIFFUSE_TEMPLATE: Final[str] = "learned_dhi_w_m2_fold{fold}"
"""Column holding the learned separation model's diffuse, for rows scored on one named fold."""

SHARED_FEATURES: Final[tuple[str, ...]] = (
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "extraterrestrial_horizontal_w_m2",
    "temp_c",
    "hour_of_day",
    "day_of_year",
)
"""Features every arm gets.

Solar geometry and season are in here deliberately. A fixed-tilt array's sensitivity to the
beam/diffuse split is partly a function of sun position, which a tree can absorb from these, so
giving every arm the geometry makes the global-irradiance-only arm as strong as it can be. That
makes any advantage arm C shows a lower bound on what a transposition model would extract.
"""

ARM_FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "A_global_only": ("ghi_w_m2",),
    "B_erbs": ("ghi_w_m2", "erbs_bhi_w_m2", "erbs_dhi_w_m2"),
    "C_era5_split": ("ghi_w_m2", "bhi_w_m2", "dhi_w_m2"),
    "D_direct_fraction": ("ghi_w_m2", "direct_fraction"),
    "B_disc": ("ghi_w_m2", "disc_bhi_w_m2", "disc_dhi_w_m2"),
    "B_learned": ("ghi_w_m2", LEARNED_BEAM_TEMPLATE, LEARNED_DIFFUSE_TEMPLATE),
}
"""The irradiance columns each arm is shown, on top of `SHARED_FEATURES`.

A name carrying `{fold}` is resolved against the fold being scored, which is what keeps arm
`B_learned`'s separation model out of its own test fold. Every other name is left alone, because
formatting a string with no placeholder in it returns the string.
"""

HEADLINE_CONTRAST: Final[tuple[str, str]] = ("C_era5_split", "B_erbs")
"""The one contrast named before the experiment ran, so it cannot be picked after the fact.

Every other contrast below is exploratory. The distinction matters because this script computes
dozens of nominally-95% intervals, and a handful of those will exclude zero by chance alone.
"""

CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    HEADLINE_CONTRAST,
    ("C_era5_split", "A_global_only"),
    ("B_erbs", "A_global_only"),
    ("D_direct_fraction", "A_global_only"),
    ("D_direct_fraction", "B_erbs"),
    ("B_disc", "A_global_only"),
    ("C_era5_split", "B_learned"),
    ("B_learned", "B_erbs"),
    ("B_learned", "A_global_only"),
)
"""Every (treatment, reference) pairing an interval is computed for."""

SENSITIVITY_ARMS: Final[tuple[str, ...]] = (
    "A_global_only",
    "B_erbs",
    "C_era5_split",
    "B_learned",
)
"""The arms the second hyperparameter setting is run on.

The second setting exists to check that an arm ordering is a property of the features rather than
of the settings. That check is worth having for the two contrasts a decision rests on, so it covers
the headline contrast, its reference, and the learned separation model the headline is read
against.
"""

CONTROL_ARMS: Final[tuple[str, ...]] = (
    "A_global_only",
    "B_erbs",
    "C_era5_split",
    "B_learned",
)
"""The arms run against the synthetic target that the split is guaranteed to help predict.

The learned separation model belongs here too: on a target built from the true split, an instrument
that could not separate a derived split from the real one would have no business reporting that it
can on the meters.
"""


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

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
"""Each (arm, site, fold) is fitted once per seed.

Seed-to-seed variation is the noise floor any arm-to-arm difference has to clear, so it is measured
rather than assumed, and the bootstrap draws a seed as well as a set of months so that the noise
reaches the interval instead of being averaged out of it.
"""

N_FOLDS: Final[int] = 5
"""Contiguous time blocks, each used once as the test fold.

Blocks are contiguous rather than random because neighbouring hours share a weather system, so a
random split would put near-copies of a test row in the training set. They are cut inside each
site's own span rather than across the whole roster's, because the roster's span is seven years and
the newest site has two and a half: global blocks would leave that site with three empty test folds
and train the remaining two on its own future.
"""

QUANTILE_LEVELS: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
"""Quantiles for the continuous ranked probability score.

The score is the integral of twice the pinball loss over every level in (0, 1), approximated here by
a Riemann sum: the level spacing times twice the summed loss. The same levels are used for every
arm, so the approximation cannot favour one.
"""

QUANTILE_LEVEL_SPACING: Final[float] = 0.1

N_BOOTSTRAP_RESAMPLES: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 20260920

MAX_CONCURRENT_FITS: Final[int] = 8
"""How many (arm, site) fits to run at once, each on `THREADS_PER_FIT` cores."""

THREADS_PER_FIT: Final[int] = 4


def _add_time_features(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Add the calendar features and the month label the block bootstrap resamples on."""
    return dataset.with_columns(
        hour_of_day=pl.col("time").dt.hour(),
        day_of_year=pl.col("time").dt.ordinal_day(),
        month=pl.col("time").dt.strftime("%Y-%m"),
    )


def _assign_folds(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Cut each site's own span into `N_FOLDS` contiguous blocks of whole months.

    Args:
        dataset: Rows carrying `site` and the `month` label.

    Returns:
        `dataset` with an integer `fold` column.
    """
    month_rank = pl.col("month").rank(method="dense").over("site")
    month_count = pl.col("month").n_unique().over("site")
    fold = ((month_rank - 1) * N_FOLDS // month_count).clip(upper_bound=N_FOLDS - 1)
    return dataset.with_columns(fold=fold.cast(pl.Int32))


def _crps(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
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


def _booster_parameters(*, hyper_parameters: HyperParameters, seed: int) -> dict[str, object]:
    """Translate the settings above into XGBoost's own parameter names."""
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


def _features_for(*, arm: str, fold: int) -> list[str]:
    """Return the feature columns one arm is shown when the named fold is the one being scored.

    Args:
        arm: The key into `ARM_FEATURES`.
        fold: The fold about to be held out.

    Returns:
        The shared features followed by the arm's own irradiance columns.
    """
    return [*SHARED_FEATURES, *(column.format(fold=fold) for column in ARM_FEATURES[arm])]


def _fit_one_fold(
    *,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: list[str],
    target: str,
    hyper_parameters: HyperParameters,
    seed: int,
    with_quantiles: bool,
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

    Returns:
        The point predictions, and the quantile predictions or `None`.
    """
    train_matrix = xgb.DMatrix(train.select(features).to_numpy(), label=train[target].to_numpy())
    test_matrix = xgb.DMatrix(test.select(features).to_numpy())
    shared = _booster_parameters(hyper_parameters=hyper_parameters, seed=seed)
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


def _run_site_arm(
    *,
    site_rows: pl.DataFrame,
    arm: str,
    setting_name: str,
    target: str,
    hyper_parameters: HyperParameters,
    with_quantiles: bool,
) -> pl.DataFrame:
    """Produce out-of-fold losses for one arm at one site, one row per (test row, seed).

    Args:
        site_rows: Every row for one site, already carrying `fold` and `month`.
        arm: The key into `ARM_FEATURES`.
        setting_name: A label for the hyperparameter setting, carried into the results.
        target: The column to predict.
        hyper_parameters: The setting to fit at.
        with_quantiles: Whether to score the continuous ranked probability score too.

    Returns:
        One row per (time, seed) with the losses.
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
        features = _features_for(arm=arm, fold=fold)
        actual = test[target].to_numpy()
        for seed in SEEDS:
            point, quantiles = _fit_one_fold(
                train=train,
                test=test,
                features=features,
                target=target,
                hyper_parameters=hyper_parameters,
                seed=seed,
                with_quantiles=with_quantiles,
            )
            # Every loss column is pinned to Float64. The real target is Float32 and the synthetic
            # control target is Float64, so without the cast the two runs produce frames that
            # cannot be stacked.
            crps = (
                pl.Series(_crps(actual=actual, quantiles=quantiles), dtype=pl.Float64)
                if quantiles is not None
                else pl.lit(None, dtype=pl.Float64)
            )
            capped_point = clamp_to_cap(prediction=point, cap_mw=test["cap_mw"])
            capped_crps = (
                pl.Series(
                    _crps(
                        actual=actual,
                        quantiles=clamp_to_cap(prediction=quantiles, cap_mw=test["cap_mw"]),
                    ),
                    dtype=pl.Float64,
                )
                if quantiles is not None
                else pl.lit(None, dtype=pl.Float64)
            )
            outputs.append(
                test.select("site", "time", "month", "fold", "effective_capacity_mw", "constrained")
                .cast({"effective_capacity_mw": pl.Float64})
                .with_columns(
                    arm=pl.lit(arm),
                    setting=pl.lit(setting_name),
                    target=pl.lit(target),
                    seed=pl.lit(seed, dtype=pl.Int32),
                    absolute_error_mw=pl.Series(np.abs(actual - point), dtype=pl.Float64),
                    signed_error_mw=pl.Series(point - actual, dtype=pl.Float64),
                    crps_mw=crps,
                    absolute_error_capped_mw=pl.Series(
                        np.abs(actual - capped_point), dtype=pl.Float64
                    ),
                    signed_error_capped_mw=pl.Series(capped_point - actual, dtype=pl.Float64),
                    crps_capped_mw=capped_crps,
                )
            )
    return pl.concat(outputs)


def _run_all(
    *, dataset: pl.DataFrame, jobs: list[tuple[str, str, str, HyperParameters, bool]]
) -> pl.DataFrame:
    """Run every (arm, site) job concurrently and concatenate the losses.

    XGBoost releases the interpreter lock while it trains, so threads give real parallelism here
    without the cost of shipping a copy of the frame to a subprocess.

    Args:
        dataset: The full frame, already carrying `fold` and `month`.
        jobs: One tuple per (arm, setting name, target, settings, whether to score quantiles).

    Returns:
        Every job's losses, stacked.
    """
    sites = sorted(dataset["site"].unique().to_list())
    outputs: list[pl.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FITS) as pool:
        futures = {}
        for arm, setting_name, target, hyper_parameters, with_quantiles in jobs:
            for site in sites:
                future = pool.submit(
                    _run_site_arm,
                    site_rows=dataset.filter(pl.col("site") == site),
                    arm=arm,
                    setting_name=setting_name,
                    target=target,
                    hyper_parameters=hyper_parameters,
                    with_quantiles=with_quantiles,
                )
                futures[future] = (setting_name, arm, site)
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            setting_name, arm, site = futures[future]
            outputs.append(future.result())
            _LOG.info("%d/%d done: %s / %s / site %s", done, len(futures), setting_name, arm, site)
    return pl.concat(outputs)


def _paired_differences(
    *, losses: pl.DataFrame, treatment: str, reference: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-seed, per-row difference in one metric, and each row's month.

    Args:
        losses: Per-row losses holding both arms, restricted to the scope wanted.
        treatment: The arm whose metric is being compared.
        reference: The arm it is compared against.
        metric: The loss column to difference.

    Returns:
        An array of shape (n_seeds, n_rows) of treatment-minus-reference differences, and the month
        label of each row.
    """
    paired = (
        losses.filter(pl.col("arm") == reference)
        .select("site", "time", "seed", "month", reference=pl.col(metric))
        .join(
            losses.filter(pl.col("arm") == treatment).select(
                "site", "time", "seed", treatment=pl.col(metric)
            ),
            on=["site", "time", "seed"],
            how="inner",
        )
        .sort("seed", "site", "time")
    )
    by_seed = [
        paired.filter(pl.col("seed") == seed).select(
            "month", difference=pl.col("treatment") - pl.col("reference")
        )
        for seed in SEEDS
    ]
    differences = np.stack([frame["difference"].to_numpy() for frame in by_seed])
    return differences, by_seed[0]["month"].to_numpy()


def _bootstrap_difference(
    *, losses: pl.DataFrame, treatment: str, reference: str, metric: str
) -> dict[str, float]:
    """Bootstrap the paired arm-to-arm difference, resampling whole months and a seed.

    Six meters inside a box roughly 25 km by 23 km share their weather, and in fact resolve to only
    two ERA5 grid cells, so the effective sample size is the number of independent weather episodes
    rather than the number of site-hours. Resampling whole calendar months keeps each episode's rows
    together, and resampling the *same* months for both arms keeps the comparison paired.

    Each resample also draws one of the seeds, for both arms alike. Without that, the interval would
    treat the seed-averaged loss as a fixed quantity and exclude a source of variation measured at
    the same order as the effect itself.

    Args:
        losses: Per-row losses for both arms, already restricted to the scope wanted.
        treatment: The arm whose metric is being compared.
        reference: The arm it is compared against.
        metric: The loss column to difference.

    Returns:
        The point estimate, the 2.5th and 97.5th percentiles, and what the estimate rests on.
    """
    differences, months = _paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric=metric
    )
    unique_months, month_index = np.unique(months, return_inverse=True)
    rows_by_month = [np.flatnonzero(month_index == index) for index in range(len(unique_months))]

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = np.empty(N_BOOTSTRAP_RESAMPLES)
    for resample in range(N_BOOTSTRAP_RESAMPLES):
        seed_index = generator.integers(0, differences.shape[0])
        drawn = generator.integers(0, len(unique_months), size=len(unique_months))
        rows = np.concatenate([rows_by_month[index] for index in drawn])
        resampled[resample] = differences[seed_index, rows].mean()

    return {
        "difference": float(differences.mean()),
        "lower_95": float(np.percentile(resampled, 2.5)),
        "upper_95": float(np.percentile(resampled, 97.5)),
        "seed_spread": float(differences.mean(axis=1).std()),
        "n_rows": differences.shape[1],
        "n_months": len(unique_months),
    }


def _per_fold_differences(
    *, losses: pl.DataFrame, treatment: str, reference: str, metric: str
) -> list[float]:
    """Return the arm-to-arm difference within each fold separately.

    Five folds agreeing in sign is the cheapest robustness statistic available here, and it is one
    the block bootstrap cannot give: the bootstrap treats months as the unit of independence, while
    each site rests on only five trained models per arm.

    Args:
        losses: Per-row losses for both arms.
        treatment: The arm whose metric is being compared.
        reference: The arm it is compared against.
        metric: The loss column to difference.

    Returns:
        One difference per fold, in fold order.
    """
    differences: list[float] = []
    for fold in range(N_FOLDS):
        in_fold = losses.filter(pl.col("fold") == fold)
        if in_fold.is_empty():
            continue
        paired, _ = _paired_differences(
            losses=in_fold, treatment=treatment, reference=reference, metric=metric
        )
        differences.append(float(paired.mean()))
    return differences


def _direct_fraction_predictability(*, dataset: pl.DataFrame) -> dict[str, float]:
    """Measure how much of ERA5's direct fraction the global-only arm could already have known.

    A separation model reads global irradiance and the sun's position and returns a diffuse
    fraction. If ERA5's own direct fraction were a deterministic function of what the global-only
    arm is shown, arm C could hold no information arm A lacks, and a null result would say nothing
    about the split. This diagnostic is what rules that out.

    Its predictors are exactly the global-only arm's feature set. Scoring it in sample would shrink
    the residual by fitting noise, biasing the number towards "there is nothing to find"; using only
    the clearness index and the zenith would bias it the other way, by ignoring what a tree can
    recover from azimuth, hour and season.

    **The withholding is by calendar month rather than by fold label**, which is the same rule
    `_add_learned_split` follows and for the same reason. Folds are cut inside each site's own span,
    so one fold number names a different calendar period at each site, and a model that dropped only
    the rows carrying the scored fold's label would still train on other sites' rows at the scored
    fold's own hours. On a reanalysis those other sites are often the same grid cell, so that path
    leaks the answer exactly. Scoring one site's fold at a time and excluding the months it covers
    closes the path, at the cost of one fit per site and fold rather than one per fold.

    Args:
        dataset: The full frame, already carrying `fold` and `month`.

    Returns:
        The out-of-fold unexplained variance fraction, and the spreads behind it.
    """
    features = [*SHARED_FEATURES, *ARM_FEATURES["A_global_only"]]
    residuals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for site in sorted(dataset["site"].unique().to_list()):
        for fold in range(N_FOLDS):
            test = dataset.filter((pl.col("site") == site) & (pl.col("fold") == fold))
            if test.is_empty():
                continue
            train = dataset.filter(~pl.col("month").is_in(test["month"].unique().to_list()))
            if train.is_empty():
                continue
            model = xgb.train(
                {
                    **_booster_parameters(hyper_parameters=PRIMARY_HYPER_PARAMETERS, seed=0),
                    "objective": "reg:squarederror",
                },
                xgb.DMatrix(
                    train.select(features).to_numpy(), label=train["direct_fraction"].to_numpy()
                ),
                num_boost_round=PRIMARY_HYPER_PARAMETERS["num_boost_round"],
            )
            actual = test["direct_fraction"].to_numpy()
            residuals.append(actual - model.predict(xgb.DMatrix(test.select(features).to_numpy())))
            targets.append(actual)

    residual = np.concatenate(residuals)
    target = np.concatenate(targets)
    return {
        "unexplained_variance_fraction": float(residual.var() / target.var()),
        "residual_standard_deviation": float(residual.std()),
        "direct_fraction_standard_deviation": float(target.std()),
        "n_rows": len(target),
    }


def _add_learned_split(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Add the best split a model can *derive* from the global-only arm's own features.

    Arm C could beat arm B for either of two reasons, and they carry opposite decisions. The
    published beam may hold information no function of global irradiance and solar geometry can
    recover, in which case the field is worth asking a supplier for. Or the product may simply
    publish a better separation model than Erbs, in which case the same gain is available locally
    for nothing. Erbs alone cannot tell those apart, because Erbs is one fixed correlation rather
    than the best available one.

    This arm is the discriminator. Its beam is a prediction of the product's own direct fraction
    from exactly arm A's feature set, so every value it carries is a function of what arm A already
    holds. It is a far more faithful separation model than Erbs, which makes it the demanding
    reference Erbs cannot be. If arm C still beats it, the advantage is information rather than
    representation.

    **Every value is withheld from the model that produced it, and the withholding is by timestamp
    rather than by fold label.** Folds are cut inside each site's own span, so one fold number is a
    different calendar period at each site, and a model that merely dropped the rows labelled with
    that fold number would still train on other sites' rows at the scored fold's own hours — on the
    reanalysis those other sites are the same grid cell, so the leak would be exact. Excluding the
    calendar months themselves closes that path.

    **Training rows are held out too, not only the scored fold.** A column that is sharper on the
    rows the arm trains on than on the rows it is scored on gets over-trusted by the power model,
    which penalises this arm for a reason that has nothing to do with the split. So for each scored
    fold the training rows are filled in by an inner cross-validation that also withholds their own
    months. Excluding two folds' months is symmetric in the two, so the fits are cached and the
    count comes to ninety rather than a hundred and fifty.

    Args:
        dataset: The full frame, already carrying `fold` and `month`.

    Returns:
        The frame with a beam and a diffuse column for each fold that can be scored.
    """
    features = [*SHARED_FEATURES, *ARM_FEATURES["A_global_only"]]
    predictors = xgb.DMatrix(dataset.select(features).to_numpy())
    global_irradiance = dataset["ghi_w_m2"].to_numpy()
    fitted: dict[tuple[str, frozenset[int]], np.ndarray] = {}

    def _prediction_without(*, site: str, withheld: frozenset[int]) -> np.ndarray:
        """Predict every row from a model trained without the months in one site's named folds."""
        key = (site, withheld)
        if key not in fitted:
            excluded = (
                dataset.filter((pl.col("site") == site) & pl.col("fold").is_in(list(withheld)))[
                    "month"
                ]
                .unique()
                .to_list()
            )
            train = dataset.filter(~pl.col("month").is_in(excluded))
            model = xgb.train(
                {
                    **_booster_parameters(hyper_parameters=PRIMARY_HYPER_PARAMETERS, seed=0),
                    "objective": "reg:squarederror",
                },
                xgb.DMatrix(
                    train.select(features).to_numpy(), label=train["direct_fraction"].to_numpy()
                ),
                num_boost_round=PRIMARY_HYPER_PARAMETERS["num_boost_round"],
            )
            fitted[key] = model.predict(predictors)
        return fitted[key]

    sites = sorted(dataset["site"].unique().to_list())
    site_labels = dataset["site"].to_numpy()
    fold_labels = dataset["fold"].to_numpy()
    columns: dict[str, pl.Series] = {}
    for scored in range(N_FOLDS):
        if dataset.filter(pl.col("fold") == scored).is_empty():
            continue
        fraction = np.full(dataset.height, np.nan)
        for site in sites:
            for fold in range(N_FOLDS):
                rows = (site_labels == site) & (fold_labels == fold)
                if not rows.any():
                    continue
                withheld = frozenset({scored} if fold == scored else {scored, fold})
                fraction[rows] = _prediction_without(site=site, withheld=withheld)[rows]
        # Clipping to a fraction keeps the pair a genuine split of this arm's own global
        # irradiance, so no arm differs from another in what its two components sum to.
        beam = np.clip(fraction, 0.0, 1.0) * global_irradiance
        columns[LEARNED_BEAM_TEMPLATE.format(fold=scored)] = pl.Series(beam, dtype=pl.Float64)
        columns[LEARNED_DIFFUSE_TEMPLATE.format(fold=scored)] = pl.Series(
            global_irradiance - beam, dtype=pl.Float64
        )
        _LOG.info(
            "learned separation built for scored fold %d (%d fits so far)", scored, len(fitted)
        )
    return dataset.with_columns(**columns)


def _intervals_for(
    *, losses: pl.DataFrame, setting_name: str, target: str, sites: list[str]
) -> list[dict[str, object]]:
    """Compute every contrast's interval, pooled and per site.

    Args:
        losses: Per-row losses for one setting and one target.
        setting_name: The setting these losses came from.
        target: The column the models predicted.
        sites: Every site label present.

    Returns:
        One record per (contrast, metric, scope).
    """
    metrics = [
        "absolute_error_capped_fraction_of_capacity",
        "absolute_error_fraction_of_capacity",
        "absolute_error_mw",
    ]
    if losses["crps_mw"].null_count() < losses.height:
        metrics.append("crps_mw")

    records: list[dict[str, object]] = []
    arms_present = set(losses["arm"].unique().to_list())
    for treatment, reference in CONTRASTS:
        if not {treatment, reference} <= arms_present:
            continue
        for metric in metrics:
            for scope in ("all_sites", *sites):
                scoped = losses if scope == "all_sites" else losses.filter(pl.col("site") == scope)
                is_headline = (
                    (treatment, reference) == HEADLINE_CONTRAST
                    and scope == "all_sites"
                    and metric == metrics[0]
                    and setting_name == "primary"
                )
                records.append(
                    {
                        "setting": setting_name,
                        "target": target,
                        "treatment": treatment,
                        "reference": reference,
                        "is_headline": is_headline,
                        "metric": metric,
                        "scope": scope,
                        "per_fold_differences": (
                            _per_fold_differences(
                                losses=losses,
                                treatment=treatment,
                                reference=reference,
                                metric=metric,
                            )
                            if scope == "all_sites"
                            else []
                        ),
                        **_bootstrap_difference(
                            losses=scoped, treatment=treatment, reference=reference, metric=metric
                        ),
                    }
                )
        _LOG.info("%s: %s vs %s bootstrapped", setting_name, treatment, reference)
    return records


def main() -> int:
    """Run every arm, the controls and the bootstrap, and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--suffix",
        default="",
        help="Selects a variant build of the same source, and keeps its results beside the main.",
    )
    arguments = parser.parse_args()
    source = f"{arguments.source}{arguments.suffix}"
    results_dir = results_dir_for(source=source)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset = with_export_cap(
        dataset=_assign_folds(
            dataset=_add_time_features(
                dataset=drop_commissioning_ramp(
                    dataset=pl.read_parquet(dataset_path_for(source=source))
                )
            )
        )
    )
    sites = sorted(dataset["site"].unique().to_list())
    _LOG.info(
        "dataset: %d rows, %d sites, %d months",
        dataset.height,
        len(sites),
        dataset["month"].n_unique(),
    )

    diagnostic = _direct_fraction_predictability(dataset=dataset)
    _LOG.info("direct-fraction predictability diagnostic: %s", diagnostic)

    dataset = _add_learned_split(dataset=dataset)

    jobs: list[tuple[str, str, str, HyperParameters, bool]] = [
        (arm, "primary", "power_mw", PRIMARY_HYPER_PARAMETERS, True) for arm in ARM_FEATURES
    ]
    jobs += [
        (arm, "sensitivity", "power_mw", SENSITIVITY_HYPER_PARAMETERS, False)
        for arm in SENSITIVITY_ARMS
    ]
    jobs += [
        (arm, "positive_control", "synthetic_power_mw", PRIMARY_HYPER_PARAMETERS, False)
        for arm in CONTROL_ARMS
    ]

    losses = _run_all(dataset=dataset, jobs=jobs).with_columns(
        absolute_error_fraction_of_capacity=pl.col("absolute_error_mw")
        / pl.col("effective_capacity_mw"),
        absolute_error_capped_fraction_of_capacity=pl.col("absolute_error_capped_mw")
        / pl.col("effective_capacity_mw"),
    )
    losses.write_parquet(results_dir / "per_row_losses.parquet")

    records: list[dict[str, object]] = []
    for setting_name, setting_target in (
        ("primary", "power_mw"),
        ("sensitivity", "power_mw"),
        ("positive_control", "synthetic_power_mw"),
    ):
        records += _intervals_for(
            losses=losses.filter(pl.col("setting") == setting_name),
            setting_name=setting_name,
            target=setting_target,
            sites=sites,
        )
    pl.DataFrame(records).write_parquet(results_dir / "bootstrap_intervals.parquet")

    summary = (
        losses.group_by("setting", "arm", "site")
        .agg(
            mae_mw=pl.col("absolute_error_mw").mean(),
            mae_fraction_of_capacity=pl.col("absolute_error_fraction_of_capacity").mean(),
            crps_mw=pl.col("crps_mw").mean(),
            bias_mw=pl.col("signed_error_mw").mean(),
            mae_capped_mw=pl.col("absolute_error_capped_mw").mean(),
            mae_capped_fraction_of_capacity=pl.col(
                "absolute_error_capped_fraction_of_capacity"
            ).mean(),
            crps_capped_mw=pl.col("crps_capped_mw").mean(),
            constrained_rows=pl.col("constrained").sum() // len(SEEDS),
            n_rows=pl.len() // len(SEEDS),
        )
        .sort("setting", "arm", "site")
    )
    summary.write_parquet(results_dir / "per_site_summary.parquet")
    (results_dir / "diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
    _LOG.info("results written to %s", results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
