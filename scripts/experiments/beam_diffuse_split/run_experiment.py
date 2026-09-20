"""Train one XGBoost PV forecast per arm and measure what the beam/diffuse split is worth.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>. It reads the frame
`build_dataset.py` wrote and writes two parquet files of results plus a JSON summary.

Every arm sees identical rows, identical folds, identical seeds, identical hyperparameters and
identical non-irradiance features. The only thing that changes between arms is which irradiance
columns the model is shown:

- **A — global only**: ERA5 `ssrd` as global horizontal irradiance.
- **B — separation model**: `ssrd` plus the direct-normal and diffuse-horizontal estimates the Erbs
  separation model derives from `ssrd` alone.
- **C — the model's own split**: `ssrd`, ERA5's own `fdir` beam flux, and the diffuse remainder.
- **D — direct fraction**: `ssrd` plus `fdir / ssrd`.
- **B-DISC — separation-model sensitivity**: arm B with the DISC separation model instead of Erbs.

Arm C against arm B is the comparison the experiment exists for: B re-expresses information arm A
already has, whereas C adds information the radiation scheme computed and `ssrd` alone does not
carry.

Because ERA5 is a reanalysis rather than a forecast, what this measures is the *information
content* of the split, not forecast skill.

Run it with `uv run --no-project` plus `--with polars --with xgboost --with numpy`.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import polars as pl
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("run_experiment")

REPO_DATA_DIR: Final[Path] = Path("/home/jack/dev/nged-substation-forecast/data")
DATASET_PATH: Final[Path] = REPO_DATA_DIR / "ERA5" / "beam_diffuse_dataset.parquet"
RESULTS_DIR: Final[Path] = REPO_DATA_DIR / "ERA5" / "beam_diffuse_results"

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
    "B_erbs": ("ghi_w_m2", "erbs_dni_w_m2", "erbs_dhi_w_m2"),
    "C_era5_split": ("ghi_w_m2", "bhi_w_m2", "dhi_w_m2"),
    "D_direct_fraction": ("ghi_w_m2", "direct_fraction"),
    "B_disc": ("ghi_w_m2", "disc_dni_w_m2", "disc_dhi_w_m2"),
}
"""The irradiance columns each arm is shown, on top of `SHARED_FEATURES`."""

REFERENCE_ARM: Final[str] = "A_global_only"
"""Every reported difference is an arm's metric minus this arm's."""


class HyperParameters(TypedDict):
    """The XGBoost settings held fixed across every arm, site, fold and seed."""

    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    reg_lambda: float
    num_boost_round: int


PRIMARY_HYPER_PARAMETERS: Final[HyperParameters] = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20.0,
    "reg_lambda": 1.0,
    "num_boost_round": 500,
}
"""Fixed, not tuned.

Tuning per arm would let the tuner's own noise decide which arm wins, and early stopping would give
each arm a different number of rounds for reasons unrelated to the irradiance features. Both are
channels through which an arm could win without carrying more information, so neither is used. The
settings are ordinary defaults for a few tens of thousands of rows.
"""

SENSITIVITY_HYPER_PARAMETERS: Final[HyperParameters] = {
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50.0,
    "reg_lambda": 5.0,
    "num_boost_round": 1200,
}
"""A shallower, more heavily regularised alternative.

A single hyperparameter setting can favour an arm by accident — a wider feature set changes how
much a fixed number of rounds overfits. Running a second, deliberately different setting says
whether the arm ordering is a property of the features or of the settings.
"""

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
"""Three seeds per fit.

Seed-to-seed variation is the noise floor any arm-to-arm difference has to clear, so it is measured
rather than assumed. Per-row absolute errors are averaged over seeds before any metric is taken.
"""

N_FOLDS: Final[int] = 5
"""Contiguous time blocks, each used once as the test fold.

Blocks are contiguous rather than random because neighbouring hours share a weather system, so a
random split would put near-copies of a test row in the training set.
"""

QUANTILE_LEVELS: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
"""Quantiles for the continuous ranked probability score.

The score is approximated as twice the mean pinball loss over these levels, which is the Riemann
approximation of its integral form. The same levels are used for every arm, so the approximation
cannot favour one.
"""

N_BOOTSTRAP_RESAMPLES: Final[int] = 5000
BOOTSTRAP_SEED: Final[int] = 20260920


def _add_time_features(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Add the calendar features and the month label the block bootstrap resamples on."""
    return dataset.with_columns(
        hour_of_day=pl.col("time").dt.hour(),
        day_of_year=pl.col("time").dt.ordinal_day(),
        month=pl.col("time").dt.strftime("%Y-%m"),
    )


def _assign_folds(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Cut the whole span into `N_FOLDS` contiguous blocks of whole months.

    Blocks are whole months and are shared across sites, so a fold boundary falls at the same
    instant for every site and every arm.

    Args:
        dataset: Rows carrying the `month` label.

    Returns:
        `dataset` with an integer `fold` column.
    """
    months = sorted(dataset["month"].unique().to_list())
    block_of_month = {
        month: min(index * N_FOLDS // len(months), N_FOLDS - 1)
        for index, month in enumerate(months)
    }
    return dataset.with_columns(
        fold=pl.col("month").replace_strict(block_of_month, return_dtype=pl.Int32)
    )


def _pinball_losses(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return the per-row mean pinball loss across `QUANTILE_LEVELS`.

    Args:
        actual: Observed power, shape (n_rows,).
        quantiles: Predicted quantiles, shape (n_rows, n_levels), in the order of
            `QUANTILE_LEVELS`.

    Returns:
        The per-row mean pinball loss, shape (n_rows,).
    """
    # XGBoost's multi-quantile head can return crossing quantiles; sorting each row repairs that
    # without changing any individual level's calibration much, and is applied to every arm alike.
    sorted_quantiles = np.sort(quantiles, axis=1)
    levels = np.asarray(QUANTILE_LEVELS)[None, :]
    difference = actual[:, None] - sorted_quantiles
    losses = np.where(difference >= 0, difference * levels, -difference * (1.0 - levels))
    return losses.mean(axis=1)


def _fit_and_predict(
    *,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: list[str],
    hyper_parameters: HyperParameters,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one point model and one quantile model, and predict the test fold with both.

    Args:
        train: The training rows.
        test: The test rows.
        features: The feature columns to show the model.
        hyper_parameters: Settings shared by every arm.
        seed: The XGBoost random seed.

    Returns:
        The point predictions, shape (n_test,), and the quantile predictions, shape
        (n_test, n_levels).
    """
    train_matrix = xgb.DMatrix(
        train.select(features).to_numpy(), label=train["power_mw"].to_numpy()
    )
    test_matrix = xgb.DMatrix(test.select(features).to_numpy())
    shared = {
        "max_depth": hyper_parameters["max_depth"],
        "eta": hyper_parameters["learning_rate"],
        "subsample": hyper_parameters["subsample"],
        "colsample_bytree": hyper_parameters["colsample_bytree"],
        "min_child_weight": hyper_parameters["min_child_weight"],
        "lambda": hyper_parameters["reg_lambda"],
        "tree_method": "hist",
        "seed": seed,
        "nthread": 4,
    }
    rounds = hyper_parameters["num_boost_round"]

    point_model = xgb.train(
        {**shared, "objective": "reg:absoluteerror"}, train_matrix, num_boost_round=rounds
    )
    quantile_model = xgb.train(
        {
            **shared,
            "objective": "reg:quantileerror",
            "quantile_alpha": np.asarray(QUANTILE_LEVELS),
        },
        train_matrix,
        num_boost_round=rounds,
    )
    quantile_predictions = np.atleast_2d(quantile_model.predict(test_matrix))
    return point_model.predict(test_matrix), quantile_predictions


def _run_arm(
    *,
    dataset: pl.DataFrame,
    arm: str,
    hyper_parameters: HyperParameters,
    setting_name: str,
) -> pl.DataFrame:
    """Produce out-of-fold losses for one arm, at one hyperparameter setting.

    Args:
        dataset: The full frame, already carrying `fold` and `month`.
        arm: The key into `ARM_FEATURES`.
        hyper_parameters: The setting to fit at.
        setting_name: A label for the setting, carried into the results.

    Returns:
        One row per (site, time) with `absolute_error_mw` and `crps_mw`, both averaged over seeds.
    """
    features = [*SHARED_FEATURES, *ARM_FEATURES[arm]]
    outputs: list[pl.DataFrame] = []
    for (site,), site_rows in dataset.group_by(["site"], maintain_order=True):
        for fold in range(N_FOLDS):
            test = site_rows.filter(pl.col("fold") == fold)
            train = site_rows.filter(pl.col("fold") != fold)
            if test.is_empty() or train.is_empty():
                continue
            actual = test["power_mw"].to_numpy()
            absolute_errors = np.zeros((len(SEEDS), test.height))
            crps_values = np.zeros((len(SEEDS), test.height))
            for seed_index, seed in enumerate(SEEDS):
                point, quantiles = _fit_and_predict(
                    train=train,
                    test=test,
                    features=features,
                    hyper_parameters=hyper_parameters,
                    seed=seed,
                )
                absolute_errors[seed_index] = np.abs(actual - point)
                crps_values[seed_index] = 2.0 * _pinball_losses(actual=actual, quantiles=quantiles)
            outputs.append(
                test.select("site", "time", "month", "fold", "power_mw").with_columns(
                    arm=pl.lit(arm),
                    setting=pl.lit(setting_name),
                    absolute_error_mw=pl.Series(absolute_errors.mean(axis=0)),
                    crps_mw=pl.Series(crps_values.mean(axis=0)),
                    absolute_error_seed_spread_mw=pl.Series(absolute_errors.std(axis=0)),
                )
            )
        _LOG.info("%s / %s: site %s done", setting_name, arm, site)
    return pl.concat(outputs)


def _bootstrap_difference(
    *,
    losses: pl.DataFrame,
    arm: str,
    metric: str,
) -> dict[str, float]:
    """Bootstrap the paired arm-minus-reference difference in one metric, resampling whole months.

    Six sites inside a 34 km box share their weather, so the effective sample size is the number of
    independent weather episodes rather than the number of site-hours. Resampling whole calendar
    months keeps each episode's rows together, and resampling the *same* months for both arms keeps
    the comparison paired.

    Args:
        losses: Per-row losses for the reference arm and `arm`, already restricted to the sites the
            interval is wanted for.
        arm: The arm to compare against `REFERENCE_ARM`.
        metric: Either `absolute_error_mw` or `crps_mw`.

    Returns:
        The point estimate and the 2.5th and 97.5th percentiles of the difference.
    """
    paired = (
        losses.filter(pl.col("arm") == REFERENCE_ARM)
        .select("site", "time", "month", reference=pl.col(metric))
        .join(
            losses.filter(pl.col("arm") == arm).select("site", "time", treatment=pl.col(metric)),
            on=["site", "time"],
            how="inner",
        )
    )
    difference = (paired["treatment"] - paired["reference"]).to_numpy()
    months = paired["month"].to_numpy()
    unique_months, month_index = np.unique(months, return_inverse=True)
    rows_by_month = [np.flatnonzero(month_index == index) for index in range(len(unique_months))]

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = np.empty(N_BOOTSTRAP_RESAMPLES)
    for resample in range(N_BOOTSTRAP_RESAMPLES):
        drawn = generator.integers(0, len(unique_months), size=len(unique_months))
        rows = np.concatenate([rows_by_month[index] for index in drawn])
        resampled[resample] = difference[rows].mean()

    return {
        "difference": float(difference.mean()),
        "lower_95": float(np.percentile(resampled, 2.5)),
        "upper_95": float(np.percentile(resampled, 97.5)),
        "n_rows": len(difference),
        "n_months": len(unique_months),
    }


def _direct_fraction_predictability(*, dataset: pl.DataFrame) -> dict[str, float]:
    """Measure how much of ERA5's direct fraction a separation model could already have known.

    A separation model reads the clearness index and the sun's position and returns a diffuse
    fraction. If ERA5's own direct fraction were a deterministic function of those two, arm C could
    hold no information arm A lacks, and a null result would say nothing about the split. This
    diagnostic is what rules that out, and is worth reading before any arm-to-arm number.

    Args:
        dataset: The full frame.

    Returns:
        The fraction of the direct fraction's variance left unexplained, and its residual spread.
    """
    daylight = dataset.filter(pl.col("extraterrestrial_horizontal_w_m2") > 10.0)
    clearness = (
        daylight["ghi_w_m2"].to_numpy() / daylight["extraterrestrial_horizontal_w_m2"].to_numpy()
    )
    predictors = np.column_stack([clearness, daylight["solar_zenith_deg"].to_numpy()])
    target = daylight["direct_fraction"].to_numpy()

    matrix = xgb.DMatrix(predictors, label=target)
    model = xgb.train(
        {"objective": "reg:squarederror", "max_depth": 6, "eta": 0.05, "seed": 0},
        matrix,
        num_boost_round=400,
    )
    residual = target - model.predict(matrix)
    return {
        "unexplained_variance_fraction": float(residual.var() / target.var()),
        "residual_standard_deviation": float(residual.std()),
        "direct_fraction_standard_deviation": float(target.std()),
        "n_rows": len(target),
    }


def main() -> int:
    """Run every arm at both hyperparameter settings and write the results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = _assign_folds(dataset=_add_time_features(dataset=pl.read_parquet(DATASET_PATH)))
    _LOG.info(
        "dataset: %d rows, %d sites, %d months",
        dataset.height,
        dataset["site"].n_unique(),
        dataset["month"].n_unique(),
    )

    diagnostic = _direct_fraction_predictability(dataset=dataset)
    _LOG.info("direct-fraction predictability diagnostic: %s", diagnostic)

    settings = {
        "primary": PRIMARY_HYPER_PARAMETERS,
        "sensitivity": SENSITIVITY_HYPER_PARAMETERS,
    }
    all_losses = pl.concat(
        [
            _run_arm(
                dataset=dataset,
                arm=arm,
                hyper_parameters=hyper_parameters,
                setting_name=setting_name,
            )
            for setting_name, hyper_parameters in settings.items()
            for arm in ARM_FEATURES
        ]
    )
    all_losses.write_parquet(RESULTS_DIR / "per_row_losses.parquet")

    intervals: list[dict[str, object]] = []
    for setting_name in settings:
        setting_losses = all_losses.filter(pl.col("setting") == setting_name)
        for arm in ARM_FEATURES:
            if arm == REFERENCE_ARM:
                continue
            for metric in ("absolute_error_mw", "crps_mw"):
                for scope in ("all_sites", *sorted(dataset["site"].unique().to_list())):
                    scoped = (
                        setting_losses
                        if scope == "all_sites"
                        else setting_losses.filter(pl.col("site") == scope)
                    )
                    interval = _bootstrap_difference(losses=scoped, arm=arm, metric=metric)
                    intervals.append(
                        {
                            "setting": setting_name,
                            "arm": arm,
                            "metric": metric,
                            "scope": scope,
                            **interval,
                        }
                    )
            _LOG.info("%s / %s: bootstrap done", setting_name, arm)

    pl.DataFrame(intervals).write_parquet(RESULTS_DIR / "bootstrap_intervals.parquet")

    summary = (
        all_losses.group_by("setting", "arm", "site")
        .agg(
            mae_mw=pl.col("absolute_error_mw").mean(),
            crps_mw=pl.col("crps_mw").mean(),
            seed_spread_mw=pl.col("absolute_error_seed_spread_mw").mean(),
            n_rows=pl.len(),
        )
        .sort("setting", "arm", "site")
    )
    summary.write_parquet(RESULTS_DIR / "per_site_summary.parquet")
    (RESULTS_DIR / "diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
    _LOG.info("results written to %s", RESULTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
