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

Run it with `uv run python studies/beam_diffuse_split/run_experiment.py`.
"""

import argparse
import concurrent.futures
import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from commissioning import drop_commissioning_ramp
from export_cap import with_export_cap
from sources import SOURCE_CHOICES, STUDY_DATA_DIR
from studies.bootstrap import bootstrap_difference, per_fold_differences
from studies.cross_validation import (
    N_FOLDS,
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
    HyperParameters,
    assign_folds,
    booster_parameters,
    out_of_fold_losses,
)
from studies.fractions_skill_score import MONTH_FORMAT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("run_experiment")

DEFAULT_SOURCE: Final[str] = "open-meteo"
"""The reanalysis route every run uses, the Copernicus archive being too slow to iterate on.

`verify_era5_sources.py` is what establishes that the mirror carries ERA5's own fields, and `cams`
is the other instrument rather than another route to this one.
"""


def dataset_path_for(*, source: str) -> Path:
    """Return the frame `build_dataset.py` wrote for one ERA5 source."""
    return STUDY_DATA_DIR / f"beam_diffuse_dataset_{source}.parquet"


def results_dir_for(*, source: str) -> Path:
    """Return where one run's results are written."""
    return STUDY_DATA_DIR / f"beam_diffuse_results_{source}"


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


MAX_CONCURRENT_FITS: Final[int] = 8
"""How many (arm, site) fits to run at once, each on `THREADS_PER_FIT` cores."""


def _add_time_features(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Add the calendar features and the month label the block bootstrap resamples on."""
    return dataset.with_columns(
        hour_of_day=pl.col("time").dt.hour(),
        day_of_year=pl.col("time").dt.ordinal_day(),
        month=pl.col("time").dt.strftime(MONTH_FORMAT),
    )


Job = tuple[str, str, str, tuple[str, ...], HyperParameters, bool]
"""One (arm, setting name, target, features, settings, whether to score quantiles) to fit.

The features are carried by the job rather than looked up by arm name, so a script with arms of its
own builds its jobs from its own table instead of patching this module's.
"""


def features_for(*, arm: str) -> tuple[str, ...]:
    """Return the feature columns one of this experiment's arms is shown.

    Args:
        arm: The key into `ARM_FEATURES`.

    Returns:
        The shared features followed by the arm's own irradiance columns, some carrying `{fold}`.
    """
    return (*SHARED_FEATURES, *ARM_FEATURES[arm])


def _run_all(*, dataset: pl.DataFrame, jobs: list[Job]) -> pl.DataFrame:
    """Run every (arm, site) job concurrently and concatenate the losses.

    XGBoost releases the interpreter lock while it trains, so threads give real parallelism here
    without the cost of shipping a copy of the frame to a subprocess.

    Args:
        dataset: The full frame, already carrying `fold`, `month`, `cap_mw` and `constrained`.
        jobs: The fits to run.

    Returns:
        Every job's losses, stacked, labelled with the arm, the setting and the target.
    """
    sites = sorted(dataset["site"].unique().to_list())
    outputs: list[pl.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FITS) as pool:
        futures = {}
        for arm, setting_name, target, features, hyper_parameters, with_quantiles in jobs:
            for site in sites:
                future = pool.submit(
                    out_of_fold_losses,
                    site_rows=dataset.filter(pl.col("site") == site),
                    features=features,
                    target=target,
                    hyper_parameters=hyper_parameters,
                    with_quantiles=with_quantiles,
                )
                futures[future] = (setting_name, arm, target, site)
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            setting_name, arm, target, site = futures[future]
            outputs.append(
                future.result().with_columns(
                    arm=pl.lit(arm), setting=pl.lit(setting_name), target=pl.lit(target)
                )
            )
            _LOG.info("%d/%d done: %s / %s / site %s", done, len(futures), setting_name, arm, site)
    return pl.concat(outputs)


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
                    **booster_parameters(hyper_parameters=PRIMARY_HYPER_PARAMETERS, seed=0),
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
                    **booster_parameters(hyper_parameters=PRIMARY_HYPER_PARAMETERS, seed=0),
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
                            per_fold_differences(
                                losses=losses,
                                treatment=treatment,
                                reference=reference,
                                metric=metric,
                            )
                            if scope == "all_sites"
                            else []
                        ),
                        **bootstrap_difference(
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
        dataset=assign_folds(
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

    jobs: list[Job] = [
        (arm, "primary", "power_mw", features_for(arm=arm), PRIMARY_HYPER_PARAMETERS, True)
        for arm in ARM_FEATURES
    ]
    jobs += [
        (
            arm,
            "sensitivity",
            "power_mw",
            features_for(arm=arm),
            SENSITIVITY_HYPER_PARAMETERS,
            False,
        )
        for arm in SENSITIVITY_ARMS
    ]
    jobs += [
        (
            arm,
            "positive_control",
            "synthetic_power_mw",
            features_for(arm=arm),
            PRIMARY_HYPER_PARAMETERS,
            False,
        )
        for arm in CONTROL_ARMS
    ]

    losses = _run_all(dataset=dataset, jobs=jobs)
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
