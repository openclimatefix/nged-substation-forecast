"""Ask the beam/diffuse question again with a fitted physical model instead of XGBoost.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**A null result from a gradient-boosted tree is ambiguous, and this script removes the ambiguity.**
The tree is shown the beam and the diffuse fluxes as two more columns and has to discover from data
that one of them should be projected onto a tilted plane and the other should not. The physical
model in `physics_model.py` is given that projection, so if the split carries information about a
site's output, a model built around the projection is where the information should surface. Up to
five free parameters per site are fitted on each training fold and scored on the held-out fold, on
exactly the rows, folds and months `run_experiment.py` uses, so the two instruments' numbers sit
beside each other. Arm `P_A` identifies only three of the five: with no split there is no
transposition to do, so its tilt and azimuth never move from where the optimiser starts.

The arms differ only in which beam and diffuse fluxes the transposition is handed. Arm `P_A` is
handed none, and for it the plane-of-array irradiance is the global horizontal irradiance itself:
with one number there is no transposition to do, which is the physical statement of what a site
loses when its weather feed carries no direct beam.

Arm `P_E_blended` exists to answer a different question — whether several splits together beat the
best single one. It is given all three beam estimates at once and fits the weights of a convex
combination, so it can reproduce any single arm and is free to do better.

**The seed means something different here from what it means in `run_experiment.py`.** There it
reseeds XGBoost, so the spread across seeds measures how much of a difference is fitting noise.
Here it only moves the optimiser's random restarts, and every seed shares the fixed starting point
that `restart_basins.py` shows reaching the lowest loss in 19 of 30 fits, so the seed-to-seed
spread this script reports is a few parts in a million and says the seeds inherit one winning start
rather than that the noise floor is that low. The bootstrap's seed draw
likewise adds nothing to this instrument's intervals.

Run it with `uv run --with netcdf4 python
studies/beam_diffuse_split/run_physics_experiment.py`, then
`python studies/beam_diffuse_split/run_physics_experiment.py --source cams
The long dependency list is `export_cap.py` reaching into `build_dataset.py`
for the site roster, which is what maps NGED's `time_series_id` to an anonymous label.
"""

import argparse
import concurrent.futures
import logging
import sys
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import polars as pl
from commissioning import drop_commissioning_ramp
from export_cap import clamp_to_cap, with_export_cap
from physics_model import MIN_COS_ZENITH, Geometry, power_mw
from run_experiment import (
    N_FOLDS,
    SEEDS,
    _add_time_features,
    _assign_folds,
    _bootstrap_difference,
    _per_fold_differences,
    dataset_path_for,
)
from scipy.optimize import minimize
from sources import SOURCE_CHOICES, STUDY_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("run_physics_experiment")


def results_dir_for(*, source: str) -> Path:
    """Return where this instrument's results for one source are written."""
    return STUDY_DATA_DIR / "ERA5" / f"beam_diffuse_physics_{source}"


ARM_SPLITS: Final[dict[str, tuple[str, ...]]] = {
    "P_A_global_only": (),
    "P_B_erbs": ("erbs",),
    "P_B_disc": ("disc",),
    "P_C_source_split": ("source",),
    "P_E_blended": ("source", "erbs", "disc"),
}
"""Which beam estimates each arm's transposition is handed.

`source` is the irradiance product's own beam field — ERA5's `fdir` or the CAMS service's beam
horizontal irradiance — and the other two are separation models run on the global irradiance the
same product published. An arm handed more than one fits the weights of a convex combination.
"""

BEAM_COLUMNS: Final[dict[str, str]] = {
    "source": "bhi_w_m2",
    "erbs": "erbs_bhi_w_m2",
    "disc": "disc_bhi_w_m2",
}
"""The dataset column holding each split's beam-on-horizontal flux."""

HEADLINE_CONTRAST: Final[tuple[str, str]] = ("P_C_source_split", "P_B_erbs")
"""The comparison this instrument exists for, declared before the run.

The same contrast `run_experiment.py` calls its headline: the product's own split against a
separation model's estimate of the same split. Everything against `P_A_global_only` measures
something else — what transposition itself buys — and would be a different claim.
"""

CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    HEADLINE_CONTRAST,
    ("P_C_source_split", "P_A_global_only"),
    ("P_B_erbs", "P_A_global_only"),
    ("P_B_disc", "P_A_global_only"),
    ("P_E_blended", "P_C_source_split"),
    ("P_E_blended", "P_B_erbs"),
)
"""Every (treatment, reference) pair reported, headline first."""

CONTROL_ARMS: Final[tuple[str, ...]] = ("P_A_global_only", "P_B_erbs", "P_C_source_split")
"""The arms run against the synthetic transposed-plane target.

**The control is only worth reading because the target is built outside this model's hypothesis
class.** `build_dataset._add_synthetic_control_target` gives each site its own tilt and azimuth,
none of them the values the optimiser starts from, and transposes the sky diffuse by the Hay-Davies
model where this one assumes an isotropic sky. So the fit has to find geometry it was not handed,
under a sky model it does not implement, which is the situation a real meter puts it in. What the
control then measures is the arm-to-arm difference this instrument produces when the split
genuinely matters — the threshold below which a difference on the real meters says nothing.
"""

N_RESTARTS: Final[int] = 8
"""How many starting points each fit is run from: one fixed vector of zeros and seven random.

Tilt and azimuth enter through a cosine, so the objective has more than one local minimum — a panel
facing east and one facing west fit a symmetric day almost equally well. `restart_basins.py`
measures how many: of 64 independent random starts, a median of 3 reach the lowest loss found. It
also measures what raising this number would buy, which is nothing that moves an arm ordering.
"""

MAX_ITERATIONS: Final[int] = 3000
"""The optimiser's iteration ceiling, reached only by a start that is going nowhere.

Powell's method is the optimiser because the objective is a mean absolute error, which has a kink
wherever a residual changes sign. On a test fit it reached the same minimum as a gradient method and
a lower one than a Nelder-Mead simplex, in a third of the simplex's time.
"""

TILT_CEILING_DEGREES: Final[float] = 60.0
"""The steepest tilt the fit may choose. Ground-mounted arrays in GB sit well inside it."""

AZIMUTH_HALF_RANGE_DEGREES: Final[float] = 90.0
"""How far either side of due south the fitted azimuth may swing."""

TEMPERATURE_COEFFICIENT_CENTRE: Final[float] = -0.003
"""The middle of the range the fitted temperature coefficient may take, per degree Celsius."""

TEMPERATURE_COEFFICIENT_HALF_RANGE: Final[float] = 0.005
"""How far either side of the centre the fitted temperature coefficient may swing.

The range is -0.008 to +0.002 per degree Celsius. A crystalline-silicon module's maximum-power
coefficient sits between -0.0045 and -0.0025, so the range contains every module this fleet could
be built from and is wide enough that a binding bound means the model is absorbing something other
than temperature. It reaches slightly past zero for the same reason: a fit that settles just below
zero and a fit pinned against zero are different findings, and a range ending at zero cannot tell
them apart.
"""

START_SPREAD: Final[float] = 1.0
"""The standard deviation of the random starting points, in the unbounded parameter space."""


def _geometry_for(*, rows: pl.DataFrame, arm: str, weights: np.ndarray | None) -> Geometry:
    """Build the model's inputs for one arm, blending the splits it was handed.

    Args:
        rows: The rows to build inputs for.
        arm: The key into `ARM_SPLITS`.
        weights: The convex weights over the arm's splits, or `None` for an arm with no split.

    Returns:
        The sun's position and the irradiance fields this arm sees.
    """
    zenith = np.radians(rows["solar_zenith_deg"].to_numpy())
    global_horizontal = rows["ghi_w_m2"].to_numpy()
    splits = ARM_SPLITS[arm]
    if not splits:
        beam = None
        diffuse = None
    else:
        beams = np.stack([rows[BEAM_COLUMNS[split]].to_numpy() for split in splits])
        beam = np.clip((beams * np.asarray(weights)[:, None]).sum(axis=0), 0.0, global_horizontal)
        diffuse = global_horizontal - beam
    return Geometry(
        cos_zenith=np.maximum(np.cos(zenith), MIN_COS_ZENITH),
        sin_zenith=np.sin(zenith),
        solar_azimuth_rad=np.radians(rows["solar_azimuth_deg"].to_numpy()),
        global_horizontal=global_horizontal,
        beam_horizontal=beam,
        diffuse_horizontal=diffuse,
        air_temperature_c=rows["temp_c"].to_numpy(),
    )


class FittedParameters(NamedTuple):
    """The physical parameters one fit settles on.

    Attributes:
        tilt_rad: Panel tilt from horizontal, in radians.
        azimuth_rad: Panel azimuth, clockwise from north, in radians.
        capacity_mw: Output at standard test conditions.
        temperature_coefficient: Fractional change in output per degree above reference.
        clip_mw: The inverter's own ceiling.
        weights: The convex weights over the arm's splits, or a single one for an arm with at most
            one split.
    """

    tilt_rad: float
    azimuth_rad: float
    capacity_mw: float
    temperature_coefficient: float
    clip_mw: float
    weights: np.ndarray


def _unpack(*, parameters: np.ndarray, arm: str, capacity_guess: float) -> FittedParameters:
    """Map the optimiser's unbounded vector onto the model's physical parameters.

    Every bounded quantity is reached through a `tanh`, and every positive one through an
    exponential, so the optimiser runs unconstrained and no parameter can leave its range.

    Args:
        parameters: The unbounded vector the optimiser proposes.
        arm: The key into `ARM_SPLITS`, which fixes how many weights the vector carries.
        capacity_guess: The site's 99th-percentile metered output, which the scale is measured
            against. It is not the registered capacity, which this experiment never reads.

    Returns:
        The physical parameters, and the convex weights over the arm's splits.
    """
    n_splits = len(ARM_SPLITS[arm])
    if n_splits <= 1:
        weights = np.ones(max(n_splits, 1))
    else:
        logits = np.concatenate([[0.0], parameters[5 : 5 + n_splits - 1]])
        weights = np.exp(logits - logits.max())
        weights = weights / weights.sum()
    return FittedParameters(
        tilt_rad=float(np.radians(TILT_CEILING_DEGREES * (1.0 + np.tanh(parameters[0])) / 2.0)),
        azimuth_rad=float(np.radians(180.0 + AZIMUTH_HALF_RANGE_DEGREES * np.tanh(parameters[1]))),
        capacity_mw=float(capacity_guess * np.exp(parameters[2])),
        temperature_coefficient=float(
            TEMPERATURE_COEFFICIENT_CENTRE
            + TEMPERATURE_COEFFICIENT_HALF_RANGE * np.tanh(parameters[3])
        ),
        clip_mw=float(capacity_guess * np.exp(parameters[4])),
        weights=weights,
    )


def _n_parameters(*, arm: str) -> int:
    """Return how long the optimiser's vector is for one arm."""
    return 5 + max(len(ARM_SPLITS[arm]) - 1, 0)


def _predict(
    *, rows: pl.DataFrame, parameters: np.ndarray, arm: str, capacity_guess: float
) -> np.ndarray:
    """Run the physical model over one set of rows at one parameter vector."""
    fitted = _unpack(parameters=parameters, arm=arm, capacity_guess=capacity_guess)
    geometry = _geometry_for(rows=rows, arm=arm, weights=fitted.weights)
    return power_mw(
        geometry=geometry,
        tilt_rad=fitted.tilt_rad,
        azimuth_rad=fitted.azimuth_rad,
        capacity_mw=fitted.capacity_mw,
        temperature_coefficient=fitted.temperature_coefficient,
        clip_mw=fitted.clip_mw,
    )


def _fit(
    *, train: pl.DataFrame, arm: str, target: str, capacity_guess: float, seed: int
) -> np.ndarray:
    """Fit the model's parameters on one training fold, from several random starts.

    The objective is mean absolute error, which is the metric the arms are compared on, so the fit
    is not optimising one loss and being scored on another.

    Args:
        train: The training rows.
        arm: The key into `ARM_SPLITS`.
        target: The column to fit.
        capacity_guess: The site's 99th-percentile metered output.
        seed: Chooses the random starting points.

    Returns:
        The best parameter vector found.
    """
    actual = train[target].to_numpy().astype(np.float64)

    def objective(parameters: np.ndarray) -> float:
        modelled = _predict(
            rows=train, parameters=parameters, arm=arm, capacity_guess=capacity_guess
        )
        return float(np.abs(modelled - actual).mean())

    generator = np.random.default_rng(seed)
    best_parameters = np.zeros(_n_parameters(arm=arm))
    best_loss = objective(best_parameters)
    for restart in range(N_RESTARTS):
        start = (
            np.zeros(_n_parameters(arm=arm))
            if restart == 0
            else generator.normal(scale=START_SPREAD, size=_n_parameters(arm=arm))
        )
        result = minimize(objective, start, method="Powell", options={"maxiter": MAX_ITERATIONS})
        if result.fun < best_loss:
            best_loss = float(result.fun)
            best_parameters = result.x
    return best_parameters


def _run_site_arm(
    *, site_rows: pl.DataFrame, arm: str, setting_name: str, target: str
) -> pl.DataFrame:
    """Produce out-of-fold losses for one arm at one site, one row per (test row, seed).

    Args:
        site_rows: Every row for one site, already carrying `fold` and `month`.
        arm: The key into `ARM_SPLITS`.
        setting_name: A label carried into the results.
        target: The column to predict.

    Returns:
        One row per (time, seed) with the losses.
    """
    capacity_guess = float(site_rows["effective_capacity_mw"][0])
    outputs: list[pl.DataFrame] = []
    for fold in range(N_FOLDS):
        test = site_rows.filter(pl.col("fold") == fold)
        # Curtailed hours are dropped from the fit for the same reason the tree drops them, and
        # for one more: an hour held at a low cap looks exactly like a small inverter, so fitting
        # on it drags the clip parameter down and mis-states the site's geometry.
        train = site_rows.filter((pl.col("fold") != fold) & ~pl.col("constrained"))
        if test.is_empty() or train.is_empty():
            continue
        actual = test[target].to_numpy().astype(np.float64)
        for seed in SEEDS:
            parameters = _fit(
                train=train, arm=arm, target=target, capacity_guess=capacity_guess, seed=seed
            )
            modelled = _predict(
                rows=test, parameters=parameters, arm=arm, capacity_guess=capacity_guess
            )
            capped = clamp_to_cap(prediction=modelled, cap_mw=test["cap_mw"])
            outputs.append(
                test.select("site", "time", "month", "fold", "effective_capacity_mw", "constrained")
                .cast({"effective_capacity_mw": pl.Float64})
                .with_columns(
                    arm=pl.lit(arm),
                    setting=pl.lit(setting_name),
                    target=pl.lit(target),
                    seed=pl.lit(seed, dtype=pl.Int32),
                    absolute_error_mw=pl.Series(np.abs(modelled - actual), dtype=pl.Float64),
                    signed_error_mw=pl.Series(modelled - actual, dtype=pl.Float64),
                    crps_mw=pl.lit(None, dtype=pl.Float64),
                    absolute_error_capped_mw=pl.Series(np.abs(capped - actual), dtype=pl.Float64),
                    signed_error_capped_mw=pl.Series(capped - actual, dtype=pl.Float64),
                    crps_capped_mw=pl.lit(None, dtype=pl.Float64),
                )
            )
    return pl.concat(outputs)


def _fitted_parameters(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Refit each arm on every site's whole span, to report what geometry the fit settles on.

    These parameters are not used for any score — every number in the results comes from a fit that
    never saw its test fold. They exist so the write-up can say whether the fitted tilt and azimuth
    are physically sensible, which is the cheapest check that the model is doing what it claims.

    **The clip is reported only where it binds.** A clip fitted above the site's own highest reading
    never limits the modelled power, so the optimiser has no gradient on it and leaves it wherever
    its starting point put it. Reporting that starting point as a fitted inverter ceiling would
    invite a reader to interpret a number the data never constrained, so an unidentified clip is
    written as null instead.

    **Curtailed hours are excluded here too.** An hour the operator held at a low cap is
    indistinguishable from an undersized inverter to a model with a clip parameter, so leaving
    those hours in would report a fitted ceiling that describes the network rather than the plant.

    Args:
        dataset: The full frame, already carrying `constrained`.

    Returns:
        One row per (arm, site) with the fitted geometry.
    """
    records: list[dict[str, object]] = []
    for arm in ARM_SPLITS:
        for site in sorted(dataset["site"].unique().to_list()):
            rows = dataset.filter((pl.col("site") == site) & ~pl.col("constrained"))
            capacity_guess = float(rows["effective_capacity_mw"][0])
            parameters = _fit(
                train=rows, arm=arm, target="power_mw", capacity_guess=capacity_guess, seed=0
            )
            fitted = _unpack(parameters=parameters, arm=arm, capacity_guess=capacity_guess)
            highest_output_mw = float(rows["power_mw"].to_numpy().max())
            records.append(
                {
                    "arm": arm,
                    "site": site,
                    "tilt_degrees": float(np.degrees(fitted.tilt_rad)),
                    "azimuth_degrees": float(np.degrees(fitted.azimuth_rad)),
                    "capacity_fraction_of_p99": fitted.capacity_mw / capacity_guess,
                    "temperature_coefficient_per_c": fitted.temperature_coefficient,
                    "clip_fraction_of_p99": (
                        fitted.clip_mw / capacity_guess
                        if fitted.clip_mw < highest_output_mw
                        else None
                    ),
                    "weights": [float(weight) for weight in fitted.weights],
                }
            )
    return pl.DataFrame(records)


def _run_all(*, dataset: pl.DataFrame, jobs: list[tuple[str, str, str]]) -> pl.DataFrame:
    """Run every (arm, site) job concurrently and concatenate the losses."""
    sites = sorted(dataset["site"].unique().to_list())
    outputs: list[pl.DataFrame] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
        futures = {}
        for arm, setting_name, target in jobs:
            for site in sites:
                future = pool.submit(
                    _run_site_arm,
                    site_rows=dataset.filter(pl.col("site") == site),
                    arm=arm,
                    setting_name=setting_name,
                    target=target,
                )
                futures[future] = (setting_name, arm, site)
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            setting_name, arm, site = futures[future]
            outputs.append(future.result())
            _LOG.info("%d/%d done: %s / %s / site %s", done, len(futures), setting_name, arm, site)
    return pl.concat(outputs)


def _intervals_for(
    *, losses: pl.DataFrame, setting_name: str, target: str, sites: list[str]
) -> list[dict[str, object]]:
    """Compute every contrast's bootstrap interval, pooled and per site."""
    metrics = [
        "absolute_error_capped_fraction_of_capacity",
        "absolute_error_fraction_of_capacity",
        "absolute_error_mw",
    ]
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
    """Fit every arm at every site and fold, bootstrap the contrasts, and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="cams")
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
    _LOG.info("dataset: %d rows, %d sites", dataset.height, len(sites))

    jobs = [(arm, "primary", "power_mw") for arm in ARM_SPLITS]
    jobs += [(arm, "positive_control", "synthetic_power_mw") for arm in CONTROL_ARMS]
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
            bias_mw=pl.col("signed_error_mw").mean(),
            mae_capped_mw=pl.col("absolute_error_capped_mw").mean(),
            mae_capped_fraction_of_capacity=pl.col(
                "absolute_error_capped_fraction_of_capacity"
            ).mean(),
            constrained_rows=pl.col("constrained").sum() // len(SEEDS),
            n_rows=pl.len() // len(SEEDS),
        )
        .sort("setting", "arm", "site")
    )
    summary.write_parquet(results_dir / "per_site_summary.parquet")
    _fitted_parameters(dataset=dataset).write_parquet(results_dir / "fitted_parameters.parquet")
    _LOG.info("wrote results to %s", results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
