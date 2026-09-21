"""Ask whether calibrating the physical model's output with a tree closes its gap to XGBoost.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**The physical model loses to XGBoost by about 0.9 percentage points of P99 output, and that gap
has two possible causes with different remedies.** The physical model may be missing information
the tree finds — soiling, shading, curtailment, a drifting capacity — in which case a tree given
the physical model's output should recover the gap. Or the physical model's five parameters may be
the wrong shape for these sites, in which case correcting its output after the fact will not help,
because the error is already baked into the transposition.

Three arms separate those. `H_physics_only` gives a tree the physical model's prediction and
nothing else, which is the narrow reading of "calibrate the physical model": it can rescale, bend
and clip that one number but cannot consult the weather. `H_physics_plus_time` adds the
non-irradiance columns every other arm sees, so the calibration can vary by season, temperature and
hour without ever seeing irradiance again. `H_physics_plus_all` gives the tree the physical model's
prediction on top of its own full feature set, which asks the opposite question: does the physical
model's structure carry anything a tree with the same weather has not already found?

**Every physical-model prediction used here is withheld from the tree that consumes it, and the
withholding is by calendar month.** Folds are cut inside each site's own span, so dropping rows by
fold label would still let a physical fit see the scored fold's own hours through another site's
rows. The scored fold's months are excluded, and the training rows are filled in by an inner
cross-validation that excludes their own months too — a column that is sharper where the tree
trains than where it is scored gets over-trusted, which would flatter these arms for a reason that
has nothing to do with calibration. That is the construction
`run_experiment._add_learned_split` uses, and for the same reason.

Run it with `uv run --no-project` plus `--with polars --with numpy --with xgboost --with scipy
--with pvlib --with xarray --with netcdf4 --with pandas --with deltalake`, then
`python scripts/experiments/beam_diffuse_split/run_hybrid_experiment.py --source cams
--alignment shifted`. The long dependency list is `export_cap.py` reaching into
`build_dataset.py` for the site roster, which is what maps NGED's `time_series_id` to an
anonymous label.
"""

import argparse
import logging
import sys
from typing import Final

import numpy as np
import polars as pl
import run_experiment
from commissioning import drop_commissioning_ramp
from export_cap import with_export_cap
from run_experiment import (
    N_FOLDS,
    PRIMARY_HYPER_PARAMETERS,
    _add_time_features,
    _assign_folds,
    _bootstrap_difference,
    _run_all,
    dataset_path_for,
    results_dir_for,
)
from run_physics_experiment import _fit, _predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("run_hybrid_experiment")

PHYSICS_TEMPLATE: Final[str] = "physics_mw_fold{fold}"
"""Column holding the physical model's prediction, for rows scored on one named fold."""

PHYSICS_ARM: Final[str] = "P_C_source_split"
"""Which physical-model arm is calibrated.

The arm given the source's own beam and diffuse fluxes, so the tree is calibrating the physical
model at its best-informed setting rather than at a deliberately starved one.
"""

PHYSICS_SEED: Final[int] = 0
"""Which optimiser restart seed the calibrated physical fits use.

`run_physics_experiment` reports a seed-to-seed spread of a few parts in a million, because
best-of-eight restarts lands on the same minimum every time, so one seed is enough here and three
would only multiply the fit count.
"""

HYBRID_ARMS: Final[dict[str, tuple[tuple[str, ...], bool]]] = {
    "H_physics_only": ((PHYSICS_TEMPLATE,), False),
    "H_physics_plus_time": ((PHYSICS_TEMPLATE,), True),
    "H_physics_plus_all": ((PHYSICS_TEMPLATE, "ghi_w_m2", "bhi_w_m2", "dhi_w_m2"), True),
}
"""Each arm's irradiance-side columns, and whether it also sees the shared non-irradiance features.

`H_physics_only` is the only arm in this experiment that drops the shared features, because the
narrow reading of "calibrate the physical model" is a function of its output alone.
"""

CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("H_physics_only", "P_C_physics"),
    ("H_physics_plus_time", "P_C_physics"),
    ("H_physics_plus_all", "P_C_physics"),
    ("H_physics_only", "C_era5_split"),
    ("H_physics_plus_time", "C_era5_split"),
    ("H_physics_plus_all", "C_era5_split"),
)
"""What to compare: each calibrated arm against the physical model and against XGBoost."""


def _add_physics_predictions(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Add the physical model's out-of-fold prediction, one column per scored fold.

    Args:
        dataset: The full frame, already carrying `fold` and `month`.

    Returns:
        The frame with a `physics_mw_fold{k}` column for every fold that can be scored.
    """
    fitted: dict[tuple[str, frozenset[int]], np.ndarray] = {}
    columns: dict[str, np.ndarray] = {}
    sites = sorted(dataset["site"].unique().to_list())

    for scored_fold in range(N_FOLDS):
        prediction = np.full(dataset.height, np.nan)
        for site in sites:
            site_mask = (dataset["site"] == site).to_numpy()
            site_rows = dataset.filter(pl.col("site") == site)
            capacity_guess = float(site_rows["effective_capacity_mw"][0])

            # Rows in the scored fold are predicted by a fit that excluded that fold's months; every
            # other row is predicted by a fit that excluded the scored fold's months *and its own*,
            # so no row's value was ever seen by the fit that produced it.
            for own_fold in range(N_FOLDS):
                withheld = frozenset({scored_fold, own_fold})
                key = (site, withheld)
                if key not in fitted:
                    excluded = (
                        site_rows.filter(pl.col("fold").is_in(list(withheld)))["month"]
                        .unique()
                        .to_list()
                    )
                    train = site_rows.filter(~pl.col("month").is_in(excluded))
                    if train.is_empty():
                        continue
                    fitted[key] = _fit(
                        train=train,
                        arm=PHYSICS_ARM,
                        target="power_mw",
                        capacity_guess=capacity_guess,
                        seed=PHYSICS_SEED,
                    )
                if key not in fitted:
                    continue
                target_mask = site_mask & (dataset["fold"] == own_fold).to_numpy()
                if not target_mask.any():
                    continue
                prediction[target_mask] = _predict(
                    rows=dataset.filter(pl.Series(target_mask)),
                    parameters=fitted[key],
                    arm=PHYSICS_ARM,
                    capacity_guess=capacity_guess,
                )
        columns[PHYSICS_TEMPLATE.format(fold=scored_fold)] = prediction
        _LOG.info(
            "physics column built for scored fold %d (%d fits so far)", scored_fold, len(fitted)
        )

    return dataset.with_columns(
        **{name: pl.Series(values, dtype=pl.Float64) for name, values in columns.items()}
    )


def main() -> int:
    """Score the calibrated arms against the physical model and against XGBoost."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("cds", "open-meteo", "cams"), default="cams")
    parser.add_argument("--alignment", choices=("as-labelled", "shifted"), default="shifted")
    arguments = parser.parse_args()

    dataset = with_export_cap(
        dataset=_assign_folds(
            dataset=_add_time_features(
                dataset=drop_commissioning_ramp(
                    dataset=pl.read_parquet(
                        dataset_path_for(source=arguments.source, alignment=arguments.alignment)
                    )
                )
            )
        )
    )
    _LOG.info("dataset: %d rows, %d sites", dataset.height, dataset["site"].n_unique())
    dataset = _add_physics_predictions(dataset=dataset)

    # `_run_all` resolves an arm's columns through the module-level table, so the new arms are
    # registered there rather than threaded through every helper. Throwaway code, one process.
    for arm, (columns, _with_shared) in HYBRID_ARMS.items():
        run_experiment.ARM_FEATURES[arm] = columns
    original_features_for = run_experiment._features_for

    def _features_for(*, arm: str, fold: int) -> list[str]:
        """Resolve one arm's columns, dropping the shared features where an arm forgoes them."""
        if arm in HYBRID_ARMS and not HYBRID_ARMS[arm][1]:
            return [column.format(fold=fold) for column in HYBRID_ARMS[arm][0]]
        return original_features_for(arm=arm, fold=fold)

    # ty rejects this as `invalid-assignment` and prints both sides of the comparison
    # identically, because it treats a module-level `def` as its own nominal type rather than
    # as its signature. Any replacement function is unassignable, however well it matches. The
    # signal to delete the suppression is ty reporting `unused-ignore-comment` here.
    run_experiment._features_for = _features_for  # ty: ignore[invalid-assignment]

    jobs = [
        (arm, "primary", "power_mw", PRIMARY_HYPER_PARAMETERS, False)
        for arm in (*HYBRID_ARMS, "C_era5_split", "A_global_only")
    ]
    losses = _run_all(dataset=dataset, jobs=jobs).with_columns(
        absolute_error_fraction_of_capacity=pl.col("absolute_error_mw")
        / pl.col("effective_capacity_mw"),
        absolute_error_capped_fraction_of_capacity=pl.col("absolute_error_capped_mw")
        / pl.col("effective_capacity_mw"),
    )

    # The physical model's own score, on the identical rows, read straight from its per-row losses.
    physics_dir = (
        run_experiment.REPO_DATA_DIR
        / "ERA5"
        / f"beam_diffuse_physics_{arguments.source}_{arguments.alignment}"
    )
    physics = (
        pl.read_parquet(physics_dir / "per_row_losses.parquet")
        .filter(
            (pl.col("arm") == PHYSICS_ARM)
            & (pl.col("setting") == "primary")
            & (pl.col("target") == "power_mw")
        )
        .with_columns(
            arm=pl.lit("P_C_physics"),
            absolute_error_fraction_of_capacity=pl.col("absolute_error_mw")
            / pl.col("effective_capacity_mw"),
        )
    )
    losses = pl.concat([losses, physics.select(losses.columns)], how="vertical")

    results_dir = results_dir_for(source=arguments.source, alignment=arguments.alignment)
    results_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "## Calibrating the physical model with a tree",
        "",
        "| Arm | MAE (% of P99 output) | Hours |",
        "|---|---|---|",
    ]
    for arm in (*HYBRID_ARMS, "P_C_physics", "C_era5_split", "A_global_only"):
        rows = losses.filter(pl.col("arm") == arm)
        if rows.is_empty():
            continue
        mae = float(rows["absolute_error_fraction_of_capacity"].to_numpy().mean()) * 100
        lines.append(f"| {arm} | {mae:.3f} | {rows.select('site', 'time').n_unique():,} |")

    lines += [
        "",
        "| Contrast | ΔMAE (pp of P99 output) | 95% interval | Excludes zero? |",
        "|---|---|---|---|",
    ]
    for treatment, reference in CONTRASTS:
        scoped = losses.filter(pl.col("arm").is_in([treatment, reference]))
        if scoped["arm"].n_unique() < 2:
            continue
        interval = _bootstrap_difference(
            losses=scoped,
            treatment=treatment,
            reference=reference,
            metric="absolute_error_fraction_of_capacity",
        )
        excludes = interval["lower_95"] * interval["upper_95"] > 0
        lines.append(
            f"| {treatment} − {reference} | {interval['difference'] * 100:+.4f} | "
            f"[{interval['lower_95'] * 100:+.4f}, {interval['upper_95'] * 100:+.4f}] | "
            f"{'**yes**' if excludes else 'no'} |"
        )

    report = "\n".join(lines) + "\n"
    (results_dir / "hybrid.md").write_text(report)
    losses.write_parquet(results_dir / "hybrid_per_row_losses.parquet")
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
