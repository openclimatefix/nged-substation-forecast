"""Test whether the physical model's arms disagree about the beam field or about panel geometry.

The physical model reports the opposite sign from XGBoost: handed the weather product's own split
it scores *worse* than handed an Erbs split. The write-up's explanation is that its two arms do not
differ only in the beam field they are given — each arm fits its own tilt and azimuth, and the arm
given the published split settles several degrees shallower. That is a confound the XGBoost arms do
not have, because every XGBoost arm sees identical non-irradiance features.

This settles it. Every arm is refitted with the tilt and azimuth held at the values the Erbs arm
settled on for that site and fold, so the arms differ only in their beam and diffuse columns, as
the XGBoost arms do. If the sign survives, the disagreement is about the beam field. If it does
not, the write-up's explanation is right and the physical model's ordering was geometry all along.

Run it from this directory as:

```bash
uv run --no-project --with polars --with numpy --with scipy --with pvlib --with deltalake \
    python shared_geometry.py --source cams --alignment piecewise
```
"""

import argparse
import concurrent.futures
import logging
import sys
from typing import Final

import numpy as np
import polars as pl
from commissioning import drop_commissioning_ramp
from export_cap import clamp_to_cap, with_export_cap
from run_experiment import (
    N_FOLDS,
    SEEDS,
    _add_time_features,
    _assign_folds,
    _bootstrap_difference,
    dataset_path_for,
)
from run_physics_experiment import MAX_ITERATIONS, START_SPREAD, _n_parameters, _predict
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("shared_geometry")

REFERENCE_ARM: Final[str] = "P_B_erbs"
"""The arm whose fitted tilt and azimuth every other arm is held to.

The Erbs arm is the reference the headline contrast is taken against, so holding the geometry at
its values leaves the treatment arm carrying the whole of the change.
"""

ARMS: Final[tuple[str, ...]] = ("P_B_erbs", "P_C_source_split", "P_B_disc")
"""The arms refitted here. Each is handed one beam estimate, so none fits split weights."""

GEOMETRY_INDICES: Final[tuple[int, int]] = (0, 1)
"""Which entries of the optimiser's vector carry tilt and azimuth."""

N_RESTARTS: Final[int] = 8
"""Starting points per fit, matching `run_physics_experiment`."""

# The seeds come from `run_experiment`, because the block bootstrap draws one of them per
# resample and would otherwise see a ragged array.


def _fit(
    *,
    train: pl.DataFrame,
    arm: str,
    capacity_guess: float,
    held_geometry: tuple[float, float] | None,
    seed: int,
) -> np.ndarray:
    """Fit one arm on one training fold, optionally with tilt and azimuth held fixed.

    Args:
        train: The training rows.
        arm: The key into `run_physics_experiment.ARM_SPLITS`.
        capacity_guess: The site's 99th-percentile metered output.
        held_geometry: The raw tilt and azimuth entries to hold, or `None` to fit them too.
        seed: Chooses the random starting points.

    Returns:
        The best full parameter vector found, geometry entries included.
    """
    actual = train["power_mw"].to_numpy().astype(np.float64)
    n_parameters = _n_parameters(arm=arm)
    tilt_index, azimuth_index = GEOMETRY_INDICES

    def expand(free: np.ndarray) -> np.ndarray:
        """Put the free entries back into a full parameter vector."""
        if held_geometry is None:
            return free
        full = np.empty(n_parameters)
        full[tilt_index], full[azimuth_index] = held_geometry
        full[azimuth_index + 1 :] = free
        return full

    def objective(free: np.ndarray) -> float:
        modelled = _predict(
            rows=train, parameters=expand(free), arm=arm, capacity_guess=capacity_guess
        )
        return float(np.abs(modelled - actual).mean())

    n_free = n_parameters if held_geometry is None else n_parameters - len(GEOMETRY_INDICES)
    generator = np.random.default_rng(seed)
    best_free = np.zeros(n_free)
    best_loss = objective(best_free)
    for restart in range(N_RESTARTS):
        start = (
            np.zeros(n_free) if restart == 0 else generator.normal(scale=START_SPREAD, size=n_free)
        )
        result = minimize(objective, start, method="Powell", options={"maxiter": MAX_ITERATIONS})
        if result.fun < best_loss:
            best_loss = float(result.fun)
            best_free = result.x
    return expand(best_free)


def _losses_for_site(*, site_rows: pl.DataFrame, site: str) -> pl.DataFrame:
    """Produce out-of-fold losses for every arm at one site, under both geometry schemes."""
    capacity_guess = float(site_rows["effective_capacity_mw"][0])
    outputs: list[pl.DataFrame] = []
    for fold in range(N_FOLDS):
        test = site_rows.filter(pl.col("fold") == fold)
        train = site_rows.filter((pl.col("fold") != fold) & ~pl.col("constrained"))
        if test.is_empty() or train.is_empty():
            continue
        actual = test["power_mw"].to_numpy().astype(np.float64)
        for seed in SEEDS:
            reference = _fit(
                train=train,
                arm=REFERENCE_ARM,
                capacity_guess=capacity_guess,
                held_geometry=None,
                seed=seed,
            )
            held = (float(reference[GEOMETRY_INDICES[0]]), float(reference[GEOMETRY_INDICES[1]]))
            for arm in ARMS:
                for scheme, geometry in (("free", None), ("shared", held)):
                    parameters = _fit(
                        train=train,
                        arm=arm,
                        capacity_guess=capacity_guess,
                        held_geometry=geometry,
                        seed=seed,
                    )
                    modelled = _predict(
                        rows=test, parameters=parameters, arm=arm, capacity_guess=capacity_guess
                    )
                    capped = clamp_to_cap(prediction=modelled, cap_mw=test["cap_mw"])
                    outputs.append(
                        test.select("site", "time", "month", "effective_capacity_mw")
                        .cast({"effective_capacity_mw": pl.Float64})
                        .with_columns(
                            arm=pl.lit(arm),
                            scheme=pl.lit(scheme),
                            seed=pl.lit(seed, dtype=pl.Int32),
                            absolute_error_capped_fraction_of_capacity=(
                                pl.Series(np.abs(capped - actual), dtype=pl.Float64)
                                / pl.col("effective_capacity_mw")
                            ),
                        )
                    )
    _LOG.info("site %s done", site)
    return pl.concat(outputs)


def main() -> int:
    """Refit every arm under both geometry schemes and print the contrasts side by side."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("cds", "open-meteo", "cams"), default="cams")
    parser.add_argument(
        "--alignment", choices=("as-labelled", "shifted", "piecewise"), default="piecewise"
    )
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

    sites = sorted(dataset["site"].unique().to_list())
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(
                _losses_for_site, site_rows=dataset.filter(pl.col("site") == site), site=site
            )
            for site in sites
        ]
        losses = pl.concat([future.result() for future in futures])

    metric = "absolute_error_capped_fraction_of_capacity"
    print(f"\n### physical model, {arguments.source}, {arguments.alignment} stamps\n")
    print("| Scheme | Arm | MAE (% of P99 output) |")
    print("|---|---|---|")
    for scheme in ("free", "shared"):
        for arm in ARMS:
            rows = losses.filter((pl.col("scheme") == scheme) & (pl.col("arm") == arm))
            print(f"| {scheme} | {arm} | {rows[metric].mean() * 100:.3f} |")

    print("\n| Scheme | Contrast | ΔMAE (pp of P99 output) | 95% interval |")
    print("|---|---|---|---|")
    for scheme in ("free", "shared"):
        scoped = losses.filter(pl.col("scheme") == scheme)
        for treatment in ("P_C_source_split", "P_B_disc"):
            result = _bootstrap_difference(
                losses=scoped, treatment=treatment, reference=REFERENCE_ARM, metric=metric
            )
            point = result["difference"] * 100
            low = result["lower_95"] * 100
            high = result["upper_95"] * 100
            print(
                f"| {scheme} | {treatment} − {REFERENCE_ARM} | {point:+.4f} "
                f"| [{low:+.4f}, {high:+.4f}] |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
