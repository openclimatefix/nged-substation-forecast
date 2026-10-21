"""Known-answer test 1 of experiment 126: the augmented code reproduces B0 where it must.

Each check is a limit in which an arm has to equal B0, so a failure means the arm's columns or
penalties are wrong, not that the data disagree.

1. B0 built from the augmented code equals `studies.dp_gam.fit_gam` to 1e-8 relative in the
   parameters, at every decision site and the replicate and scored cuts.
2. Each interaction arm at a penalty weight of 1e10 has B0's fitted values to 1e-5 points. (The
   plan said 1e-6. Development runs before any fit showed a floor of 0.8e-6 to 1.8e-6 points at
   every penalty weight, from the conditioning of the normal equations, so the plan was amended
   to 1e-5 in its own commit before the experiment ran.)
3. The hinge arm at 15.5 degrees C has B0's coefficients exactly.
4. The pieces of a split column partition the rows, so the pieces sum to the column.
5. B0's filtered correction from `score_year` equals `run_kalman` and `correction_at` called
   directly on the stored B0 residuals.
6. No training row ends after its cut.

    uv run python ml-experiments/00114-*/e00126_test1.py
"""

import sys
from typing import Final

import e00126_common as common
import e00126_design as design
import e00126_pipeline as pipeline
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    PRIOR_START,
    FilterHyperparameters,
    correction_at,
    day_ahead_initialisation,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import assert_no_lookahead, fit_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

TIGHT_PENALTY_WEIGHT: Final[float] = 1e10
PARAMETER_TOLERANCE: Final[float] = 1e-8
FITTED_VALUE_TOLERANCE_POINTS: Final[float] = 1e-5
TEST_CUTS: Final[tuple[str, ...]] = ("replicate", "scored")


def check_b0(*, cache: FeatureCache, vintages: pl.DataFrame, site: str, label: str) -> list[str]:
    """Run checks 1, 2, 3, 4, and 6 at one site and cut and return the failures."""
    failures: list[str] = []
    cut = common.FIT_CUTS[label]
    base = design.build_base(cache=cache, vintages=vintages, site=site, cut=cut)
    assert_no_lookahead(cache=cache, index=base.index, cut=cut)
    if cache.end_times[base.index].max() > common.to_64(moment=cut):
        failures.append(f"{site} {label}: a training row ends after the cut")
    fit = design.fit_arm(cache=cache, base=base, arm=design.B0)
    reference = fit_gam(
        cache=cache,
        site=site,
        specs=[common.base_spec()],
        cut=cut,
        window=common.base_window(),
        season_years=common.SEASON_YEARS,
        pinned=pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut),
    )
    relative = float(
        np.abs(reference.theta[0] - fit.theta).max() / np.abs(reference.theta[0]).max()
    )
    print(f"{site} {label}: B0 against fit_gam, max relative difference {relative:.2e}")
    if relative > PARAMETER_TOLERANCE:
        failures.append(f"{site} {label}: B0 differs from fit_gam by {relative:.2e}")
    index = common.scoring_index(
        cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    b0_values = design.predict_arm(cache=cache, base=base, fit=fit, index=index)
    for name in pipeline.INTERACTION_ARMS:
        arm = design.interaction_arm(name=name, penalty_weight=TIGHT_PENALTY_WEIGHT)
        arm_fit = design.fit_arm(cache=cache, base=base, arm=arm)
        gap = float(
            np.abs(
                design.predict_arm(cache=cache, base=base, fit=arm_fit, index=index) - b0_values
            ).max()
        )
        print(f"{site} {label}: {name} at lambda 1e10 against B0, max fitted-value gap {gap:.2e}")
        if gap > FITTED_VALUE_TOLERANCE_POINTS:
            failures.append(f"{site} {label}: {name} at lambda 1e10 differs from B0 by {gap:.2e}")
    hinge = design.fit_arm(
        cache=cache, base=base, arm=design.threshold_arm(threshold_c=common.BASE_THRESHOLD_C)
    )
    hinge_gap = float(np.abs(hinge.theta - fit.theta).max() / np.abs(fit.theta).max())
    print(f"{site} {label}: hinge at 15.5 against B0, max relative difference {hinge_gap:.2e}")
    if hinge_gap > PARAMETER_TOLERANCE:
        failures.append(f"{site} {label}: the hinge at 15.5 differs from B0 by {hinge_gap:.2e}")
    for factors in (("daytype",), ("subband",), ("daytype", "subband")):
        masks = design.piece_masks(cache=cache, index=base.index, factors=factors)
        if not np.array_equal(masks.sum(axis=1), np.ones(base.n)):
            failures.append(f"{site} {label}: pieces of {factors} do not partition the rows")
    return failures


def check_filter(*, cache: FeatureCache, vintages: pl.DataFrame, site: str) -> list[str]:
    """Run check 5 at one site and return the failures."""
    run = pipeline.fit_year(
        cache=cache, vintages=vintages, site=site, fit_label="tuning", arm_name="b0"
    )
    replica = pipeline.fit_year(
        cache=cache, vintages=vintages, site=site, fit_label="replica_tuning", arm_name="b0"
    )
    assert run is not None
    assert replica is not None
    hyper = FilterHyperparameters.from_array(values=np.array(PRIOR_START))
    scored = pipeline.score_year(
        cache=cache,
        runs={"tuning": run, "replica_tuning": replica},
        label="tuning",
        hyper=hyper,
    )
    trace = run_kalman(
        residual=run.residual_stream(cache=cache), band=cache.calendar.band, hyper=hyper
    )
    valid = cache.end_times[run.index]
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times,
        initialisation=day_ahead_initialisation(end_times=valid, days_ahead=1),
        delay_hours=3.5,
    )
    direct = correction_at(
        trace=trace, cutoff=cutoff, target=run.index, band=cache.calendar.band[run.index]
    )
    gap = float(np.abs(direct.total - scored.correction.total).max())
    print(f"{site}: filtered correction against run_kalman called directly, max gap {gap:.2e}")
    return [] if gap == 0.0 else [f"{site}: the filtered correction differs by {gap:.2e}"]


def main() -> None:
    """Run every check and exit non-zero if any fails."""
    common.configure_threads()
    cache = common.load_cache()
    vintages = read_vintages()
    failures: list[str] = []
    for site in common.DECISION_SITES:
        for label in TEST_CUTS:
            failures += check_b0(cache=cache, vintages=vintages, site=site, label=label)
        failures += check_filter(cache=cache, vintages=vintages, site=site)
    if failures:
        print("\nTEST 1 FAILED")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)
    print("\nTEST 1 PASSED")


if __name__ == "__main__":
    main()
