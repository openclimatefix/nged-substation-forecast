"""Known-answer test 1 of experiment 121: the new columns reproduce B0 where they must.

Each check is a limit in which an arm has to equal B0, so a failure means the arm's columns or
penalties are wrong, not that the data disagree.

1. B0 built from this code equals `studies.dp_gam.fit_gam` to 1e-8 relative in the parameters.
2. MH-5 with only the 15.5 degree hinge has B0's coefficients (in B0's column order) to 1e-8.
3. MH, MH-sh, NS-L, NS-U, and NS-H with the added columns' ridge multiplied by 1e10 have B0's
   fitted values to 1e-5 points.
4. NS-L and NS-U columns agree on a winter day and differ by two half-hours on a summer day, and
   NS-H's columns are zero outside 00:00 to 07:00 UTC.
5. B0's filtered correction from `score_year` equals `run_kalman` and `correction_at` called
   directly on the stored B0 residuals.
6. No training row ends after its cut.

    uv run python ml-experiments/00121-*/e00121_test1.py [--cuts tuning replicate scored]
"""

import argparse
import sys
from dataclasses import replace
from typing import Final

import e00121_common as common
import e00121_design as design
import e00121_pipeline as pipeline
import numpy as np
import polars as pl
import torch
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

PARAMETER_TOLERANCE: Final[float] = 1e-8
FITTED_VALUE_TOLERANCE_POINTS: Final[float] = 1e-5
TIGHT_SHRINK: Final[float] = 1e10


def check_b0(*, cache: FeatureCache, vintages: pl.DataFrame, site: str, label: str) -> list[str]:
    """Run checks 1, 2, 3, and 6 at one site and cut and return the failures."""
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
    # Check 2: MH-5 with only the 15.5 degree hinge.
    single = replace(design.MH5, name="mh5_single", thresholds_c=(common.BASE_THRESHOLD_C,))
    single_fit = design.fit_arm(cache=cache, base=base, arm=single)
    heating = base.layout.slices["heating"]
    keep = [j for j in range(len(fit.theta)) if not heating.start <= j < heating.stop]
    in_b0_order = np.zeros_like(fit.theta)
    in_b0_order[keep] = single_fit.theta[: len(keep)]
    in_b0_order[heating] = single_fit.theta[len(keep) :]
    single_gap = float(np.abs(in_b0_order - fit.theta).max() / np.abs(fit.theta).max())
    print(f"{site} {label}: MH-5 at 15.5 against B0, max relative difference {single_gap:.2e}")
    if single_gap > PARAMETER_TOLERANCE:
        failures.append(f"{site} {label}: MH-5 at 15.5 differs from B0 by {single_gap:.2e}")
    # Check 3: the added blocks forced to zero.
    index = common.scoring_index(
        cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    b0_values = design.predict_arm(cache=cache, base=base, fit=fit, index=index)
    for arm in (design.MH, design.NS_L, design.NS_U, design.NS_H, design.NS_HW, design.NS_UT):
        tight = replace(arm, name=f"{arm.name}_tight", shrink=TIGHT_SHRINK)
        arm_fit = design.fit_arm(cache=cache, base=base, arm=tight)
        values = design.predict_arm(cache=cache, base=base, fit=arm_fit, index=index)
        gap = float(np.abs(values - b0_values).max())
        print(
            f"{site} {label}: {arm.name} at ridge x1e10 against B0, max fitted-value gap {gap:.2e}"
        )
        if gap > FITTED_VALUE_TOLERANCE_POINTS:
            failures.append(
                f"{site} {label}: {arm.name} at ridge x1e10 differs from B0 by {gap:.2e}"
            )
    failures += check_signs(cache=cache, base=base, site=site, label=label)
    return failures


def check_signs(*, cache: FeatureCache, base: design.BaseFit, site: str, label: str) -> list[str]:
    """Check each arm's sign flags and added coefficients, and print its added effective df."""
    failures: list[str] = []
    reference = design.effective_degrees_of_freedom(
        gram=torch.as_tensor(base.x0.T @ base.x0),
        penalty=design.penalty_for(cache=cache, base=base, arm=design.B0, x=base.x0),
    )
    for arm in design.all_arms().values():
        if arm.kind == "b0":
            continue
        block, constrained = design.added_block(
            cache=cache, site=site, index=base.index, arm=arm, cut=base.cut
        )
        x = design.arm_matrix(
            cache=cache, layout=base.layout, site=site, index=base.index, x0=base.x0, arm=arm
        )
        penalty = design.penalty_for(cache=cache, base=base, arm=arm, x=x)
        n_added = block.shape[1]
        if not np.all(penalty.constrained[-n_added:] == constrained):
            failures.append(f"{site} {label}: {arm.name} has the wrong sign flags")
        if constrained != (arm.kind not in ("night_l", "night_u")):
            failures.append(f"{site} {label}: {arm.name} has the wrong sign constraint")
        theta = design.fit_arm(cache=cache, base=base, arm=arm).theta
        if constrained and theta[-n_added:].min() < -1e-12:
            failures.append(f"{site} {label}: {arm.name} has a negative added coefficient")
        edf = design.effective_degrees_of_freedom(gram=torch.as_tensor(x.T @ x), penalty=penalty)
        print(f"{site} {label}: {arm.name} added effective df {edf - reference:.1f}")
    return failures


def check_night_columns(*, cache: FeatureCache) -> list[str]:
    """Run check 4 on the whole grid and return the failures."""
    failures: list[str] = []
    grid = np.arange(len(cache.end_times))
    ends = cache.end_times
    for label, day in (("winter", "2024-01-15"), ("summer", "2024-07-15")):
        rows = grid[(ends.astype("datetime64[D]") == np.datetime64(day))]
        utc = design.night_columns(cache=cache, index=rows, local=False)
        local = design.night_columns(cache=cache, index=rows, local=True)
        if len(rows) != design.HALF_HOURS_PER_DAY:
            failures.append(f"{label} day holds {len(rows)} rows, not 48")
            continue
        utc_halves = design.start_half_hour(cache=cache, index=rows, local=False)
        local_halves = design.start_half_hour(cache=cache, index=rows, local=True)
        shift = (local_halves - utc_halves) % design.HALF_HOURS_PER_DAY
        expected = 0 if label == "winter" else 2
        if not np.all(shift == expected):
            failures.append(f"{label}: local minus UTC half-hour is not {expected} half-hours")
        if label == "winter" and not np.array_equal(utc, local):
            failures.append("winter night columns differ between the UTC and local clocks")
        if label == "summer" and np.array_equal(utc, local):
            failures.append("summer night columns do not differ between the clocks")
        if utc.sum(axis=1).max() > 1:
            failures.append(f"{label}: a row falls in two night columns")
        if int(utc.sum()) != design_night_rows(label=label):
            failures.append(f"{label}: night rows of the day number {int(utc.sum())}, not 20")
    day = grid[ends.astype("datetime64[D]") == np.datetime64("2024-01-15")]
    columns = design.night_columns(cache=cache, index=day, local=False)
    by_end = {str(ends[r].astype("datetime64[m]"))[11:]: columns[k] for k, r in enumerate(day)}
    for end_time, expected in (("22:00", None), ("22:30", 0), ("08:00", 19), ("08:30", None)):
        row = by_end[end_time]
        got = None if row.sum() == 0 else int(np.argmax(row))
        if got != expected:
            failures.append(f"row ending {end_time} sets night column {got}, expected {expected}")
    rows = grid[grid % 997 == 0]
    hinge = design.night_hinge_columns(cache=cache, site=common.DECISION_SITES[0], index=rows)
    half = design.start_half_hour(cache=cache, index=rows, local=False)
    outside = half >= common.NS_H_END_HALF_HOUR_UTC
    if np.any(hinge[outside] != 0.0):
        failures.append("NS-H has a non-zero column outside 00:00 to 07:00 UTC")
    print(f"night columns: {len(failures)} failures")
    return failures


def design_night_rows(*, label: str) -> int:
    """Return the number of night half-hours a single UTC-day grid holds for the UTC clock."""
    del label
    return common.NIGHT_HALF_HOURS


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
    return [] if gap <= 1e-12 else [f"{site}: the filtered correction differs by {gap:.2e}"]


def main() -> None:
    """Run every check and exit non-zero if any fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts", nargs="+", default=["tuning"])
    arguments = parser.parse_args()
    common.configure_threads()
    cache = common.load_cache()
    vintages = read_vintages()
    failures: list[str] = check_night_columns(cache=cache)
    for site in common.DECISION_SITES:
        for label in arguments.cuts:
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
