"""Known-answer self-tests of the injection builder and the scores of experiment 00045.

Prints PASS or FAIL with the numbers for each check and exits non-zero if any fails. The tests use
the tuning farms only.

    POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        uv run python \
        ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_inject_selftest.py
"""

import sys
from typing import Final

import e00045_common as common
import e00045_inject as inject
import e00045_metrics as metrics
import e00045_model as model
import numpy as np
import polars as pl

SITES: Final[tuple[str, ...]] = ("D16", "D10")
"""Donington and Spilsby: two Primaries with clean series."""
_failures: list[str] = []


def report(*, name: str, passed: bool, detail: str) -> None:
    """Print one check's outcome and remember a failure."""
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    if not passed:
        _failures.append(name)


def test_unit_counts() -> None:
    """Each split has 3 farms x 4 seeds x 8 Primaries x 14 cases = 1,344 units, none shared."""
    tuning, scoring = inject.units_for(split="tuning"), inject.units_for(split="scoring")
    arithmetic = len(inject.TUNING_FARMS) * len(inject.TUNING_SEEDS) * len(common.PRIMARIES) * 14
    shared_farms = set(inject.TUNING_FARMS) & set(inject.SCORING_FARMS)
    shared_seeds = set(inject.TUNING_SEEDS) & set(inject.SCORING_SEEDS)
    report(
        name="unit counts",
        passed=len(tuning) == len(scoring) == arithmetic == 1344
        and len(inject.CASES) == 14
        and not shared_farms
        and not shared_seeds
        and len(set(tuning)) == 1344,
        detail=(
            f"tuning {len(tuning)}, scoring {len(scoring)}, plan arithmetic {arithmetic}; "
            f"shared farms {sorted(shared_farms)}, shared seeds {sorted(shared_seeds)}"
        ),
    )


def test_split_guard() -> None:
    """A scoring farm or seed is refused in the tuning split."""
    refused = 0
    for farm, seeds in (("D", inject.TUNING_SEEDS), ("A", inject.SCORING_SEEDS)):
        try:
            inject.assert_split(farm=farm, seeds=seeds, split="tuning")
        except ValueError:
            refused += 1
    report(
        name="tuning split refuses scoring farms and seeds",
        passed=refused == 2,
        detail=f"{refused} of 2 refused",
    )


def test_baseline_keeps_weekly_noise(*, inputs: inject.Inputs) -> None:
    """The baseline's smooth PV is the known capacity and its weekly projection onto `u` is not 0.

    A baseline built by subtracting a free weekly fit would pass the first half and fail the
    second: every weekly capacity would sit exactly on the known value, and the known-answer test
    would hold none of the weekly noise that it exists to measure.
    """
    smooth_worst, noise_floor = 0.0, np.inf
    detail = []
    for site in SITES:
        base = inject.primary_base(inputs=inputs, site=site)
        for farm in (None, "B"):
            rows = (
                base.table
                if farm is None
                else inject.unit_rows(base=base, inputs=inputs, farm=farm)
            )
            design = model.design_from_rows_table(
                table=rows.with_columns(power=pl.col("baseline_power"))
            )
            if farm is None:
                smooth = model.fit_season_year_constant(design=design)
                smooth_worst = max(smooth_worst, float(np.nanmax(np.abs(smooth - base.capacity))))
            free = inject.unconstrained_weekly_fit(design=design)
            spread = float(np.nanmedian(np.abs(free - base.capacity)) / base.capacity)
            noise_floor = min(noise_floor, spread)
            detail.append(f"{site}{'' if farm is None else farm} {spread:.3f}")
    report(
        name="baseline keeps weekly noise around the known capacity",
        passed=smooth_worst < 1e-6 and noise_floor > 0.01,
        detail=(
            f"season-year fit of the baseline is the known capacity (largest gap "
            f"{smooth_worst:.1e} MW); median |free weekly fit - capacity| / capacity: "
            f"{', '.join(detail)}"
        ),
    )


def test_constant_path_has_no_trend(*, inputs: inject.Inputs) -> None:
    """The constant-path baseline holds a known constant capacity and has no trend."""
    worst = 0.0
    detail = []
    for site in SITES:
        base = inject.primary_base(inputs=inputs, site=site)
        farm_rows = inject.unit_rows(base=base, inputs=inputs, farm="A")
        unit = inject.Unit(primary=site, farm="A", seed=0, case="constant_path")
        rows = inject.permute_u(table=farm_rows, rng=inject._rng(unit=unit))
        power = rows["power"].to_numpy() - base.capacity * rows["u"].to_numpy()
        design = model.design_from_rows_table(table=rows.with_columns(power=pl.Series(power)))
        free = inject.unconstrained_weekly_fit(design=design)
        weeks = np.nonzero(np.isfinite(free))[0]
        slope, intercept = np.polyfit(weeks, free[weeks], 1)
        residual = free[weeks] - (slope * weeks + intercept)
        slope_se = residual.std(ddof=2) / np.sqrt(((weeks - weeks.mean()) ** 2).sum())
        z = abs(slope) / slope_se
        worst = max(worst, float(z))
        drift = slope * 52 * 5 / base.capacity
        detail.append(f"{site} slope {drift:+.3f} of capacity per 5 years, {z:.1f} standard errors")
    report(
        name="constant-path baseline has no trend",
        passed=worst < 3.0,
        detail="; ".join(detail),
    )


def _noise_free_error(*, inputs: inject.Inputs, site: str, case: inject.CaseName) -> float:
    """Inject `case` with `u` as the farm shape; return the largest increment error.

    The paired difference of two unconstrained fits is exact here because the injected shape is `u`.
    """
    base = inject.primary_base(inputs=inputs, site=site)
    rows = inject.unit_rows(base=base, inputs=inputs, farm="A").with_columns(
        farm_norm=pl.col("u") / pl.col("u").quantile(0.99)
    )
    unit = inject.Unit(primary=site, farm="A", seed=1, case=case)
    injection = inject.build_injection(unit=unit, base=base, farm_rows=rows, inputs=inputs)
    base_power = rows["baseline_power"].to_numpy()
    with_injection = base_power + injection.delta_for(rows)

    def fit(power: np.ndarray) -> np.ndarray:
        design = model.design_from_rows_table(table=rows.with_columns(power=pl.Series(power)))
        return inject.unconstrained_weekly_fit(design=design)

    c1, c0 = fit(with_injection), fit(base_power)
    observed = inject.observed_weeks(table=rows)
    increment = inject.estimate_path(
        with_injection=c1,
        baseline=c0,
        ratio=1.0,
        variant="real_plus_injection",
        base_capacity=base.capacity,
    )
    return float(np.abs(increment[observed] - injection.increment[observed]).max())


def test_noise_free_recovery(*, inputs: inject.Inputs) -> None:
    """With the farm shape equal to `u`, the free fit returns the injected path to 1e-6 MW."""
    cases: tuple[inject.CaseName, ...] = ("ramp_50", "step_30", "fall_30_26", "ratio_constant_farm")
    worst = {
        case: max(_noise_free_error(inputs=inputs, site=s, case=case) for s in SITES)
        for case in cases
    }
    report(
        name="noise-free known path in, recovered path out (free fit, farm shape = u)",
        passed=max(worst.values()) < 1e-6,
        detail=", ".join(f"{case} {err:.1e} MW" for case, err in worst.items()),
    )


def test_farm_gaps(*, inputs: inject.Inputs) -> None:
    """No row of a dropped farm-week survives, and every kept week has 3 usable days."""
    base = inject.primary_base(inputs=inputs, site="D16")
    problems = 0
    kept_weeks = 0
    for farm in inject.TUNING_FARMS:
        rows = inject.unit_rows(base=base, inputs=inputs, farm=farm)
        dropped = set(
            inputs.farm_weeks.filter((pl.col("farm") == farm) & pl.col("dropped"))["week"].to_list()
        )
        weeks = set(rows["week"].unique().to_list())
        problems += len(weeks & dropped)
        per_week = (
            rows.with_columns(day=common.day_expr(time=pl.col("time")))
            .group_by("week", "day")
            .agg(n=pl.len())
            .filter(pl.col("n") >= common.MIN_USABLE_HALF_HOURS_PER_DAY)
            .group_by("week")
            .agg(days=pl.len())
        )
        problems += int((per_week["days"] < 3).sum())
        kept_weeks += len(weeks)
    report(
        name="farm-week gaps leave no rows and no thin weeks",
        passed=problems == 0 and kept_weeks > 0,
        detail=f"{problems} offending weeks in {kept_weeks} kept farm-weeks (3 farms, Donington)",
    )


def test_transfers(*, inputs: inject.Inputs) -> None:
    """A transfer outside the mask has a true increment; one inside is masked and has none."""
    base = inject.primary_base(inputs=inputs, site="D10")
    rows = inject.unit_rows(base=base, inputs=inputs, farm="A")
    outside = inject.build_injection(
        unit=inject.Unit(primary="D10", farm="A", seed=2, case="transfer_outside_mask"),
        base=base,
        farm_rows=rows,
        inputs=inputs,
    )
    inside = inject.build_injection(
        unit=inject.Unit(primary="D10", farm="A", seed=2, case="transfer_inside_mask"),
        base=base,
        farm_rows=rows,
        inputs=inputs,
    )
    donor = inject.donor_of(primary="D10")
    expected_weeks = inject.TRANSFER_DAYS / 7
    total = float(outside.increment.sum() / (inject.TRANSFER_SHARE * inputs.base_capacity[donor]))
    window = inside.masked_window
    assert window is not None
    masked_rows = inject.unit_rows(base=base, inputs=inputs, farm="A", masked_window=window)
    hit = rows.filter((pl.col("time") > window[0]) & (pl.col("time") <= window[1]))
    left = masked_rows.filter((pl.col("time") > window[0]) & (pl.col("time") <= window[1]))
    delta = outside.delta_for(rows)
    report(
        name="transfers: true increment outside the mask, none inside",
        passed=abs(total - expected_weeks) < 1e-9
        and not inside.increment.any()
        and window is not None
        and outside.masked_window is None
        and hit.height > 0
        and left.height == 0
        and delta.max() > 0,
        detail=(
            f"outside increment sums to {total:.3f} donor-capacity weeks "
            f"(30 days = {expected_weeks:.3f}); "
            f"inside: {hit.height} rows in the window before masking, {left.height} after"
        ),
    )


def test_steps_and_seeds(*, inputs: inject.Inputs) -> None:
    """Step seeds alternate seasons, one seed in four sits on a month boundary, a seed repeats."""
    base = inject.primary_base(inputs=inputs, site="D16")
    rows = inject.unit_rows(base=base, inputs=inputs, farm="B")
    seasons, boundaries = [], 0
    for seed in inject.TUNING_SEEDS:
        unit = inject.Unit(primary="D16", farm="B", seed=seed, case="step_20")
        inc = inject.build_injection(unit=unit, base=base, farm_rows=rows, inputs=inputs).increment
        week = int(np.argmax(inc > 0))
        seasons.append(week_is_summer := common.week_start(week=week).month in common.SUMMER_MONTHS)
        boundaries += common.week_start(week=week).day <= 7 and week_is_summer in (True, False)
        assert np.isclose(inc.max(), 0.20 * base.capacity)
    again = inject.build_injection(
        unit=inject.Unit(primary="D16", farm="B", seed=0, case="step_20"),
        base=base,
        farm_rows=rows,
        inputs=inputs,
    ).increment
    first = inject.build_injection(
        unit=inject.Unit(primary="D16", farm="B", seed=0, case="step_20"),
        base=base,
        farm_rows=rows,
        inputs=inputs,
    ).increment
    other = inject.build_injection(
        unit=inject.Unit(primary="D16", farm="B", seed=1, case="step_20"),
        base=base,
        farm_rows=rows,
        inputs=inputs,
    ).increment
    report(
        name="step placement: seasons alternate, a seed repeats, seeds differ",
        passed=seasons == [True, False, True, False]
        and np.array_equal(first, again)
        and not np.array_equal(first, other),
        detail=f"summer flags for seeds 0 to 3: {seasons}; step sizes equal 20% of capacity",
    )


def test_seasonal_load(*, inputs: inject.Inputs) -> None:
    """The seasonal load peaks near 10% of capacity at midsummer midday and is zero in winter."""
    base = inject.primary_base(inputs=inputs, site="D16")
    rows = inject.unit_rows(base=base, inputs=inputs, farm="A")
    unit = inject.Unit(primary="D16", farm="B", seed=0, case="seasonal_load_independent")
    inj = inject.build_injection(unit=unit, base=base, farm_rows=rows, inputs=inputs)
    sun = inject.build_injection(
        unit=inject.Unit(primary="D16", farm="B", seed=0, case="seasonal_load_sunshine"),
        base=base,
        farm_rows=rows,
        inputs=inputs,
    )
    delta, delta_sun = inj.delta_for(rows), sun.delta_for(rows)
    month = rows["time"].dt.month().to_numpy()
    peak = delta[(month == 6)].max() / base.capacity
    winter = np.abs(delta[np.isin(month, (11, 12, 1))]).max() / base.capacity
    report(
        name="seasonal load: weather-independent and sunshine-scaled",
        passed=bool(
            0.09 < peak <= 0.1 + 1e-9
            and winter < 0.02
            and not inj.increment.any()
            and np.all(delta_sun <= 1.2 * delta + 1e-12)
            and not np.allclose(delta, delta_sun)
        ),
        detail=f"June peak {peak:.3f} of capacity, winter peak {winter:.4f}; no true increment",
    )


def test_placebo(*, inputs: inject.Inputs) -> None:
    """The permutation keeps power and each month's set of days, and changes `u`."""
    table = model.usable_rows_table(half_hourly=inputs.half_hourly, weeks=inputs.weeks, site="D16")
    permuted = inject.permute_u(table=table, rng=np.random.default_rng([0, 45]))
    joined = table.join(permuted, on="time", suffix="_perm")
    same_power = bool((joined["power"] == joined["power_perm"]).all())
    moved = float(np.mean(joined["u"].to_numpy() != joined["u_perm"].to_numpy()))
    kept = permuted.height / table.height
    report(
        name="placebo permutation leaves power in place and moves u",
        passed=same_power and moved > 0.9 and kept > 0.9,
        detail=f"{moved:.2f} of rows have a new u; {kept:.3f} of rows kept (donor day had the row)",
    )


def test_metrics() -> None:
    """Toy paths with known answers for the step count, the fall share, the drift, and the error."""
    path = np.full(100, 4.0)
    path[30:] += 1.0
    path[60:] += 0.6
    steps = metrics.count_steps(path=path, threshold=0.5)
    ramp = np.r_[np.full(20, 4.0), np.linspace(4.0, 5.0, 10), np.full(40, 5.0)]
    ramp_steps = metrics.count_steps(path=ramp, threshold=0.4)
    exempt = metrics.rise_weeks(path=path, threshold=0.5)
    truth = np.full(60, 4.0)
    truth[20:40] = 2.8
    estimate = np.full(60, 4.0)
    estimate[20:40] = 3.4
    weeks = np.ones(60, dtype=bool)
    share = metrics.fall_recovered(
        estimate=estimate, truth=truth, weeks=weeks, start=20, duration=20
    )
    drift_path = np.linspace(4.0, 4.0 + 5 * 0.2, 5 * 52)
    drift = metrics.ratchet_drift_per_5_years(path=drift_path, reference=4.0)
    error = metrics.path_error(estimate=truth * 1.1, truth=truth, weeks=weeks)
    report(
        name="metrics on toy paths",
        passed=steps == 2
        and ramp_steps == 2
        and exempt[28]
        and not exempt[5]
        and abs(share - 0.5) < 1e-12
        and abs(drift - 0.25) < 0.01
        and abs(error - 0.1) < 1e-12,
        detail=(
            f"steps {steps} (2); a 1.0 MW ramp over 10 weeks counted {ramp_steps} (2, one per "
            f"4-week stretch); fall share {share:.3f} (0.5); drift {drift:.3f} (0.25); "
            f"error {error:.3f} (0.1)"
        ),
    )


def main() -> None:
    """Run every self-test."""
    inputs = inject.load_inputs()
    test_unit_counts()
    test_split_guard()
    test_baseline_keeps_weekly_noise(inputs=inputs)
    test_noise_free_recovery(inputs=inputs)
    test_constant_path_has_no_trend(inputs=inputs)
    test_farm_gaps(inputs=inputs)
    test_transfers(inputs=inputs)
    test_steps_and_seeds(inputs=inputs)
    test_seasonal_load(inputs=inputs)
    test_placebo(inputs=inputs)
    test_metrics()
    print("ALL PASS" if not _failures else f"FAILED: {_failures}")
    sys.exit(1 if _failures else 0)


if __name__ == "__main__":
    main()
