"""The positive control: can the template posterior recover an easy battery?

The aggregate is the calendar replica of one demand series (a monthly mean profile with almost no
noise) minus a simulated 2-hour merchant battery at a 40% share of the real series' 99th percentile
absolute flow. The battery follows the N2EX day-ahead price through `lp_schedule`, with physical
parameters that are not on the estimator's grid: a one-way efficiency drawn from 0.88 to 0.95, the
state-of-charge limits drawn from 0% to 10% and 90% to 100%, and a cap of 1 or 2 cycles a day.

Two truths are run. The *on-grid* truth uses a template the estimator can reproduce exactly (usable
duration 2.2 hours, a one-way efficiency of sqrt(0.88), limits of 5% and 95%, a cap of 1 cycle a
day), so a failure there is a code fault. The *off-grid* truth is the plan's: its physical
parameters are drawn off the grid, so it measures what the grid's coarseness costs.

Pass rule, per block, setting, and truth: the 90% credible interval for the merchant power holds the
true power, the interval for the merchant energy holds the true usable energy, and each posterior
median lies within 10% of its truth. If any block fails, nothing else is scored.

The demand series is the primary with the fewest missing half-hours, a rule fixed before the run.

Run: `OMP_NUM_THREADS=2 uv run python \
studies/unmetered_battery_capacity/capacity_positive_control.py`
Writes `report_positive_control.md` and `positive_control_draws.parquet`.
"""

import time
from typing import Final, Literal

import numpy as np
import polars as pl
from capacity_inputs import (
    BLOCK_NAMES,
    OUTPUT_DIR,
    day_ahead_on_grid,
    nged_series,
    window_half_hours,
)
from capacity_templates import (
    DURATIONS_HOURS,
    ROUND_TRIP_EFFICIENCIES,
    SETTING_NAMES,
    SettingNameType,
    combo_grid,
    fit_block,
    load_templates,
)
from studies.battery_capacity import calendar_replica
from studies.battery_dispatch import lp_schedule
from studies.template_posterior import posterior_draws

TruthType = Literal["on_grid", "off_grid"]
TRUTHS: Final[tuple[TruthType, ...]] = ("on_grid", "off_grid")
ON_GRID_USABLE_HOURS: Final[float] = 2.2
SHARE: Final[float] = 0.40
NAMEPLATE_HOURS: Final[float] = 2.0
N_DRAWS: Final[int] = 1000
SEED: Final[int] = 20261008
MEDIAN_TOLERANCE: Final[float] = 0.10


def simulated_battery(*, truth: TruthType, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Return a unit battery's export-positive schedule and its usable duration.

    Args:
        truth: `on_grid` or `off_grid`.
        rng: Draws the off-grid physical parameters.

    Returns:
        The one-megawatt schedule on the window grid, and the usable duration in hours at full
        power.
    """
    if truth == "on_grid":
        soc_min, soc_max, eta_one_way, cap = 0.05, 0.95, float(np.sqrt(0.88)), 1.0
        nameplate = ON_GRID_USABLE_HOURS / (soc_max - soc_min)
    else:
        soc_min = rng.uniform(0.0, 0.10)
        soc_max = rng.uniform(0.90, 1.0)
        eta_one_way, cap = rng.uniform(0.88, 0.95), float(rng.choice([1.0, 2.0]))
        nameplate = NAMEPLATE_HOURS
    schedule = lp_schedule(
        prices=day_ahead_on_grid(),
        energy_hours=nameplate,
        eta_one_way=eta_one_way,
        soc_min=soc_min,
        soc_max=soc_max,
        cycles_per_day_cap=cap,
    )
    return schedule, (soc_max - soc_min) * nameplate


def run_setting(
    *, setting: SettingNameType, truth: TruthType, y_true: np.ndarray, replica: np.ndarray
) -> list[dict]:
    """Fit the positive control in every block under one setting and truth.

    Args:
        setting: The setting.
        truth: Whether the simulated battery is on or off the estimator's grid.
        y_true: The real series, for the share's scale.
        replica: The calendar replica of the series.

    Returns:
        One row per block.
    """
    rng = np.random.default_rng(SEED)
    unit, usable_hours = simulated_battery(truth=truth, rng=rng)
    power = SHARE * float(np.nanquantile(np.abs(y_true), 0.99))
    aggregate = replica - power * unit
    candidates, nuisance = load_templates(setting=setting)
    grid = combo_grid(setting=setting)
    rows = []
    for block, name in enumerate(BLOCK_NAMES):
        started = time.monotonic()
        posterior = fit_block(
            series=aggregate,
            block=block,
            candidates=candidates,
            nuisance=nuisance,
            setting=setting,
            rng=np.random.default_rng(SEED + block),
        )
        fit_seconds = time.monotonic() - started
        combos, powers = posterior_draws(
            posterior=posterior, n_draws=N_DRAWS, rng=np.random.default_rng(SEED + 10 + block)
        )
        merchant_duration_index = np.unravel_index(combos, grid.axes)[0]
        merchant_power = powers[:, 0]
        merchant_energy = merchant_power * np.array(DURATIONS_HOURS)[merchant_duration_index]
        true_energy = power * usable_hours
        row = {
            "setting": setting,
            "truth": truth,
            "block": name,
            "true_power_mw": power,
            "true_energy_mwh": true_energy,
            "fit_seconds": fit_seconds,
            "orthant_evaluations": posterior.evaluation.orthant_evaluations,
            "combinations": len(grid.log_prior),
            "log_bayes_factor": posterior.log_bayes_factor,
            "rho": posterior.rho,
            "sigma_mw": float(np.sqrt(posterior.sigma2)),
            "other_classes_median_mw": float(np.median(powers[:, 1:].sum(axis=1))),
        }
        for label, draws, true_value in (
            ("power", merchant_power, power),
            ("energy", merchant_energy, true_energy),
        ):
            q05, q25, q50, q75, q95 = np.quantile(draws, [0.05, 0.25, 0.5, 0.75, 0.95])
            row |= {
                f"{label}_q05": q05,
                f"{label}_q25": q25,
                f"{label}_median": q50,
                f"{label}_q75": q75,
                f"{label}_q95": q95,
                f"{label}_in_90": bool(q05 <= true_value <= q95),
                f"{label}_median_error": float(q50 / true_value - 1.0),
                f"{label}_pass": bool(
                    q05 <= true_value <= q95 and abs(q50 / true_value - 1.0) <= MEDIAN_TOLERANCE
                ),
            }
        rows.append(row)
        print(
            f"{setting} {truth} {name}: fit {fit_seconds:.1f} s, P {row['power_median']:.2f} vs "
            f"{power:.2f}, E {row['energy_median']:.2f} vs {true_energy:.2f}, "
            f"BF {posterior.log_bayes_factor:.1f}",
            flush=True,
        )
    return rows


SINGLE_FACTOR_VARIANTS: Final[dict[str, dict[str, float]]] = {
    "nothing off the grid (usable 2.2 h, round trip 0.88, limits 5-95%, cap 1)": {},
    "usable duration 2.0 h (between grid values 1.35 and 2.2)": {"usable": 2.0},
    "round trip 0.846 (one-way 0.92; between grid values 0.83 and 0.88)": {"one_way": 0.92},
    "limits 0% and 100%": {"soc_min": 0.0, "soc_max": 1.0},
    "cap of 2 cycles a day": {"cycles_per_day_cap": 2.0},
}
"""One physical parameter at a time moved off the on-grid truth, to attribute the off-grid loss."""


def single_factor_rows(*, y_true: np.ndarray, replica: np.ndarray) -> list[dict]:
    """Fit block 0 in the standard setting with one parameter at a time off the grid.

    Args:
        y_true: The real series, for the share's scale.
        replica: Its calendar replica.

    Returns:
        One row per variant: the ratio of the merchant power's posterior median to the truth, the
        90% interval as ratios, and the grid duration and efficiency the posterior chose.
    """
    candidates, nuisance = load_templates(setting="standard")
    grid = combo_grid(setting="standard")
    power = SHARE * float(np.nanquantile(np.abs(y_true), 0.99))
    rows = []
    for label, change in SINGLE_FACTOR_VARIANTS.items():
        soc_min, soc_max = change.get("soc_min", 0.05), change.get("soc_max", 0.95)
        usable = change.get("usable", ON_GRID_USABLE_HOURS)
        unit = lp_schedule(
            prices=day_ahead_on_grid(),
            energy_hours=usable / (soc_max - soc_min),
            eta_one_way=change.get("one_way", float(np.sqrt(0.88))),
            soc_min=soc_min,
            soc_max=soc_max,
            cycles_per_day_cap=change.get("cycles_per_day_cap", 1.0),
        )
        posterior = fit_block(
            series=replica - power * unit,
            block=0,
            candidates=candidates,
            nuisance=nuisance,
            setting="standard",
            rng=np.random.default_rng(SEED),
        )
        combos, powers = posterior_draws(
            posterior=posterior, n_draws=N_DRAWS, rng=np.random.default_rng(SEED + 1)
        )
        low, median, high = np.quantile(powers[:, 0] / power, [0.05, 0.5, 0.95])
        axes = np.unravel_index(combos, grid.axes)
        rows.append(
            {
                "variant": label,
                "power_ratio_q05": low,
                "power_ratio_median": median,
                "power_ratio_q95": high,
                "modal_merchant_duration_hours": DURATIONS_HOURS[
                    int(np.bincount(axes[0]).argmax())
                ],
                "modal_efficiency": ROUND_TRIP_EFFICIENCIES[int(np.bincount(axes[3]).argmax())],
            }
        )
    return rows


def real_block_timing() -> dict[str, float]:
    """Time the posterior of one real aggregate: a primary in one block, with no added battery.

    Returns:
        The seconds of the fit, the orthant probabilities computed, and the combinations kept.
    """
    candidates, nuisance = load_templates(setting="standard")
    label = min(
        (k for k in nged_series() if k.startswith("S")),
        key=lambda k: int(np.isnan(nged_series()[k]).sum()),
    )
    started = time.monotonic()
    posterior = fit_block(
        series=nged_series()[label],
        block=0,
        candidates=candidates,
        nuisance=nuisance,
        setting="standard",
        rng=np.random.default_rng(SEED),
    )
    return {
        "seconds": time.monotonic() - started,
        "orthant_evaluations": float(posterior.evaluation.orthant_evaluations),
        "combinations_kept": float((~posterior.evaluation.pruned).sum()),
    }


def compute_budget(*, seconds_per_sum: float) -> list[str]:
    """Return the report lines that project the cost of rungs 1 to 3 from a measured fit time.

    Args:
        seconds_per_sum: The measured time of one fit.

    Returns:
        Report lines.
    """
    sums = {"rung 1": 924, "rung 2": 308, "rung 3 (5 fleet draws)": 4048}
    cut_sums = {"rung 1": 924, "rung 2": 308, "rung 3 (3 fleet draws)": 2992}
    lines = []
    for label, counts in (("planned", sums), ("after cut 1", cut_sums)):
        total = sum(counts.values()) * 2
        hours = total * seconds_per_sum / 3600
        lines.append(
            f"- {label}: {sum(counts.values())} sums at 2 settings = {total} fits, "
            f"{hours:.1f} h on one core, {hours / 4:.1f} h on 4 workers."
        )
    return lines


def main() -> None:
    """Run the positive control under both settings and write the report."""
    nged = nged_series()
    missing = {k: int(np.isnan(v).sum()) for k, v in nged.items() if k.startswith("S")}
    label = min(missing, key=missing.__getitem__)
    y_true = nged[label]
    replica = calendar_replica(output=y_true, half_hour_end_time=window_half_hours())
    rows = []
    for setting in SETTING_NAMES:
        for truth in TRUTHS:
            rows += run_setting(setting=setting, truth=truth, y_true=y_true, replica=replica)
    timing = real_block_timing()
    frame = pl.DataFrame(rows)
    frame.write_parquet(OUTPUT_DIR / "positive_control_draws.parquet")

    def passes(truth: str, setting: str) -> bool:
        part = frame.filter((pl.col("truth") == truth) & (pl.col("setting") == setting))
        return bool(part["power_pass"].all() and part["energy_pass"].all())

    passed = passes("off_grid", "standard")
    lines = [
        "# Positive control",
        "",
        f"Demand series: {label} (fewest missing half-hours of the 8 primaries: {missing[label]}).",
        (
            f"Share {SHARE:.0%} of the series' p99 absolute flow, a {NAMEPLATE_HOURS:.0f}-hour "
            "nameplate merchant battery on the N2EX price; the battery's physical parameters are "
            "drawn off the estimator's grid."
        ),
        "",
        (
            f"**Plan's positive control (off-grid truth, standard setting): "
            f"{'PASS' if passed else 'FAIL'}.** On-grid truth, standard setting (a code check): "
            f"{'PASS' if passes('on_grid', 'standard') else 'FAIL'}. The off-grid truth under the "
            f"sensitivity setting: {'PASS' if passes('off_grid', 'sensitivity') else 'FAIL'}. The "
            "on-grid truth is built from the standard setting's limits and cap, which the "
            "sensitivity setting's templates differ from by design, so it is not a check there."
        ),
        "",
        frame.select(
            "setting", "truth", "block", "true_power_mw", "power_q05", "power_median", "power_q95",
            "power_in_90", "power_median_error", "true_energy_mwh", "energy_q05",
            "energy_median", "energy_q95", "energy_in_90", "energy_median_error",
            "other_classes_median_mw", "log_bayes_factor", "rho", "sigma_mw",
        ).write_csv(separator="|"),
        "## Which off-grid parameter costs the power estimate (block Sep-Nov, standard setting)",
        "",
        pl.DataFrame(single_factor_rows(y_true=y_true, replica=replica)).write_csv(separator="|"),
        "## Timing and the compute budget",
        "",
        (
            f"Grid: {len(combo_grid(setting='standard').log_prior)} combinations of 6 power "
            f"columns ({len(DURATIONS_HOURS)} x {len(DURATIONS_HOURS)} x {len(DURATIONS_HOURS)} "
            f"durations x {len(ROUND_TRIP_EFFICIENCIES)} efficiencies), 144 candidate columns."
        ),
        f"One real sum (a primary in one block, no added battery): {timing}.",
        "The positive control's fits prune most combinations and are faster:",
        "",
        frame.select(
            "setting", "truth", "block", "fit_seconds", "orthant_evaluations", "combinations"
        ).write_csv(separator="|"),
    ]  # fmt: skip
    lines += ["", *compute_budget(seconds_per_sum=timing["seconds"])]
    (OUTPUT_DIR / "report_positive_control.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
