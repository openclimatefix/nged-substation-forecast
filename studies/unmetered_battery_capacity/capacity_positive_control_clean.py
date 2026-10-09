"""The clean positive control for the differentiable battery estimator, pass rule committed first.

The first positive control (`capacity_positive_control_dp.py`) passed at 2 of 3 blocks after the
estimator had been refined using diagnostics from a scored block, with one fixed off-grid truth
that lay 9% and 11% of a node spacing from the fine stack's nodes. The first science review
(`s2_science1.md`, M2) therefore asked for a control that no tuning has touched. This script is
that control, and nothing in the estimator, the stacks, or the priors changes before it runs.

**Design.**

- Series: the calendar replicas of S1, S3, S5, S7, S8, BSP2, and GSP1, the seven demand-like series
  that no earlier diagnostic touched (S2 and S6 were the tuning series, S4 and GSP2 are excluded
  from the study, and BSP1 holds NGED battery A).
- Truths: 10 fresh draws (a seed no earlier script used) of a 2-hour-nameplate merchant battery
  dispatched by the day-by-day linear programme on the N2EX price. Each draw takes the one-way
  efficiency from 0.88 to 0.95, the state-of-charge limits from 0% to 10% and 90% to 100%, and a cap
  of 1 or 2 cycles a day. A draw is rejected when its usable duration or its round-trip efficiency
  lies within 25% of a node spacing of a node of the fine stack (`DURATION_NODES`,
  `EFFICIENCY_NODES`), so every accepted truth sits at least a quarter of a spacing from the nearest
  node in both dimensions. Each draw's fractional position inside its cell is recorded.
- Share: 40% of the real series' 99th percentile absolute flow, in all four blocks.
- Fits: 10 truths x 7 series x 4 blocks = 280 fits on the calendar replicas, standard setting.
  The same truths are also added to the real demand of the same series, and its coverage is
  reported (280 further fits), because the replica's integrated autocorrelation time (1 to 4) is
  not the real demand's (about 49).

**Pass rule, committed before the first run and not changed afterwards.** The control passes if the
90% credible interval holds both the true merchant power and the true merchant energy in at least
two thirds of the 280 replica fits (at least 187). The medians are reported, with their relative
error, and do not enter the pass rule. The real-demand fits are reported and do not enter the pass
rule. If the control fails, the finding is that the estimator's intervals do not cover an off-node
truth, and it is reported as such.

Run: `uv run python studies/unmetered_battery_capacity/capacity_positive_control_clean.py`
Writes `positive_control_clean_draws.parquet` and `report_positive_control_clean.md`.
"""

import math
import time
from typing import Final

import numpy as np
import polars as pl
from capacity_inputs import (
    BLOCK_NAMES,
    OUTPUT_DIR,
    day_ahead_on_grid,
    nged_series,
    window_half_hours,
)
from capacity_runs import p99_flow
from capacity_stacks import DURATION_NODES, EFFICIENCY_NODES
from capacity_state_space import (
    ADAM_STAGES,
    STARTS,
    block_problems,
    estimator,
    posterior_summary,
    start_parameters,
)
from studies.battery_capacity import calendar_replica
from studies.battery_dispatch import lp_schedule

SERIES: Final[tuple[str, ...]] = ("S1", "S3", "S5", "S7", "S8", "BSP2", "GSP1")
N_TRUTHS: Final[int] = 10
SEED: Final[int] = 20261021
SHARE: Final[float] = 0.40
NAMEPLATE_HOURS: Final[float] = 2.0
MIN_DISTANCE_FROM_NODE: Final[float] = 0.25
"""The smallest accepted distance from a node, as a fraction of the node spacing."""
PASS_FRACTION: Final[float] = 2.0 / 3.0
MAX_DRAW_ATTEMPTS: Final[int] = 10_000


def cell_position(*, usable_hours: float, round_trip: float) -> tuple[float, float]:
    """Return a truth's fractional position inside its fine-stack cell, in duration and efficiency.

    A position of 0 or 1 is a node and 0.5 is the middle of the cell.

    Args:
        usable_hours: The usable duration in hours at full power.
        round_trip: The round-trip efficiency.

    Returns:
        The positions in duration (in log hours, since the nodes are log-spaced) and efficiency.
    """
    log_nodes = np.log(DURATION_NODES)
    duration = (math.log(usable_hours) - log_nodes[0]) / (log_nodes[1] - log_nodes[0])
    efficiency = (round_trip - EFFICIENCY_NODES[0]) / (EFFICIENCY_NODES[1] - EFFICIENCY_NODES[0])
    return float(duration % 1.0), float(efficiency % 1.0)


def distance_from_node(*, position: float) -> float:
    """Return how far a cell position is from the nearest node, as a fraction of the spacing."""
    return min(position, 1.0 - position)


def draw_truths(*, n_truths: int, seed: int) -> list[dict[str, float]]:
    """Draw off-node truths by rejection.

    Args:
        n_truths: The number of accepted draws.
        seed: Seeds the draws.

    Returns:
        For each accepted draw, `soc_min`, `soc_max`, `eta_one_way`, `cycles_cap`, `usable_hours`,
        `round_trip`, and the cell positions `duration_position` and `efficiency_position`.

    Raises:
        RuntimeError: If too many draws are rejected.
    """
    rng = np.random.default_rng(seed)
    accepted: list[dict[str, float]] = []
    for _ in range(MAX_DRAW_ATTEMPTS):
        soc_min, soc_max = rng.uniform(0.0, 0.10), rng.uniform(0.90, 1.0)
        eta_one_way, cap = rng.uniform(0.88, 0.95), float(rng.choice([1.0, 2.0]))
        usable = (soc_max - soc_min) * NAMEPLATE_HOURS
        round_trip = eta_one_way**2
        d_position, e_position = cell_position(usable_hours=usable, round_trip=round_trip)
        if min(distance_from_node(position=d_position), distance_from_node(position=e_position)) < (
            MIN_DISTANCE_FROM_NODE
        ):
            continue
        accepted.append(
            {
                "soc_min": soc_min,
                "soc_max": soc_max,
                "eta_one_way": eta_one_way,
                "cycles_cap": cap,
                "usable_hours": usable,
                "round_trip": round_trip,
                "duration_position": d_position,
                "efficiency_position": e_position,
            }
        )
        if len(accepted) == n_truths:
            return accepted
    raise RuntimeError(f"only {len(accepted)} of {n_truths} truths accepted")


def schedule_for(*, truth: dict[str, float]) -> np.ndarray:
    """Return a truth's one-megawatt schedule on the N2EX price."""
    return lp_schedule(
        prices=day_ahead_on_grid(),
        energy_hours=NAMEPLATE_HOURS,
        eta_one_way=truth["eta_one_way"],
        soc_min=truth["soc_min"],
        soc_max=truth["soc_max"],
        cycles_per_day_cap=truth["cycles_cap"],
    )


def run(*, demand_kind: str) -> pl.DataFrame:
    """Fit every truth on every series in every block, on replicas or on real demand.

    Args:
        demand_kind: `replica` or `real`.

    Returns:
        One row per series, truth, and block.
    """
    nged = nged_series()
    truths = draw_truths(n_truths=N_TRUTHS, seed=SEED)
    schedules = [schedule_for(truth=t) for t in truths]
    aggregates = np.zeros((len(SERIES), len(truths), len(window_half_hours())))
    powers = np.zeros(len(SERIES))
    for g, label in enumerate(SERIES):
        base = (
            calendar_replica(output=nged[label], half_hour_end_time=window_half_hours())
            if demand_kind == "replica"
            else nged[label]
        )
        powers[g] = SHARE * p99_flow(nged[label])
        for k, schedule in enumerate(schedules):
            aggregates[g, k] = base - powers[g] * schedule
    rows = []
    for block, name in enumerate(BLOCK_NAMES):
        started = time.monotonic()
        fit = estimator(setting="standard", block=block).fit(
            problems=block_problems(block=block, aggregates=aggregates),
            starts=start_parameters(),
            stages=ADAM_STAGES,
        )
        seconds = time.monotonic() - started
        print(
            f"{demand_kind} block {name}: {seconds:.0f} s for "
            f"{len(SERIES) * len(truths) * STARTS} fits",
            flush=True,
        )
        for g, label in enumerate(SERIES):
            for k, truth in enumerate(truths):
                summary = posterior_summary(
                    fit=fit, group=g, lane=k, rng=np.random.default_rng(SEED + 100 * block + g)
                )
                true_power = float(powers[g])
                true_energy = true_power * truth["usable_hours"]
                row = {
                    "demand": demand_kind,
                    "series": label,
                    "truth": k,
                    "block": name,
                    "true_power_mw": true_power,
                    "true_energy_mwh": true_energy,
                    **{f"truth_{key}": value for key, value in truth.items()},
                    **summary,
                }
                for quantity, true_value in (("power", true_power), ("energy", true_energy)):
                    has = bool(summary["has_interval"])
                    row[f"{quantity}_in_90"] = bool(
                        has
                        and summary[f"merchant_{quantity}_q05"]
                        <= true_value
                        <= summary[f"merchant_{quantity}_q95"]
                    )
                    median = summary.get(
                        f"merchant_{quantity}_median", summary[f"merchant_{quantity}_point"]
                    )
                    row[f"{quantity}_median_error"] = median / true_value - 1.0
                    row[f"{quantity}_width_over_truth"] = (
                        (summary[f"merchant_{quantity}_q95"] - summary[f"merchant_{quantity}_q05"])
                        / true_value
                        if has
                        else float("nan")
                    )
                rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def summarise(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return the control's headline numbers for one set of fits."""
    both = (frame["power_in_90"] & frame["energy_in_90"]).to_numpy()
    power_error = frame["power_median_error"].to_numpy()
    energy_error = frame["energy_median_error"].to_numpy()
    return {
        "fits": float(frame.height),
        "both_in_90": float(both.sum()),
        "both_rate": float(both.mean()),
        "power_in_90_rate": float(frame["power_in_90"].to_numpy().mean()),
        "energy_in_90_rate": float(frame["energy_in_90"].to_numpy().mean()),
        "median_power_error": float(np.median(power_error)),
        "median_abs_power_error": float(np.median(np.abs(power_error))),
        "median_abs_energy_error": float(np.median(np.abs(energy_error))),
        "without_interval": float((~frame["has_interval"].to_numpy()).sum()),
    }


def report_lines(*, replica: pl.DataFrame, real: pl.DataFrame) -> list[str]:
    """Return the report text."""
    result = summarise(frame=replica)
    needed = math.ceil(PASS_FRACTION * replica.height)
    passed = result["both_in_90"] >= needed
    lines = [
        "# Clean positive control",
        "",
        (
            f"**Pass rule (committed before the run): {'PASS' if passed else 'FAIL'}.** On the "
            f"calendar replicas of {len(SERIES)} untouched series, {N_TRUTHS} off-node truths, "
            f"4 blocks ({replica.height} fits), the 90% interval holds both the merchant power and "
            f"the merchant energy in {int(result['both_in_90'])} fits "
            f"({result['both_rate']:.1%}); at least {needed} (two thirds) were required."
        ),
        "",
        (
            "This control supersedes the earlier pass of `capacity_positive_control_dp.py`, which "
            "passed 2 of 3 scored blocks with a single truth after the estimator had been refined "
            "using diagnostics from a scored block, and whose truth lay 9% and 11% of a node "
            "spacing from the nodes of the fine stack."
        ),
        "",
        "## The truths",
        "",
        (
            f"Each truth sits at least {MIN_DISTANCE_FROM_NODE:.0%} of a node spacing from the "
            "nearest node of the fine stack in both dimensions (position 0 or 1 is a node, 0.5 "
            "the middle of a cell)."
        ),
        "",
        replica.group_by("truth")
        .agg(
            usable_hours=pl.col("truth_usable_hours").first(),
            round_trip=pl.col("truth_round_trip").first(),
            cycles_cap=pl.col("truth_cycles_cap").first(),
            duration_position=pl.col("truth_duration_position").first(),
            efficiency_position=pl.col("truth_efficiency_position").first(),
        )
        .sort("truth")
        .with_columns(pl.col(pl.Float64).round(4))
        .write_csv(separator="|"),
        "## Coverage and error",
        "",
    ]
    for name, frame in (("Calendar replicas (the pass rule)", replica), ("Real demand", real)):
        s = summarise(frame=frame)
        lines += [
            (
                f"**{name}: {int(s['fits'])} fits.** The 90% interval holds both "
                f"power and energy in {int(s['both_in_90'])} ({s['both_rate']:.1%}), power in "
                f"{s['power_in_90_rate']:.1%}, and energy in "
                f"{s['energy_in_90_rate']:.1%}. The median relative power error is "
                f"{s['median_power_error']:+.2%} (median absolute "
                f"{s['median_abs_power_error']:.2%}); the median absolute energy error is "
                f"{s['median_abs_energy_error']:.2%}. {int(s['without_interval'])} fits have no "
                "interval."
            ),
            "",
            "By block:",
            "",
            frame.group_by("block")
            .agg(
                n=pl.len(),
                both_in_90=(pl.col("power_in_90") & pl.col("energy_in_90")).mean(),
                power_in_90=pl.col("power_in_90").mean(),
                energy_in_90=pl.col("energy_in_90").mean(),
                median_power_error=pl.col("power_median_error").median(),
                median_tau=pl.col("tau").median(),
                median_power_width_over_truth=pl.col("power_width_over_truth").median(),
            )
            .with_columns(pl.col(pl.Float64).round(4))
            .write_csv(separator="|"),
            "By series:",
            "",
            frame.group_by("series")
            .agg(
                n=pl.len(),
                both_in_90=(pl.col("power_in_90") & pl.col("energy_in_90")).mean(),
                median_power_error=pl.col("power_median_error").median(),
            )
            .sort("series")
            .with_columns(pl.col(pl.Float64).round(4))
            .write_csv(separator="|"),
            "By truth:",
            "",
            frame.group_by("truth")
            .agg(
                n=pl.len(),
                both_in_90=(pl.col("power_in_90") & pl.col("energy_in_90")).mean(),
                median_power_error=pl.col("power_median_error").median(),
            )
            .sort("truth")
            .with_columns(pl.col(pl.Float64).round(4))
            .write_csv(separator="|"),
        ]
    return lines


def main() -> None:
    """Run the clean control and write its report."""
    replica = run(demand_kind="replica")
    real = run(demand_kind="real")
    pl.concat([replica, real], how="diagonal_relaxed").write_parquet(
        OUTPUT_DIR / "positive_control_clean_draws.parquet"
    )
    lines = report_lines(replica=replica, real=real)
    (OUTPUT_DIR / "report_positive_control_clean.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
