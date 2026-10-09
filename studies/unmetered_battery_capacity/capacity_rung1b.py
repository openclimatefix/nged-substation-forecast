"""Rung 1b: simulated merchant batteries dispatched by policies the estimator does not contain.

Rung 1's truth is the day-by-day linear programme on the N2EX price, which is the family the
estimator interpolates, so its detection and calibration are a best case. The first science review
(M1) asked for the same 9 series, 4 blocks, 3 durations, and 7 shares with truths from outside
that family. Two policies are run:

- `rank_rule`: `studies.battery_dispatch.rank_rule_schedule` on the N2EX price. Each day charges at
  full power in its cheapest half-hours and discharges in its dearest, with no efficiency, limits,
  or cap, and no energy balance. The nameplate duration is the usable duration.
- `noisy_price`: the linear programme on the N2EX price multiplied by `1 + e` with `e` independent
  and normal with standard deviation 0.2 at every half-hour, a stand-in for a day-ahead forecast
  error, with the physical parameters drawn as in rung 1.

The thresholds come from rung 1's 36 blocks with no added battery, so these rungs fit batteries
only. Writes `rung1b_<policy>_posteriors.parquet`.

Run: `OMP_NUM_THREADS=2 uv run python studies/unmetered_battery_capacity/capacity_rung1b.py`
"""

from typing import Final, Literal

import numpy as np
import polars as pl
from capacity_inputs import OUTPUT_DIR, day_ahead_on_grid, demand_series
from capacity_runs import (
    NAMEPLATE_HOURS,
    ONE_WAY_EFFICIENCY_RANGE,
    SHARES,
    SOC_MAX_RANGE,
    SOC_MIN_RANGE,
    fit_all_blocks,
    p99_flow,
)
from studies.battery_dispatch import lp_schedule, rank_rule_schedule

PolicyType = Literal["rank_rule", "noisy_price"]
POLICIES: Final[tuple[PolicyType, ...]] = ("rank_rule", "noisy_price")
SEED: Final[int] = 20261030
PRICE_NOISE_SD: Final[float] = 0.2
HALF_HOURS_PER_HOUR: Final[int] = 2


def policy_battery(
    *, policy: PolicyType, nameplate_hours: float, seed: int
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Return a one-megawatt schedule from a policy outside the estimator's family.

    Args:
        policy: Which policy dispatches the battery.
        nameplate_hours: The energy capacity in hours at full power.
        seed: Seeds the draws.

    Returns:
        The schedule (positive for export), the usable duration in hours at full power, and the
        drawn parameters (empty for the rank rule).
    """
    if policy == "rank_rule":
        schedule = rank_rule_schedule(
            prices=day_ahead_on_grid(),
            duration_half_hours=round(nameplate_hours * HALF_HOURS_PER_HOUR),
        )
        return schedule, nameplate_hours, {}
    rng = np.random.default_rng(seed)
    eta_one_way = rng.uniform(*ONE_WAY_EFFICIENCY_RANGE)
    soc_min, soc_max = rng.uniform(*SOC_MIN_RANGE), rng.uniform(*SOC_MAX_RANGE)
    cap = float(rng.choice([1.0, 2.0]))
    prices = day_ahead_on_grid()
    noisy = prices * (1.0 + rng.normal(0.0, PRICE_NOISE_SD, size=prices.shape))
    schedule = lp_schedule(
        prices=noisy,
        energy_hours=nameplate_hours,
        eta_one_way=eta_one_way,
        soc_min=soc_min,
        soc_max=soc_max,
        cycles_per_day_cap=cap,
    )
    drawn = {
        "soc_min": soc_min,
        "soc_max": soc_max,
        "round_trip": eta_one_way**2,
        "cycles_cap": cap,
    }
    return schedule, (soc_max - soc_min) * nameplate_hours, drawn


def build(*, policy: PolicyType) -> tuple[np.ndarray, list[list[dict]]]:
    """Build one policy's aggregates and identifiers.

    Args:
        policy: The dispatch policy.

    Returns:
        Aggregates of shape (series, batteries x shares, 17,520) and the identifiers and truth of
        each.
    """
    aggregates, metadata = [], []
    for g, (label, demand) in enumerate(demand_series().items()):
        p99 = p99_flow(demand)
        lanes, meta = [], []
        for h, hours in enumerate(NAMEPLATE_HOURS):
            unit, usable, drawn = policy_battery(
                policy=policy, nameplate_hours=hours, seed=SEED + 100 * g + h
            )
            for share in SHARES:
                lanes.append(demand - share * p99 * unit)
                meta.append(
                    {
                        "rung": f"rung1b_{policy}",
                        "series": label,
                        "nameplate_hours": hours,
                        "share": share,
                        "true_power_mw": share * p99,
                        "true_usable_hours": usable,
                        "true_energy_mwh": share * p99 * usable,
                        **{f"truth_{k}": v for k, v in drawn.items()},
                    }
                )
        aggregates.append(np.stack(lanes))
        metadata.append(meta)
    return np.stack(aggregates), metadata


def main() -> None:
    """Fit both policies and save the posteriors."""
    for policy in POLICIES:
        aggregates, metadata = build(policy=policy)
        posteriors = fit_all_blocks(aggregates=aggregates, metadata=metadata, label=policy)
        posteriors.write_parquet(OUTPUT_DIR / f"rung1b_{policy}_posteriors.parquet")
        print(
            policy,
            posteriors.group_by("share").agg(pl.col("log_bayes_factor").median()).sort("share"),
        )


if __name__ == "__main__":
    main()
