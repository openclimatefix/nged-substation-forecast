"""Rung 3: real public batteries, alone and in fleets, added to real demand.

The batteries are 23 units: the four of the prior battery study (`NAMED_BATTERIES`), the four with
the smallest registered power among the census list's battery-hint BMUs that have at least 95% of
their half-hours (the rule fixed here), and 15 fleets of 2, 4, and 8 BMUs (a fixed seed, 5 draws
per size) drawn from the 98 battery-hint BMUs with a registered generation capacity above zero (3
of the 101 list none). Each unit's settled output (B1610) is scaled so that its registered
generation capacity (a fleet's sum of registered capacities) is a share of 5%, 10%, 20%, or 40% of
a demand series' 99th percentile absolute flow, and subtracted from the series.

The power truth is the registered capacity; the metered 99th percentile and, for a fleet, the
coincident peak (the 99th percentile of the fleet's summed output) are saved beside it. No public
energy capacity exists, so the energy reference is the smallest capacity that holds the unit's
state of charge in the block (the prior study's rung 3 fit, with the one-way efficiency chosen to
minimise it between 0.80 and 0.98).

Rung 3 is 23 units x 9 series x 4 blocks x 4 shares = 3,312 sums. Writes `rung3_posteriors.parquet`
and `rung3_units.parquet`.

Run: `OMP_NUM_THREADS=2 uv run python studies/unmetered_battery_capacity/capacity_rung3.py`
"""

from typing import Final

import numpy as np
import polars as pl
from capacity_inputs import (
    OUTPUT_DIR,
    block_slices,
    demand_series,
    public_battery_output,
    public_battery_registry,
)
from capacity_runs import N_BLOCKS, fit_all_blocks, p99_flow
from studies.battery_capacity import smallest_capacity

NAMED_BATTERIES: Final[tuple[str, ...]] = ("T_LKSDB-1", "E_DOLLB-1", "T_THURB-1", "T_OCHLB-1")
FLEET_SIZES: Final[tuple[int, ...]] = (2, 4, 8)
FLEET_DRAWS: Final[int] = 5
SHARES_RUNG3: Final[tuple[float, ...]] = (0.05, 0.10, 0.20, 0.40)
MIN_PRESENT: Final[float] = 0.95
ETA_GRID: Final[np.ndarray] = np.arange(0.80, 0.9801, 0.01)
SEED: Final[int] = 20261011


def energy_reference(*, output_mw: np.ndarray) -> tuple[float, float]:
    """Return the smallest holding capacity in MWh and the one-way efficiency that gives it.

    Args:
        output_mw: A block's output in MW, positive for export, NaN where missing (treated as 0).

    Returns:
        The capacity and the efficiency.
    """
    energy = np.nan_to_num(output_mw) * 0.5
    capacities = [smallest_capacity(output_mwh=energy, eta=float(e))[0] for e in ETA_GRID]
    best = int(np.argmin(capacities))
    return float(capacities[best]), float(ETA_GRID[best])


def units() -> list[dict]:
    """Return the 23 units: member BMUs, registered power, and the summed output on the grid."""
    registry = public_battery_registry().filter(pl.col("generation_capacity_mw") > 0)
    capacity = dict(zip(registry["elexon_bmu_id"], registry["generation_capacity_mw"], strict=True))
    outputs = {b: public_battery_output(bmu_id=b) for b in capacity}
    present = {b: float(np.isfinite(o).mean()) for b, o in outputs.items()}
    eligible = [b for b in capacity if present[b] >= MIN_PRESENT and b not in NAMED_BATTERIES]
    smallest = sorted(eligible, key=lambda b: (capacity[b], b))[:4]
    rng = np.random.default_rng(SEED)
    groups = [("named", (b,)) for b in NAMED_BATTERIES] + [("smallest", (b,)) for b in smallest]
    for size in FLEET_SIZES:
        for _ in range(FLEET_DRAWS):
            members = tuple(sorted(rng.choice(sorted(capacity), size=size, replace=False)))
            groups.append((f"fleet{size}", members))
    out = []
    for index, (kind, members) in enumerate(groups):
        summed = np.sum([np.nan_to_num(outputs[b]) for b in members], axis=0)
        missing = np.any([~np.isfinite(outputs[b]) for b in members], axis=0)
        out.append(
            {
                "unit": f"U{index + 1:02d}",
                "kind": kind,
                "members": list(members),
                "registered_mw": float(sum(capacity[b] for b in members)),
                "metered_p99_mw": float(np.quantile(np.abs(summed), 0.99)),
                "output_mw": np.where(missing, np.nan, summed),
            }
        )
    return out


def build() -> tuple[np.ndarray, list[list[dict]], list[dict]]:
    """Build the aggregates and their identifiers.

    Returns:
        Aggregates of shape (series, units x shares, 17,520), identifiers with truth for each, and
        a description of each unit.
    """
    series = demand_series()
    all_units = units()
    aggregates, metadata = [], []
    for label, demand in series.items():
        p99 = p99_flow(demand)
        lanes, meta = [], []
        for unit in all_units:
            for share in SHARES_RUNG3:
                scale = share * p99 / unit["registered_mw"]
                added = np.nan_to_num(unit["output_mw"]) * scale
                lanes.append(demand - added)
                reference = []
                for block in range(N_BLOCKS):
                    capacity, _ = energy_reference(
                        output_mw=unit["output_mw"][block_slices()[block]] * scale
                    )
                    reference.append(capacity)
                meta.append(
                    {
                        "rung": "rung3",
                        "series": label,
                        "unit": unit["unit"],
                        "kind": unit["kind"],
                        "n_members": len(unit["members"]),
                        "share": share,
                        "true_power_mw": share * p99,
                        "coincident_peak_mw": scale * unit["metered_p99_mw"],
                        "energy_reference_by_block_mwh": reference,
                    }
                )
        aggregates.append(np.stack(lanes))
        metadata.append(meta)
    descriptions = [
        {k: v for k, v in unit.items() if k != "output_mw"} | {"members": ";".join(unit["members"])}
        for unit in all_units
    ]
    return np.stack(aggregates), metadata, descriptions


def main() -> None:
    """Fit rung 3 and save the posteriors."""
    aggregates, metadata, descriptions = build()
    pl.DataFrame(descriptions).write_parquet(OUTPUT_DIR / "rung3_units.parquet")
    posteriors = fit_all_blocks(aggregates=aggregates, metadata=metadata, label="rung 3")
    posteriors = posteriors.with_columns(
        true_energy_reference_mwh=pl.struct("block", "energy_reference_by_block_mwh").map_elements(
            lambda s: s["energy_reference_by_block_mwh"][s["block"]], return_dtype=pl.Float64
        )
    ).drop("energy_reference_by_block_mwh")
    posteriors.write_parquet(OUTPUT_DIR / "rung3_posteriors.parquet")


if __name__ == "__main__":
    main()
