"""Rung 3 on calendar replicas: real public batteries added to a demand series with almost no noise.

Rung 3 added 23 real public batteries and fleets to real demand and found the posterior barely
moved. Two explanations compete: real demand noise hides the battery, or real dispatch lies outside
the estimator's family. The first science review (S3) asked for the same units on the calendar
replicas of the same nine series (a monthly mean profile with almost no noise). With the noise
nearly gone, a posterior that still does not find the battery points at the dispatch, not at the
noise.

The units are rung 3's 23 (`capacity_rung3.units`), at shares of 10% and 40% of the real series'
99th percentile absolute flow, plus one lane with no battery per series: 9 series x (23 x 2 + 1)
lanes x 4 blocks. Writes `rung3_replica_posteriors.parquet`.

Run: `OMP_NUM_THREADS=2 uv run python studies/unmetered_battery_capacity/capacity_rung3_replica.py`
"""

from typing import Final

import numpy as np
from capacity_inputs import OUTPUT_DIR, demand_series, window_half_hours
from capacity_rung3 import units
from capacity_runs import fit_all_blocks, p99_flow
from studies.battery_capacity import calendar_replica

SHARES_REPLICA: Final[tuple[float, ...]] = (0.10, 0.40)


def build() -> tuple[np.ndarray, list[list[dict]]]:
    """Build the aggregates and identifiers.

    Returns:
        Aggregates of shape (series, lanes, 17,520) and the identifiers and truth of each lane.
    """
    all_units = units()
    aggregates, metadata = [], []
    for label, demand in demand_series().items():
        p99 = p99_flow(demand)
        replica = calendar_replica(output=demand, half_hour_end_time=window_half_hours())
        lanes = [replica]
        meta = [
            {
                "rung": "rung3_replica",
                "series": label,
                "unit": "none",
                "kind": "none",
                "share": 0.0,
                "true_power_mw": 0.0,
                "coincident_peak_mw": 0.0,
            }
        ]
        for unit in all_units:
            for share in SHARES_REPLICA:
                scale = share * p99 / unit["registered_mw"]
                lanes.append(replica - np.nan_to_num(unit["output_mw"]) * scale)
                meta.append(
                    {
                        "rung": "rung3_replica",
                        "series": label,
                        "unit": unit["unit"],
                        "kind": unit["kind"],
                        "share": share,
                        "true_power_mw": share * p99,
                        "coincident_peak_mw": scale * unit["metered_p99_mw"],
                    }
                )
        aggregates.append(np.stack(lanes))
        metadata.append(meta)
    return np.stack(aggregates), metadata


def main() -> None:
    """Fit the replicas and save the posteriors."""
    aggregates, metadata = build()
    fit_all_blocks(aggregates=aggregates, metadata=metadata, label="rung 3 replica").write_parquet(
        OUTPUT_DIR / "rung3_replica_posteriors.parquet"
    )


if __name__ == "__main__":
    main()
