"""Rung 1: a simulated merchant battery with an exact answer, on real demand.

For each of the nine demand-like series, three merchant batteries (nameplate durations of 1, 2, and
4 hours) are dispatched by the day-by-day linear programme on the N2EX price with physical
parameters drawn off the estimator's grid: the one-way efficiency uniform on 0.88 to 0.95, the
state-of-charge limits on 0% to 10% and 90% to 100%, and a cap of 1 or 2 cycles a day. Each is
subtracted from its series at shares of 0.5%, 1%, 2%, 5%, 10%, 20%, and 40% of the series' 99th
percentile absolute flow. The null of each series is the same aggregate with no battery.

Rung 1 is 9 series x 4 blocks x 3 durations x 7 shares = 756 sums, plus 36 blocks with no added
battery. Writes `rung1_posteriors.parquet` and `rung1_step_tail.parquet`.

Run: `OMP_NUM_THREADS=2 uv run python studies/unmetered_battery_capacity/capacity_rung1.py`
"""

import numpy as np
import polars as pl
from capacity_inputs import OUTPUT_DIR, demand_series
from capacity_runs import (
    NAMEPLATE_HOURS,
    SHARES,
    fit_all_blocks,
    p99_flow,
    simulated_merchant_battery,
    step_tail_table,
)

SEED = 20261009


def build() -> tuple[np.ndarray, list[list[dict]]]:
    """Build the rung's aggregates and their identifiers.

    Returns:
        Aggregates of shape (series, lanes, 17,520) and, for each, the identifiers and truth.
    """
    series = demand_series()
    aggregates, metadata = [], []
    for g, (label, demand) in enumerate(series.items()):
        p99 = p99_flow(demand)
        lanes = [demand]
        meta = [
            {
                "rung": "rung1",
                "series": label,
                "nameplate_hours": 0.0,
                "share": 0.0,
                "true_power_mw": 0.0,
                "true_usable_hours": 0.0,
                "true_energy_mwh": 0.0,
            }
        ]
        for h, hours in enumerate(NAMEPLATE_HOURS):
            unit, usable, drawn = simulated_merchant_battery(
                nameplate_hours=hours, seed=SEED + 100 * g + h
            )
            for share in SHARES:
                lanes.append(demand - share * p99 * unit)
                meta.append(
                    {
                        "rung": "rung1",
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
    """Fit rung 1 and save the posteriors and the step tails."""
    aggregates, metadata = build()
    posteriors = fit_all_blocks(aggregates=aggregates, metadata=metadata, label="rung 1")
    posteriors.write_parquet(OUTPUT_DIR / "rung1_posteriors.parquet")
    step_tail_table(aggregates=aggregates, metadata=metadata).write_parquet(
        OUTPUT_DIR / "rung1_step_tail.parquet"
    )
    print(
        posteriors.group_by("share")
        .agg(
            pl.col("log_bayes_factor").median().alias("median_log_bf"),
            pl.col("has_interval").mean().alias("share_with_interval"),
        )
        .sort("share")
    )


if __name__ == "__main__":
    main()
