"""Rung 7c, run 1: how much of rung 7b's stable fitted solar capacity comes from the penalty terms.

Rung 7b's fitted solar capacity barely moves with the assumed battery power. The linear programme
charges 0.1 per megawatt of battery throughput and nothing for solar capacity. At a large
battery power the residual reaches zero in many half-hours, so many solutions fit equally well, and
the penalty terms settle which one the solver returns. This run refits rung 7b's model on the 25
real aggregate BMUs and their calendar replicas at `P = 50` and `200` MW while varying two costs:

- the battery throughput cost, per megawatt of charging or discharging, against 1 per megawatt of
  residual; and
- a cost per megawatt of fitted solar AC capacity, which rung 7b sets to zero.

Each solar cost also gets a `P = 0` fit, which has no battery, to be the reference for the
residual.

Writes `rung7c_costs_fits.parquet` (one row per BMU, source, power, and cost setting).

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_rung7c_costs.py`.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
import polars as pl
from battery_inputs import OUTPUT_DIR
from battery_rung7 import (
    DURATION_HOURS,
    MAX_WORKERS,
    ONE_WAY_EFFICIENCY,
    SOURCES,
    WINDOW_HALF_HOURS,
    aggregate_bmus,
)
from battery_rung7b import SOLVER_METHOD, bmu_inputs
from studies.battery_joint_lp import DEFAULT_THROUGHPUT_PENALTY, fit_joint_solar_battery

COST_POWERS_MW: Final[tuple[float, float]] = (50.0, 200.0)
"""The assumed battery powers; `P = 0` is added for each solar cost."""
SETTINGS: Final[tuple[tuple[float, float], ...]] = (
    (0.08, 0.0),
    (DEFAULT_THROUGHPUT_PENALTY, 0.0),
    (0.3, 0.0),
    (1.0, 0.0),
    (DEFAULT_THROUGHPUT_PENALTY, 0.01),
    (DEFAULT_THROUGHPUT_PENALTY, 0.05),
)
"""The (battery throughput cost, solar cost) pairs. The default throughput cost with no solar cost
is rung 7b's setting."""


def run_bmu(bmu_id: str) -> list[dict]:
    """Fit one BMU's real output and calendar replica under every cost setting.

    Args:
        bmu_id: The aggregate BMU's identifier.

    Returns:
        One row per source, assumed power, and cost setting, including the `P = 0` references.
    """
    basis, design, outputs = bmu_inputs(bmu_id=bmu_id)
    rows: list[dict] = []
    for source in SOURCES:
        aggregate = outputs[source]
        finite = np.isfinite(aggregate)
        jobs = [
            (0.0, DEFAULT_THROUGHPUT_PENALTY, solar) for solar in sorted({s for _, s in SETTINGS})
        ]
        jobs += [
            (power, throughput, solar) for power in COST_POWERS_MW for throughput, solar in SETTINGS
        ]
        for power, throughput, solar in jobs:
            fit = fit_joint_solar_battery(
                aggregate_mw=aggregate,
                solar_basis=basis,
                power_mw=power,
                energy_mwh=DURATION_HOURS * power,
                one_way_efficiency=ONE_WAY_EFFICIENCY,
                throughput_penalty=throughput,
                window_half_hours=WINDOW_HALF_HOURS,
                signed_columns=design,
                solar_weight_penalty=solar,
                method=SOLVER_METHOD,
            )
            rows.append(
                {
                    "bmu": bmu_id,
                    "source": source,
                    "power_mw": power,
                    "throughput_cost": throughput,
                    "solar_cost": solar,
                    "fitted_ac_mw": float(fit.solar_weights.sum()),
                    "residual_abs_sum_mw": float(np.abs(fit.residual_mw[finite]).sum()),
                    "share_exact_fit": float((np.abs(fit.residual_mw[finite]) < 1e-6).mean()),
                }
            )
    return rows


def main() -> None:
    """Fit all 25 BMUs and write the fits."""
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(run_bmu, aggregate_bmus()))
    fits = pl.DataFrame([r for rows in results for r in rows], infer_schema_length=None)
    fits.write_parquet(OUTPUT_DIR / "rung7c_costs_fits.parquet")
    print(
        fits.group_by("source", "power_mw", "throughput_cost", "solar_cost")
        .agg(pl.col("fitted_ac_mw").sum())
        .sort("source", "power_mw", "throughput_cost", "solar_cost")
    )


if __name__ == "__main__":
    main()
