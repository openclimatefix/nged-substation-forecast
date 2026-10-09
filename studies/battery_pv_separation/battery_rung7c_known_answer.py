"""Rung 7c, run 2: a known-answer test of rung 7b's model on sums that contain a demand-like part.

Rung 7b fitted `solar + calendar baseline + battery` to the 25 real aggregate BMUs, where the
answer is unknown. This run builds sums with a known answer and fits the same model:

    aggregate = solar half + battery half + demand-like half.

- The solar half is one of rung 4's four solar sets, scaled so that its 99th-percentile output is
  25 MW or 50 MW (0 MW for the false-alarm sums).
- The battery half is one of rung 4's four batteries, scaled so that its 99th-percentile absolute
  output is `100 MW x (1 - share)`, where `share` is the solar half's 99th percentile over 100 MW.
- The solar study found no cloud signal in 16 of the 25 aggregate BMUs. The demand-like half is
  the real output of one of the three of those 16 with the largest 99th-percentile absolute output,
  or that BMU's calendar replica, which has no weather noise at all. It is scaled so that its
  99th-percentile absolute output is 50 MW, keeping its sign. The solar study found no cloud signal
  in that BMU, so the known solar is taken to be the solar half alone.

Each sum is fitted with rung 7b's model (four fleet curves under the regional sky of the solar set,
the seasonal calendar baseline, and a battery of the assumed power, 2 hours of energy, and a
one-way efficiency of 0.92) to half-hourly levels in windows of about 4 weeks, for assumed powers
of `POWERS_MW` and the true battery power. The solar study's difference separation (fleet curves
fitted to four-hour changes, then the baseline) is fitted to the same sums for reference.

Writes `rung7c_known_answer_fits.parquet` (one row per sum and assumed power).

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_rung7c_known_answer.py`.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
import polars as pl
from battery_inputs import BATTERIES, OUTPUT_DIR
from battery_rung4 import _prepare_skies
from battery_rung6 import score_fit, true_battery_sizes
from battery_rung7 import (
    DURATION_HOURS,
    MAX_WORKERS,
    ONE_WAY_EFFICIENCY,
    WINDOW_HALF_HOURS,
)
from battery_rung7b import SOLVER_METHOD
from battery_synthetic import (
    AGGREGATE_P99_MW,
    SOLAR_SETS,
    grid_index,
    output_on_grid,
    window_half_hours,
)
from studies.battery_capacity import calendar_replica
from studies.battery_joint_lp import fit_joint_solar_battery
from studies.pv_fit import fit_plant
from studies.pv_separation import baseline_design, separate_by_differences

POWERS_MW: Final[tuple[float, ...]] = (0.0, 20.0, 50.0, 100.0, 200.0)
"""The assumed battery powers; the true power is added for each sum."""
SOLAR_P99_MW: Final[tuple[float, ...]] = (0.0, 25.0, 50.0)
"""The solar half's 99th-percentile output. Zero is the false-alarm sum."""
DEMAND_P99_MW: Final[float] = 50.0
DEMAND_BMUS: Final[tuple[str, ...]] = ("2__BEDGE001", "V__NFLEX003", "2__ABGAS000")
"""The three aggregate BMUs with the largest 99th-percentile absolute output among the 16 that have
no cloud signal (rung 7b's table at `P = 0`)."""
DEMAND_KINDS: Final[tuple[str, ...]] = ("real", "replica")


def demand_series() -> dict[tuple[str, str], np.ndarray]:
    """Return the six demand-like halves, each scaled to a 99th-percentile absolute output of 50 MW.

    Returns:
        The scaled series on the window grid, keyed by BMU and kind (`real` or `replica`).
    """
    series = {}
    for bmu_id in DEMAND_BMUS:
        real = output_on_grid(bmu_id=bmu_id)
        for kind, raw in (
            ("real", real),
            ("replica", calendar_replica(output=real, half_hour_end_time=window_half_hours())),
        ):
            series[(bmu_id, kind)] = raw * DEMAND_P99_MW / np.nanquantile(np.abs(raw), 0.99)
    return series


def run_set(set_name: str) -> list[dict]:
    """Fit every sum that uses one solar set.

    Args:
        set_name: A key of `SOLAR_SETS`.

    Returns:
        One row per battery, solar size, demand-like half, and assumed power.
    """
    members = SOLAR_SETS[set_name]
    solar_raw = np.nansum([output_on_grid(bmu_id=b) for b in members], axis=0)
    solar_raw[np.any([~np.isfinite(output_on_grid(bmu_id=b)) for b in members], axis=0)] = np.nan
    inputs = _prepare_skies(members=members, solar_raw=solar_raw)["regional"]
    index = grid_index(half_hour_end_time=inputs.sky.half_hour_end_time)
    physical = fit_plant(
        sky=inputs.sky,
        output_mw=solar_raw[index],
        orientation="free",
        usable=np.isfinite(solar_raw[index]),
    )
    if physical is None:
        raise RuntimeError(f"No reference fit for {set_name}")
    design = baseline_design(half_hour_end_time=window_half_hours(), flexibility="seasonal")
    # Rung 7b's model fitted to the solar half alone: the best that model can do on this solar half.
    alone = fit_joint_solar_battery(
        aggregate_mw=solar_raw,
        solar_basis=inputs.basis,
        power_mw=0.0,
        energy_mwh=0.0,
        window_half_hours=WINDOW_HALF_HOURS,
        signed_columns=design,
        method=SOLVER_METHOD,
    )
    fleet_alone_ac_mw = float(alone.solar_weights.sum())
    sizes = true_battery_sizes()
    demands = demand_series()
    rows: list[dict] = []
    for battery_id in BATTERIES:
        battery_raw = output_on_grid(bmu_id=battery_id)
        for solar_p99 in SOLAR_P99_MW:
            factor = solar_p99 / np.nanquantile(solar_raw, 0.99)
            solar = solar_raw * factor
            true_power = AGGREGATE_P99_MW - solar_p99
            battery = battery_raw * true_power / sizes[battery_id][0]
            for (demand_bmu, kind), demand in demands.items():
                aggregate = solar + battery + demand
                difference = separate_by_differences(
                    output_mw=aggregate, basis=inputs.basis, design=design
                )
                difference_ac_mw = (
                    float("nan") if difference is None else difference.total_capacity_mw
                )
                for power in sorted({*POWERS_MW, true_power}):
                    fit = fit_joint_solar_battery(
                        aggregate_mw=aggregate,
                        solar_basis=inputs.basis,
                        power_mw=power,
                        energy_mwh=DURATION_HOURS * power,
                        one_way_efficiency=ONE_WAY_EFFICIENCY,
                        window_half_hours=WINDOW_HALF_HOURS,
                        signed_columns=design,
                        method=SOLVER_METHOD,
                    )
                    row = {
                        "solar_set": set_name,
                        "battery": battery_id,
                        "solar_p99_mw": solar_p99,
                        "demand_bmu": demand_bmu,
                        "demand_kind": kind,
                        "power_mw": power,
                        "true_power_mw": true_power,
                        "reference_ac_mw": physical.parameters.ac_capacity_mw * factor,
                        "fleet_alone_ac_mw": fleet_alone_ac_mw * factor,
                        "difference_ac_mw": difference_ac_mw,
                    }
                    if solar_p99 > 0:
                        row |= score_fit(
                            fit=fit,
                            inputs=inputs,
                            solar=solar,
                            battery=battery,
                            true_soc=np.full_like(solar, np.nan),
                            reference_ac_mw=row["reference_ac_mw"],
                            battery_scores=False,
                        )
                    else:
                        row["fitted_ac_mw"] = float(fit.solar_weights.sum())
                    rows.append(row)
    return rows


def main() -> None:
    """Fit all sums and write the fits."""
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(SOLAR_SETS))) as pool:
        results = list(pool.map(run_set, list(SOLAR_SETS)))
    fits = pl.DataFrame([r for rows in results for r in rows], infer_schema_length=None)
    fits.write_parquet(OUTPUT_DIR / "rung7c_known_answer_fits.parquet")
    print(
        fits.group_by("solar_p99_mw", "demand_kind", "power_mw")
        .agg(pl.col("fitted_ac_mw").mean())
        .sort("solar_p99_mw", "demand_kind", "power_mw")
    )


if __name__ == "__main__":
    main()
