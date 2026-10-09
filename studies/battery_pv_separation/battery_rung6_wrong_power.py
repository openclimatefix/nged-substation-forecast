"""Rung 6 with a wrong battery power: A8 given half and double the true power.

Rung 6 gave the joint model the true battery's 99th-percentile power and rung 3's fitted energy
capacity. Rungs 7 and 7b sweep the power because the real BMUs' batteries are unknown. This run
refits rung 6's `A8` on the regional sky with the power, and with the energy capacity to keep the
duration, scaled by 0.5 and by 2:

- `A8_p05` half the true power and half the energy capacity.
- `A8_p2` double the true power and double the energy capacity.

Rung 6's `A8`, `A0b`, and `A3b` are read from `rung6_fits.parquet`, not refitted.

Writes `rung6_wrong_power_fits.parquet` (one row per aggregate and arm).

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_rung6_wrong_power.py`.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Final

import numpy as np
import polars as pl
from battery_inputs import BATTERIES, OUTPUT_DIR
from battery_rung4 import _prepare_skies
from battery_rung6 import (
    ONE_WAY_EFFICIENCY,
    WINDOW_HALF_HOURS,
    score_fit,
    true_battery_sizes,
    true_soc_on_grid,
)
from battery_synthetic import AGGREGATE_P99_MW, SHARES, SOLAR_SETS, grid_index, output_on_grid
from studies.battery_joint_lp import fit_joint_solar_battery
from studies.pv_fit import fit_plant

POWER_FACTORS: Final[dict[str, float]] = {"A8_p05": 0.5, "A8_p2": 2.0}
"""Each arm's power and energy capacity as a multiple of the true battery's."""
SKY: Final[str] = "regional"


def run_set(set_name: str) -> list[dict]:
    """Fit both arms to every battery and share of one solar set.

    Args:
        set_name: A key of `SOLAR_SETS`.

    Returns:
        One row per battery, share, and arm.
    """
    members = SOLAR_SETS[set_name]
    solar_raw = np.nansum([output_on_grid(bmu_id=b) for b in members], axis=0)
    solar_raw[np.any([~np.isfinite(output_on_grid(bmu_id=b)) for b in members], axis=0)] = np.nan
    inputs = _prepare_skies(members=members, solar_raw=solar_raw)[SKY]
    index = grid_index(half_hour_end_time=inputs.sky.half_hour_end_time)
    reference = fit_plant(
        sky=inputs.sky,
        output_mw=solar_raw[index],
        orientation="free",
        usable=np.isfinite(solar_raw[index]),
    )
    if reference is None:
        raise RuntimeError(f"No reference fit for {set_name}")
    sizes = true_battery_sizes()
    rows: list[dict] = []
    for battery_id in BATTERIES:
        battery_raw = output_on_grid(bmu_id=battery_id)
        raw_p99, raw_energy = sizes[battery_id]
        raw_soc = true_soc_on_grid(bmu_id=battery_id)
        for share in SHARES:
            factor = share * AGGREGATE_P99_MW / np.nanquantile(solar_raw, 0.99)
            solar = solar_raw * factor
            scale = (1.0 - share) * AGGREGATE_P99_MW / raw_p99
            battery = battery_raw * scale
            aggregate = solar + battery
            reference_ac_mw = reference.parameters.ac_capacity_mw * factor
            for arm, power_factor in POWER_FACTORS.items():
                fit = fit_joint_solar_battery(
                    aggregate_mw=aggregate,
                    solar_basis=inputs.basis,
                    power_mw=power_factor * (1.0 - share) * AGGREGATE_P99_MW,
                    energy_mwh=power_factor * raw_energy * scale,
                    one_way_efficiency=ONE_WAY_EFFICIENCY,
                    window_half_hours=WINDOW_HALF_HOURS,
                )
                rows.append(
                    {
                        "solar_set": set_name,
                        "battery": battery_id,
                        "share": share,
                        "sky": SKY,
                        "arm": arm,
                        "reference_ac_mw": reference_ac_mw,
                        **score_fit(
                            fit=fit,
                            inputs=inputs,
                            solar=solar,
                            battery=battery,
                            true_soc=raw_soc * scale,
                            reference_ac_mw=reference_ac_mw,
                        ),
                    }
                )
    return rows


def main() -> None:
    """Fit all aggregates and write the fits."""
    with ProcessPoolExecutor(max_workers=len(SOLAR_SETS)) as pool:
        results = list(pool.map(run_set, list(SOLAR_SETS)))
    fits = pl.DataFrame([r for rows in results for r in rows], infer_schema_length=None)
    fits.write_parquet(OUTPUT_DIR / "rung6_wrong_power_fits.parquet")
    print(
        fits.group_by("arm", "share").agg(pl.col("nmae_of_solar_p99").mean()).sort("arm", "share")
    )


if __name__ == "__main__":
    main()
