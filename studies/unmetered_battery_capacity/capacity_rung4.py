"""Rung 4: NGED battery A inside a bulk supply point's flow (a real known answer).

A pre-plan check found that NGED battery A's half-hour changes correlate with one bulk supply point
(BSP1) at -0.28 over the study year (negative, as expected when export reduces import) and at 0.07
or below with every primary and both grid supply points. The correlation is evidence of connection,
not a record of the network's topology. NGED battery A's metered output is a few percent of that
flow's 99th percentile.

The rung scores the flow in five forms, indexed by the multiple `m` of NGED battery A inside it:
`m = 0` adds the metered output back (a matched null for the same flow), `m = 1` is the flow as
metered, and `m = 2, 4, 11` subtract 1, 3, and 10 further copies. The truth is `m` times the
metered 99th percentile output (power) and `m` times the smallest holding capacity of the metered
output in the block (the energy reference). Exploratory, one site.

Writes `rung4_posteriors.parquet`. Run:
`OMP_NUM_THREADS=2 uv run python studies/unmetered_battery_capacity/capacity_rung4.py`
"""

from typing import Final

import numpy as np
import polars as pl
from capacity_inputs import OUTPUT_DIR, block_slices, nged_series
from capacity_rung3 import energy_reference
from capacity_runs import N_BLOCKS, fit_all_blocks, p99_flow

MULTIPLES: Final[tuple[int, ...]] = (0, 1, 2, 4, 11)


def main() -> None:
    """Fit the five forms of the flow and save the posteriors."""
    nged = nged_series()
    flow, battery = nged["BSP1"], np.nan_to_num(nged["battery_A"])
    battery_p99 = p99_flow(battery)
    lanes, meta = [], []
    for m in MULTIPLES:
        lanes.append(flow + (1 - m) * battery)
        reference = [
            energy_reference(output_mw=battery[block_slices()[b]])[0] * m for b in range(N_BLOCKS)
        ]
        meta.append(
            {
                "rung": "rung4",
                "series": "BSP1",
                "multiple": m,
                "true_power_mw": m * battery_p99,
                "energy_reference_by_block_mwh": reference,
            }
        )
    posteriors = fit_all_blocks(aggregates=np.stack(lanes)[None], metadata=[meta], label="rung 4")
    posteriors = posteriors.with_columns(
        true_energy_reference_mwh=pl.struct("block", "energy_reference_by_block_mwh").map_elements(
            lambda s: s["energy_reference_by_block_mwh"][s["block"]], return_dtype=pl.Float64
        )
    ).drop("energy_reference_by_block_mwh")
    posteriors.write_parquet(OUTPUT_DIR / "rung4_posteriors.parquet")


if __name__ == "__main__":
    main()
