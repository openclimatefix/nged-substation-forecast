"""The power of the rung 5 screen: how often the real template set ranks first when a battery is in.

Rung 5 ranks each primary's log Bayes factor for the real templates against 12 placebo template
sets and reads "the real set ranks first of 13" as evidence of a battery. The first science review
(M7) pointed out that nobody had run that screen on a series with a known battery, so a null result
could not be read as "no battery". This script runs the same 13 sets (the coarse stacks, so that no
set is favoured by a finer interpolation) on the nine demand-like series with:

- no added battery (rung 1's null lane), which gives the rate at which the real set ranks first by
  chance, for the screen's real placebos, whether or not they are exchangeable;
- rung 1's simulated merchant batteries at shares of 10% and 40%, all three nameplate durations;
- the four named public batteries of rung 3 at a 40% share.

Writes `screen_power_posteriors.parquet`, one row per series, lane, template set, and block.

Run: `OMP_NUM_THREADS=2 uv run python studies/unmetered_battery_capacity/capacity_screen_power.py`
"""

from typing import Final

import numpy as np
import polars as pl
from capacity_inputs import OUTPUT_DIR, demand_series
from capacity_rung1 import build as build_rung1
from capacity_rung3 import NAMED_BATTERIES, units
from capacity_rung5 import template_sets
from capacity_runs import N_BLOCKS, fit_block, p99_flow, posterior_row
from capacity_state_space import estimator

MERCHANT_SHARES: Final[tuple[float, ...]] = (0.10, 0.40)
PUBLIC_SHARE: Final[float] = 0.40


def build() -> tuple[np.ndarray, list[list[dict]]]:
    """Build the screen's aggregates and their identifiers.

    Returns:
        Aggregates of shape (series, lanes, 17,520) and the identifiers and truth of each lane.
    """
    aggregates, metadata = build_rung1()
    named = [u for u in units() if u["kind"] == "named"]
    assert [u["members"][0] for u in named] == list(NAMED_BATTERIES)
    series = demand_series()
    lanes_out, meta_out = [], []
    for g, (label, demand) in enumerate(series.items()):
        keep = [
            i
            for i, m in enumerate(metadata[g])
            if m["share"] == 0.0 or any(np.isclose(m["share"], s) for s in MERCHANT_SHARES)
        ]
        lanes = [aggregates[g, i] for i in keep]
        meta = [
            {**metadata[g][i], "lane_kind": "null" if metadata[g][i]["share"] == 0 else "merchant"}
            for i in keep
        ]
        p99 = p99_flow(demand)
        for unit in named:
            scale = PUBLIC_SHARE * p99 / unit["registered_mw"]
            lanes.append(demand - np.nan_to_num(unit["output_mw"]) * scale)
            meta.append(
                {
                    "rung": "screen_power",
                    "series": label,
                    "nameplate_hours": 0.0,
                    "share": PUBLIC_SHARE,
                    "true_power_mw": PUBLIC_SHARE * p99,
                    "unit": unit["unit"],
                    "lane_kind": "public",
                }
            )
        lanes_out.append(np.stack(lanes))
        meta_out.append(meta)
    return np.stack(lanes_out), meta_out


def main() -> None:
    """Fit every lane under every template set and save the log Bayes factors."""
    aggregates, metadata = build()
    rows = []
    for template_set in template_sets():
        for block in range(N_BLOCKS):
            model = estimator(
                setting="standard",
                block=block,
                stack_path=template_set["stack"],
                window_shift_hours=template_set["shift"],
            )
            fit, seconds = fit_block(block=block, aggregates=aggregates, model=model)
            print(f"screen {template_set['name']} block {block}: {seconds:.0f} s", flush=True)
            for g, group in enumerate(metadata):
                for lane, meta in enumerate(group):
                    row = posterior_row(fit=fit, group=g, lane=lane, seed=1000 * block + g)
                    rows.append(
                        {
                            **meta,
                            "lane": lane,
                            "template_set": template_set["name"],
                            "set_kind": template_set["kind"],
                            "block": block,
                            "log_bayes_factor": row["log_bayes_factor"],
                            "merchant_power_point": row["merchant_power_point"],
                        }
                    )
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        OUTPUT_DIR / "screen_power_posteriors.parquet"
    )


if __name__ == "__main__":
    main()
