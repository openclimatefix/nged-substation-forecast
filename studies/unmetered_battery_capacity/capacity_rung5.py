"""Rung 5: the screen of the 8 NGED primaries metered in MW, with a within-series placebo.

No primary is known to be battery-free, so a threshold built from the other primaries cannot claim
that one holds a battery. The screen's evidence is a within-series placebo instead: each primary's
log Bayes factor for the real templates is ranked against its log Bayes factors for 12 placebo
template sets on the same primary: the tariff windows moved by -3, -2, -1, +1, +2, and +3 hours,
and the N2EX and Agile prices taken from 6 other weeks (3 earlier and 3 later). A real battery
class should beat every placebo, and 1 in 13 sets would rank first by chance.

All 13 sets use the coarse stacks (`stacks_coarse_*.npz`), so that no set is favoured by a finer
interpolation. The primary is labelled S1 to S8. Writes `rung5_posteriors.parquet`, one row per
primary, block, and template set.

Run: `OMP_NUM_THREADS=2 uv run python studies/unmetered_battery_capacity/capacity_rung5.py`
"""

from typing import Final

import numpy as np
import polars as pl
from capacity_inputs import OUTPUT_DIR, nged_series
from capacity_runs import N_BLOCKS, fit_block, posterior_row
from capacity_stacks import COARSE_PRICE_SHIFT_WEEKS, coarse_stack_path
from capacity_state_space import estimator

PRIMARIES: Final[tuple[str, ...]] = tuple(f"S{i}" for i in range(1, 9))
WINDOW_SHIFTS_HOURS: Final[tuple[float, ...]] = (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)


def template_sets() -> list[dict]:
    """Return the 13 template sets: the real one and the 12 placebos."""
    sets = [{"name": "real", "kind": "real", "stack": coarse_stack_path(weeks=0), "shift": 0.0}]
    sets += [
        {
            "name": f"windows{h:+.0f}h",
            "kind": "window_placebo",
            "stack": coarse_stack_path(weeks=0),
            "shift": h,
        }
        for h in WINDOW_SHIFTS_HOURS
    ]
    sets += [
        {
            "name": f"prices{w:+d}w",
            "kind": "price_placebo",
            "stack": coarse_stack_path(weeks=w),
            "shift": 0.0,
        }
        for w in COARSE_PRICE_SHIFT_WEEKS
    ]
    return sets


def main() -> None:
    """Fit every primary under every template set and save the posteriors."""
    nged = nged_series()
    aggregates = np.stack([nged[label] for label in PRIMARIES])[:, None, :]
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
            print(f"rung 5 {template_set['name']} block {block}: {seconds:.0f} s", flush=True)
            for g, label in enumerate(PRIMARIES):
                rows.append(
                    {
                        "rung": "rung5",
                        "series": label,
                        "template_set": template_set["name"],
                        "set_kind": template_set["kind"],
                        "block": block,
                        "fit_seconds": seconds,
                        **posterior_row(fit=fit, group=g, lane=0, seed=1000 * block + g),
                    }
                )
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        OUTPUT_DIR / "rung5_posteriors.parquet"
    )


if __name__ == "__main__":
    main()
