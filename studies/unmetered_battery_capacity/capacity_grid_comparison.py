"""The grid estimator on rung 1, for the comparison with the differentiable estimator.

The subset is all of rung 1 (the 756 sums with a simulated merchant battery, the same aggregates
the differentiable estimator fits), run in the standard setting only, on 4 worker processes with
`OMP_NUM_THREADS=2`. The plan allows the grid estimator at most 2 hours of compute for this
comparison; the script prints its wall-clock time and its core-hours of fitting. Writes
`rung1_grid_posteriors.parquet`: per sum, the merchant power and energy posterior median and 5% and
95% quantiles, the true values, and the log Bayes factor.

Run: `OMP_NUM_THREADS=2 uv run python \
studies/unmetered_battery_capacity/capacity_grid_comparison.py`
"""

import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from typing import Final

import numpy as np
import polars as pl
from capacity_inputs import OUTPUT_DIR
from capacity_rung1 import build
from capacity_templates import DURATIONS_HOURS, combo_grid, fit_block, load_templates
from studies.template_posterior import posterior_draws

MAX_WORKERS: Final[int] = 4
N_DRAWS: Final[int] = 1000
SEED: Final[int] = 20261012


@cache
def _templates() -> tuple[np.ndarray, np.ndarray]:
    return load_templates(setting="standard")


def _fit_group(task: tuple[int, int, np.ndarray, list[dict]]) -> list[dict]:
    """Fit every lane of one series in one block with the grid estimator."""
    group, block, lanes, meta = task
    candidates, nuisance = _templates()
    grid = combo_grid(setting="standard")
    rows = []
    for lane, (aggregate, info) in enumerate(zip(lanes, meta, strict=True)):
        t0 = time.monotonic()
        posterior = fit_block(
            series=aggregate,
            block=block,
            candidates=candidates,
            nuisance=nuisance,
            setting="standard",
            rng=np.random.default_rng(SEED + 100 * group + lane),
        )
        combos, powers = posterior_draws(
            posterior=posterior,
            n_draws=N_DRAWS,
            rng=np.random.default_rng(SEED + 10_000 + 100 * group + lane),
        )
        durations = np.array(DURATIONS_HOURS)[np.unravel_index(combos, grid.axes)[0]]
        merchant_power = powers[:, 0]
        row = {
            **info,
            "block": block,
            "grid_log_bayes_factor": float(posterior.log_bayes_factor),
            "grid_fit_seconds": time.monotonic() - t0,
        }
        for name, values in (("power", merchant_power), ("energy", merchant_power * durations)):
            q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
            row |= {f"grid_{name}_q05": q05, f"grid_{name}_median": q50, f"grid_{name}_q95": q95}
        rows.append(row)
    return rows


def main() -> None:
    """Run the grid estimator on rung 1 and save its posteriors."""
    aggregates, metadata = build()
    tasks = [
        (g, block, aggregates[g], metadata[g])
        for g in range(aggregates.shape[0])
        for block in range(4)
    ]
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for finished in pool.map(_fit_group, tasks):
            rows += finished
            print(f"{len(rows)} sums after {time.monotonic() - started:.0f} s", flush=True)
    wall = time.monotonic() - started
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        grid_wall_seconds_total=pl.lit(wall)
    )
    frame.write_parquet(OUTPUT_DIR / "rung1_grid_posteriors.parquet")
    print(
        f"{frame.height} sums, {wall:.0f} s wall on {MAX_WORKERS} workers "
        f"({frame['grid_fit_seconds'].sum() / 3600:.2f} core-hours of fitting)"
    )


if __name__ == "__main__":
    main()
