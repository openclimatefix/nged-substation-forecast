"""Build and save the schedule stacks that the differentiable estimator interpolates.

For each price taker (the merchant battery on the N2EX day-ahead price, and the Agile tariff on the
Agile price) this script solves the day-by-day linear programme at every node of a grid of usable
durations and round-trip efficiencies, for cycle caps of 1 and 2 a day. Interpolating between
nodes costs an error in the schedule that is proportional to the node spacing. With the fine grid
below, a positive control against the linear programme's own dispatch recovers the power to about
0.5%. The Agile tariff's cap is 1 in both layers (the standard setting's cap), so its cap-2 layer is
a copy and is not solved again.

The fine stacks (`stacks.npz`) serve rungs 1 to 4. The coarse stacks (`stacks_coarse_<k>w.npz`,
with the prices taken from `k` weeks later, `k = 0` being the real prices) serve rung 5, whose 13
template sets must all use the same interpolation so that none is favoured.

Each file holds `merchant` and `agile`, of shape (nodes, 17,520) in float32, with the node index
`(i_duration * n_efficiency + i_efficiency) * 2 + cap`, and the node grids.

Run: `OMP_NUM_THREADS=1 uv run python \
studies/unmetered_battery_capacity/capacity_stacks.py [coarse]`
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Final

import numpy as np
from capacity_inputs import OUTPUT_DIR, agile_prices, day_ahead_on_grid, window_half_hours
from studies.battery_templates import STANDARD_LP_SETTINGS, agile_template, merchant_template

DURATION_NODES: Final[np.ndarray] = np.geomspace(0.6, 6.0, 100)
"""Fine usable durations in hours at full power, log-spaced (a ratio of 1.0235 between nodes)."""
EFFICIENCY_NODES: Final[np.ndarray] = np.linspace(0.75, 0.975, 19)
"""Fine round-trip efficiencies, 0.0125 apart."""
COARSE_DURATION_NODES: Final[np.ndarray] = np.geomspace(0.6, 6.0, 23)
COARSE_EFFICIENCY_NODES: Final[np.ndarray] = np.linspace(0.75, 0.975, 7)
COARSE_PRICE_SHIFT_WEEKS: Final[tuple[int, ...]] = (-3, -2, -1, 1, 2, 3)
CYCLE_CAPS: Final[tuple[float, float]] = (1.0, 2.0)
MAX_WORKERS: Final[int] = 4
STACKS_PATH: Final = OUTPUT_DIR / "stacks.npz"
DAYS_PER_WEEK: Final[int] = 7
HALF_HOURS_PER_DAY: Final[int] = 48


def coarse_stack_path(*, weeks: int) -> Path:
    """Return where the coarse stacks with prices from `weeks` weeks later live."""
    return OUTPUT_DIR / f"stacks_coarse_{weeks:+d}w.npz"


def node_index(*, duration: int, efficiency: int, cap: int, n_efficiency: int) -> int:
    """Return a node's position in a stack."""
    return (duration * n_efficiency + efficiency) * len(CYCLE_CAPS) + cap


def _build(
    task: tuple[str, int, int, int, float, float, int, int],
) -> tuple[str, int, np.ndarray]:
    """Build one node's schedule; the worker function of the process pool."""
    unit, duration, efficiency, cap, d_value, e_value, shift_days, n_efficiency = task
    settings = replace(STANDARD_LP_SETTINGS, cycles_per_day_cap=CYCLE_CAPS[cap])
    if unit == "merchant":
        prices = np.roll(day_ahead_on_grid(), -shift_days * HALF_HOURS_PER_DAY)
        column = merchant_template(
            day_ahead_prices=prices,
            duration_hours=d_value,
            round_trip_efficiency=e_value,
            settings=settings,
        )
    else:
        agile = agile_prices().with_columns(
            time=agile_prices()["time"] - timedelta(days=shift_days)
        )
        column = agile_template(
            half_hour_end_time=window_half_hours(),
            agile_prices=agile,
            duration_hours=d_value,
            round_trip_efficiency=e_value,
            settings=STANDARD_LP_SETTINGS,
        )
    position = node_index(
        duration=duration, efficiency=efficiency, cap=cap, n_efficiency=n_efficiency
    )
    return unit, position, column


def build(
    *, duration_nodes: np.ndarray, efficiency_nodes: np.ndarray, shift_days: int = 0
) -> dict[str, np.ndarray]:
    """Solve every node's linear programme.

    Args:
        duration_nodes: The usable durations in hours.
        efficiency_nodes: The round-trip efficiencies.
        shift_days: Takes the prices from this many days later (a placebo when nonzero).

    Returns:
        The stacks by unit name, shape (nodes, 17,520).
    """
    n_d, n_e = len(duration_nodes), len(efficiency_nodes)
    stacks = {
        unit: np.zeros((n_d * n_e * len(CYCLE_CAPS), len(window_half_hours())), dtype=np.float32)
        for unit in ("merchant", "agile")
    }
    tasks = [
        (unit, d, e, c, float(duration_nodes[d]), float(efficiency_nodes[e]), shift_days, n_e)
        for unit in stacks
        for d in range(n_d)
        for e in range(n_e)
        for c in range(len(CYCLE_CAPS))
        if unit == "merchant" or c == 0
    ]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for unit, position, column in pool.map(_build, tasks, chunksize=8):
            stacks[unit][position] = column
    agile = stacks["agile"].reshape(-1, len(CYCLE_CAPS), stacks["agile"].shape[-1])
    agile[:, 1] = agile[:, 0]
    return stacks


def save(
    *, path: Path, duration_nodes: np.ndarray, efficiency_nodes: np.ndarray, shift: int
) -> None:
    """Build one set of stacks and save it."""
    started = time.monotonic()
    stacks = build(
        duration_nodes=duration_nodes,
        efficiency_nodes=efficiency_nodes,
        shift_days=shift * DAYS_PER_WEEK,
    )
    np.savez_compressed(
        path, duration_nodes=duration_nodes, efficiency_nodes=efficiency_nodes, **stacks
    )
    print(f"{path.name}: {stacks['merchant'].shape[0]} nodes in {time.monotonic() - started:.0f} s")


def main() -> None:
    """Build the fine stacks, or with the argument `coarse` the coarse stacks of rung 5."""
    if len(sys.argv) > 1 and sys.argv[1] == "coarse":
        for weeks in (0, *COARSE_PRICE_SHIFT_WEEKS):
            save(
                path=coarse_stack_path(weeks=weeks),
                duration_nodes=COARSE_DURATION_NODES,
                efficiency_nodes=COARSE_EFFICIENCY_NODES,
                shift=weeks,
            )
    else:
        save(
            path=STACKS_PATH,
            duration_nodes=DURATION_NODES,
            efficiency_nodes=EFFICIENCY_NODES,
            shift=0,
        )
        print(f"stacks of {len(DURATION_NODES) * len(EFFICIENCY_NODES) * 2} nodes saved")


if __name__ == "__main__":
    main()
