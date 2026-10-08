"""Build and save the schedule stacks that the differentiable estimator interpolates.

For each price taker (the merchant battery on the N2EX day-ahead price, and the Agile tariff on the
Agile price) this script solves the day-by-day linear programme at every node of a fine grid of
usable durations and round-trip efficiencies, for cycle caps of 1 and 2 a day. Interpolating
between nodes costs an error in the schedule that is proportional to the node spacing: with the
grid below, a positive control against the linear programme's own dispatch recovers the power to
about 0.5%. The Agile tariff's cap is 1 in both layers (the standard setting's cap), so its cap-2
layer is a copy and is not solved again.

Saved to `stacks.npz`: `merchant` and `agile`, each of shape (nodes, 17,520) in float32, with the
node index `(i_duration * n_efficiency + i_efficiency) * 2 + cap`, and the node grids.

Run: `OMP_NUM_THREADS=1 uv run python studies/unmetered_battery_capacity/capacity_stacks.py`
"""

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Final

import numpy as np
from capacity_inputs import OUTPUT_DIR, agile_prices, day_ahead_on_grid, window_half_hours
from studies.battery_templates import STANDARD_LP_SETTINGS, agile_template, merchant_template

DURATION_NODES: Final[np.ndarray] = np.geomspace(0.6, 6.0, 100)
"""Usable durations in hours at full power, log-spaced (a ratio of 1.0235 between neighbours)."""
EFFICIENCY_NODES: Final[np.ndarray] = np.linspace(0.75, 0.975, 19)
"""Round-trip efficiencies, 0.0125 apart."""
CYCLE_CAPS: Final[tuple[float, float]] = (1.0, 2.0)
MAX_WORKERS: Final[int] = 4
STACKS_PATH: Final = OUTPUT_DIR / "stacks.npz"


def node_index(*, duration: int, efficiency: int, cap: int) -> int:
    """Return a node's position in a stack."""
    return (duration * len(EFFICIENCY_NODES) + efficiency) * len(CYCLE_CAPS) + cap


def _build(task: tuple[str, int, int, int]) -> tuple[str, int, np.ndarray]:
    """Build one node's schedule; the worker function of the process pool."""
    unit, duration, efficiency, cap = task
    settings = replace(STANDARD_LP_SETTINGS, cycles_per_day_cap=CYCLE_CAPS[cap])
    if unit == "merchant":
        column = merchant_template(
            day_ahead_prices=day_ahead_on_grid(),
            duration_hours=float(DURATION_NODES[duration]),
            round_trip_efficiency=float(EFFICIENCY_NODES[efficiency]),
            settings=settings,
        )
    else:
        column = agile_template(
            half_hour_end_time=window_half_hours(),
            agile_prices=agile_prices(),
            duration_hours=float(DURATION_NODES[duration]),
            round_trip_efficiency=float(EFFICIENCY_NODES[efficiency]),
            settings=STANDARD_LP_SETTINGS,
        )
    return unit, node_index(duration=duration, efficiency=efficiency, cap=cap), column


def build() -> dict[str, np.ndarray]:
    """Solve every node's linear programme.

    Returns:
        The stacks by unit name, shape (nodes, 17,520).
    """
    n_nodes = len(DURATION_NODES) * len(EFFICIENCY_NODES) * len(CYCLE_CAPS)
    stacks = {
        unit: np.zeros((n_nodes, len(window_half_hours())), dtype=np.float32)
        for unit in ("merchant", "agile")
    }
    tasks = [
        (unit, d, e, c)
        for unit in stacks
        for d in range(len(DURATION_NODES))
        for e in range(len(EFFICIENCY_NODES))
        for c in range(len(CYCLE_CAPS))
        if unit == "merchant" or c == 0
    ]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for unit, position, column in pool.map(_build, tasks, chunksize=8):
            stacks[unit][position] = column
    agile = stacks["agile"].reshape(-1, len(CYCLE_CAPS), stacks["agile"].shape[-1])
    agile[:, 1] = agile[:, 0]
    return stacks


def main() -> None:
    """Build the stacks and save them."""
    started = time.monotonic()
    stacks = build()
    np.savez_compressed(
        STACKS_PATH,
        duration_nodes=DURATION_NODES,
        efficiency_nodes=EFFICIENCY_NODES,
        **stacks,
    )
    print(
        f"{len(stacks)} stacks of {stacks['merchant'].shape[0]} nodes in "
        f"{time.monotonic() - started:.0f} s -> {STACKS_PATH}"
    )


if __name__ == "__main__":
    main()
