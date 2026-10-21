"""Time the batched Kalman filter grid against looping the CPU filter, on a tuning-sized grid.

The grid is the shape of the DP filter tuning: 6 values of log `q`, 5 of logit `phi`, 4 of log
`sigma_z`, and 4 of log `sigma_eps` (480 settings), for 16 sites of one year of half-hours with 5%
of the readings missing, in 4 hour bands. The CPU time is `studies.dp_assimilation.run_kalman`
timed on a few stream-setting pairs and scaled to the whole grid.

    uv run --no-sync python studies/dp_filter_grid_benchmark/benchmark_filter_grid.py
"""

import itertools
import time
from typing import Final

import numpy as np
import torch
from studies.dp_assimilation import FilterHyperparameters, run_kalman
from studies.dp_filter_grid import run_kalman_grid

N_SITES: Final[int] = 16
N_ROWS: Final[int] = 17520
N_BANDS: Final[int] = 4
N_TIMED_CPU_RUNS: Final[int] = 3
LOG_Q: Final[tuple[float, ...]] = tuple(np.log([0.01, 0.05, 0.2, 0.8, 3.0, 10.0]))
LOGIT_PHI: Final[tuple[float, ...]] = (1.0, 2.5, 4.2, 5.5, 7.0)
LOG_SIGMA_Z: Final[tuple[float, ...]] = tuple(np.log([0.1, 0.35, 1.0, 2.5]))
LOG_SIGMA_EPS: Final[tuple[float, ...]] = tuple(np.log([0.3, 0.7, 1.5, 3.0]))


def main() -> None:
    """Print the CPU loop time, the grid time on each device, and the ratios."""
    rng = np.random.default_rng(0)
    residuals = rng.normal(0.0, 1.0, (N_SITES, N_ROWS))
    residuals[rng.random(residuals.shape) < 0.05] = np.nan
    band = (np.arange(N_ROWS) % 48) // 12
    grid = np.array(list(itertools.product(LOG_Q, LOGIT_PHI, LOG_SIGMA_Z, LOG_SIGMA_EPS)))
    start = time.perf_counter()
    for values in grid[:N_TIMED_CPU_RUNS]:
        run_kalman(
            residual=residuals[0], band=band, hyper=FilterHyperparameters.from_array(values=values)
        )
    per_run = (time.perf_counter() - start) / N_TIMED_CPU_RUNS
    loop_seconds = per_run * N_SITES * len(grid)
    print(f"{N_SITES} sites x {len(grid)} settings x {N_ROWS} half-hours")
    print(f"looping run_kalman: {per_run:.3f} s per run, about {loop_seconds:.0f} s for the grid")
    devices = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]
    for device in devices:
        run_kalman_grid(
            residuals=residuals[:2], band=band, hyper=grid[:4], n_bands=N_BANDS, device=device
        )
        start = time.perf_counter()
        run_kalman_grid(
            residuals=residuals,
            band=band,
            hyper=grid,
            n_bands=N_BANDS,
            device=device,
            keep_states=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        print(f"grid on {device}: {seconds:.1f} s, {loop_seconds / seconds:.0f}x the loop")


if __name__ == "__main__":
    main()
