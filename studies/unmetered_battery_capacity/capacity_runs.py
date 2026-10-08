"""Machinery the rungs share: simulated batteries, block fits, posterior rows, and the step tail.

Every rung builds aggregates (a demand series with a battery subtracted), fits them block by block
with the differentiable estimator, and saves one row per sum: the identifiers, the truth, the
posterior's quantiles for the merchant class and for the other units, the log Bayes factor, and the
Laplace approximation itself (`theta` and the flattened `covariance`), so a chart needs no refit.
"""

import time
from typing import Final

import numpy as np
import polars as pl
import torch
from capacity_inputs import block_slices, day_ahead_on_grid
from capacity_state_space import (
    ADAM_STAGES,
    block_problems,
    estimator,
    posterior_summary,
    start_parameters,
)
from capacity_templates import block_free_columns, nuisance_candidates
from studies.battery_dispatch import lp_schedule
from studies.battery_state_space import Estimator, FitResult

SHARES: Final[tuple[float, ...]] = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
"""Battery power as a fraction of the demand series' 99th percentile absolute flow."""
NAMEPLATE_HOURS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0)
"""The simulated merchant batteries' nameplate durations, which lie between the grid's values."""
ONE_WAY_EFFICIENCY_RANGE: Final[tuple[float, float]] = (0.88, 0.95)
SOC_MIN_RANGE: Final[tuple[float, float]] = (0.0, 0.10)
SOC_MAX_RANGE: Final[tuple[float, float]] = (0.90, 1.0)
STEP_TAIL_QUANTILE: Final[float] = 0.995
STEP_TAIL_NORMAL_QUANTILE: Final[float] = 2.81
MAD_TO_SD: Final[float] = 1.4826
N_BLOCKS: Final[int] = 4


def p99_flow(series: np.ndarray) -> float:
    """Return the 99th percentile of a series' absolute flow in MW, ignoring missing values."""
    return float(np.nanquantile(np.abs(series), 0.99))


def simulated_merchant_battery(
    *, nameplate_hours: float, seed: int
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Draw one merchant battery whose physical parameters are off the estimator's grid.

    The one-way efficiency, the state-of-charge limits, and the cap of 1 or 2 cycles a day are
    drawn, and the schedule is the day-by-day linear programme on the N2EX price.

    Args:
        nameplate_hours: The energy capacity in hours at full power.
        seed: Seeds the draw.

    Returns:
        The one-megawatt schedule on the window grid (positive for export), the usable duration in
        hours at full power, and the drawn parameters (`soc_min`, `soc_max`, `round_trip`,
        `cycles_cap`).
    """
    rng = np.random.default_rng(seed)
    eta_one_way = rng.uniform(*ONE_WAY_EFFICIENCY_RANGE)
    soc_min, soc_max = rng.uniform(*SOC_MIN_RANGE), rng.uniform(*SOC_MAX_RANGE)
    cap = float(rng.choice([1.0, 2.0]))
    schedule = lp_schedule(
        prices=day_ahead_on_grid(),
        energy_hours=nameplate_hours,
        eta_one_way=eta_one_way,
        soc_min=soc_min,
        soc_max=soc_max,
        cycles_per_day_cap=cap,
    )
    parameters = {
        "soc_min": soc_min,
        "soc_max": soc_max,
        "round_trip": eta_one_way**2,
        "cycles_cap": cap,
    }
    return schedule, (soc_max - soc_min) * nameplate_hours, parameters


def step_tail_power(
    *, aggregate: np.ndarray, free: np.ndarray, valid: np.ndarray
) -> tuple[float, float]:
    """Return the model-free step statistic of one block: the power a step of the residual implies.

    The baseline and solar columns are fitted by least squares; the residual's half-hour changes
    `dr` give `P_step = max(0, (q99.5(|dr|) - 2.81 * sigma) / 2)` with `sigma = 1.4826 * MAD(dr)`.
    A battery switching from full charge to full discharge adds a step of up to twice its power.

    Args:
        aggregate: The block's aggregate in MW.
        free: The block's free columns.
        valid: Where the aggregate and every column are finite.

    Returns:
        The step-tail power in MW and `sigma_step`, the robust standard deviation of the residual's
        half-hour changes in MW (the noise unit).
    """
    rows = np.where(valid)[0]
    coefficients, *_ = np.linalg.lstsq(free[rows], aggregate[rows], rcond=None)
    residual = np.full(len(aggregate), np.nan)
    residual[rows] = aggregate[rows] - free[rows] @ coefficients
    change = np.diff(residual)
    change = change[np.isfinite(change)]
    sigma = MAD_TO_SD * float(np.median(np.abs(change - np.median(change))))
    tail = float(np.quantile(np.abs(change), STEP_TAIL_QUANTILE))
    return max(0.0, (tail - STEP_TAIL_NORMAL_QUANTILE * sigma) / 2.0), sigma


def fit_block(
    *, block: int, aggregates: np.ndarray, model: Estimator | None = None, verbose: bool = False
) -> tuple[FitResult, float]:
    """Fit one block's aggregates and time it.

    Args:
        block: The block's index.
        aggregates: Shape (groups, lanes, 17,520), the import-positive aggregates in MW.
        model: The estimator; the standard-setting estimator of the block by default.
        verbose: Print the Adam progress.

    Returns:
        The fit and the seconds it took, including the synchronisation of the device.
    """
    started = time.monotonic()
    model = model or estimator(setting="standard", block=block)
    problems = block_problems(block=block, aggregates=aggregates)
    fit = model.fit(
        problems=problems, starts=start_parameters(), stages=ADAM_STAGES, verbose=verbose
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return fit, time.monotonic() - started


def posterior_row(*, fit: FitResult, group: int, lane: int, seed: int) -> dict:
    """Return one sum's posterior as a table row.

    Args:
        fit: The fit.
        group: The group index.
        lane: The lane within the group.
        seed: Seeds the draws for the quantiles.

    Returns:
        The summary of the best start (see `capacity_state_space.posterior_summary`), the log
        Bayes factor of the model with batteries against the model with none (NaN without a
        covariance), and the Laplace approximation (`theta`, `covariance`) of the best start.
    """
    start = int(fit.best_start[group, lane])
    row = posterior_summary(fit=fit, group=group, lane=lane, rng=np.random.default_rng(seed))
    row["log_bayes_factor"] = float(
        fit.log_evidence[group, lane, start] - fit.null_log_evidence[group, lane, start]
    )
    row["theta"] = fit.theta[group, lane, start].tolist()
    row["covariance"] = fit.covariance[group, lane, start].ravel().tolist()
    return row


def fit_all_blocks(
    *, aggregates: np.ndarray, metadata: list[list[dict]], label: str
) -> pl.DataFrame:
    """Fit every block of a rung and return the table of posterior rows.

    Args:
        aggregates: Shape (groups, lanes, 17,520), identical in every block.
        metadata: For each group and lane, the identifiers and truth to store beside the posterior.
        label: Names the rung in the timing lines.

    Returns:
        One row per group, lane, and block.
    """
    rows = []
    for block in range(N_BLOCKS):
        fit, seconds = fit_block(block=block, aggregates=aggregates)
        n_fits = fit.theta.shape[0] * fit.theta.shape[1] * fit.theta.shape[2]
        print(
            f"{label} block {block}: {seconds:.0f} s on "
            f"{'GPU' if torch.cuda.is_available() else 'CPU'} for {n_fits} fits",
            flush=True,
        )
        for g, group in enumerate(metadata):
            for lane, meta in enumerate(group):
                rows.append(
                    {
                        **meta,
                        "block": block,
                        "fit_seconds": seconds,
                        **posterior_row(fit=fit, group=g, lane=lane, seed=1000 * block + g),
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def block_valid_and_free(*, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a block's free columns and the rows where the solar columns are finite."""
    return block_free_columns(block=block, nuisance=nuisance_candidates())


def step_tail_table(*, aggregates: np.ndarray, metadata: list[list[dict]]) -> pl.DataFrame:
    """Return the step-tail power of every sum in every block.

    Args:
        aggregates: Shape (groups, lanes, 17,520).
        metadata: As in `fit_all_blocks`.

    Returns:
        One row per group, lane, and block with `step_tail_power_mw` and `sigma_step_mw`.
    """
    rows = []
    for block in range(N_BLOCKS):
        free, solar_ok = block_valid_and_free(block=block)
        slice_ = block_slices()[block]
        for g, group in enumerate(metadata):
            for lane, meta in enumerate(group):
                y = aggregates[g, lane, slice_]
                valid = np.isfinite(y) & solar_ok
                power, sigma = step_tail_power(aggregate=y, free=free, valid=valid)
                rows.append(
                    {**meta, "block": block, "step_tail_power_mw": power, "sigma_step_mw": sigma}
                )
    return pl.DataFrame(rows, infer_schema_length=None)
