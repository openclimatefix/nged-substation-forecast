"""The differentiable battery estimator wired to this study's signals, blocks, and priors.

Six units in three classes: the merchant battery (a price taker on the N2EX day-ahead price), the
Agile tariff (a price taker on the Agile price, in the domestic class), the commercial and
industrial red-band battery, and the three fixed-window domestic tariffs. Each class has its own
usable duration and round-trip efficiency, with the grid estimator's priors.

The differentiable estimator reads the same free columns as the grid estimator (the monthly
baseline, the four solar curves, and the three charge-only nuisance columns), profiled out by
projection.
"""

import os
from functools import cache
from typing import Final

import numpy as np
import torch
from capacity_inputs import (
    agile_prices,
    block_slices,
    day_ahead_on_grid,
    window_half_hours,
)
from capacity_templates import (
    CLASS_NAMES,
    DURATION_PRIORS,
    EFFICIENCY_PRIOR_MEAN,
    POWER_PRIOR_FRACTION_OF_P99,
    SETTINGS,
    SettingNameType,
    beta_from_interval,
    block_free_columns,
    nuisance_candidates,
)
from studies.battery_state_space import (
    Estimator,
    FitResult,
    Layout,
    Priors,
    Problems,
    Sharpness,
    daily_rank_fraction,
    make_problems,
    natural_draws,
    natural_point,
)
from studies.battery_templates import TARIFF_WINDOWS, agile_days, window_coverage

UNIT_NAMES: Final[tuple[str, ...]] = (
    "merchant",
    "agile",
    "commercial_and_industrial",
    "intelligent_octopus_go",
    "octopus_go",
    "octopus_flux",
)
LAYOUT: Final = Layout(unit_class=(0, 2, 1, 2, 2, 2), n_classes=3, n_rank=2)
"""Units 0 and 1 are rank units; the class indices follow `CLASS_NAMES`."""
FIXED_WINDOWS: Final[tuple[str, ...]] = (
    "red_band",
    "intelligent_octopus_go",
    "octopus_go",
    "octopus_flux",
)
STARTS: Final[int] = 3
ADAM_STAGES: Final[tuple[tuple[int, float, Sharpness], ...]] = (
    (120, 0.08, Sharpness(rank=20.0, smoothing=1e-2)),
    (120, 0.04, Sharpness(rank=60.0, smoothing=1e-3)),
    (160, 0.02, Sharpness(rank=150.0, smoothing=1e-4)),
)
"""Adam stages: iterations, learning rate, and annealed sharpness; the last is the final model."""
N_DRAWS: Final[int] = 2000


def device() -> torch.device:
    """Return `cuda` when it is available and `STUDIES_FORCE_CPU` is unset, otherwise `cpu`."""
    if torch.cuda.is_available() and not os.environ.get("STUDIES_FORCE_CPU"):
        return torch.device("cuda")
    return torch.device("cpu")


def priors(*, setting: SettingNameType) -> Priors:
    """Return the grid estimator's duration and efficiency priors for the three classes.

    Args:
        setting: The setting, which scales the duration spread and widens the efficiency range.

    Returns:
        The log-normal duration priors and the Beta efficiency priors.
    """
    spec = SETTINGS[setting]
    alpha, beta = beta_from_interval(
        mean=EFFICIENCY_PRIOR_MEAN,
        low=spec.efficiency_interval[0],
        high=spec.efficiency_interval[1],
    )
    return Priors(
        log_duration_mean=np.log([DURATION_PRIORS[name][0] for name in CLASS_NAMES]),
        log_duration_sd=np.array(
            [DURATION_PRIORS[name][1] * spec.duration_sd_multiplier for name in CLASS_NAMES]
        ),
        efficiency_alpha=np.full(len(CLASS_NAMES), alpha),
        efficiency_beta=np.full(len(CLASS_NAMES), beta),
    )


@cache
def signals_numpy() -> dict[str, np.ndarray]:
    """Return the policy signals on the study year's half-hour grid.

    Returns:
        The keyword arguments of `studies.battery_state_space.make_signals` other than the device.
    """
    grid = window_half_hours()
    n_grid = len(grid)
    merchant_rank, merchant_valid = daily_rank_fraction(prices=day_ahead_on_grid())
    day_prices, slot_index = agile_days(half_hour_end_time=grid, agile_prices=agile_prices())
    agile_rank_by_slot, agile_valid_by_slot = daily_rank_fraction(prices=day_prices.ravel())
    agile_rank = np.full(n_grid, 0.5)
    agile_valid = np.zeros(n_grid, dtype=bool)
    flat_index = slot_index.ravel()
    inside = (flat_index >= 0) & (flat_index < n_grid)
    agile_rank[flat_index[inside]] = agile_rank_by_slot[inside]
    agile_valid[flat_index[inside]] = agile_valid_by_slot[inside]
    coverage = [
        window_coverage(half_hour_end_time=grid, window=TARIFF_WINDOWS[n]) for n in FIXED_WINDOWS
    ]
    return {
        "rank_fraction": np.stack([merchant_rank, agile_rank]),
        "rank_valid": np.stack([merchant_valid, agile_valid]),
        "fixed_charge": np.stack([c for c, _ in coverage]),
        "fixed_discharge": np.stack([d for _, d in coverage]),
    }


def estimator(*, setting: SettingNameType, block: int) -> Estimator:
    """Build the estimator for one setting and one block on the best available device."""
    rows = block_slices()[block]
    return Estimator(
        layout=LAYOUT,
        signals_numpy={name: values[:, rows] for name, values in signals_numpy().items()},
        priors=priors(setting=setting),
        device=device(),
    )


def start_parameters() -> np.ndarray:
    """Return the `STARTS` initial parameter vectors, with `log P` relative to the power scale.

    The three starts differ in the merchant battery's power, duration, and thresholds; the other
    units start small.

    Returns:
        Shape (starts, parameters).
    """
    layout = LAYOUT
    rows = []
    for merchant_power, duration, threshold in ((-0.7, 1.0, 0.1), (0.4, 2.0, 0.2), (1.1, 4.0, 0.3)):
        theta = np.zeros(layout.n_parameters)
        theta[layout.power] = -3.0
        theta[0] = merchant_power
        theta[layout.log_duration] = np.log([duration, 1.5, 2.0])
        theta[layout.logit_efficiency] = np.log(0.87 / 0.13)
        theta[layout.threshold_charge] = np.log(2 * threshold / (1 - 2 * threshold))
        theta[layout.threshold_discharge] = np.log(2 * threshold / (1 - 2 * threshold))
        rows.append(theta)
    return np.stack(rows)


def block_problems(
    *, block: int, aggregates: np.ndarray, extra_valid: np.ndarray | None = None
) -> Problems:
    """Assemble the groups of one block.

    Args:
        block: The block's index.
        aggregates: Shape (groups, lanes, 17,520), each an import-positive aggregate over the
            whole year in MW, NaN where missing.
        extra_valid: Optional extra mask of shape (groups, 17,520).

    Returns:
        The problems, with each group's free columns projected off.
    """
    rows = block_slices()[block]
    free, solar_ok = block_free_columns(block=block, nuisance=nuisance_candidates())
    block_aggregate = aggregates[:, :, rows]
    valid = np.isfinite(block_aggregate).all(axis=1) & solar_ok[None, :]
    if extra_valid is not None:
        valid &= extra_valid[:, rows]
    scale = np.stack(
        [
            [
                max(
                    POWER_PRIOR_FRACTION_OF_P99
                    * float(np.quantile(np.abs(lane[valid[group]]), 0.99)),
                    1e-6,
                )
                for lane in block_aggregate[group]
            ]
            for group in range(block_aggregate.shape[0])
        ]
    )
    return make_problems(
        aggregate=np.nan_to_num(block_aggregate),
        valid=valid,
        free=[free] * block_aggregate.shape[0],
        power_scale=scale,
    )


def posterior_summary(
    *, fit: FitResult, group: int, lane: int, rng: np.random.Generator
) -> dict[str, float | bool]:
    """Summarise one aggregate's best start: medians and 5/25/75/95% quantiles of MW and MWh.

    Args:
        fit: The fit.
        group: The group index.
        lane: The lane index within the group.
        rng: The random generator for the draws.

    Returns:
        For the merchant class and for the sum of the other units: `<name>_<quantity>_<q>` entries,
        the log-posterior loss, the integrated autocorrelation time, whether an interval exists,
        and the spread of the merchant power across the starts.
    """
    start = int(fit.best_start[group, lane])
    theta = fit.theta[group, lane, start]
    row: dict[str, float | bool] = {
        "loss": float(fit.loss[group, lane, start]),
        "tau": float(fit.tau[group, lane, start]),
        "has_interval": bool(fit.has_interval[group, lane, start]),
        "at_bound": bool(fit.at_bound[group, lane, start]),
        "start_spread_merchant_power": float(
            np.ptp(np.exp(fit.theta[group, lane, :, 0])) / np.exp(theta[0])
        ),
    }
    point = natural_point(theta=theta, layout=LAYOUT)
    row |= {
        "merchant_power_point": float(point["power"][0]),
        "merchant_energy_point": float(point["energy"][0]),
        "merchant_duration_point": float(point["duration"][0]),
        "merchant_efficiency_point": float(point["efficiency"][0]),
        "other_units_power_point": float(point["power"][1:].sum()),
    }
    if not row["has_interval"]:
        return row
    draws = natural_draws(
        theta=theta,
        covariance=fit.covariance[group, lane, start],
        layout=LAYOUT,
        n_draws=N_DRAWS,
        rng=rng,
    )
    quantities = {
        "merchant_power": draws["power"][:, 0],
        "merchant_energy": draws["energy"][:, 0],
        "merchant_duration": draws["duration"][:, 0],
        "merchant_efficiency": draws["efficiency"][:, 0],
        "other_units_power": draws["power"][:, 1:].sum(axis=1),
    }
    for name, values in quantities.items():
        for label, level in (
            ("q05", 0.05),
            ("q25", 0.25),
            ("median", 0.5),
            ("q75", 0.75),
            ("q95", 0.95),
        ):
            row[f"{name}_{label}"] = float(np.quantile(values, level))
    return row
