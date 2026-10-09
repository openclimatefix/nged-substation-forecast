"""The differentiable battery estimator wired to this study's signals, blocks, and priors.

Six units in three classes: the merchant battery (a price taker on the N2EX day-ahead price), the
Agile tariff (a price taker on the Agile price, in the domestic class), the commercial and
industrial red-band battery, and the three fixed-window domestic tariffs. The two price takers
interpolate the schedule stacks of `capacity_stacks.py`. Each class has its own
usable duration and round-trip efficiency, with the grid estimator's priors.

The differentiable estimator reads the same free columns as the grid estimator (the monthly
baseline, the four solar curves, and the three charge-only nuisance columns), profiled out by
projection.
"""

import os
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Final

import numpy as np
import torch
from capacity_inputs import block_slices, window_half_hours
from capacity_stacks import STACKS_PATH
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
    make_problems,
    natural_draws,
    natural_point,
)
from studies.battery_templates import (
    TARIFF_WINDOWS,
    TariffNameType,
    TariffWindow,
    window_coverage,
)

UNIT_NAMES: Final[tuple[str, ...]] = (
    "merchant",
    "agile",
    "commercial_and_industrial",
    "intelligent_octopus_go",
    "octopus_go",
    "octopus_flux",
)
LAYOUT: Final = Layout(unit_class=(0, 2, 1, 2, 2, 2), n_classes=3, n_stack=2)
"""Units 0 and 1 are stack units; the class indices follow `CLASS_NAMES`."""
FIXED_WINDOWS: Final[tuple[TariffNameType, ...]] = (
    "red_band",
    "intelligent_octopus_go",
    "octopus_go",
    "octopus_flux",
)
STARTS: Final[int] = 3
ADAM_STAGES: Final[tuple[tuple[int, float, Sharpness], ...]] = (
    (150, 0.08, Sharpness(smoothing=1e-2)),
    (150, 0.04, Sharpness(smoothing=1e-3)),
    (200, 0.02, Sharpness(smoothing=1e-4)),
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


def shifted_window(*, name: TariffNameType, hours: float) -> TariffWindow:
    """Return a tariff window with every edge moved by some hours (a placebo window)."""
    window = TARIFF_WINDOWS[name]
    return replace(
        window,
        charge_start_hour=window.charge_start_hour + hours,
        charge_end_hour=window.charge_end_hour + hours,
        discharge_start_hour=window.discharge_start_hour + hours,
        discharge_end_hour=window.discharge_end_hour + hours,
    )


@cache
def signals_numpy(
    *, stack_path: Path = STACKS_PATH, window_shift_hours: float = 0.0
) -> dict[str, np.ndarray]:
    """Return the policy signals on the study year's half-hour grid.

    Args:
        stack_path: Which saved stacks the price takers interpolate.
        window_shift_hours: Moves every tariff window by this many hours (zero for the real
            windows, nonzero for a placebo).

    Returns:
        The keyword arguments of `studies.battery_state_space.make_signals` other than the device
        and dtype: the merchant and Agile stacks, their node grids, and the window indicators.
    """
    grid = window_half_hours()
    saved = np.load(stack_path)
    coverage = [
        window_coverage(
            half_hour_end_time=grid, window=shifted_window(name=name, hours=window_shift_hours)
        )
        for name in FIXED_WINDOWS
    ]
    return {
        "stacks": np.stack([saved["merchant"], saved["agile"]]),
        "duration_nodes": saved["duration_nodes"],
        "efficiency_nodes": saved["efficiency_nodes"],
        "fixed_charge": np.stack([c for c, _ in coverage]),
        "fixed_discharge": np.stack([d for _, d in coverage]),
    }


def _block_signals(
    *, rows: slice, stack_path: Path, window_shift_hours: float
) -> dict[str, np.ndarray]:
    """Cut the signals to a block's half-hours."""
    signals = signals_numpy(stack_path=stack_path, window_shift_hours=window_shift_hours)
    return {
        "stacks": signals["stacks"][:, :, rows],
        "duration_nodes": signals["duration_nodes"],
        "efficiency_nodes": signals["efficiency_nodes"],
        "fixed_charge": signals["fixed_charge"][:, rows],
        "fixed_discharge": signals["fixed_discharge"][:, rows],
    }


def estimator(
    *,
    setting: SettingNameType,
    block: int,
    stack_path: Path = STACKS_PATH,
    window_shift_hours: float = 0.0,
) -> Estimator:
    """Build the estimator for one setting and one block on the best available device.

    Args:
        setting: The setting.
        block: The block's index.
        stack_path: Which saved stacks the price takers interpolate.
        window_shift_hours: Moves every tariff window by this many hours (a placebo).

    Returns:
        The estimator.
    """
    return Estimator(
        layout=LAYOUT,
        signals_numpy=_block_signals(
            rows=block_slices()[block],
            stack_path=stack_path,
            window_shift_hours=window_shift_hours,
        ),
        priors=priors(setting=setting),
        device=device(),
    )


def start_parameters() -> np.ndarray:
    """Return the `STARTS` initial parameter vectors, with `log P` relative to the power scale.

    The three starts differ in the merchant battery's power and duration; the other units start
    small.

    Returns:
        Shape (starts, parameters).
    """
    layout = LAYOUT
    rows = []
    for merchant_power, duration in ((-0.7, 1.0), (0.4, 2.0), (1.1, 4.0)):
        theta = np.zeros(layout.n_parameters)
        theta[layout.power] = -3.0
        theta[0] = merchant_power
        theta[layout.log_duration] = np.log([duration, 1.5, 2.0])
        theta[layout.logit_efficiency] = np.log(0.87 / 0.13)
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
    """Summarise one aggregate's best start: point values and quantiles of MW and MWh per class.

    Args:
        fit: The fit.
        group: The group index.
        lane: The lane index within the group.
        rng: The random generator for the draws.

    Returns:
        The log-posterior loss, the integrated autocorrelation time, whether an interval exists,
        whether a parameter sits on its bound, the spread of the merchant power across the starts,
        the point values (`<class>_power_point`, `<class>_energy_point`, the merchant class's
        `merchant_duration_point`, `merchant_efficiency_point`, and `merchant_cap_weight_point`),
        and, where an interval exists, the 5, 25, 50, 75, and 95% quantiles
        (`<class>_<quantity>_<q>`) of every class's power and energy and of the merchant class's
        duration and efficiency. `<class>` is a name in `CLASS_NAMES`.
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
    row |= {k: float(v) for k, v in _class_quantities(natural=point).items()}
    row |= {
        "merchant_duration_point": float(point["duration"][0]),
        "merchant_efficiency_point": float(point["efficiency"][0]),
        "merchant_cap_weight_point": float(point["cap_weight"][0]),
    }
    row = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in row.items()}
    if not row["has_interval"]:
        return row
    draws = natural_draws(
        theta=theta,
        covariance=fit.covariance[group, lane, start],
        layout=LAYOUT,
        n_draws=N_DRAWS,
        rng=rng,
    )
    quantities = _class_quantities(natural=draws)
    quantities["merchant_duration"] = draws["duration"][:, 0]
    quantities["merchant_efficiency"] = draws["efficiency"][:, 0]
    for key, values in quantities.items():
        name = key.removesuffix("_point")
        for label, level in (
            ("q05", 0.05),
            ("q25", 0.25),
            ("median", 0.5),
            ("q75", 0.75),
            ("q95", 0.95),
        ):
            row[f"{name}_{label}"] = float(np.quantile(values, level))
    return row


def _class_quantities(*, natural: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Sum the units of each class, and keep each unit: `<class>_power_point`, `unit_<unit>_...`.

    Works for one optimum (1-D arrays) or for draws (2-D arrays with one row per draw).
    """
    out = {}
    classes = np.array(LAYOUT.unit_class)
    for index, name in enumerate(CLASS_NAMES):
        members = np.where(classes == index)[0]
        out[f"{name}_power_point"] = natural["power"][..., members].sum(axis=-1)
        out[f"{name}_energy_point"] = natural["energy"][..., members].sum(axis=-1)
    for index, name in enumerate(UNIT_NAMES):
        out[f"unit_{name}_power_point"] = natural["power"][..., index]
        out[f"unit_{name}_energy_point"] = natural["energy"][..., index]
    return out
