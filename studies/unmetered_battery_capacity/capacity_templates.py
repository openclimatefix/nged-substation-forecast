"""Build, save, and load the battery templates and the combination grid of the template posterior.

A *setting* is one set of the fixed choices: `standard` has the state-of-charge limits at 5% and
95%, a cap of 1 cycle a day for the Agile price taker, and the priors of the plan, and `sensitivity`
has the limits at 0% and 100%, a cap of 2 cycles a day for the Agile price taker, the duration
priors' log standard deviations doubled, and the efficiency prior's 95% range widened to 0.70 to
0.95. The merchant class's cycle cap (1 or 2 a day) is a grid dimension in both settings.

For each setting this script builds 168 candidate columns over the whole study year: for each of 4
round-trip efficiencies, 6 durations times 2 cycle caps for the merchant class, 6 durations for the
commercial and industrial class, and 6 durations times 4 tariffs for the domestic class. It also
builds 3 charge-only nuisance columns (one per fixed tariff window start). The columns are saved to
`templates_<setting>.parquet`, and `fit_block` fits one aggregate in one block from them.

Run: `OMP_NUM_THREADS=2 uv run python studies/unmetered_battery_capacity/capacity_templates.py`
"""

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from capacity_inputs import (
    OUTPUT_DIR,
    agile_prices,
    block_slices,
    day_ahead_on_grid,
    solar_columns,
    window_half_hours,
)
from scipy.optimize import brentq
from scipy.stats import beta as beta_distribution
from scipy.stats import norm
from studies.battery_templates import (
    DOMESTIC_TARIFFS,
    SENSITIVITY_LP_SETTINGS,
    STANDARD_LP_SETTINGS,
    TARIFF_WINDOWS,
    LpSettings,
    TariffNameType,
    charge_only_template,
    domestic_template,
    merchant_template,
    window_template,
)
from studies.pv_separation import baseline_design
from studies.template_posterior import ComboGrid, SumPosterior, fit_aggregate

SettingNameType = Literal["standard", "sensitivity"]
SETTING_NAMES: Final[tuple[SettingNameType, ...]] = ("standard", "sensitivity")
DURATIONS_HOURS: Final[tuple[float, ...]] = (0.5, 0.82, 1.35, 2.2, 3.6, 6.0)
"""Six log-spaced usable durations, in hours at full power."""
ROUND_TRIP_EFFICIENCIES: Final[tuple[float, ...]] = (0.78, 0.83, 0.88, 0.93)
MERCHANT_CYCLE_CAPS: Final[tuple[float, ...]] = (1.0, 2.0)
"""The merchant class's cycles-a-day cap values, each with prior probability one half."""
CLASS_NAMES: Final[tuple[str, ...]] = ("merchant", "commercial_and_industrial", "domestic")
NUISANCE_WINDOWS: Final[tuple[TariffNameType, ...]] = (
    "intelligent_octopus_go",
    "octopus_go",
    "octopus_flux",
)
NUISANCE_CHARGE_HOURS: Final[float] = 3.0
POWER_PRIOR_FRACTION_OF_P99: Final[float] = 0.2
"""The scale of the half-normal prior of every class power, as a fraction of the aggregate's 99th
percentile absolute flow."""
MAX_WORKERS: Final[int] = 4
TWO_SIDED_95: Final[float] = 1.959964


@dataclass(frozen=True)
class SettingSpec:
    """The fixed choices of one setting.

    Attributes:
        lp: The state-of-charge limits and cycle cap of the price-taker schedules.
        duration_sd_multiplier: Multiplies the log standard deviation of every duration prior.
        efficiency_interval: The 95% prior interval of the round-trip efficiency.
    """

    lp: LpSettings
    duration_sd_multiplier: float
    efficiency_interval: tuple[float, float]


SETTINGS: Final[dict[SettingNameType, SettingSpec]] = {
    "standard": SettingSpec(STANDARD_LP_SETTINGS, 1.0, (0.80, 0.92)),
    "sensitivity": SettingSpec(SENSITIVITY_LP_SETTINGS, 2.0, (0.70, 0.95)),
}
EFFICIENCY_PRIOR_MEAN: Final[float] = 0.87
DURATION_PRIORS: Final[dict[str, tuple[float, float]]] = {
    "merchant": (2.0, float(np.log(4.0 / 1.0) / (2 * TWO_SIDED_95))),
    "commercial_and_industrial": (1.5, float(np.log(3.0 / 1.0) / (2 * TWO_SIDED_95))),
    "domestic": (2.0, float(np.log(3.5 / 1.2) / (2 * TWO_SIDED_95))),
}
"""The median, in hours, and the log standard deviation of each class's log-normal duration prior:
95% between 1 and 4 hours (merchant), 1 and 3 hours (commercial and industrial), and 1.2 and 3.5
hours (domestic)."""


def duration_log_prior(*, class_name: str, setting: SettingNameType) -> np.ndarray:
    """Return the normalised log prior weight of each duration grid value.

    The weight of a grid value is the log-normal density at the value times the value (the density
    per unit of log duration), normalised over the log-spaced grid.

    Args:
        class_name: The battery class.
        setting: The setting.

    Returns:
        Six log weights that sum to 1 after exponentiation.
    """
    median, sd = DURATION_PRIORS[class_name]
    z = (np.log(DURATIONS_HOURS) - np.log(median)) / (sd * SETTINGS[setting].duration_sd_multiplier)
    log_weight = norm.logpdf(z)
    return log_weight - np.logaddexp.reduce(log_weight)


def beta_from_interval(*, mean: float, low: float, high: float) -> tuple[float, float]:
    """Return the Beta distribution of a given mean whose central 95% interval is `(low, high)`.

    Args:
        mean: The distribution's mean.
        low: The 2.5th percentile.
        high: The 97.5th percentile.

    Returns:
        The shape parameters `(alpha, beta)`.
    """

    def width_error(concentration: float) -> float:
        a, b = mean * concentration, (1 - mean) * concentration
        return float(beta_distribution.ppf(0.975, a, b) - beta_distribution.ppf(0.025, a, b)) - (
            high - low
        )

    concentration = brentq(width_error, 2.0, 1e5)
    return mean * concentration, (1 - mean) * concentration


def efficiency_log_prior(*, setting: SettingNameType) -> np.ndarray:
    """Return the normalised log prior weight of each efficiency grid value.

    Args:
        setting: The setting.

    Returns:
        Four log weights that sum to 1 after exponentiation.
    """
    low, high = SETTINGS[setting].efficiency_interval
    a, b = beta_from_interval(mean=EFFICIENCY_PRIOR_MEAN, low=low, high=high)
    log_weight = beta_distribution.logpdf(ROUND_TRIP_EFFICIENCIES, a, b)
    return log_weight - np.logaddexp.reduce(log_weight)


def candidate_names() -> list[str]:
    """Return the 168 candidate column names, in column order.

    Returns:
        For each efficiency: merchant durations at cap 1, merchant durations at cap 2, commercial
        and industrial durations, then for each domestic duration the four tariffs.
    """
    names = []
    for efficiency in ROUND_TRIP_EFFICIENCIES:
        for cap in MERCHANT_CYCLE_CAPS:
            names += [f"merchant_cap{cap:.0f}_d{d:.2f}_e{efficiency:.2f}" for d in DURATIONS_HOURS]
        names += [f"commercial_and_industrial_d{d:.2f}_e{efficiency:.2f}" for d in DURATIONS_HOURS]
        for d in DURATIONS_HOURS:
            names += [f"domestic_{t}_d{d:.2f}_e{efficiency:.2f}" for t in DOMESTIC_TARIFFS]
    return names


COLUMNS_PER_EFFICIENCY: Final[int] = 42


def column_index(
    *, efficiency: int, class_name: str, duration: int, tariff: int = 0, cap: int = 0
) -> int:
    """Return a candidate column's position.

    Args:
        efficiency: Index into `ROUND_TRIP_EFFICIENCIES`.
        class_name: One of `CLASS_NAMES`.
        duration: Index into `DURATIONS_HOURS`.
        tariff: Index into `DOMESTIC_TARIFFS` for the domestic class.
        cap: Index into `MERCHANT_CYCLE_CAPS` for the merchant class.

    Returns:
        The column's index in the candidate array.
    """
    n_d = len(DURATIONS_HOURS)
    base = efficiency * COLUMNS_PER_EFFICIENCY
    if class_name == "merchant":
        return base + cap * n_d + duration
    if class_name == "commercial_and_industrial":
        return base + len(MERCHANT_CYCLE_CAPS) * n_d + duration
    return base + (len(MERCHANT_CYCLE_CAPS) + 1) * n_d + duration * len(DOMESTIC_TARIFFS) + tariff


@cache
def combo_grid(*, setting: SettingNameType) -> ComboGrid:
    """Return the 1,728 combinations of durations, merchant cycle cap, and efficiency.

    The flat index runs over `(merchant duration, merchant cap, commercial duration, domestic
    duration, efficiency)` with the efficiency varying fastest. Each combination has 6 power
    columns: the merchant column, the commercial and industrial column, and the four domestic
    tariff columns.

    Args:
        setting: The setting, which fixes the priors.

    Returns:
        The grid, with `axes = (6, 2, 6, 6, 4)`.
    """
    n_d, n_e, n_cap = len(DURATIONS_HOURS), len(ROUND_TRIP_EFFICIENCIES), len(MERCHANT_CYCLE_CAPS)
    prior_m = duration_log_prior(class_name="merchant", setting=setting)
    prior_c = duration_log_prior(class_name="commercial_and_industrial", setting=setting)
    prior_d = duration_log_prior(class_name="domestic", setting=setting)
    prior_e = efficiency_log_prior(setting=setting)
    log_prior_cap = -np.log(n_cap)
    combos = []
    log_prior = []
    for dm in range(n_d):
        for cap in range(n_cap):
            for dc in range(n_d):
                for dd in range(n_d):
                    for e in range(n_e):
                        columns = [
                            column_index(efficiency=e, class_name="merchant", duration=dm, cap=cap),
                            column_index(
                                efficiency=e, class_name="commercial_and_industrial", duration=dc
                            ),
                            *[
                                column_index(
                                    efficiency=e, class_name="domestic", duration=dd, tariff=t
                                )
                                for t in range(len(DOMESTIC_TARIFFS))
                            ],
                        ]
                        combos.append(columns)
                        log_prior.append(
                            prior_m[dm] + log_prior_cap + prior_c[dc] + prior_d[dd] + prior_e[e]
                        )
    return ComboGrid(
        combos=np.array(combos),
        log_prior=np.array(log_prior),
        axes=(n_d, n_cap, n_d, n_d, n_e),
    )


def _build_one(task: tuple[SettingNameType, int, int, str, int, int]) -> tuple[int, np.ndarray]:
    """Build one candidate column; the worker function of the process pool."""
    setting, efficiency, duration, class_name, tariff, cap = task
    position = column_index(
        efficiency=efficiency, class_name=class_name, duration=duration, tariff=tariff, cap=cap
    )
    lp = SETTINGS[setting].lp
    grid = window_half_hours()
    eta = ROUND_TRIP_EFFICIENCIES[efficiency]
    d = DURATIONS_HOURS[duration]
    if class_name == "merchant":
        column = merchant_template(
            day_ahead_prices=day_ahead_on_grid(),
            duration_hours=d,
            round_trip_efficiency=eta,
            settings=replace(lp, cycles_per_day_cap=MERCHANT_CYCLE_CAPS[cap]),
        )
    elif class_name == "commercial_and_industrial":
        column = window_template(
            half_hour_end_time=grid,
            window=TARIFF_WINDOWS["red_band"],
            duration_hours=d,
            round_trip_efficiency=eta,
        )
    else:
        column = domestic_template(
            tariff=DOMESTIC_TARIFFS[tariff],
            half_hour_end_time=grid,
            agile_prices=agile_prices(),
            duration_hours=d,
            round_trip_efficiency=eta,
            settings=lp,
        )
    return position, column


def build_candidates(*, setting: SettingNameType) -> np.ndarray:
    """Build the 168 candidate template columns over the study year.

    Args:
        setting: The setting.

    Returns:
        Shape (17,520, 168): one-megawatt schedules, positive for export.
    """
    tasks: list[tuple[SettingNameType, int, int, str, int, int]] = []
    for e in range(len(ROUND_TRIP_EFFICIENCIES)):
        for d in range(len(DURATIONS_HOURS)):
            tasks.extend(
                (setting, e, d, "merchant", 0, cap) for cap in range(len(MERCHANT_CYCLE_CAPS))
            )
            tasks.append((setting, e, d, "commercial_and_industrial", 0, 0))
            tasks.extend((setting, e, d, "domestic", t, 0) for t in range(len(DOMESTIC_TARIFFS)))
    out = np.zeros((len(window_half_hours()), len(tasks)))
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for position, column in pool.map(_build_one, tasks, chunksize=2):
            out[:, position] = column
    return out


def nuisance_candidates() -> np.ndarray:
    """Return the three charge-only nuisance columns, one per fixed tariff window start.

    Returns:
        Shape (17,520, 3).
    """
    grid = window_half_hours()
    return np.column_stack(
        [
            charge_only_template(
                half_hour_end_time=grid,
                window=TARIFF_WINDOWS[name],
                charge_hours=NUISANCE_CHARGE_HOURS,
            )
            for name in NUISANCE_WINDOWS
        ]
    )


def templates_path(*, setting: SettingNameType) -> Path:
    """Return where a setting's saved templates live."""
    return OUTPUT_DIR / f"templates_{setting}.parquet"


def load_templates(*, setting: SettingNameType) -> tuple[np.ndarray, np.ndarray]:
    """Load a setting's saved candidate and nuisance columns.

    Args:
        setting: The setting.

    Returns:
        The candidates (17,520 by 168) and the nuisance columns (17,520 by 3).
    """
    frame = pl.read_parquet(templates_path(setting=setting))
    candidates = frame.select(candidate_names()).to_numpy()
    nuisance = frame.select([f"nuisance_{n}" for n in NUISANCE_WINDOWS]).to_numpy()
    return candidates, nuisance


@cache
def _solar_and_grid() -> tuple[np.ndarray, pl.Series]:
    return solar_columns(), window_half_hours()


def block_free_columns(*, block: int, nuisance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a block's free columns and the rows where they are finite.

    The free columns are the monthly calendar baseline (a column per local half-hour and day type,
    and another per local half-hour and month, with the columns that are zero throughout the block
    dropped), the four solar fleet curves, and the three charge-only nuisance columns.

    Args:
        block: The block's index.
        nuisance: The year's nuisance columns.

    Returns:
        The free columns of the block, and a mask of rows where the solar columns are finite.
    """
    solar, grid = _solar_and_grid()
    rows = block_slices()[block]
    baseline = baseline_design(half_hour_end_time=grid[rows], flexibility="monthly")
    baseline = baseline[:, baseline.any(axis=0)]
    free = np.hstack([baseline, solar[rows], nuisance[rows]])
    return free, np.isfinite(solar[rows]).all(axis=1)


def fit_block(
    *,
    series: np.ndarray,
    block: int,
    candidates: np.ndarray,
    nuisance: np.ndarray,
    setting: SettingNameType,
    rng: np.random.Generator,
) -> SumPosterior:
    """Fit one aggregate in one block.

    Args:
        series: The aggregate over the whole year in MW, import-positive, NaN where missing.
        block: The block's index.
        candidates: The year's candidate template columns (export-positive).
        nuisance: The year's nuisance columns.
        setting: The setting.
        rng: The random generator.

    Returns:
        The posterior over the combination grid.
    """
    rows = block_slices()[block]
    free, solar_ok = block_free_columns(block=block, nuisance=nuisance)
    y = series[rows]
    valid = np.isfinite(y) & solar_ok
    scale = POWER_PRIOR_FRACTION_OF_P99 * float(np.quantile(np.abs(y[valid]), 0.99))
    return fit_aggregate(
        free=free,
        candidates=-candidates[rows],
        target=np.nan_to_num(y),
        valid=valid,
        grid=combo_grid(setting=setting),
        power_prior_scale=max(scale, 1e-6),
        rng=rng,
    )


def main() -> None:
    """Build and save the templates of both settings."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nuisance = nuisance_candidates()
    for setting in SETTING_NAMES:
        started = time.monotonic()
        candidates = build_candidates(setting=setting)
        columns = {name: candidates[:, k] for k, name in enumerate(candidate_names())}
        columns |= {f"nuisance_{n}": nuisance[:, k] for k, n in enumerate(NUISANCE_WINDOWS)}
        pl.DataFrame({"half_hour_end_time": window_half_hours(), **columns}).write_parquet(
            templates_path(setting=setting)
        )
        print(f"{setting}: {candidates.shape[1]} columns in {time.monotonic() - started:.0f} s")
    for setting in SETTING_NAMES:
        grid = combo_grid(setting=setting)
        prior_mass = np.exp(grid.log_prior).sum()
        print(
            setting, "combinations:", grid.combos.shape, "prior mass:", round(float(prior_mass), 6)
        )


if __name__ == "__main__":
    main()
