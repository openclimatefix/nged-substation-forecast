"""Tune each monitor's one parameter on a week-shuffled placebo, and freeze it.

**The tuning reads no injected loss.** For each monitor, the parameter is the most sensitive one in
its grid whose false alarms per farm-year, summed over `TUNING_SHUFFLES` week-shuffled copies of the
five pool farms' signals, are at most `TARGET_FALSE_ALARMS_PER_FARM_YEAR`. The result is written to
`e00072_frozen_config.json`, which the scoring script reads and never changes.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
from e00072_inject import (
    METHODS,
    MethodType,
    farm_years,
    make_alarm_function,
    tuned_parameter,
    week_shuffle,
)
from e00072_signal import POOL_SITES, clean_baseline, load_record
from studies.guards import refuse_to_overwrite

TUNING_SHUFFLES: Final[int] = 20
TUNING_SEED_BASE: Final[int] = 0
"""Tuning shuffle `k` uses seed `TUNING_SEED_BASE + k`; the scoring seeds start at 1000."""

TARGET_FALSE_ALARMS_PER_FARM_YEAR: Final[float] = 0.35

GRIDS: Final[dict[MethodType, tuple[float, ...]]] = {
    "changepoint": (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0),
    "rolling_median": tuple(round(0.04 + 0.02 * step, 2) for step in range(14)),
    "hyst": tuple(round(0.04 + 0.02 * step, 2) for step in range(14)),
}
"""Each monitor's candidate parameters: penalty multiplier, threshold, and drop."""

FROZEN_PATH: Final[Path] = Path(__file__).parent / "e00072_frozen_config.json"


def placebo_signals(*, seed: int) -> list[tuple[np.ndarray, float]]:
    """Return the five pool farms' week-shuffled signals for one seed, with their farm-years.

    Args:
        seed: The shuffle's seed.

    Returns:
        For each pool farm, in `POOL_SITES` order, the shuffled signal and the years from the
        unshuffled signal's first valid day to its last.
    """
    record = load_record()
    baseline = clean_baseline(record=record)
    rng = np.random.default_rng(seed)
    signals = []
    for site in POOL_SITES:
        farm = record.sites.index(site)
        signal = baseline.signal[farm].copy()
        if site == "A":
            signal[baseline.low_months] = np.nan
        signals.append(
            (week_shuffle(signal=signal, dates=record.dates, rng=rng), farm_years(signal=signal))
        )
    return signals


def false_alarm_rates(
    *, seeds: range, method: MethodType, grid: tuple[float, ...]
) -> dict[float, float]:
    """Return a monitor's false alarms per farm-year on the placebo, for each parameter.

    Args:
        seeds: The shuffle seeds.
        method: Which monitor.
        grid: The candidate parameters.

    Returns:
        A mapping from parameter to alarms per farm-year, pooled over seeds and farms.
    """
    shuffled = [placebo_signals(seed=seed) for seed in seeds]
    years = sum(farm_span for signals in shuffled for _, farm_span in signals)
    rates = {}
    for parameter in grid:
        alarm_function = make_alarm_function(method=method, parameter=parameter)
        count = sum(len(alarm_function(signal)) for signals in shuffled for signal, _ in signals)
        rates[parameter] = count / years
    return rates


def main() -> None:
    """Tune the three monitors and write the frozen configuration."""
    refuse_to_overwrite(paths=[FROZEN_PATH])
    seeds = range(TUNING_SEED_BASE, TUNING_SEED_BASE + TUNING_SHUFFLES)
    frozen: dict[str, object] = {
        "target": TARGET_FALSE_ALARMS_PER_FARM_YEAR,
        "tuning_seeds": [seeds.start, seeds.stop],
    }
    for method in METHODS:
        rates = false_alarm_rates(seeds=seeds, method=method, grid=GRIDS[method])
        chosen = tuned_parameter(rates=rates, target=TARGET_FALSE_ALARMS_PER_FARM_YEAR)
        print(method, "chosen", chosen, {key: round(value, 3) for key, value in rates.items()})
        frozen[method] = {"parameter": chosen, "rates": {str(k): v for k, v in rates.items()}}
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2) + "\n")


if __name__ == "__main__":
    main()
