"""Known-answer self-tests of experiment 72's monitors. Run before tuning or scoring.

Every test prints one line and the script exits non-zero if any fails.
"""

import sys

import numpy as np
from e00072_detectors import (
    MIN_SEGMENT_VALID_DAYS,
    change_point_alarms,
    hysteresis_alarms,
    optimal_partition,
    rolling_median_alarms,
)
from e00072_inject import week_shuffle

SEED: int = 7
FROZEN_SCALE: float = 6.0
DAYS: int = 600
STEP_DAY: int = 300


def _noisy(*, step: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Return a signal of `DAYS` days with a multiplicative loss from `STEP_DAY`, gaps of 60%."""
    signal = np.log(1.0 - step) * (np.arange(DAYS) >= STEP_DAY) + rng.normal(0.0, noise, DAYS)
    signal[rng.random(DAYS) < 0.6] = np.nan
    return signal


def main() -> None:
    """Run the self-tests and exit with status 1 if one fails."""
    rng = np.random.default_rng(SEED)
    failures = []

    def check(name: str, passed: bool) -> None:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        if not passed:
            failures.append(name)

    flat = np.zeros(DAYS)
    check("constant signal: no alarm", change_point_alarms(signal=flat, penalty_scale=2.0) == [])
    check(
        "constant signal: one segment",
        optimal_partition(values=flat, penalty=1.0, min_length=MIN_SEGMENT_VALID_DAYS) == [0],
    )
    clean = _noisy(step=0.25, noise=0.0, rng=rng)
    alarms = change_point_alarms(signal=clean, penalty_scale=2.0)
    check("noiseless 25% step: one alarm", len(alarms) == 1)
    if alarms:
        first_valid_after = int(np.flatnonzero(np.isfinite(clean[STEP_DAY:]))[0]) + STEP_DAY
        check("noiseless 25% step: on the step", abs(alarms[0].day - first_valid_after) <= 1)
        check("noiseless 25% step: size within 1 point", abs(alarms[0].size - 0.25) <= 0.01)
    noisy_ok = True
    for _ in range(20):
        noisy = _noisy(step=0.25, noise=0.08, rng=rng)
        found = change_point_alarms(signal=noisy, penalty_scale=FROZEN_SCALE)
        first_valid = int(np.flatnonzero(np.isfinite(noisy[STEP_DAY:]))[0]) + STEP_DAY
        noisy_ok &= (
            len(found) == 1
            and abs(found[0].day - first_valid) <= 14
            and abs(found[0].size - 0.25) <= 0.05
        )
    check("8% noise, 25% step, 20 repeats: one alarm, on the step, size within 5 points", noisy_ok)
    step_signal = np.concatenate([np.zeros(60), np.full(60, np.log(0.75))])
    rolling = rolling_median_alarms(signal=step_signal, threshold=0.12)
    check(
        "rolling median: one alarm near a noiseless 25% step",
        len(rolling) == 1 and abs(rolling[0].day - 60) <= 14,
    )
    check(
        "rolling median: no alarm on a flat signal",
        rolling_median_alarms(signal=np.zeros(120), threshold=0.12) == [],
    )
    hyst = hysteresis_alarms(ratio=np.exp(step_signal), drop=0.10)
    check(
        "hysteresis: one alarm within 7 days of a noiseless 25% step",
        len(hyst) == 1 and 60 <= hyst[0].day <= 67,
    )
    rise = np.log(1.25) * (np.arange(DAYS) >= STEP_DAY) + rng.normal(0.0, 0.01, DAYS)
    check("rise: no alarm", change_point_alarms(signal=rise, penalty_scale=2.0) == [])
    two_levels = np.concatenate([np.zeros(28), np.ones(28)])
    check(
        "zero penalty: splits exactly where the level changes",
        optimal_partition(values=two_levels, penalty=0.0, min_length=MIN_SEGMENT_VALID_DAYS)
        == [0, 28],
    )
    check(
        "huge penalty: one segment",
        optimal_partition(values=two_levels, penalty=1e6, min_length=MIN_SEGMENT_VALID_DAYS) == [0],
    )
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2024-01-01"))
    marked = np.arange(len(dates), dtype=float)
    shuffled = week_shuffle(signal=marked, dates=dates, rng=rng)
    check("week shuffle: same values", sorted(shuffled) == sorted(marked))
    check("week shuffle: moves values", not np.array_equal(shuffled, marked))
    years = dates.astype("datetime64[Y]").astype(int)
    day_of_year = (dates - dates.astype("datetime64[Y]").astype("datetime64[D]")).astype(int)
    source_week = np.minimum(day_of_year // 7, 51)[shuffled.astype(int)]
    check(
        "week shuffle: weeks stay in their week of the year",
        np.array_equal(source_week, np.minimum(day_of_year // 7, 51)),
    )
    check(
        "week shuffle: some day comes from another year",
        bool((years[shuffled.astype(int)] != years).any()),
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
