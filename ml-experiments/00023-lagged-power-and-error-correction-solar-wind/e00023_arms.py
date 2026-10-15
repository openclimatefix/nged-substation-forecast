"""Apply the 14-day correction and save each arm's predictions for experiment 00023.

Reads the pooled mixtures of `e00023_fit.py`, shifts the `mixture` and `cal_both` arms by the 14-day
correction where an arm has one, checks that `mixture` reproduces experiment 00002, and writes one
prediction set per arm under `data/ml-experiments/00023-.../predictions/`. A set holds the keys,
`seed`, `lead_hours`, `actual_norm`, `technology`, `band`, `trivial`, and the 9 quantile columns.

The correction for the run at time `t` is the mean of `observed - median` over the non-trivial cases
of earlier runs with a valid time in `[t - 14 d, t - 2 d]`, taking for each valid time (and series)
the forecast of the latest earlier run. Fewer than `MIN_CORRECTION_CASES` such cases means no shift.
It is applied to the run's non-trivial cases (95% quantile above 1% of capacity), clipped to 0 to 1
and sorted. `trivial` is the uncorrected `mixture` arm's flag, shared by every arm.

    uv run python ml-experiments/00023-*/e00023_arms.py
"""

from datetime import timedelta
from typing import Final

import numpy as np
import polars as pl
from e00023_common import (
    CORRECTION_DELAY_DAYS,
    CORRECTION_WINDOW_DAYS,
    EXPERIMENT_ID,
    KEY_COLUMNS,
    LEVEL_COLUMNS,
    MIN_CORRECTION_CASES,
    ROW_SET_ID,
    SEEDS,
    SLUG,
    TECHNOLOGIES,
    TRIVIAL_LEVEL,
    UPPER_INDEX,
    WORK_DIR,
    TechnologyType,
    crps_estimate,
    scale_about_median,
    validation_cases,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

from studies import experiment_store

MIXTURE_DIR: Final = WORK_DIR / "mixture"
PUBLISHED_CRPS: Final[dict[str, float]] = {"wind": 11.11, "solar": 5.05}
"""Experiment 00002's `b_mixture_full` mean CRPS per technology (points of capacity)."""
REPRODUCTION_TOLERANCE: Final[float] = 0.05
MEDIAN_INDEX: Final[int] = 4
CORRECTED_FROM: Final[dict[str, tuple[str, str]]] = {
    "mix_c14": ("mixture", "mean"),
    "cal_both_c14": ("cal_both", "mean"),
    "mix_c14_median": ("mixture", "median"),
}
"""Each corrected arm: the arm it corrects and the statistic of the window's errors. The median
statistic is post hoc."""


def read_mixture(
    *, cases: pl.DataFrame, technology: TechnologyType, arm: str, seed: int
) -> np.ndarray:
    """Return an arm's mixture quantiles for a technology's cases, in `cases` order.

    Raises:
        ValueError: If a case has no mixture.
    """
    stored = pl.read_parquet(MIXTURE_DIR / f"{arm}_{technology}_seed{seed}.parquet")
    joined = cases.select(KEY_COLUMNS).join(
        stored, on=list(KEY_COLUMNS), how="left", validate="1:1"
    )
    if joined[LEVEL_COLUMNS[0]].null_count():
        msg = f"{arm} {technology} seed {seed}: some cases have no mixture"
        raise ValueError(msg)
    return joined.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)


def apply_correction(
    *, cases: pl.DataFrame, mixture: np.ndarray, statistic: str = "mean"
) -> np.ndarray:
    """Return the quantiles after the 14-day correction, run by run from earlier runs only.

    Args:
        cases: The technology's validation cases.
        mixture: Shape (cases, 9) quantiles to correct.
        statistic: `mean` (the pre-registered correction) or `median` (post hoc) of the window's
            errors.
    """
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    valid = cases["valid_time"].to_numpy()
    init = cases["power_fcst_init_time"].to_numpy()
    series = cases["time_series_id"].to_numpy()
    scorable = mixture[:, UPPER_INDEX] > TRIVIAL_LEVEL
    error = actual - mixture[:, MEDIAN_INDEX]
    window = np.timedelta64(timedelta(days=CORRECTION_WINDOW_DAYS))
    delay = np.timedelta64(timedelta(days=CORRECTION_DELAY_DAYS))
    out = mixture.copy()
    for run in np.unique(init):
        target = np.flatnonzero((init == run) & scorable)
        earlier = np.flatnonzero(
            (init < run) & scorable & (valid >= run - window) & (valid <= run - delay)
        )
        if earlier.size == 0:
            continue
        # The latest earlier run's forecast for each series and valid time.
        order = earlier[np.lexsort((init[earlier], valid[earlier], series[earlier]))]
        last = np.ones(order.size, dtype=bool)
        same = (series[order][1:] == series[order][:-1]) & (valid[order][1:] == valid[order][:-1])
        last[:-1] = ~same
        used = order[last]
        if used.size < MIN_CORRECTION_CASES:
            continue
        out[target] = scale_about_median(
            quantiles=mixture[target],
            shift=float(np.median(error[used]) if statistic == "median" else error[used].mean()),
            factor=1.0,
        )
    return out


def main() -> None:
    """Build every arm for every seed, check the reproduction, and save the prediction sets."""
    all_cases = validation_cases()
    frames: dict[str, list[pl.DataFrame]] = {}
    for technology in TECHNOLOGIES:
        cases = all_cases.filter(pl.col("technology") == technology)
        for seed in SEEDS:
            base = read_mixture(cases=cases, technology=technology, arm="mixture", seed=seed)
            here = float(
                100.0 * crps_estimate(quantiles=base, actual=cases["actual_norm"].to_numpy()).mean()
            )
            print(f"{technology} seed {seed}: mixture scores {here:.3f}", flush=True)
            if abs(here - PUBLISHED_CRPS[technology]) > REPRODUCTION_TOLERANCE:
                msg = f"{technology}: mixture scores {here:.3f}, experiment 00002 published it"
                raise ValueError(msg)
            trivial = pl.Series("trivial", base[:, UPPER_INDEX] <= TRIVIAL_LEVEL)
            arms: dict[str, np.ndarray] = {
                arm: read_mixture(cases=cases, technology=technology, arm=arm, seed=seed)
                for arm in ("mixture", "cal0", "cal_lags", "cal_sums", "cal_both")
            }
            for corrected, (source, statistic) in CORRECTED_FROM.items():
                arms[corrected] = apply_correction(
                    cases=cases, mixture=arms[source], statistic=statistic
                )
            for arm, quantiles in arms.items():
                frames.setdefault(arm, []).append(
                    cases.with_columns(seed=pl.lit(seed), trivial=trivial).hstack(
                        pl.DataFrame(quantiles.astype(np.float32), schema=list(LEVEL_COLUMNS))
                    )
                )
    for arm, pieces in frames.items():
        stored = (
            ML_EXPERIMENTS_DATA_DIR
            / f"{EXPERIMENT_ID:05d}-{SLUG}"
            / "predictions"
            / arm.replace("_", "-")
        )
        if stored.exists():
            continue
        experiment_store.write_predictions(
            experiment_id=EXPERIMENT_ID,
            name=arm.replace("_", "-"),
            frame=pl.concat(pieces),
            kind="quantiles",
            row_set_id=ROW_SET_ID,
        )


if __name__ == "__main__":
    main()
