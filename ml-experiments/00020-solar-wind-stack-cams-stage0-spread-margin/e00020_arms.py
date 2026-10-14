"""Apply the spread factor and save each arm's predictions for experiment 00020.

Reads the pooled mixtures of `e00020_fit.py`, scales them about the median where an arm has the
spread factor, and writes one prediction set per arm under
`data/ml-experiments/00020-.../predictions/`. A set holds the keys, `seed`, `lead_hours`,
`actual_norm`, `technology`, `band`, `trivial`, and the 9 quantile columns. The factor is the grid
value nearest 90% coverage of the 5% to 95% band, fitted for each scored month on the arm's own
non-trivial cases with valid times before the first run of that month (at least `MIN_FIT_CASES` of
them; otherwise the month is left uncorrected). Solar uses one factor, and wind one factor for 0 to
2 days and one for 3 to 15 days.

`trivial` is the uncorrected `base` mixture's flag (95% quantile at most 1% of capacity), shared by
every arm of a technology so that coverage uses one set of cases.

    uv run python ml-experiments/00020-*/e00020_arms.py
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00020_common import (
    ARM_SOURCE,
    EXPERIMENT_ID,
    KEY_COLUMNS,
    LEVEL_COLUMNS,
    MIN_FIT_CASES,
    ROW_SET_ID,
    SEEDS,
    SHORT_BANDS,
    TECHNOLOGIES,
    TRIVIAL_LEVEL,
    UPPER_INDEX,
    WORK_DIR,
    ArmType,
    TechnologyType,
    VariantType,
    crps_estimate,
    fit_spread_factor,
    scale_about_median,
    validation_cases,
)

from studies import experiment_store

MIXTURE_DIR: Final[Path] = WORK_DIR / "mixture"
PUBLISHED_CRPS: Final[dict[str, float]] = {"wind": 11.11, "solar": 5.05}
"""Experiment 00002's `b_mixture_full` mean CRPS per technology (points of capacity)."""
REPRODUCTION_TOLERANCE: Final[float] = 0.05


def read_mixture(
    *, cases: pl.DataFrame, technology: TechnologyType, variant: VariantType, seed: int
) -> np.ndarray:
    """Return a variant's mixture quantiles for a technology's cases, in `cases` order.

    Args:
        cases: The technology's validation cases.
        technology: The technology whose mixtures to read.
        variant: The fitted-and-applied variant.
        seed: The fitting seed.

    Returns:
        Array of shape (cases, 9) of mixture quantiles, a fraction of capacity.

    Raises:
        ValueError: If a case has no mixture.
    """
    stored = pl.read_parquet(MIXTURE_DIR / f"{variant}_{technology}_seed{seed}.parquet")
    joined = cases.select(KEY_COLUMNS).join(
        stored, on=list(KEY_COLUMNS), how="left", validate="1:1"
    )
    if joined[LEVEL_COLUMNS[0]].null_count():
        msg = f"{variant} {technology} seed {seed}: some cases have no mixture"
        raise ValueError(msg)
    return joined.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)


def apply_spread(
    *, cases: pl.DataFrame, mixture: np.ndarray, technology: TechnologyType
) -> np.ndarray:
    """Return the quantiles after the spread factor, fitted month by month on earlier cases."""
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    valid = cases["valid_time"].to_numpy()
    init = cases["power_fcst_init_time"].to_numpy()
    month = cases["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    short = np.isin(cases["band"].to_numpy(), list(SHORT_BANDS))
    scorable = mixture[:, UPPER_INDEX] > TRIVIAL_LEVEL
    out = mixture.copy()
    groups = [np.ones_like(short)] if technology == "solar" else [short, ~short]
    for month_label in sorted(set(month)):
        in_month = month == month_label
        cutoff = init[in_month].min()
        for group in groups:
            target = in_month & group & scorable
            fit_index = np.flatnonzero((valid < cutoff) & group & scorable)
            if not target.any() or fit_index.size < MIN_FIT_CASES:
                continue
            factor = fit_spread_factor(quantiles=mixture[fit_index], actual=actual[fit_index])
            out[target] = scale_about_median(quantiles=mixture[target], shift=0.0, factor=factor)
    return out


def main() -> None:
    """Build every arm for every seed, check the reproduction, and save the prediction sets."""
    all_cases = validation_cases()
    frames: dict[ArmType, list[pl.DataFrame]] = {}
    for technology in TECHNOLOGIES:
        cases = all_cases.filter(pl.col("technology") == technology)
        for seed in SEEDS:
            base = read_mixture(cases=cases, technology=technology, variant="base", seed=seed)
            trivial = pl.Series("trivial", base[:, UPPER_INDEX] <= TRIVIAL_LEVEL)
            here = float(
                100.0 * crps_estimate(quantiles=base, actual=cases["actual_norm"].to_numpy()).mean()
            )
            print(f"{technology} seed {seed}: base scores {here:.3f}", flush=True)
            if abs(here - PUBLISHED_CRPS[technology]) > REPRODUCTION_TOLERANCE:
                msg = (
                    f"{technology}: base scores {here:.3f}, experiment 00002 scored it as published"
                )
                raise ValueError(msg)
            cache: dict[tuple[VariantType, bool], np.ndarray] = {}
            for arm, (variant, spread) in ARM_SOURCE[technology].items():
                if (variant, spread) not in cache:
                    mixture = read_mixture(
                        cases=cases, technology=technology, variant=variant, seed=seed
                    )
                    cache[variant, spread] = (
                        apply_spread(cases=cases, mixture=mixture, technology=technology)
                        if spread
                        else mixture
                    )
                frames.setdefault(arm, []).append(
                    cases.with_columns(seed=pl.lit(seed), trivial=trivial).hstack(
                        pl.DataFrame(
                            cache[variant, spread].astype(np.float32), schema=list(LEVEL_COLUMNS)
                        )
                    )
                )
    for arm, pieces in frames.items():
        experiment_store.write_predictions(
            experiment_id=EXPERIMENT_ID,
            name=arm.replace("_", "-"),
            frame=pl.concat(pieces),
            kind="quantiles",
            row_set_id=ROW_SET_ID,
        )


if __name__ == "__main__":
    main()
