"""Save each arm's pooled-mixture predictions for experiment 00026.

Reads the pooled mixtures of `e00026_fit.py`, checks that arm A0 reproduces the benchmark of
experiments 00002 and 00020 (solar 5.03 and wind 11.05 within 0.05 points, per seed), and writes one
prediction set per arm under `data/ml-experiments/00026-.../predictions/`. A set holds the keys,
`seed`, `lead_hours`, `actual_norm`, `technology`, `band`, `trivial` (A0's flag, shared by every
arm), and the 9 quantile columns.

    uv run python ml-experiments/00026-*/e00026_arms.py
"""

from typing import Final

import numpy as np
import polars as pl
from e00026_common import (
    ARMS,
    EXPERIMENT_ID,
    KEY_COLUMNS,
    LEVEL_COLUMNS,
    ROW_SET_ID,
    SEEDS,
    SLUG,
    TECHNOLOGIES,
    TRIVIAL_LEVEL,
    UPPER_INDEX,
    WORK_DIR,
    crps_estimate,
    validation_cases,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

from studies import experiment_store

MIXTURE_DIR: Final = WORK_DIR / "mixture"
PUBLISHED_CRPS: Final[dict[str, float]] = {"wind": 11.05, "solar": 5.03}
REPRODUCTION_TOLERANCE: Final[float] = 0.05


def read_mixture(*, cases: pl.DataFrame, technology: str, arm: str, seed: int) -> np.ndarray:
    """Return an arm's mixture quantiles for a technology's cases, in `cases` order.

    Args:
        cases: The technology's validation cases.
        technology: `solar` or `wind`.
        arm: The arm whose mixtures to read.
        seed: The fitting seed.

    Returns:
        Array of shape (cases, 9) of mixture quantiles, a fraction of capacity.

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


def main() -> None:
    """Build every arm for every seed, check the reproduction, and save the prediction sets."""
    all_cases = validation_cases()
    frames: dict[str, list[pl.DataFrame]] = {arm: [] for arm in ARMS}
    for technology in TECHNOLOGIES:
        cases = all_cases.filter(pl.col("technology") == technology)
        for seed in SEEDS:
            base = read_mixture(cases=cases, technology=technology, arm="a0", seed=seed)
            here = float(
                100.0 * crps_estimate(quantiles=base, actual=cases["actual_norm"].to_numpy()).mean()
            )
            print(f"{technology} seed {seed}: A0 scores {here:.3f}", flush=True)
            if abs(here - PUBLISHED_CRPS[technology]) > REPRODUCTION_TOLERANCE:
                msg = f"{technology}: A0 scores {here:.3f}, the benchmark is {PUBLISHED_CRPS}"
                raise ValueError(msg)
            trivial = pl.Series("trivial", base[:, UPPER_INDEX] <= TRIVIAL_LEVEL)
            for arm in ARMS:
                quantiles = read_mixture(cases=cases, technology=technology, arm=arm, seed=seed)
                frames[arm].append(
                    cases.with_columns(seed=pl.lit(seed), trivial=trivial).hstack(
                        pl.DataFrame(quantiles.astype(np.float32), schema=list(LEVEL_COLUMNS))
                    )
                )
    for arm, pieces in frames.items():
        stored = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "predictions" / arm
        if stored.exists():
            continue
        experiment_store.write_predictions(
            experiment_id=EXPERIMENT_ID,
            name=arm,
            frame=pl.concat(pieces),
            kind="quantiles",
            row_set_id=ROW_SET_ID,
        )


if __name__ == "__main__":
    main()
