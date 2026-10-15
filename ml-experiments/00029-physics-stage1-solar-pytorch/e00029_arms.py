"""Save each arm's predictions for experiment 00029.

Reads the pooled mixtures of `e00029_fit.py` and writes one prediction set per arm under
`data/ml-experiments/00029-.../predictions/`. A set holds the keys, `seed`, `lead_hours`,
`actual_norm`, `technology`, `band`, `trivial`, and the 9 quantile columns, as experiment 00020's
sets do. `trivial` is experiment 00020's `base` flag for the same case.

    uv run --no-sync python ml-experiments/00029-*/e00029_arms.py
"""

from typing import Final

import numpy as np
import polars as pl
from e00029_common import (
    EXPERIMENT_ID,
    KEY_COLUMNS,
    LEVEL_COLUMNS,
    ROW_SET_ID,
    SEEDS,
    UPPER_INDEX,
    crps_estimate,
    validation_cases,
)
from e00029_fit import ARMS, MIXTURE_DIR

from studies import experiment_store

BENCHMARK_ID: Final[int] = 20
PUBLISHED_SOLAR_CRPS: Final[float] = 5.05
REPRODUCTION_TOLERANCE: Final[float] = 0.05


def main() -> None:
    """Check the benchmark reproduces, then write a prediction set for every arm."""
    cases = validation_cases().with_columns(technology=pl.lit("solar"))
    base = experiment_store.read_predictions(
        experiment_id=BENCHMARK_ID, name="base", row_set_id=ROW_SET_ID
    ).filter(pl.col("time_series_id").is_in(cases["time_series_id"].unique().to_list()))
    here = float(
        100.0
        * crps_estimate(
            quantiles=base.select(LEVEL_COLUMNS).to_numpy().astype(np.float64),
            actual=base["actual_norm"].to_numpy().astype(np.float64),
        ).mean()
    )
    print(f"benchmark `base` scores {here:.3f}", flush=True)
    if abs(here - PUBLISHED_SOLAR_CRPS) > REPRODUCTION_TOLERANCE:
        msg = f"base scores {here:.3f}; experiment 00020 published {PUBLISHED_SOLAR_CRPS}"
        raise ValueError(msg)
    flags = base.select(*KEY_COLUMNS, "trivial").unique(subset=KEY_COLUMNS)
    cases = cases.join(flags, on=list(KEY_COLUMNS), how="left", validate="1:1")
    if cases["trivial"].null_count():
        msg = "some cases have no trivial flag in the benchmark"
        raise ValueError(msg)
    for arm in ARMS:
        pieces = []
        for seed in SEEDS:
            stored = pl.read_parquet(MIXTURE_DIR / f"{arm}_seed{seed}.parquet")
            joined = cases.join(stored, on=list(KEY_COLUMNS), how="left", validate="1:1")
            if joined[LEVEL_COLUMNS[UPPER_INDEX]].null_count():
                msg = f"{arm} seed {seed}: some cases have no mixture"
                raise ValueError(msg)
            pieces.append(joined.with_columns(seed=pl.lit(seed)))
        experiment_store.write_predictions(
            experiment_id=EXPERIMENT_ID,
            name=arm.replace("_", "-"),
            frame=pl.concat(pieces),
            kind="quantiles",
            row_set_id=ROW_SET_ID,
        )
        print(f"wrote {arm}", flush=True)


if __name__ == "__main__":
    main()
