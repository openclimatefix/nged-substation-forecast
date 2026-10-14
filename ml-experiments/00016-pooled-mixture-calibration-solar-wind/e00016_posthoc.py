"""Post hoc arm of experiment 00016: a spread factor per technology and two lead groups.

The planned `spread_band` arm needs 5,000 earlier non-trivial cases per lead band, which the
short-lead wind bands reach only late in the validation year. This arm pools the bands into two
groups (0 to 2 days, and 3 to 15 days) so that the fit has more cases, and is otherwise the
`spread` arm: the factor nearest 90% coverage, fitted on cases with valid times before the first
run of the scored month, non-trivial cases only. It is not part of the pre-registered contrasts.

    uv run python ml-experiments/00016-*/e00016_posthoc.py
"""

import numpy as np
import polars as pl
from e00016_common import (
    EXPERIMENT_ID,
    LEVEL_COLUMNS,
    MIN_FIT_CASES,
    ROW_SET_ID,
    SEEDS,
    TECHNOLOGIES,
    TRIVIAL_LEVEL,
    UPPER_INDEX,
    fit_spread_factor,
    scale_about_median,
    validation_cases,
)
from e00016_fit import mixture_for_seed

from studies import experiment_store

SHORT_BANDS: frozenset[str] = frozenset({"0 to 24 h", "1 to 2 days"})
SET_NAME: str = "spread-two-groups"


def correct_two_groups(*, cases: pl.DataFrame, mixture: np.ndarray) -> np.ndarray:
    """Return the quantiles after a spread factor per technology and lead group, month by month."""
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    valid = cases["valid_time"].to_numpy()
    init = cases["power_fcst_init_time"].to_numpy()
    technology = cases["technology"].to_numpy()
    short = np.isin(cases["band"].to_numpy(), list(SHORT_BANDS))
    month = cases["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    scorable = mixture[:, UPPER_INDEX] > TRIVIAL_LEVEL
    out = mixture.copy()
    for month_label in sorted(set(month)):
        in_month = month == month_label
        cutoff = init[in_month].min()
        for tech in TECHNOLOGIES:
            for is_short in (True, False):
                group = (technology == tech) & (short == is_short) & scorable
                target = in_month & group
                fit_index = np.flatnonzero((valid < cutoff) & group)
                if not target.any() or fit_index.size < MIN_FIT_CASES:
                    continue
                factor = fit_spread_factor(quantiles=mixture[fit_index], actual=actual[fit_index])
                out[target] = scale_about_median(
                    quantiles=mixture[target], shift=0.0, factor=factor
                )
    return out


def main() -> None:
    """Build the post hoc arm for every seed and save it as a prediction set."""
    cases = validation_cases()
    pieces = []
    for seed in SEEDS:
        mixture = mixture_for_seed(seed=seed, cases=cases)
        corrected = correct_two_groups(cases=cases, mixture=mixture)
        trivial = pl.Series("trivial", mixture[:, UPPER_INDEX] <= TRIVIAL_LEVEL)
        pieces.append(
            cases.with_columns(seed=pl.lit(seed), trivial=trivial).hstack(
                pl.DataFrame(corrected.astype(np.float32), schema=list(LEVEL_COLUMNS))
            )
        )
    experiment_store.write_predictions(
        experiment_id=EXPERIMENT_ID,
        name=SET_NAME,
        frame=pl.concat(pieces),
        kind="quantiles",
        row_set_id=ROW_SET_ID,
    )


if __name__ == "__main__":
    main()
