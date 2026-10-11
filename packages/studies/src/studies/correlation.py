"""The Pearson correlation of a prediction with the measured value, with a month-resampled interval.

Written for the ERA5 variable ladder study, planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1097>. The interval resamples
whole calendar months and one fitting seed, the same design `studies.bootstrap` uses for a mean
absolute error, so a correlation and an error beside it rest on the same resampling.
"""

from typing import Final, TypedDict

import numpy as np

from studies.bootstrap import BOOTSTRAP_SEED, N_BOOTSTRAP_RESAMPLES

SUMMARY_COLUMNS: Final[int] = 6
"""The summary sums kept per month and seed: the row count, then the sums of x, y, x², y², x·y."""


class CorrelationInterval(TypedDict):
    """A pooled correlation and its interval at the level asked for, and what it rests on."""

    correlation: float
    lower: float
    upper: float
    n_rows: int
    n_months: int


def _correlation_from_sums(*, sums: np.ndarray) -> float:
    """Return the Pearson correlation from the summary sums of a set of rows.

    Args:
        sums: The row count, then the sums of x, y, x², y², and x·y, as one array.

    Returns:
        The correlation, or not-a-number if either variable has no variance.
    """
    count, sum_x, sum_y, sum_xx, sum_yy, sum_xy = sums
    covariance = count * sum_xy - sum_x * sum_y
    variance_x = count * sum_xx - sum_x**2
    variance_y = count * sum_yy - sum_y**2
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(covariance / np.sqrt(variance_x * variance_y))


def _monthly_sums(
    *, actual: np.ndarray, prediction: np.ndarray, month_index: np.ndarray, n_months: int
) -> np.ndarray:
    """Return the summary sums of every (seed, month), about the global means.

    Args:
        actual: The measured values, shape (n_rows,).
        prediction: The predictions, shape (n_seeds, n_rows).
        month_index: Each row's month as an integer from 0 to `n_months - 1`.
        n_months: How many distinct months there are.

    Returns:
        An array of shape (n_seeds, n_months, `SUMMARY_COLUMNS`).
    """
    centred_actual = actual - actual.mean()
    sums = np.empty((prediction.shape[0], n_months, SUMMARY_COLUMNS))
    for seed_position, seed_prediction in enumerate(prediction):
        centred_prediction = seed_prediction - seed_prediction.mean()
        columns = (
            np.ones_like(centred_actual),
            centred_actual,
            centred_prediction,
            centred_actual**2,
            centred_prediction**2,
            centred_actual * centred_prediction,
        )
        for column, values in enumerate(columns):
            sums[seed_position, :, column] = np.bincount(
                month_index, weights=values, minlength=n_months
            )
    return sums


def pooled_correlation_interval(
    *, actual: np.ndarray, prediction: np.ndarray, months: np.ndarray, level: float = 95.0
) -> CorrelationInterval:
    """Return the correlation over all rows, and an interval from resampling whole months.

    The point estimate is the mean over the seeds of each seed's correlation with the measured
    value. Each resample draws one seed and then a set of months with replacement, in that order,
    and takes the correlation over the drawn months' rows.

    Args:
        actual: The measured values, shape (n_rows,).
        prediction: The out-of-fold predictions, shape (n_seeds, n_rows), row for row with `actual`.
        months: Each row's month label, shape (n_rows,).
        level: The interval's coverage in percent, above 1 and below 100 (95.0, never 0.95).

    Returns:
        The correlation, the interval, and the number of rows and months it rests on.

    Raises:
        ValueError: If `level` is not above 1 and below 100, or the shapes disagree.
    """
    if not 1.0 < level < 100.0:
        msg = f"level is a coverage in percent, above 1 and below 100 (95.0, not 0.95): {level}"
        raise ValueError(msg)
    if prediction.ndim != 2 or not prediction.shape[1] == actual.shape[0] == months.shape[0]:
        msg = (
            f"expected prediction (n_seeds, n_rows) beside actual and months of n_rows rows, got "
            f"{prediction.shape}, {actual.shape}, {months.shape}"
        )
        raise ValueError(msg)
    unique_months, month_index = np.unique(months, return_inverse=True)
    n_months = len(unique_months)
    sums = _monthly_sums(
        actual=actual, prediction=prediction, month_index=month_index, n_months=n_months
    )
    point = float(
        np.mean([_correlation_from_sums(sums=seed_sums.sum(axis=0)) for seed_sums in sums])
    )

    # The seed is drawn before the months, as `studies.bootstrap` does.
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = np.empty(N_BOOTSTRAP_RESAMPLES)
    for resample in range(N_BOOTSTRAP_RESAMPLES):
        seed_position = generator.integers(0, prediction.shape[0])
        drawn = generator.integers(0, n_months, size=n_months)
        counts = np.bincount(drawn, minlength=n_months)
        resampled[resample] = _correlation_from_sums(sums=counts @ sums[seed_position])
    tail = (100.0 - level) / 2.0
    return {
        "correlation": point,
        "lower": float(np.nanpercentile(resampled, tail)),
        "upper": float(np.nanpercentile(resampled, 100.0 - tail)),
        "n_rows": int(actual.shape[0]),
        "n_months": n_months,
    }
