"""Scoring of a fitted plant on the ERA5 hold-out, shared by Stages 2, 3, and 3b.

A plant gives the expected power (the median). The 11 quantiles are the expected power plus the
empirical quantiles of the training residuals in 14 bins of expected power (experiment 230's
`ResidualQuantiles`, fitted on every training row, the experiment 210 mask included, as 230 did).
The score is experiment 33's quantile score in points of capacity; the hold-out is 2024-04 to
2025-06 with ERA5 as the weather, so it is an upper bound on what the plant can do.
"""

from dataclasses import dataclass

import e00230_core as core230  # ty: ignore[unresolved-import]
import e00347_common as common
import e00347_plant as plant
import numpy as np


@dataclass(frozen=True)
class Evaluation:
    """Per-row hold-out scores of one plant at one farm.

    Attributes:
        score: Quantile score in points of capacity, one per hold-out row.
        median: The plant's expected power on the hold-out rows.
        target: The observed fractions on the hold-out rows.
        kept: True where experiment 210's mask keeps the row.
        month: The calendar month label of each row.
        quantiles: Shape (rows, 11) hold-out quantiles.
    """

    score: np.ndarray
    median: np.ndarray
    target: np.ndarray
    kept: np.ndarray
    month: np.ndarray
    quantiles: np.ndarray

    def mean_score(self, *, rows: np.ndarray | None = None) -> float:
        """Return the mean score over `rows` (all hold-out rows when `None`)."""
        selection = np.ones(self.score.size, dtype=bool) if rows is None else rows
        return float(self.score[selection].mean())

    def r_squared(self, *, rows: np.ndarray) -> float:
        """Return the share of variance of the observed fraction the median explains on `rows`."""
        actual, forecast = self.target[rows], self.median[rows]
        return float(1 - ((actual - forecast) ** 2).sum() / ((actual - actual.mean()) ** 2).sum())

    def mean_absolute_error(self, *, rows: np.ndarray) -> float:
        """Return the mean absolute error of the median, in points of capacity."""
        return float(np.abs(self.target[rows] - self.median[rows]).mean() * 100.0)


def evaluate_expected(
    *, farm: common.FarmData, expected_train: np.ndarray, expected_holdout: np.ndarray
) -> Evaluation:
    """Score hold-out expected powers with residual quantiles fitted on the training rows."""
    train, holdout = farm.train, farm.holdout
    residuals = core230.ResidualQuantiles.fit(expected=expected_train, observed=train.target)
    quantiles = residuals.quantiles(expected=expected_holdout)
    score = common.score_points(quantiles=quantiles, actual=holdout.target)
    return Evaluation(
        score=score,
        median=expected_holdout,
        target=holdout.target,
        kept=~holdout.idle_mask,
        month=holdout.month,
        quantiles=quantiles,
    )


def evaluate_plant(
    *, farm: common.FarmData, parameters: plant.PlantParameters, prior_scale: float
) -> Evaluation:
    """Score a plant on the hold-out rows of `farm`."""
    train, holdout = farm.train, farm.holdout
    expected_train = plant.expected_power(
        parameters=parameters, weather=train.weather, prior_scale=prior_scale
    )
    expected_holdout = plant.expected_power(
        parameters=parameters, weather=holdout.weather, prior_scale=prior_scale
    )
    return evaluate_expected(
        farm=farm, expected_train=expected_train, expected_holdout=expected_holdout
    )
