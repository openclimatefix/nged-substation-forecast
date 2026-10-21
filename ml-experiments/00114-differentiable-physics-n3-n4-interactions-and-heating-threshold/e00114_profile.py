"""The heating-threshold profile of experiment 114: one design, 15 hinges, month-block weightings.

**One design holds every threshold.** The union design is B0's columns followed by one block of 16
hinge columns per threshold of the grid. A fit at threshold `t` uses B0's columns with the heating
block replaced, in place, by the hinge block of `t`, so B0's penalty and sign bounds apply
unchanged. A month-block bootstrap resample is a vector of month counts, and its Gram matrix is the
count-weighted sum of per-month Gram matrices, so one pass over the rows serves every threshold,
every weighting, and (for a synthetic target) every draw.
"""

from dataclasses import dataclass

import e00114_common as common
import e00114_design as design
import numpy as np
import torch
from studies.dp_calendar import SMOOTHING_HOURS
from studies.dp_features import FeatureCache
from studies.dp_gam import N_BANDS

N_HINGE_COLUMNS: int = len(SMOOTHING_HOURS) * N_BANDS
BASE_THRESHOLD_POSITION: int = common.HINGE_GRID_C.index(common.BASE_THRESHOLD_C)
ARGMIN_INTERVAL_PERCENTILES: tuple[float, float] = (10.0, 90.0)
"""The percentiles of the bootstrap arg-mins that bound an 80% interval."""
GRID_HALF_STEP_C: float = common.HINGE_STEP_C / 2.0


@dataclass(frozen=True)
class UnionDesign:
    """B0's design with a hinge block for every threshold, and its per-month Gram matrices.

    Attributes:
        base: B0's inputs at the cut.
        x: The union design, shape (n, p0 + 15 * 16).
        grams: The per-month Gram matrices of `x`.
        penalty: B0's penalty and sign bounds.
    """

    base: design.BaseFit
    x: np.ndarray
    grams: design.MonthGrams
    penalty: design.ArmPenalty

    def columns(self, *, threshold_index: int) -> np.ndarray:
        """Return the union columns of the fit at one threshold, in B0's column order.

        Args:
            threshold_index: Position of the threshold in `common.HINGE_GRID_C`.

        Returns:
            Column indices into `x`: B0's, with the heating block replaced by the threshold's hinge.
        """
        columns = np.arange(self.base.x0.shape[1])
        heating = self.base.layout.slices["heating"]
        first = self.base.x0.shape[1] + threshold_index * N_HINGE_COLUMNS
        columns[heating] = np.arange(first, first + N_HINGE_COLUMNS)
        return columns


def build_union(*, cache: FeatureCache, base: design.BaseFit) -> UnionDesign:
    """Build the union design and its monthly Gram matrices.

    Args:
        cache: The feature cache.
        base: B0's inputs at the cut.

    Returns:
        The union design.
    """
    blocks = [base.x0]
    blocks += [
        design.hinge_block(cache=cache, site=base.site, index=base.index, threshold_c=threshold)
        for threshold in common.HINGE_GRID_C
    ]
    x = np.hstack(blocks)
    grams = design.MonthGrams.build(x=x, month_of_row=base.month_of_row, n_months=len(base.months))
    penalty = design.arm_penalty(layout=base.layout, arm=design.B0, n_rows=base.n)
    return UnionDesign(base=base, x=x, grams=grams, penalty=penalty)


def stratified_counts(
    *, quarter_of_month: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Return month counts of a month-block bootstrap that keeps every calendar quarter's size.

    Each resample draws, with replacement and within each calendar quarter, as many of the
    quarter's months as the quarter holds.

    Args:
        quarter_of_month: The calendar quarter (0 to 3) of each month.
        n_resamples: The number of resamples.
        rng: The random generator.

    Returns:
        Counts of shape (n_resamples, number of months).
    """
    counts = np.zeros((n_resamples, len(quarter_of_month)), dtype=np.float64)
    for quarter in range(4):
        members = np.flatnonzero(quarter_of_month == quarter)
        if len(members) == 0:
            continue
        draws = rng.integers(0, len(members), size=(n_resamples, len(members)))
        for resample in range(n_resamples):
            counts[resample, members] = np.bincount(draws[resample], minlength=len(members))
    return counts


def weighting_set(*, quarter_of_month: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return the full sample's counts followed by `common.N_BOOTSTRAP` stratified resamples."""
    full = np.ones((1, len(quarter_of_month)))
    resamples = stratified_counts(
        quarter_of_month=quarter_of_month, n_resamples=common.N_BOOTSTRAP, rng=rng
    )
    return np.concatenate([full, resamples])


@dataclass(frozen=True)
class ProfileResult:
    """The fits at every threshold for a batch of weightings.

    Attributes:
        theta: The coefficients in B0's column order, shape (n_thresholds, B, p0).
        loss: The weighted squared error (no penalty), shape (B, n_thresholds).
        penalised_loss: The squared error plus the penalty, shape (B, n_thresholds).
    """

    theta: np.ndarray
    loss: np.ndarray
    penalised_loss: np.ndarray

    def argmin_thresholds(self) -> np.ndarray:
        """Return, for each weighting, the threshold with the lowest unpenalised loss."""
        return np.array(common.HINGE_GRID_C)[np.argmin(self.loss, axis=1)]


def run_profile(
    *, union: UnionDesign, counts: np.ndarray, target: np.ndarray, only_full_sample: bool = False
) -> ProfileResult:
    """Fit B0 with the hinge at every threshold, for each weighting of a batch.

    Args:
        union: The union design.
        counts: Month counts, shape (B, M); the first row is conventionally all ones.
        target: The target for every training row.
        only_full_sample: True to fit only the first weighting.

    Returns:
        The fits and their losses.
    """
    weights = counts[:1] if only_full_sample else counts
    month_cross = union.grams.cross_products(x=union.x, target=target)
    month_squares = union.grams.squares(target=target)
    squares = weights @ month_squares
    penalty_matrix = torch.as_tensor(union.penalty.matrix)
    thetas, losses, penalised = [], [], []
    for threshold_index in range(len(common.HINGE_GRID_C)):
        columns = union.columns(threshold_index=threshold_index)
        sub = union.grams.gram[:, columns][:, :, columns]
        gram = torch.einsum("bm,mpq->bpq", torch.as_tensor(weights), sub)
        rhs = torch.as_tensor(weights) @ month_cross[:, columns]
        theta = design.solve(gram=gram, rhs=rhs, penalty=union.penalty)
        theta_t = torch.as_tensor(theta)
        loss = (
            torch.as_tensor(squares)
            - 2.0 * (theta_t * rhs).sum(dim=1)
            + torch.einsum("bp,bpq,bq->b", theta_t, gram, theta_t)
        ).numpy()
        extra = torch.einsum("bp,pq,bq->b", theta_t, penalty_matrix, theta_t).numpy()
        thetas.append(theta)
        losses.append(loss)
        penalised.append(loss + extra)
    return ProfileResult(
        theta=np.stack(thetas),
        loss=np.stack(losses, axis=1),
        penalised_loss=np.stack(penalised, axis=1),
    )


def argmin_interval(*, argmins: np.ndarray) -> tuple[float, float]:
    """Return the 80% interval of bootstrap arg-mins, widened by half a grid step each way.

    Args:
        argmins: The arg-min threshold of each bootstrap weighting.

    Returns:
        The lower and upper ends in degrees C.
    """
    low, high = np.percentile(argmins, ARGMIN_INTERVAL_PERCENTILES)
    return float(low) - GRID_HALF_STEP_C, float(high) + GRID_HALF_STEP_C


def band_slope_functionals(*, layout_heating_start: int, p0: int) -> np.ndarray:
    """Return the (p0, 4) matrix whose column b sums band b's heating columns over smoothings."""
    out = np.zeros((p0, N_BANDS))
    for smoothing in range(len(SMOOTHING_HOURS)):
        for band in range(N_BANDS):
            out[layout_heating_start + smoothing * N_BANDS + band, band] = 1.0
    return out


def variance_inflation(
    *, union: UnionDesign, result: ProfileResult, noise_variance: float
) -> float:
    """Return the bootstrap variance over the model-based variance of the band heating slopes.

    The model-based variance of a penalised fit under independent noise is `sigma^2 u^T G u` with
    `u = (G + P)^-1 v`. The bootstrap variance is the variance of the slopes across the stratified
    resamples (all weightings but the first). The ratio is the median over the four bands, at the
    base threshold of 15.5 degrees C.

    Args:
        union: The union design.
        result: The profile of the full sample and its resamples.
        noise_variance: The independent-noise variance `sigma^2` in points squared.

    Returns:
        The ratio `tau`.
    """
    base = union.base
    p0 = base.x0.shape[1]
    functionals = band_slope_functionals(
        layout_heating_start=base.layout.slices["heating"].start, p0=p0
    )
    columns = union.columns(threshold_index=BASE_THRESHOLD_POSITION)
    gram = union.grams.gram[:, columns][:, :, columns].sum(dim=0).numpy()
    matrix = gram + union.penalty.matrix
    u = np.linalg.solve(matrix, functionals)
    sandwich = noise_variance * np.einsum("pb,pq,qb->b", u, gram, u)
    slopes = result.theta[BASE_THRESHOLD_POSITION][1:] @ functionals
    bootstrap = slopes.var(axis=0, ddof=1)
    return float(np.median(bootstrap / sandwich))


def effective_nats(*, loss: np.ndarray, n_rows: int, inflation: float) -> np.ndarray:
    """Return each threshold's loss above the minimum, in effective nats.

    The unit is the loss difference over `2 * sigma^2 * tau`, with `sigma^2` the minimum loss per
    row, so that a gap of 2 nats is the usual two-log-likelihood-unit difference after allowing for
    autocorrelated noise.

    Args:
        loss: The full-sample unpenalised loss at each threshold, shape (n_thresholds,).
        n_rows: The number of training rows.
        inflation: The variance ratio `tau`.

    Returns:
        The effective nats above the minimum, shape (n_thresholds,).
    """
    sigma_squared = float(loss.min()) / n_rows
    return (loss - loss.min()) / (2.0 * sigma_squared * inflation)
