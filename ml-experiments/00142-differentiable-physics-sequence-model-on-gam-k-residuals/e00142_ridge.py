"""A ridge regression of the median correction on the sequence network's inputs.

The linear reference: the same 15 day vectors, the target day's features, and a substation
one-hot, flattened. Missing targets count as zero correction. The penalty is chosen on the
out-of-fold absolute error of the same purged month-block folds the networks use.
"""

from typing import Final

import e00142_model as model
import numpy as np

ALPHAS: Final[tuple[float, ...]] = (1e1, 1e2, 1e3, 1e4, 1e5)


def design(*, pooled: model.Pooled, index: np.ndarray, n_sites: int) -> np.ndarray:
    """Return the flattened inputs of the chosen samples."""
    onehot = np.eye(n_sites)[pooled.site[index]]
    flat = pooled.history[index].reshape(len(index), -1)
    return np.concatenate([flat, pooled.target[index], onehot], axis=1).astype(np.float64)


def fit_ridge(
    *, x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, scale, weights) of a ridge fit with an intercept."""
    mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-6
    z = (x - mean) / scale
    target_mean = y.mean(axis=0)
    gram = z.T @ z + alpha * np.eye(z.shape[1])
    weights = np.linalg.solve(gram, z.T @ (y - target_mean))
    return mean, scale, np.vstack([weights, target_mean])


def predict_ridge(
    *, fitted: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray
) -> np.ndarray:
    """Return the correction, shape (n, 48)."""
    mean, scale, weights = fitted
    return ((x - mean) / scale) @ weights[:-1] + weights[-1]


def fit_folds(
    *, pooled: model.Pooled, candidates: np.ndarray, n_folds: int, n_sites: int
) -> tuple[list, dict[int, np.ndarray], dict[int, np.ndarray], float]:
    """Choose the penalty on out-of-fold error and fit one ridge per fold at that penalty.

    Args:
        pooled: The pooled samples.
        candidates: The sample indices that may train.
        n_folds: The number of month-block folds.
        n_sites: The number of substations.

    Returns:
        The fold fits, the held-out samples of each fold, their out-of-fold predictions, and the
        penalty.
    """
    blocks = model.month_blocks(month_codes=pooled.month_code[candidates], n_folds=n_folds)
    splits = [model.purged_split(pooled=pooled, candidates=candidates, block=b) for b in blocks]
    y_all = np.nan_to_num(pooled.y, nan=0.0)
    best_alpha, best_error = ALPHAS[0], np.inf
    for alpha in ALPHAS:
        error, count = 0.0, 0
        for train, validation in splits:
            fitted = fit_ridge(
                x=design(pooled=pooled, index=train, n_sites=n_sites), y=y_all[train], alpha=alpha
            )
            predicted = predict_ridge(
                fitted=fitted, x=design(pooled=pooled, index=validation, n_sites=n_sites)
            )
            mask = np.isfinite(pooled.y[validation])
            error += float(np.abs(np.where(mask, pooled.y[validation] - predicted, 0.0)).sum())
            count += int(mask.sum())
        if error / count < best_error:
            best_alpha, best_error = alpha, error / count
    fits, held_out, oof = [], {}, {}
    for fold, (train, validation) in enumerate(splits):
        fitted = fit_ridge(
            x=design(pooled=pooled, index=train, n_sites=n_sites), y=y_all[train], alpha=best_alpha
        )
        fits.append(fitted)
        held_out[fold] = validation
        oof[fold] = predict_ridge(
            fitted=fitted, x=design(pooled=pooled, index=validation, n_sites=n_sites)
        )
    return fits, held_out, oof, best_alpha
