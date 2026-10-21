"""Month-block cross-validation, the settings grid, and the final fits of experiment 119.

A direction (F1 or F2) has a training year and a scored year. Tuning blocks are 3 consecutive
months of the training year. A block's training rows exclude any row whose run date or valid time
lies within 7 days of the block (the purge), so no actual of the tuning block reaches its fit.
"""

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import e00119_common as common
import e00119_data as data
import e00119_model as model
import numpy as np
import polars as pl
import torch

DECAY_GRID: Final[tuple[float, ...]] = (0.4, 4.0, 40.0)
SEEDS: Final[tuple[int, ...]] = (11, 12, 13, 14, 15)
SHRINK_GRID: Final[tuple[float, ...]] = (0.0, 0.5, 1.0)
PURGE: Final[timedelta] = timedelta(days=7)


@dataclass(frozen=True)
class Direction:
    """A training year and a scored year.

    Attributes:
        name: `F1` (verdict) or `F2` (replicate).
        train: Run-date range `[start, end)` of the training year.
        scored: Run-date range `[start, end)` of the scored year.
    """

    name: str
    train: tuple[datetime, datetime]
    scored: tuple[datetime, datetime]


F1: Final[Direction] = Direction("F1", data.TRAIN_F1, data.SCORED_F1)
F2: Final[Direction] = Direction("F2", data.TRAIN_F2, data.SCORED_F2)
DIRECTIONS: Final[dict[str, Direction]] = {"F1": F1, "F2": F2}


def run_mask(*, rows: pl.DataFrame, period: tuple[datetime, datetime]) -> np.ndarray:
    """Return a boolean mask of the rows whose run date lies in `[period[0], period[1])`."""
    dates = rows["run_date"]
    return ((dates >= period[0]) & (dates < period[1])).to_numpy()


def blocks_of(*, direction: Direction) -> list[tuple[datetime, datetime]]:
    """Return the tuning blocks of 3 consecutive months as `[start, end)` date ranges."""
    start, end = direction.train
    months = (end.year - start.year) * 12 + end.month - start.month
    out = []
    for first in range(0, months, 3):
        year, month = divmod(start.year * 12 + start.month - 1 + first, 12)
        block_start = datetime(year, month + 1, 1, tzinfo=UTC)
        year, month = divmod(start.year * 12 + start.month - 1 + first + 3, 12)
        out.append((block_start, datetime(year, month + 1, 1, tzinfo=UTC)))
    return out


def fold_indices(
    *, rows: pl.DataFrame, direction: Direction, block: tuple[datetime, datetime]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the training and tuning row indices of one block.

    Args:
        rows: The row table.
        direction: The direction.
        block: The tuning block.

    Returns:
        Indices into `rows`. The tuning rows are the training year's rows with run date in the
        block. The training rows are the training year's other rows whose run date and valid time
        both lie outside the block widened by the purge.
    """
    in_year = run_mask(rows=rows, period=direction.train)
    low, high = block[0] - PURGE, block[1] + PURGE
    run = rows["run_date"]
    valid = rows["valid_time"]
    near = ((run >= low) & (run < high)) | ((valid >= low) & (valid < high))
    tune = in_year & ((run >= block[0]) & (run < block[1])).to_numpy()
    train = in_year & ~near.to_numpy()
    return np.flatnonzero(train), np.flatnonzero(tune)


def n_days(*, rows: pl.DataFrame, index: np.ndarray) -> int:
    """Return the number of distinct run dates among some rows."""
    return (
        int(rows[index.tolist()]["run"].n_unique())
        if len(index) < 50_000
        else int(rows["run"].to_numpy()[index].max() - rows["run"].to_numpy()[index].min() + 1)
    )


def to_device(index: np.ndarray) -> torch.Tensor:
    """Return row indices as a device tensor."""
    return torch.from_numpy(index).to(model.DEVICE)


@dataclass
class OofResult:
    """The out-of-fold predictions of one arm over a direction's training year.

    Attributes:
        index: The training-year rows (indices into the row table), in block order.
        quantiles: The 11 quantiles (lambda 1) at those rows, from the block's fitted encoder.
        best_epochs: The early-stopped epoch of each block.
        train_loss: The final training loss of each block.
        oof_loss: Each block's out-of-fold pinball loss.
        seconds: Wall time of each block's fit.
    """

    index: np.ndarray
    quantiles: np.ndarray
    best_epochs: list[int]
    train_loss: list[float]
    oof_loss: list[float]
    seconds: list[float]


def run_oof(
    *,
    rows: pl.DataFrame,
    bundle: model.Bundle,
    arm: model.Arm,
    direction: Direction,
    decay_c: float,
    seed: int,
) -> OofResult:
    """Fit an arm on each block's training rows and predict the block's tuning rows.

    Args:
        rows: The row table the bundle was built from.
        bundle: The device bundle of every row (with the arm's weather source).
        arm: The arm.
        direction: The direction.
        decay_c: The weight-decay constant c.
        seed: The seed.

    Returns:
        The out-of-fold predictions and the per-block fit details.
    """
    indices, quantiles = [], []
    result = OofResult(np.array([], dtype=np.int64), np.empty((0, 11)), [], [], [], [])
    for block in blocks_of(direction=direction):
        train_index, tune_index = fold_indices(rows=rows, direction=direction, block=block)
        started = time.monotonic()
        train_bundle = bundle.subset(to_device(train_index))
        tune_bundle = bundle.subset(to_device(tune_index))
        days = len(np.unique(rows["run"].to_numpy()[train_index]))
        fitted = model.fit(
            arm=arm,
            train=train_bundle,
            validation=tune_bundle,
            seed=seed,
            decay_c=decay_c,
            n_days=days,
        )
        quantiles.append(model.to_numpy(model.quantiles_of(result=fitted, bundle=tune_bundle)))
        indices.append(tune_index)
        result.best_epochs.append(fitted.best_epoch)
        result.train_loss.append(fitted.train_loss[fitted.best_epoch - 1])
        result.oof_loss.append(float(fitted.val_loss[fitted.best_epoch - 1]))
        result.seconds.append(time.monotonic() - started)
    result.index = np.concatenate(indices)
    result.quantiles = np.concatenate(quantiles)
    return result


def pinball_numpy(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss per row (the quantile score), in points."""
    levels = np.array(common.QUANTILE_LEVELS)[None, :]
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)


def shrink_quantiles(
    *, k0: np.ndarray, quantiles: np.ndarray, kind: str, shrink: float
) -> np.ndarray:
    """Return the quantiles after scaling the encoder's shift by lambda.

    For the correction the shift is `quantiles - k0` before sorting, which the sorted output
    loses; the sorted shift is a close substitute and is used only to choose lambda.
    """
    shift = quantiles - k0
    out = k0 + shrink * shift
    return np.sort(out, axis=1) if kind == "c" else out


def write_decisions(*, key: str, value: object) -> None:
    """Merge one entry into `decisions.json`."""
    path = common.WORK_DIR / "decisions.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current[key] = value
    common.write_json_atomic(path=path, content=current)
