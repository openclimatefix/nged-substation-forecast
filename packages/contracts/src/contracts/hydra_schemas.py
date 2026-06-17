"""Hydra configuration schemas for the NGED substation forecast project."""

from datetime import date

from pydantic import BaseModel, Field

from contracts.power_schemas import FoldId


class DataSplitConfig(BaseModel):
    """Configuration for temporal data splitting."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date


class CvFoldConfig(BaseModel):
    """Configuration for a single expanding-window CV fold."""

    fold_id: FoldId
    train_start: date
    train_end: date
    val_start: date
    val_end: date


class CvConfig(BaseModel):
    """Configuration for expanding-window cross-validation.

    The folds list defines the evaluation protocol shared by all experiments on the
    leaderboard. All models must be evaluated against the same folds to ensure
    apples-to-apples comparison.

    min_training_months controls which time series are eligible for each fold: a time
    series is only included if it has at least this many months of data before val_start
    (and data through val_end).
    """

    folds: list[CvFoldConfig]
    min_training_months: int = Field(default=6, ge=1)
