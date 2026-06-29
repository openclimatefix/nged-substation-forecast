"""Hydra configuration schemas for the NGED substation forecast project."""

from datetime import date
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from contracts.power_schemas import FoldId


class CvFoldConfig(BaseModel):
    """Configuration for a single expanding-window CV fold.

    ``leaderboard`` distinguishes the epoch-pinned leaderboard folds (the apples-to-apples
    evaluation protocol) from optional non-leaderboard dev folds such as ``smoke_test``: a
    ``leaderboard=False`` fold runs through the identical pipeline but never feeds the leaderboard.

    ``min_training_months`` overrides ``CvConfig.min_training_months`` for this fold alone (``None``
    falls back to the config-level value). A short dev fold sets it to its train length so
    eligibility does not demand the leaderboard's longer history.
    """

    fold_id: FoldId
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    leaderboard: bool = True
    min_training_months: int | None = Field(default=None, ge=1)


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

    @property
    def fold_ids(self) -> list[str]:
        """The fold ids in declaration order (e.g. ``["2022", "2023", ...]``).

        Used to build the ``cv_experiment_folds`` partitions and to expand experiment
        registration into per-fold partition keys — always read from config, never hard-coded.
        """
        return [fold.fold_id for fold in self.folds]

    @property
    def leaderboard_fold_ids(self) -> list[str]:
        """The fold ids of the leaderboard folds only, in declaration order.

        Non-leaderboard dev folds (e.g. ``smoke_test``) are excluded. Used to expand the
        ``full_cv`` / ``register_only`` run modes and to scope leaderboard metrics.
        """
        return [fold.fold_id for fold in self.folds if fold.leaderboard]

    def get_fold(self, fold_id: str) -> CvFoldConfig:
        """Return the fold with the given ``fold_id``.

        Args:
            fold_id: The fold identifier to look up (e.g. ``"2022"``).

        Raises:
            KeyError: If no fold with that id exists in the config.
        """
        for fold in self.folds:
            if fold.fold_id == fold_id:
                return fold
        raise KeyError(f"No fold with fold_id={fold_id!r}; available folds: {self.fold_ids}")


def load_cv_config(path: Path) -> CvConfig:
    """Load and validate the cross-validation config from a YAML file.

    The CV folds are the leaderboard's evaluation protocol and must be read from
    ``conf/cv/default.yaml`` (never hard-coded) so every experiment and asset shares one
    canonical definition.

    Args:
        path: Path to the CV config YAML (e.g. ``conf/cv/default.yaml``).

    Returns:
        The validated ``CvConfig``.
    """
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return CvConfig.model_validate(raw)
