"""Configuration schemas, and the class-path round-trip that ``_target_`` strings ride on."""

import importlib
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from contracts.power_schemas import FoldId


def class_target(obj: type | object) -> str:
    """Return the fully-qualified import path of a class (or of an instance's class).

    The inverse of ``import_class``. Together they are how a config file, an MLflow experiment tag
    and a saved model's ``meta.json`` all name a Python class: a ``module.ClassName`` string that
    survives being written to disk and read back in another process.

    Args:
        obj: The class to name, or an instance whose class should be named.

    Returns:
        The ``module.ClassName`` path, e.g. ``"xgboost_forecaster.forecaster.XGBoostForecaster"``.

    Raises:
        ValueError: ``obj``'s class is not defined at module level — it is nested inside another
            class or inside a function. ``import_class`` resolves a single attribute lookup on a
            module, so such a class has no path it could resolve. Failing here, where the class is
            defined, beats emitting a target string that only breaks when something later tries to
            load it.
    """
    cls = obj if isinstance(obj, type) else type(obj)
    if "." in cls.__qualname__:
        raise ValueError(
            f"{cls.__module__}.{cls.__qualname__} is not defined at module level (it is nested "
            "inside a class or a function), so it has no importable path. Move it to module level "
            "to use it as a _target_."
        )
    return f"{cls.__module__}.{cls.__qualname__}"


def import_class(target: str) -> type:
    """Resolve a fully-qualified class path to the class object.

    The inverse of ``class_target``: imports the module and looks the class up on it.

    Args:
        target: A ``module.ClassName`` path, as produced by ``class_target``.

    Returns:
        The class named by ``target``.

    Raises:
        ValueError: ``target`` names no module path, is relative, its module cannot be imported,
            the module has no such attribute, or the attribute is not a class. A module that
            fails on its *own* imports is reported the same way — the original ``ImportError``
            is chained as ``__cause__``.
    """
    module_path, _, class_name = target.rpartition(".")
    # A leading dot would make import_module read the target as a relative import and demand a
    # package to resolve it against; that is a malformed target, so reject it here rather than
    # letting it surface as the TypeError import_module would raise.
    if not module_path or module_path.startswith("."):
        raise ValueError(f"{target!r} is not a fully-qualified class path (expected 'module.Cls').")
    try:
        module = importlib.import_module(module_path)
    except ImportError as error:
        raise ValueError(f"Cannot import module {module_path!r} of target {target!r}.") from error
    try:
        resolved = getattr(module, class_name)
    except AttributeError as error:
        raise ValueError(f"Module {module_path!r} has no attribute {class_name!r}.") from error
    if not isinstance(resolved, type):
        # TRY004 wants a TypeError, but the isinstance check is on what `target` *resolved to*,
        # not on `target` itself, which is a perfectly well-typed str. What is wrong is its value,
        # so every failure here is one ValueError and a caller needs to catch only that.
        raise ValueError(  # noqa: TRY004
            f"Target {target!r} resolved to {resolved!r}, which is not a class."
        )
    return resolved


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
    with path.open(encoding="utf-8") as file:
        return CvConfig.model_validate(yaml.safe_load(file))
