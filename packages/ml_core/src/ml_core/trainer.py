"""Base classes for ML model training."""

from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

import mlflow
from pydantic import BaseModel, ConfigDict

from ml_core.assets import FeatureAsset

# Type variable for training data requirements, bound to BaseDataRequirements.
T_TrainReq = TypeVar("T_TrainReq", bound="BaseDataRequirements")


class BaseDataRequirements(BaseModel):
    """A strictly-typed contract that maps Pydantic fields to Dagster FeatureAssets.

    This class serves as the base for both training (`T_TrainReq`) and
    inference (`T_InferReq`) data requirements. Subclasses should define fields
    where the field name matches a `FeatureAsset` value, and the type hint is a
    Patito `LazyFrame` or `DataFrame`.
    """

    # We use `ConfigDict(arbitrary_types_allowed=True)` here because Pydantic
    # natively rejects Polars DataFrames/LazyFrames as they are not standard
    # Python types. This allows us to use sane, expressive type hints while
    # still benefiting from Pydantic's validation machinery.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_required_assets(cls) -> list[FeatureAsset]:
        """Returns the list of FeatureAssets required by this class.

        We check that every field name in the Pydantic model corresponds to
        a valid value in the `FeatureAsset` Enum. If a field name is not found
        in the Enum, this method will raise a `ValueError`.
        """
        return [FeatureAsset(f) for f in cls.model_fields.keys()]


class DataRequirementsMixin:
    """Mixin to provide data requirement discovery for trainers and models."""

    # Explicitly define the requirements class to avoid fragile __orig_bases__ inspection.
    # This is used by the Dagster factory to determine which assets to inject.
    requirements_class: Type[BaseDataRequirements]

    @classmethod
    def data_requirements(cls) -> list[FeatureAsset]:
        """Auto-resolves Dagster dependencies from the requirements class."""
        if not hasattr(cls, "requirements_class"):
            raise TypeError(
                f"Class {cls.__name__} must define a 'requirements_class' attribute "
                "pointing to a BaseDataRequirements subclass."
            )
        return cls.requirements_class.get_required_assets()


class BaseTrainer(ABC, DataRequirementsMixin, Generic[T_TrainReq]):
    """Abstract base class for all ML model trainers.

    The Trainer is a heavy script that handles `LazyFrames`, targets, and
    executes the training loop. It returns a lightweight `BaseInferenceModel` artifact.

    The `Generic[T_TrainReq]` parameter binds the trainer to a specific
    `BaseDataRequirements` subclass, ensuring that the `train` method receives
    exactly the data shapes it expects with full IDE type hinting.

    Usage Lifecycle:
    1. The orchestrator (e.g. Dagster) calls `data_requirements()` to determine
       which `FeatureAsset`s to fetch from the data lake.
    2. The orchestrator fetches the data as `LazyFrames` and passes them into
       the `train()` method.
    3. The `train()` method executes the training logic and returns an
       implementation of `BaseInferenceModel` (an MLflow `PythonModel`) that
       captures everything required to run the model in production.
    """

    @abstractmethod
    def train(self, data: T_TrainReq, config: dict) -> mlflow.pyfunc.PythonModel:
        """Executes heavy training logic and returns the lightweight artifact.

        Args:
            data: The validated Pydantic payload containing the required LazyFrames.
            config: The Hydra configuration dictionary for this model run.

        Returns:
            An MLflow pyfunc-compatible model artifact (BaseInferenceModel).
        """
        pass
