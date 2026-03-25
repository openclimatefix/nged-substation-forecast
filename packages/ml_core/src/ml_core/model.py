"""Base classes for ML model inference."""

from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

import mlflow
import polars as pl

from ml_core.trainer import BaseDataRequirements

T_InferReq = TypeVar("T_InferReq", bound=BaseDataRequirements)


class BasePolarsModel(mlflow.pyfunc.PythonModel, ABC, Generic[T_InferReq]):
    """Abstract base class for all ML model artifacts.

    The Model is a lightweight, deployable mathematical artifact that handles
    eager DataFrames. It is designed to be serialized by MLflow.
    """

    # Explicitly define the inference requirements class.
    inference_requirements_class: Type[T_InferReq]

    # We use this undocumented MLflow escape hatch to completely disable MLflow's
    # aggressive type hint inspection at class definition time. This allows us to use
    # sane, expressive type hints (like pt.DataFrame[SubstationFeatures]) without
    # MLflow spamming the Dagster UI with UserWarnings about unsupported types.
    # We rely on Patito for actual runtime schema validation anyway.
    #
    # Note: Because this is an undocumented API, there is a minor risk that MLflow
    # could rename or remove this attribute in a future release. If they do, our
    # code will not crash (MLflow uses a safe `getattr` check), but the annoying
    # UserWarnings would return during Dagster startup.
    _skip_type_hint_validation = True

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: dict[str, pl.DataFrame],
        params: dict | None = None,
    ) -> pl.DataFrame:
        """Satisfies MLflow's rigid string-based signature.

        Immediately parses the raw dictionary into our strictly-typed,
        validated Pydantic payload.

        Args:
            context: MLflow context (unused).
            model_input: A dictionary mapping FeatureAsset names to Polars DataFrames.
            params: Optional inference parameters (unused).

        Returns:
            A Polars DataFrame containing the model's predictions.
        """
        if not hasattr(self, "inference_requirements_class"):
            raise TypeError(
                f"Class {self.__class__.__name__} must define an "
                "'inference_requirements_class' attribute."
            )

        # Validates data presence and types instantly
        typed_data = self.inference_requirements_class(**model_input)
        return self._run_inference(typed_data)

    @abstractmethod
    def _run_inference(self, data: T_InferReq) -> pl.DataFrame:
        """The developer writes their math here, enjoying full IDE autocomplete.

        Args:
            data: The validated Pydantic payload containing the required DataFrames.

        Returns:
            A Polars DataFrame containing the model's predictions.
        """
        pass
