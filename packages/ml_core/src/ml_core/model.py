"""Base classes for ML model inference."""

from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

import mlflow
import patito as pt
import polars as pl
from contracts.data_schemas import PowerForecast

from ml_core.assets import FeatureAsset
from ml_core.trainer import BaseDataRequirements

# Type variable for inference data requirements, bound to BaseDataRequirements.
T_InferReq = TypeVar("T_InferReq", bound=BaseDataRequirements)


class BaseInferenceModel(mlflow.pyfunc.PythonModel, ABC, Generic[T_InferReq]):
    """Abstract base class for all ML model artifacts.

    The Model is a lightweight, deployable mathematical artifact that handles
    eager `DataFrames`. It is designed to be serialized by MLflow and used
    for production inference.

    The `Generic[T_InferReq]` parameter binds the model to a specific
    `BaseDataRequirements` subclass. This allows the developer to write
    inference logic with full IDE autocomplete and type safety.

    Example:
        class MyInferenceData(BaseDataRequirements):
            weather_ecmwf_ens_0_25: pt.DataFrame[WeatherContract]

        class MyModel(BaseInferenceModel[MyInferenceData]):
            inference_requirements_class = MyInferenceData

            def _run_inference(
                self,
                data: MyInferenceData,
                context: mlflow.pyfunc.PythonModelContext | None = None,
                params: dict | None = None,
            ) -> pt.DataFrame[PowerForecast]:
                # 'data.weather_ecmwf_ens_0_25' is automatically recognized
                # as a Patito DataFrame with the correct schema.
                return self.model.predict(data.weather_ecmwf_ens_0_25)
    """

    # Explicitly define the inference requirements class.
    inference_requirements_class: Type[T_InferReq]

    # We use this undocumented MLflow escape hatch to completely disable MLflow's
    # aggressive type hint inspection at class definition time. This allows us to use
    # sane, expressive type hints (like pt.DataFrame[PowerForecast]) without
    # MLflow spamming the Dagster UI with UserWarnings about unsupported types.
    # We rely on Patito for actual runtime schema validation anyway.
    _skip_type_hint_validation = True

    @classmethod
    def data_requirements(cls) -> list[FeatureAsset]:
        """Returns the list of FeatureAssets required for inference.

        This should be called by the orchestrator or API layer to determine
        which data to fetch before calling `predict()`.
        """
        if not hasattr(cls, "inference_requirements_class"):
            raise TypeError(
                f"Class {cls.__name__} must define an 'inference_requirements_class' "
                "attribute pointing to a BaseDataRequirements subclass."
            )
        return cls.inference_requirements_class.get_required_assets()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: dict[str, pl.DataFrame],
        params: dict | None = None,
    ) -> pt.DataFrame[PowerForecast]:
        """Satisfies MLflow's rigid string-based signature.

        We cannot use `T_InferReq` as the type hint for `model_input` here
        because MLflow's internal machinery expects a standard Python `dict`
        (or a Pandas `DataFrame`). If we used our custom Pydantic class as a
        type hint, MLflow would fail to pass the data correctly.

        Instead, this method acts as a 'bouncer': it accepts the raw dictionary
        from MLflow, validates it against our strictly-typed Pydantic payload,
        and then delegates to `_run_inference` where the actual math happens.

        Args:
            context: MLflow context, used for accessing model artifacts.
            model_input: A dictionary mapping `FeatureAsset` names to Polars `DataFrames`.
            params: Optional dictionary of inference-time parameters.

        Returns:
            A Patito DataFrame containing the model's predictions.
        """
        if not hasattr(self, "inference_requirements_class"):
            raise TypeError(
                f"Class {self.__class__.__name__} must define an "
                "'inference_requirements_class' attribute."
            )

        # Validates data presence and types instantly.
        # This converts the raw dict into our strictly-typed Pydantic object.
        typed_data = self.inference_requirements_class(**model_input)

        return self._run_inference(typed_data, context=context, params=params)

    @abstractmethod
    def _run_inference(
        self,
        data: T_InferReq,
        context: mlflow.pyfunc.PythonModelContext | None = None,
        params: dict | None = None,
    ) -> pt.DataFrame[PowerForecast]:
        """The developer writes their math here, enjoying full IDE autocomplete.

        This method is separated from `predict()` to maintain a clean boundary
        between MLflow's framework requirements and our domain-specific,
        strictly-typed inference logic.

        Args:
            data: The validated Pydantic payload containing the required DataFrames.
            context: The MLflow context (passed through from predict).
            params: Optional inference parameters (passed through from predict).

        Returns:
            A Patito DataFrame containing the model's predictions.
        """
        pass
