"""Base classes for ML model inference."""

from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

import mlflow
import patito as pt
import polars as pl
from contracts.data_schemas import PowerForecast
from pydantic import BaseModel

from ml_core.trainer import BaseDataRequirements, DataRequirementsMixin

# Type variable for inference data requirements, bound to BaseDataRequirements.
T_InferReq = TypeVar("T_InferReq", bound=BaseDataRequirements)

# Type variable for inference parameters, bound to pydantic.BaseModel.
T_InferParams = TypeVar("T_InferParams", bound=BaseModel)


class BaseInferenceModel(
    mlflow.pyfunc.PythonModel, DataRequirementsMixin, ABC, Generic[T_InferReq, T_InferParams]
):
    """Abstract base class for all ML model artifacts.

    The Model is a lightweight, deployable mathematical artifact that handles
    eager `DataFrames`. It is designed to be serialized by MLflow and used
    for production inference.

    The `Generic[T_InferReq, T_InferParams]` parameters bind the model to specific
    data and parameter contracts. This allows the developer to write
    inference logic with full IDE autocomplete and type safety.

    Example:
        class MyInferenceData(BaseDataRequirements):
            weather_ecmwf_ens_0_25: pt.DataFrame[WeatherContract]

        class MyInferParams(BaseModel):
            nwp_init_time: datetime

        class MyModel(BaseInferenceModel[MyInferenceData, MyInferParams]):
            requirements_class = MyInferenceData
            params_class = MyInferParams

            def _run_inference(
                self,
                data: MyInferenceData,
                context: mlflow.pyfunc.PythonModelContext | None = None,
                params: MyInferParams | None = None,
            ) -> pt.DataFrame[PowerForecast]:
                # 'data.weather_ecmwf_ens_0_25' and 'params.nwp_init_time'
                # are automatically recognized with correct types.
                return self.model.predict(data.weather_ecmwf_ens_0_25)
    """

    # Explicitly define the inference requirements and parameters classes.
    requirements_class: Type[T_InferReq]
    params_class: Type[T_InferParams]

    # We use this undocumented MLflow escape hatch to completely disable MLflow's
    # aggressive type hint inspection at class definition time. This allows us to use
    # sane, expressive type hints (like pt.DataFrame[PowerForecast]) without
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
        if not hasattr(self, "requirements_class"):
            raise TypeError(
                f"Class {self.__class__.__name__} must define a 'requirements_class' attribute."
            )
        if not hasattr(self, "params_class"):
            raise TypeError(
                f"Class {self.__class__.__name__} must define a 'params_class' attribute."
            )

        # Validates data presence and types instantly.
        # This converts the raw dict into our strictly-typed Pydantic object.
        typed_data = self.requirements_class(**model_input)

        # Validates and parses inference parameters.
        typed_params = self.params_class(**(params or {}))

        return self._run_inference(typed_data, context=context, params=typed_params)

    @abstractmethod
    def _run_inference(
        self,
        data: T_InferReq,
        context: mlflow.pyfunc.PythonModelContext | None = None,
        params: T_InferParams | None = None,
    ) -> pt.DataFrame[PowerForecast]:
        """The developer writes their math here, enjoying full IDE autocomplete.

        This method is separated from `predict()` to maintain a clean boundary
        between MLflow's framework requirements and our domain-specific,
        strictly-typed inference logic.

        Args:
            data: The validated Pydantic payload containing the required DataFrames.
            context: The MLflow context (passed through from predict).
            params: The validated Pydantic parameters (passed through from predict).

        Returns:
            A Patito DataFrame containing the model's predictions.
        """
        pass
