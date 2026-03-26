"""XGBoost model artifact for inference."""

import logging
from datetime import datetime
from typing import Type

import mlflow
import patito as pt
import polars as pl
from contracts.data_schemas import PowerForecast, ProcessedNwp
from pydantic import BaseModel
from xgboost import XGBRegressor

from ml_core.model import BaseInferenceModel
from ml_core.trainer import BaseDataRequirements

log = logging.getLogger(__name__)


class XGBoostInferenceData(BaseDataRequirements):
    """Inference data requirements for XGBoost."""

    weather_ecmwf_ens_0_25: pt.DataFrame[ProcessedNwp]


class XGBoostInferParams(BaseModel):
    """Inference parameters for XGBoost."""

    nwp_init_time: datetime | None = None
    power_fcst_model_name: str | None = None


class XGBoostInferenceModel(BaseInferenceModel[XGBoostInferenceData, XGBoostInferParams]):
    """MLflow pyfunc wrapper for XGBoost inference.

    This class is designed to be lightweight and serializable by MLflow.
    It handles the transformation from the validated Pydantic payload
    to the format expected by the underlying XGBoost model.
    """

    requirements_class: Type[XGBoostInferenceData] = XGBoostInferenceData
    params_class: Type[XGBoostInferParams] = XGBoostInferParams

    def __init__(self, model: XGBRegressor):
        """Initialize the wrapper with a trained XGBoost model.

        Args:
            model: The trained XGBoost model.
        """
        self.model = model

    def _run_inference(
        self,
        data: XGBoostInferenceData,
        context: mlflow.pyfunc.PythonModelContext | None = None,
        params: XGBoostInferParams | None = None,
    ) -> pt.DataFrame[PowerForecast]:
        """Execute the inference logic.

        Args:
            data: The validated inference data.
            context: MLflow context (unused).
            params: The validated inference parameters.

        Returns:
            A Patito DataFrame containing the predictions.
        """
        # 🎉 PERFECT IDE TYPE HINTING 🎉
        df = data.weather_ecmwf_ens_0_25

        # Prepare features (must match training features)
        # For this example, we'll just select numeric columns
        X = df.select(pl.all().exclude(["valid_time", "h3_index", "ensemble_member"])).to_pandas()

        preds = self.model.predict(X)

        # Return predictions joined with metadata
        return pt.DataFrame[PowerForecast](
            df.select(["valid_time", "h3_index", "ensemble_member"]).with_columns(
                MW_or_MVA=pl.Series(values=preds, dtype=pl.Float32)
            )
        )
