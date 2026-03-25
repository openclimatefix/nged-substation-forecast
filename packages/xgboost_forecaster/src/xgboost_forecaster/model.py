"""XGBoost model artifact for inference."""

import logging
from typing import Type

import patito as pt
import polars as pl
from contracts.data_schemas import PowerForecast, ProcessedNwp
from xgboost import XGBRegressor

from ml_core.model import ForecastInference
from ml_core.trainer import BaseDataRequirements

log = logging.getLogger(__name__)


class XGBoostInferenceData(BaseDataRequirements):
    """Inference data requirements for XGBoost."""

    weather_ecmwf_ens_0_25: pt.DataFrame[ProcessedNwp]


class XGBoostPolarsWrapper(ForecastInference[XGBoostInferenceData]):
    """MLflow pyfunc wrapper for XGBoost inference.

    This class is designed to be lightweight and serializable by MLflow.
    It handles the transformation from the validated Pydantic payload
    to the format expected by the underlying XGBoost model.
    """

    inference_requirements_class: Type[XGBoostInferenceData] = XGBoostInferenceData

    def __init__(self, model: XGBRegressor):
        """Initialize the wrapper with a trained XGBoost model.

        Args:
            model: The trained XGBoost model.
        """
        self.model = model

    def _run_inference(self, data: XGBoostInferenceData) -> pt.DataFrame[PowerForecast]:
        """Execute the inference logic.

        Args:
            data: The validated inference data.

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
