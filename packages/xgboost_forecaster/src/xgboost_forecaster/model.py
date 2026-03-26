"""XGBoost model artifact for inference."""

import logging

import patito as pt
import polars as pl
from contracts.data_schemas import PowerForecast, ProcessedNwp
from xgboost import XGBRegressor

from ml_core.model import BaseForecaster

log = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """XGBoost implementation of the Forecaster interface.

    This class handles the transformation from strictly-typed Patito DataFrames
    to the format expected by the underlying XGBoost model.
    """

    def __init__(self, model: XGBRegressor):
        """Initialize the forecaster with a trained XGBoost model.

        Args:
            model: The trained XGBoost model.
        """
        self.model = model

    def predict(  # type: ignore[override]
        self, weather_ecmwf_ens_0_25: pt.DataFrame[ProcessedNwp], **kwargs
    ) -> pt.DataFrame[PowerForecast]:
        """Execute the inference logic.

        Args:
            weather_ecmwf_ens_0_25: The validated weather forecast data.
            **kwargs: Additional arguments (unused).

        Returns:
            A Patito DataFrame containing the predictions.
        """
        # Prepare features (must match training features)
        # For this example, we'll just select numeric columns
        X = weather_ecmwf_ens_0_25.select(
            pl.all().exclude(["valid_time", "h3_index", "ensemble_member"])
        ).to_pandas()

        preds = self.model.predict(X)

        # Return predictions joined with metadata
        return pt.DataFrame[PowerForecast](
            weather_ecmwf_ens_0_25.select(
                ["valid_time", "h3_index", "ensemble_member"]
            ).with_columns(MW_or_MVA=pl.Series(values=preds, dtype=pl.Float32))
        )
