"""XGBoost model wrapper for substation forecasting."""

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import mlflow
import patito as pt
import polars as pl
from contracts.data_schemas import InferenceParams, PowerForecast, SubstationFeatures
from xgboost import XGBRegressor

if TYPE_CHECKING:
    import patito as pt
    from contracts.data_schemas import PowerForecast, SubstationFeatures

log = logging.getLogger(__name__)


class NoPredictionsError(Exception):
    """Raised when the forecaster fails to produce any predictions."""

    pass


class XGBoostPyFuncWrapper(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for XGBoostForecaster.

    This wrapper allows the model to be used in a model-agnostic way during inference.
    It handles both global models (one model for all substations) and local models
    (one model per substation) by routing the input data to the correct model.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the underlying XGBoost models from the artifacts.

        Args:
            context: MLflow context containing the artifacts.
        """
        self.models: dict[str, XGBoostForecaster] = {}
        for name, path in context.artifacts.items():
            self.models[name] = XGBoostForecaster.load(Path(path))

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: list[pt.DataFrame[SubstationFeatures]],
        params: dict[str, Any] | None = None,
    ) -> list[pt.DataFrame[PowerForecast]]:
        """Make predictions using the loaded models.

        Note:
            MLflow's pyfunc.PythonModel.predict expects a list of inputs when using
            custom types (like Patito DataFrames) to support batching. Even though
            a single DataFrame already contains a batch of rows, we must wrap it
            in a list to satisfy MLflow's type hint inspection and avoid warnings.

        Args:
            context: MLflow context.
            model_input: A list containing a single Patito DataFrame of SubstationFeatures.
            params: Optional dictionary of parameters, including 'nwp_init_time'
                and 'power_fcst_model'.

        Returns:
            A list containing a single Patito DataFrame of PowerForecast.
        """
        if params is None:
            raise ValueError("'params' must be provided to the predict method")

        # Validate parameters using Pydantic model
        inference_params = InferenceParams(**params)
        nwp_init_time = inference_params.nwp_init_time
        power_fcst_model = (
            inference_params.power_fcst_model or XGBoostForecaster.model_name_and_version()
        )

        # Unwrap the input list. MLflow expects a list for custom types.
        if not isinstance(model_input, list) or len(model_input) == 0:
            raise ValueError("model_input must be a non-empty list of DataFrames")

        df = model_input[0]

        # If we have a global model, use it for everything
        if "global" in self.models:
            preds = self.models["global"].predict(df)
            res = df.select(["valid_time", "substation_number"]).with_columns(
                MW_or_MVA=pl.Series(values=preds, dtype=pl.Float32),
                nwp_init_time=pl.lit(nwp_init_time).cast(pl.Datetime("us", "UTC")),
                power_fcst_model=pl.lit(power_fcst_model).cast(pl.Categorical),
                ensemble_member=pl.lit(0).cast(pl.UInt8),
            )
            return [cast(pt.DataFrame[PowerForecast], res)]

        # Otherwise, route to local models
        all_preds = []
        for substation_number, group in df.group_by("substation_number"):
            sub_id_str = str(substation_number)
            if sub_id_str not in self.models:
                log.warning(f"No model found for substation {substation_number}")
                continue

            preds = self.models[sub_id_str].predict(group)
            all_preds.append(
                group.select(["valid_time", "substation_number"]).with_columns(
                    MW_or_MVA=pl.Series(values=preds, dtype=pl.Float32),
                    nwp_init_time=pl.lit(nwp_init_time).cast(pl.Datetime("us", "UTC")),
                    power_fcst_model=pl.lit(power_fcst_model).cast(pl.Categorical),
                    ensemble_member=pl.lit(0).cast(pl.UInt8),
                )
            )

        if not all_preds:
            raise NoPredictionsError(
                f"No models found for any of the substations in the input data: "
                f"{df['substation_number'].unique().to_list()}"
            )

        res = pl.concat(all_preds)
        return [cast(pt.DataFrame[PowerForecast], res)]


# TODO: Create an abstract base class that defines the universal interface to all Forecasters.
class XGBoostForecaster:
    """Wrapper around XGBoost for substation-level forecasting."""

    model_name = "xgboost"
    version = "v0.0.1"

    @staticmethod
    def model_name_and_version() -> str:
        return f"{XGBoostForecaster.model_name}_{XGBoostForecaster.version}"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the forecaster with optional XGBoost parameters.

        Args:
            params: Dictionary of XGBoost parameters.
        """
        self.params = params or self.get_default_params()
        self.model: XGBRegressor | None = None
        self.feature_names: Sequence[str] | None = None

    @staticmethod
    def get_default_params() -> dict[str, Any]:
        """Returns a dictionary of default XGBoost parameters."""
        return {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "tree_method": "hist",
            "random_state": 42,
        }

    def train(
        self,
        df: pl.DataFrame,
        target_col: str = "MW_or_MVA",
        feature_cols: list[str] | None = None,
        eval_set: list[tuple[pl.DataFrame, pl.Series]] | None = None,
    ) -> None:
        """Train the XGBoost model.

        Args:
            df: Training data as a Polars DataFrame.
            target_col: Name of the target column.
            feature_cols: List of feature column names. If None, uses all except target.
            eval_set: Optional list of (X, y) tuples for early stopping.
        """
        if feature_cols is None:
            feature_cols = [
                c
                for c in df.columns
                if c != target_col and c != "valid_time" and df[c].dtype.is_numeric()
            ]

        self.feature_names = feature_cols
        X = df.select(feature_cols)
        y = df.select(target_col)

        xgb_eval_set = None
        if eval_set:
            xgb_eval_set = []
            for X_eval_df, y_eval_series in eval_set:
                xgb_eval_set.append((X_eval_df.select(feature_cols), y_eval_series))

        model = XGBRegressor(**self.params)
        model.fit(X, y, eval_set=xgb_eval_set, verbose=False)
        self.model = model

    def predict(self, df: pl.DataFrame) -> pl.Series:
        """Make predictions using the trained model.

        Args:
            df: Input data as a Polars DataFrame.

        Returns:
            Polars Series of predictions.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model must be trained before calling predict.")

        X = df.select(self.feature_names)
        predictions = self.model.predict(X)
        return pl.Series(name="predictions", values=predictions, dtype=pl.Float32)

    def save(self, path: Path) -> None:
        """Save the model and metadata to a file.

        Args:
            path: Path to save the model to.
        """
        if self.model is None:
            raise ValueError("No model to save.")

        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        # We could also save feature names separately if needed,
        # but XGBoost models often store them if passed during fit.

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a model from a file.

        Args:
            path: Path to the saved model.

        Returns:
            An instance of XGBoostForecaster with the loaded model.
        """
        instance = cls()
        model = XGBRegressor()
        model.load_model(path)
        instance.model = model
        # Note: feature_names might need to be recovered if not stored in the model
        try:
            instance.feature_names = model.get_booster().feature_names
        except Exception:
            instance.feature_names = None
        return instance

    def get_feature_importance(self) -> pl.DataFrame:
        """Get feature importance as a Polars DataFrame."""
        if self.model is None or self.feature_names is None:
            raise ValueError("Model must be trained.")

        importance = self.model.feature_importances_
        return pl.DataFrame({"feature": self.feature_names, "importance": importance}).sort(
            "importance", descending=True
        )
