"""XGBoost model wrapper for substation forecasting."""

import logging
from collections.abc import Sequence
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

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the underlying XGBoost models from the artifacts.

        Args:
            context: MLflow context containing the artifacts.
        """
        self.models: dict[str, XGBoostForecaster] = {}
        for name, path in context.artifacts.items():
            self.models[name] = XGBoostForecaster.load(Path(path))

    def predict(  # type: ignore
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pt.DataFrame[SubstationFeatures],
        params: InferenceParams,
    ) -> pt.DataFrame[PowerForecast]:
        """Make predictions using the loaded models.

        Args:
            context: MLflow context.
            model_input: Input data as a Patito DataFrame of SubstationFeatures.
            params: Parameters for inference, including 'nwp_init_time'
                and 'power_fcst_model'.

        Returns:
            Patito DataFrame of PowerForecast.
        """
        # If params is a dict (passed by MLflow), convert to InferenceParams object.
        # We must do this manually because we've disabled MLflow's automatic
        # type hint validation/conversion.
        if isinstance(params, dict):
            params = InferenceParams(**params)

        nwp_init_time = params.nwp_init_time
        power_fcst_model = params.power_fcst_model or XGBoostForecaster.model_name_and_version()

        # If we have a global model, use it for all substations.
        if "global" in self.models:
            preds = self.models["global"].predict(model_input)
            res = model_input.select(["valid_time", "substation_number"]).with_columns(
                MW_or_MVA=pl.Series(values=preds, dtype=pl.Float32)
            )
        else:
            # Otherwise, route to local models
            all_preds = []
            for substation_number, group in model_input.group_by("substation_number"):
                sub_id_str = str(substation_number)
                if sub_id_str not in self.models:
                    log.warning(f"No model found for substation {substation_number}")
                    continue

                preds = self.models[sub_id_str].predict(group)
                all_preds.append(
                    group.select(["valid_time", "substation_number"]).with_columns(
                        MW_or_MVA=pl.Series(values=preds, dtype=pl.Float32)
                    )
                )

            if not all_preds:
                raise NoPredictionsError(
                    f"No models found for any of the substations in the input data: "
                    f"{model_input['substation_number'].unique().to_list()}"
                )

            res = pl.concat(all_preds)

        # Add common metadata columns
        res = res.with_columns(
            nwp_init_time=pl.lit(nwp_init_time).cast(pl.Datetime("us", "UTC")),
            power_fcst_model=pl.lit(power_fcst_model).cast(pl.Categorical),
            ensemble_member=pl.lit(0).cast(pl.UInt8),
        )

        return cast(pt.DataFrame[PowerForecast], res)


# TODO: Create an abstract base class (e.g. `BaseForecaster`) that defines the universal
# interface for *training* and *saving* all ML models (XGBoost, PyTorch GNN, etc.).
# While the `XGBoostPyFuncWrapper` above provides a universal interface for *inference*
# via MLflow, Dagster's training assets still need a common interface to instantiate,
# train, and save the underlying mathematical models regardless of their architecture.
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


def train_local_xgboost_model(
    substation_number: int,
    df: pl.DataFrame,
    output_path: Path,
    target_col: str = "MW_or_MVA",
    train_test_split: float = 0.8,
) -> Path:
    """Train an XGBoost model for a single substation and save it.

    Args:
        substation_number: The substation number.
        df: Dataframe containing features and target for this substation.
        output_path: Path where the trained model should be saved.
        target_col: Name of the target column.
        train_test_split: Fraction of data to use for training (rest for evaluation).

    Returns:
        The path to the saved model.

    Raises:
        ValueError: If the input dataframe is empty.
    """
    if df.is_empty():
        raise ValueError(f"No data available for substation {substation_number}")

    # Train model
    forecaster = XGBoostForecaster()

    # Split into train/eval
    df = df.sort("valid_time")
    train_size = int(len(df) * train_test_split)
    train_df = df.head(train_size)
    eval_df = df.tail(len(df) - train_size)

    feature_cols = [
        c for c in df.columns if c not in [target_col, "valid_time", "substation_number"]
    ]

    eval_set = [(eval_df, eval_df[target_col])]

    forecaster.train(
        df=train_df,
        target_col=target_col,
        feature_cols=feature_cols,
        eval_set=eval_set,
    )

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecaster.save(output_path)

    return output_path
