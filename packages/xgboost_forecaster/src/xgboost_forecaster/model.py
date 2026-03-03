"""XGBoost model wrapper for substation forecasting."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import polars as pl
import xgboost as xgb
from xgboost import XGBRegressor
from typing import cast

log = logging.getLogger(__name__)


class XGBoostForecaster:
    """Wrapper around XGBoost for substation-level forecasting."""

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
        target_col: str = "power_mw",
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
            feature_cols = [c for c in df.columns if c != target_col and c != "timestamp"]

        self.feature_names = feature_cols
        X = df.select(feature_cols).to_numpy()
        y = df.select(target_col).to_numpy().flatten()

        xgb_eval_set = None
        if eval_set:
            xgb_eval_set = []
            for X_eval_df, y_eval_series in eval_set:
                xgb_eval_set.append(
                    (X_eval_df.select(feature_cols).to_numpy(), y_eval_series.to_numpy())
                )

        self.model = cast(Any, xgb).XGBRegressor(**self.params)
        self.model.fit(X, y, eval_set=xgb_eval_set, verbose=False)

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

        X = df.select(self.feature_names).to_numpy()
        preds = self.model.predict(X)
        return pl.Series("predictions", preds)

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
        instance.model = cast(Any, xgb).XGBRegressor()
        instance.model.load_model(path)
        # Note: feature_names might need to be recovered if not stored in the model
        try:
            instance.feature_names = instance.model.get_booster().feature_names
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
