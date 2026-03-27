"""Hydra configuration schemas for the NGED substation forecast project."""

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class DataSplitConfig(BaseModel):
    """Configuration for temporal data splitting."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date


class XGBoostHyperparameters(BaseModel):
    """Hyperparameters for the XGBoost model."""

    learning_rate: float = Field(default=0.01, gt=0.0)
    n_estimators: int = Field(default=100, gt=0)
    max_depth: int = Field(default=6, gt=0)


class NwpModel(str, Enum):
    """Available NWP datasets."""

    ECMWF_ENS_0_25DEG = "ecmwf_ens_0_25deg"


class ModelFeaturesConfig(BaseModel):
    """Configuration for model features."""

    nwps: list[NwpModel] = Field(default_factory=list)
    feature_names: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Configuration for the ML model."""

    power_fcst_model_name: str = Field(
        ...,
        description=(
            "A unique identifier for this model configuration. This name is used to label "
            "predictions in the evaluation results and to identify the model in MLflow. "
            "Users should use this as free text to describe substantial differences between "
            "different versions of the same underlying model (e.g., 'xgboost_baseline', "
            "'xgboost_with_solar_features')."
        ),
    )
    hyperparameters: XGBoostHyperparameters
    features: ModelFeaturesConfig


class TrainingConfig(BaseModel):
    """Root configuration object for model training and evaluation."""

    data_split: DataSplitConfig
    model: ModelConfig
