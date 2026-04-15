"""Hydra configuration schemas for the NGED substation forecast project."""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class DataSplitConfig(BaseModel):
    """Configuration for temporal data splitting."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date


class ModelFeaturesConfig(BaseModel):
    """Configuration for model features."""

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
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    features: ModelFeaturesConfig

    # The latency between the NWP init time and when the NWP is actually downloaded and processed
    # and ready for use.
    nwp_delay_hours: int = Field(default=3)

    # The latency between the telemetry timestamp and when it is actually available for use
    # in our forecasting pipeline.
    power_telemetry_delay_hours: int = Field(default=24)


class TrainingConfig(BaseModel):
    """Root configuration object for model training and evaluation."""

    data_split: DataSplitConfig
    model: ModelConfig
    train_on_nwp_ensemble_member: Literal["control", "all"]
