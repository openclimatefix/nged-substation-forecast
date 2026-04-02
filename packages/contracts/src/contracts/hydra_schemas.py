"""Hydra configuration schemas for the NGED substation forecast project."""

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DataSplitConfig(BaseModel):
    """Configuration for temporal data splitting."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date


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
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    required_lookback_days: int = Field(default=21)
    features: ModelFeaturesConfig

    # The latency between the NWP init time and when the NWP is actually downloaded and processed
    # and ready for use.
    nwp_availability_delay_hours: int = Field(default=3)

    # The latency between the telemetry timestamp and when it is actually available for use
    # in our forecasting pipeline.
    telemetry_delay_hours: int = Field(default=24)

    # Maximum number of samples to use for training to prevent OOM errors.
    # If set, the training data will be randomly sampled before collection.
    max_training_samples: int | None = Field(default=None, gt=0)


class TrainingConfig(BaseModel):
    """Root configuration object for model training and evaluation."""

    data_split: DataSplitConfig
    model: ModelConfig
