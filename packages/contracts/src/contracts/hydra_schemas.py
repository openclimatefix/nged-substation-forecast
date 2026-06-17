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


class CvFoldConfig(BaseModel):
    """Configuration for a single expanding-window CV fold."""

    fold_id: int = Field(..., ge=1)
    train_start: date
    train_end: date
    val_start: date
    val_end: date


class CvConfig(BaseModel):
    """Configuration for expanding-window cross-validation.

    The folds list defines the evaluation protocol shared by all experiments on the
    leaderboard. All models must be evaluated against the same folds to ensure
    apples-to-apples comparison.

    min_training_months controls which time series are eligible for each fold: a time
    series is only included if it has at least this many months of data before val_start
    (and data through val_end).
    """

    folds: list[CvFoldConfig]
    min_training_months: int = Field(default=6, ge=1)


class ModelFeaturesConfig(BaseModel):
    """Configuration for model features."""

    feature_names: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Configuration for the ML model.

    This Pydantic model documents the shape expected in conf/model/*.yaml files.
    The YAML files also carry a top-level ``_target_`` key (and a nested one under
    ``model_params``) that Hydra's ``instantiate()`` uses to construct the
    ``BaseForecaster`` and its config object — those keys are not part of this
    Pydantic schema because they are Hydra-internal.
    """

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

    # Experiment tags — stored in MLflow so experiments can be grouped on the leaderboard.
    task: str = Field(
        default="",
        description=(
            "The forecasting task, e.g. 'primary_substation_demand', "
            "'solar_pv_generation', 'wind_generation'."
        ),
    )
    model_family: str = Field(
        default="",
        description=(
            "The broad model family, e.g. 'baseline_persistence', 'xgboost', "
            "'pytorch_mlp', 'pytorch_gnn'."
        ),
    )
    weather_source: str = Field(
        default="",
        description=(
            "The NWP source, e.g. 'none', 'ecmwf_control', 'full_ecmwf_ensemble', 'cerra'."
        ),
    )
    training_strategy: str = Field(
        default="",
        description=(
            "How the model is trained across lead times, e.g. "
            "'direct_multistep', 'horizon_as_feature', 'end_to_end'."
        ),
    )


class TrainingConfig(BaseModel):
    """Root configuration object for model training and evaluation."""

    data_split: DataSplitConfig
    model: ModelConfig
    train_on_nwp_ensemble_member: Literal["control_member_only", "all_members"]
