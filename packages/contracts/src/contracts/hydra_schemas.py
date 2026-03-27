"""Hydra configuration schemas for the NGED substation forecast project."""

from pydantic import BaseModel, Field


class DataSplitConfig(BaseModel):
    """Configuration for temporal data splitting."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str


class XGBoostHyperparameters(BaseModel):
    """Hyperparameters for the XGBoost model."""

    learning_rate: float = Field(default=0.01, gt=0.0)
    n_estimators: int = Field(default=100, gt=0)
    max_depth: int = Field(default=6, gt=0)


class ModelFeaturesConfig(BaseModel):
    """Configuration for model features."""

    nwp_provider: str


class ModelConfig(BaseModel):
    """Configuration for the ML model."""

    power_fcst_model_name: str
    trainer_class: str | None = None
    hyperparameters: XGBoostHyperparameters
    features: ModelFeaturesConfig


class TrainingConfig(BaseModel):
    """Root configuration object for model training and evaluation."""

    data_split: DataSplitConfig
    model: ModelConfig
