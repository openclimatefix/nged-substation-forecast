from pydantic import BaseModel, Field


class XGBoostHyperparameters(BaseModel):
    """Hyperparameters for the XGBoost model."""

    learning_rate: float = Field(default=0.01, gt=0.0)
    n_estimators: int = Field(default=100, gt=0)
    max_depth: int = Field(default=6, gt=0)
    enable_categorical: bool = Field(default=True)
