from datetime import date
import pytest
from pydantic import ValidationError
from contracts.hydra_schemas import (
    DataSplitConfig,
    ModelFeaturesConfig,
    ModelConfig,
    TrainingConfig,
)


def test_valid_training_config():
    valid_dict = {
        "data_split": {
            "train_start": date(2019, 1, 1),
            "train_end": date(2022, 12, 31),
            "test_start": date(2023, 1, 1),
            "test_end": date(2023, 12, 31),
        },
        "model": {
            "power_fcst_model_name": "xgboost",
            "hyperparameters": {
                "learning_rate": 0.01,
                "n_estimators": 100,
                "max_depth": 6,
            },
            "features": {
                "feature_names": ["temp", "wind"],
            },
        },
        "train_on_nwp_ensemble_member": "control_member_only",
    }
    config = TrainingConfig(**valid_dict)  # type: ignore
    assert config.model.power_fcst_model_name == "xgboost"
    assert config.model.hyperparameters["learning_rate"] == 0.01
    assert config.model.features.feature_names == ["temp", "wind"]


def test_missing_required_field():
    with pytest.raises(ValidationError, match="Field required"):
        DataSplitConfig(  # type: ignore
            train_end=date(2022, 12, 31),
            test_start=date(2023, 1, 1),
            test_end=date(2023, 12, 31),
        )


def test_valid_model_features_config_empty():
    config = ModelFeaturesConfig()
    assert config.feature_names == []
