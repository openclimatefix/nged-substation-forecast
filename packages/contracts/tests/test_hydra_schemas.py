from datetime import date
import pytest
from pydantic import ValidationError
from contracts.hydra_schemas import (
    DataSplitConfig,
    ModelFeaturesConfig,
    NwpModel,
    TrainingConfig,
    XGBoostHyperparameters,
)


def test_valid_training_config():
    valid_dict = {
        "data_split": {
            "train_start": "2019-01-01",
            "train_end": "2022-12-31",
            "test_start": "2023-01-01",
            "test_end": "2023-12-31",
        },
        "model": {
            "power_fcst_model_name": "xgboost",
            "trainer_class": "xgboost_forecaster.trainer.XGBoostTrainer",
            "hyperparameters": {
                "learning_rate": 0.01,
                "n_estimators": 100,
                "max_depth": 6,
            },
            "features": {
                "nwps": ["ecmwf_ens_0_25deg"],
            },
        },
    }
    config = TrainingConfig(**valid_dict)  # type: ignore
    assert config.model.power_fcst_model_name == "xgboost"
    assert config.model.hyperparameters.learning_rate == 0.01
    assert config.model.features.nwps == [NwpModel.ECMWF_ENS_0_25DEG]


def test_invalid_nwp_model():
    with pytest.raises(ValidationError, match="Input should be 'ecmwf_ens_0_25deg'"):
        ModelFeaturesConfig(nwps=["invalid_model"])  # type: ignore


def test_invalid_learning_rate():
    with pytest.raises(ValidationError, match="greater than 0"):
        XGBoostHyperparameters(learning_rate=-0.1, n_estimators=100, max_depth=6)


def test_invalid_n_estimators():
    with pytest.raises(ValidationError, match="greater than 0"):
        XGBoostHyperparameters(learning_rate=0.01, n_estimators=0, max_depth=6)


def test_missing_required_field():
    with pytest.raises(ValidationError, match="Field required"):
        DataSplitConfig(  # type: ignore
            train_end=date(2022, 12, 31),
            test_start=date(2023, 1, 1),
            test_end=date(2023, 12, 31),
        )


def test_valid_model_features_config_empty():
    config = ModelFeaturesConfig()
    assert config.nwps == []
