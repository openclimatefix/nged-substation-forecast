from typing import cast
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from contracts.hydra_schemas import TrainingConfig


def test_real_xgboost_yaml_parses_successfully():
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=["model=xgboost"])
        cfg_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
        print(f"DEBUG: {cfg_dict=}")

        # This will raise a ValidationError if the YAML doesn't match the schema
        config = TrainingConfig(**cfg_dict)

        assert config.model.power_fcst_model_name == "xgboost"
        assert config.model.hyperparameters["n_estimators"] > 0
        assert config.data_split.train_start is not None
        assert config.model.features.nwps == ["ecmwf_ens_0_25deg"]
