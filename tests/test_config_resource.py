from nged_substation_forecast.config_resource import NgedConfig


def test_nged_config_initialization():
    config = NgedConfig()
    assert config is not None
