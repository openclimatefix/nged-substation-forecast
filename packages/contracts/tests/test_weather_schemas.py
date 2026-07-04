from typing import get_args

from contracts.weather_schemas import (
    Nwp,
    NwpMetaData,
    WeatherFeature,
)


def test_nwp_metadata_load():
    # The load method defaults to the correct path in metadata/
    metadata = NwpMetaData.load()
    assert metadata.height >= 1
    assert "ECMWF_ENS_0_25_degree" in metadata["nwp_model_id"].to_list()


def test_weather_feature_literal_matches_model_fields():
    """WeatherFeature Literal must contain exactly the variables in Nwp.all_weather_var_names().

    This test catches any divergence between the static type annotation and the runtime
    model-field derivation — e.g. a new NWP variable added to Nwp but not to
    WeatherFeature, or vice versa.
    """
    assert frozenset(get_args(WeatherFeature)) == Nwp.all_weather_var_names()


def test_continuous_and_categorical_partition_all_weather_vars():
    """continuous_var_names and categorical_var_names must partition all_weather_var_names exactly."""
    all_vars = Nwp.all_weather_var_names()
    continuous = Nwp.continuous_var_names()
    categorical = frozenset(Nwp.categorical_var_names)

    assert continuous | categorical == all_vars, "union must equal all weather vars"
    assert continuous & categorical == frozenset(), "no variable should appear in both sets"
