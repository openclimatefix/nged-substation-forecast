from typing import get_args

from contracts.weather_schemas import (
    Nwp,
    WeatherFeature,
)


def test_weather_feature_literal_matches_model_fields():
    """Model features use derived polar wind; the archive stores components."""
    components = {f"wind_{axis}_{height}" for axis in ("u", "v") for height in ("10m", "100m")}
    polar = {
        f"wind_{quantity}_{height}"
        for quantity in ("speed", "direction")
        for height in ("10m", "100m")
    }
    assert frozenset(get_args(WeatherFeature)) == (Nwp.all_weather_var_names() - components) | polar


def test_continuous_and_categorical_partition_all_weather_vars():
    """continuous_var_names and categorical_var_names must partition all_weather_var_names
    exactly.
    """
    all_vars = Nwp.all_weather_var_names()
    continuous = Nwp.continuous_var_names()
    categorical = frozenset(Nwp.categorical_var_names)

    assert continuous | categorical == all_vars, "union must equal all weather vars"
    assert continuous & categorical == frozenset(), "no variable should appear in both sets"
