from typing import get_args

from contracts.weather_schemas import (
    NWP_VARIABLE_EMPTY_WINDOWS,
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


def test_every_variable_with_an_empty_window_is_a_weather_variable() -> None:
    assert frozenset(NWP_VARIABLE_EMPTY_WINDOWS) <= Nwp.all_weather_var_names()


def test_total_cloud_cover_is_a_continuous_variable() -> None:
    """Continuous variables are the ones `delta_store.nwp.write_nwp` rounds and H3 averages."""
    assert "total_cloud_cover_atmosphere" in Nwp.continuous_var_names()


def test_columns_absent_from_older_partitions_are_nullable_weather_variables() -> None:
    for column in Nwp.columns_absent_from_older_partitions:
        assert column in Nwp.all_weather_var_names()
        assert column in Nwp.nullable_columns
