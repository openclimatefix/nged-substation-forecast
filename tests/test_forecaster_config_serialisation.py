"""The invariants every ``BaseForecasterConfig`` subclass must satisfy, whatever it declares.

Both are stated on ``BaseForecasterConfig`` and neither is enforceable by the type checker:
serialisation must be canonical, and unknown keys must be rejected. This module is what makes them
enforceable rather than merely documented. It lives in the app tier, not in ``packages/ml_core``,
because enforcing them means importing every concrete forecaster — a dependency ``ml_core`` itself
must not take on.
"""

from typing import get_origin

import pytest
from ml_core.base_forecaster import BaseForecasterConfig
from pydantic import ValidationError
from xgboost_forecaster import XGBoostConfig


def _config_classes(cls: type[BaseForecasterConfig]) -> set[type[BaseForecasterConfig]]:
    """``cls`` and every class below it — ``__subclasses__()`` sees only one level."""
    return {cls}.union(*map(_config_classes, cls.__subclasses__()))


_CONFIG_CLASSES: list[type[BaseForecasterConfig]] = sorted(
    _config_classes(BaseForecasterConfig), key=lambda cls: cls.__name__
)
"""Every config class to hold to the invariants.

Only classes that have been *imported* are visible, hence the explicit ``XGBoostConfig`` import
above. Sorted so the parametrised ids stay in a stable order.
"""


def test_every_concrete_forecaster_config_is_covered() -> None:
    """Guard against the list silently emptying out if an import is dropped."""
    assert XGBoostConfig in _CONFIG_CLASSES


@pytest.mark.parametrize(
    argnames="config_cls", argvalues=_CONFIG_CLASSES, ids=lambda cls: cls.__name__
)
def test_every_config_class_forbids_extra_keys(config_cls: type[BaseForecasterConfig]) -> None:
    """An unknown key must raise, so a misspelled hyperparameter cannot register silently.

    Asserted on the behaviour rather than on the ``model_config`` flag, so a subclass that keeps
    ``extra="forbid"`` but strips unknown keys in a ``model_validator(mode="before")`` fails too.
    """
    # Splatted from a named mapping: a literal keyword is a static error (no such parameter), and
    # a literal dict trips PIE804. This is the one spelling both linters accept.
    unknown_key = {"not_a_declared_field": "x"}

    with pytest.raises(ValidationError, match="not_a_declared_field"):
        config_cls(selected_features=set(), **unknown_key)


def _set_valued_fields(config_cls: type[BaseForecasterConfig]) -> set[str]:
    """Names of ``config_cls``'s fields whose declared type is a set."""
    return {
        name
        for name, field in config_cls.model_fields.items()
        if get_origin(field.annotation) in (set, frozenset)
    }


def _fields_with_a_serialiser(config_cls: type[BaseForecasterConfig]) -> set[str]:
    """Names of ``config_cls``'s fields that declare their own serialiser."""
    return {
        field
        for decorator in config_cls.__pydantic_decorators__.field_serializers.values()
        for field in decorator.info.fields
    }


@pytest.mark.parametrize("config_cls", _CONFIG_CLASSES, ids=lambda cls: cls.__name__)
def test_every_set_valued_field_declares_a_serialiser(
    config_cls: type[BaseForecasterConfig],
) -> None:
    """A set-valued field must pin its own serialisation order.

    A ``set`` of strings iterates in a different order in every process (hash randomisation), and
    Dagster launches each job run in its own process, so a set left to serialise itself makes two
    dumps of the *same* config differ — which reads as a config change and collides with MLflow's
    write-once params.
    """
    assert _set_valued_fields(config_cls) <= _fields_with_a_serialiser(config_cls)


@pytest.mark.parametrize("config_cls", _CONFIG_CLASSES, ids=lambda cls: cls.__name__)
def test_the_declared_serialiser_actually_sorts(
    config_cls: type[BaseForecasterConfig],
) -> None:
    """Declaring a serialiser is not enough — it has to produce a canonical order."""
    features = {"wind_speed_10m", "power_lag_24h", "hour_of_day", "temperature_2m"}

    dumped = config_cls(selected_features=features).model_dump(mode="json")

    assert dumped["selected_features"] == sorted(features)
