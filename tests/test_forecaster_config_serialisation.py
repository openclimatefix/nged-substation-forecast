"""The invariants every ``BaseForecasterConfig`` subclass must satisfy, whatever it declares.

Two of them, both stated on ``BaseForecasterConfig`` itself and neither enforceable by the type
checker. **Serialisation must be canonical**: a config is compared and stored as its *serialised*
form — ``register_experiment`` stamps ``model_dump_json()`` onto the MLflow experiment as the
``config`` tag and compares a re-registration against it, and logs ``flatten_config(...)`` as
write-once MLflow params — so the dump has to be a pure function of the config's values and
nothing else. **Unknown keys must be rejected**: a subclass that re-opens ``extra`` would let a
misspelled hyperparameter through registration silently.

This module is what makes both enforceable rather than merely documented. It lives in the app
tier, not in ``packages/ml_core``, because enforcing them means importing every concrete
forecaster — a dependency ``ml_core`` itself must not take on.
"""

from typing import get_origin

import pytest
from ml_core.base_forecaster import BaseForecasterConfig
from xgboost_forecaster import XGBoostConfig

_CONFIG_CLASSES: list[type[BaseForecasterConfig]] = [
    BaseForecasterConfig,
    *BaseForecasterConfig.__subclasses__(),
]
"""Every config class to hold to the invariant.

``__subclasses__()`` only sees classes that have been imported, hence the explicit
``XGBoostConfig`` import above; it is what picks up a *future* forecaster automatically.
"""


def test_every_concrete_forecaster_config_is_covered() -> None:
    """Guard against the list silently emptying out if an import is dropped."""
    assert XGBoostConfig in _CONFIG_CLASSES


@pytest.mark.parametrize("config_cls", _CONFIG_CLASSES, ids=lambda cls: cls.__name__)
def test_every_config_class_forbids_extra_keys(config_cls: type[BaseForecasterConfig]) -> None:
    """An unknown key must raise, so a misspelled hyperparameter cannot register silently.

    Pydantic merges a parent's ``model_config`` into a subclass's, so a subclass declaring its own
    ``model_config`` keeps the strictness. What this catches is one that re-opens ``extra``
    explicitly.
    """
    assert config_cls.model_config.get("extra") == "forbid"


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
