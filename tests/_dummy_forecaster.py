"""A minimal forecaster whose config class is a ``BaseForecasterConfig`` subclass.

Every production forecaster uses ``BaseForecasterConfig`` itself as its ``CONFIG_CLASS``, so a test
that asserts only ``isinstance(config, BaseForecasterConfig)`` cannot tell whether code reached the
config through ``CONFIG_CLASS`` or constructed ``BaseForecasterConfig`` directly. This forecaster
adds one field, ``dummy_extra``, so a test can check that the value survives the round trip.

The module lives in ``tests/`` so that ``import_class`` can import it by name: the pytest
``importlib`` import mode gives a test module no importable name.
"""

from pathlib import Path
from typing import ClassVar, Self

import patito as pt
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig


class DummyForecasterConfig(BaseForecasterConfig):
    """``BaseForecasterConfig`` plus one field that only this subclass declares."""

    dummy_extra: int = 0


class DummyForecaster(BaseForecaster):
    """A forecaster that trains nothing and whose ``CONFIG_CLASS`` is ``DummyForecasterConfig``."""

    MODEL_NAME = "dummy"
    CONFIG_CLASS: ClassVar[type[BaseForecasterConfig]] = DummyForecasterConfig

    @property
    def trained_time_series_ids(self) -> list[int]:
        return []

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> Self:
        raise NotImplementedError

    def train(self, data: pt.LazyFrame[AllFeatures], time_series_ids: list[int]) -> None:
        raise NotImplementedError

    def predict(
        self, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
    ) -> pt.DataFrame[PowerForecast]:
        raise NotImplementedError
