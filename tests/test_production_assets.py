"""Unit tests for helpers in ``defs/production_assets.py`` that don't need the full
``live_forecasts`` integration fixtures in ``tests/test_live_forecasts.py``.
"""

from datetime import UTC, datetime

import pytest
from contracts.settings import Settings

from nged_substation_forecast.defs import production_assets


def _patch_delta_table(monkeypatch: pytest.MonkeyPatch, raw_init_times: set[str]) -> None:
    """Monkeypatch ``production_assets.DeltaTable`` to return canned partition values.

    Isolates ``_available_nwp_init_times``'s parsing from a real Delta table on disk — the
    function only ever reads ``DeltaTable(...).partitions()``.
    """

    class _FakeDeltaTable:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def partitions(self) -> list[dict[str, str]]:
            return [
                {"nwp_model_id": "ECMWF_ENS_0_25_degree", "init_time": value}
                for value in raw_init_times
            ]

    monkeypatch.setattr(production_assets, "DeltaTable", _FakeDeltaTable)


@pytest.mark.parametrize(
    "raw_value",
    [
        "2026-07-04 00:00:00.000000",  # delta-rs' current rendering: always six fractional digits
        "2026-07-04 00:00:00",  # a hypothetical future delta-rs release dropping trailing zeros
    ],
)
def test_available_nwp_init_times_parses_both_fractional_second_renderings(
    monkeypatch: pytest.MonkeyPatch, raw_value: str
) -> None:
    """Must survive either spelling delta-rs might emit for a whole-second ``init_time``.

    ``datetime.strptime``'s ``%f`` directive requires at least one fractional digit, so it would
    raise on the second rendering; ``datetime.fromisoformat`` (what the code uses) accepts both.
    """
    _patch_delta_table(monkeypatch, {raw_value})

    result = production_assets._available_nwp_init_times(Settings())

    assert result == [datetime(2026, 7, 4, 0, 0, tzinfo=UTC)]
