from datetime import UTC, datetime

import patito as pt
import pytest
from contracts.power_schemas import EffectiveCapacity


def _one_row(effective_capacity_mw: float) -> pt.DataFrame[EffectiveCapacity]:
    return pt.DataFrame(
        {
            "time_series_id": [123],
            "time": [datetime(2026, 1, 1, 0, 30, tzinfo=UTC)],
            "effective_capacity_mw": [effective_capacity_mw],
        }
    ).set_model(EffectiveCapacity)


def test_effective_capacity_validation() -> None:
    _one_row(10.0).cast().validate()


@pytest.mark.parametrize("bad_value", [0.0, -5.0])
def test_effective_capacity_rejects_non_positive_value(bad_value: float) -> None:
    """`effective_capacity_mw` must be strictly positive — it is the NMAE denominator, and a
    zero or negative capacity would divide every error by a value that can't happen physically.
    ``compute_effective_capacity`` relies on this: it drops non-positive P99s before they ever
    reach this contract, and this pins the boundary it drops them at."""
    with pytest.raises(Exception, match="effective_capacity_mw"):
        _one_row(bad_value).cast().validate()
