import numpy as np
from validate_weathernext3 import (
    DIRECT_RADIATION,
    OFFENDING_FRACTION_LIMIT,
    TOTAL_RADIATION,
    VARIABLES,
    Results,
    _check_radiation,
)

N_VALUES = 200_000
"""One lead time on a 200,000-cell box, so a fraction of 0.01% is 20 values."""


def _direct_check_results(*, offending: int) -> Results:
    """Run `_check_radiation` on a run whose direct radiation exceeds total in `offending` cells."""
    values = np.zeros((len(VARIABLES), 1, 1, N_VALUES), dtype=np.float32)
    values[VARIABLES.index(TOTAL_RADIATION)] = 3600.0 * 200.0
    direct = values[VARIABLES.index(DIRECT_RADIATION)]
    direct[...] = 3600.0 * 100.0
    direct[..., :offending] = 3600.0 * 400.0
    valid = np.array(["2026-06-01T12:00"], dtype="datetime64[ns]")
    results: Results = {}
    _check_radiation(values=values, valid=valid, label="run", results=results)
    return results


def test_direct_le_total_passes_below_the_direct_limit() -> None:
    offending = 100  # 0.05% of the values, above the 0.01% `ranges` limit.
    assert offending / N_VALUES > OFFENDING_FRACTION_LIMIT
    assert "direct_le_total" not in _direct_check_results(offending=offending)


def test_direct_le_total_fails_above_the_direct_limit() -> None:
    results = _direct_check_results(offending=101)
    assert results["direct_le_total"] == ["run (0.051% of values)"]


def test_direct_le_total_passes_when_no_value_offends() -> None:
    assert "direct_le_total" not in _direct_check_results(offending=0)
