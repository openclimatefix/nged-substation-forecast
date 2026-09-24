"""Tests for `studies/beam_diffuse_split/check_new_products.py`'s `_local_prominence`.

`_night_jump_lines` prints a curvature ratio against the whole day's median, which a run interval
of a few hours can raise across several adjacent hours rather than showing a clean peak at each
hour the interval divides. `_local_prominence` fixes that by dividing each hour by the mean of its
two circular neighbours instead. `test_a_run_every_three_hours_peaks_at_every_third_hour` is the
regression: a version of `_local_prominence` that used the whole-day median instead of the two
neighbours would pass a plateau spanning hours 8 to 16 without ever showing which of those hours is
the true switch, because every hour in the plateau sits above the median alike.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import polars as pl

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
"""The repo root, one level above this `tests/` directory."""

SCRIPT_PATH: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split" / "check_new_products.py"
"""The study script under test, imported by path because `studies/` is not an importable package."""


def _load_script() -> ModuleType:
    """Import `check_new_products.py` from its path in `studies/beam_diffuse_split/`.

    Its own directory goes on `sys.path` first, because the script does bare imports (`from
    sources import ...`, `from build_dataset import ...`) that only resolve when it is run
    directly, where Python puts the script's own directory there itself.
    """
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("check_new_products", SCRIPT_PATH)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_PATH.parent))
    return module


check_new_products = _load_script()


def test_a_run_every_three_hours_peaks_at_every_third_hour() -> None:
    """A plateau raised at every hour divisible by 3 becomes a peak at exactly those hours.

    The ratio is 1.0 everywhere except a raised value at hours 0, 3, 6, ..., 21 -- an idealised
    3-hourly run switch. Local prominence must show every one of those hours above 1 and every
    other hour at or below 1, which a whole-day-median ratio cannot guarantee once the raised hours
    make up a large share of the day.
    """
    hours = list(range(24))
    relative = [1.3 if hour % 3 == 0 else 1.0 for hour in hours]
    ratio = pl.DataFrame({"hour": hours, "relative": relative})

    prominence = check_new_products._local_prominence(ratio=ratio)

    values = dict(zip(prominence["hour"].to_list(), prominence["relative"].to_list(), strict=True))
    for hour in hours:
        if hour % 3 == 0:
            assert values[hour] > 1.0, (hour, values[hour])
        else:
            assert values[hour] <= 1.0, (hour, values[hour])


def test_a_flat_ratio_has_no_prominent_hour() -> None:
    """A ratio that never departs from 1 has no hour standing out from its neighbours."""
    hours = list(range(24))
    ratio = pl.DataFrame({"hour": hours, "relative": [1.0] * len(hours)})

    prominence = check_new_products._local_prominence(ratio=ratio)

    np.testing.assert_allclose(prominence["relative"].to_numpy(), 1.0)
