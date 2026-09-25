"""Tests for `studies/beam_diffuse_split/ens_forecast_horizons.py`'s `band_steps(six_hourly=True)`.

`coarsen_to_six_hourly` drops a step at a multiple of 6 hours unless the step 3 hours earlier is
present, but only for a field that is a period mean. A wind field is instantaneous, so a wind band
in 6-hourly mode must keep every step at a multiple of 6 hours, including lead 0 on day 0.
"""

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import polars as pl

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
"""The repo root, one level above this `tests/` directory."""

SCRIPT_PATH: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split" / "ens_forecast_horizons.py"
"""The study script under test, imported by path because `studies/` is not an importable package."""

ENSEMBLE_SIZE: Final[int] = 2
"""Members per run in the synthetic extract."""


def _load_script() -> ModuleType:
    """Import `ens_forecast_horizons.py` from its path, with its own directory on `sys.path`."""
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("ens_forecast_horizons", SCRIPT_PATH)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_PATH.parent))
    return module


efh = _load_script()


def _extract(*, domain: str) -> pl.DataFrame:
    """Return one site's one run, two members, on 3-hour leads 0 to 48, with every field filled."""
    leads = list(range(0, 49, 3))
    rows = [
        {
            "site": "a",
            "init_time": datetime(2025, 1, 1, tzinfo=UTC),
            "ensemble_member": member,
            "lead_hours": lead,
            "ghi_w_m2": float(lead),
            "temp_c": 10.0,
            "speed_100m": float(lead),
            "direction_100m": 90.0,
            "speed_10m": 5.0,
            "direction_10m": 90.0,
        }
        for member in range(ENSEMBLE_SIZE)
        for lead in leads
    ]
    return pl.DataFrame(rows)


def test_a_wind_band_keeps_its_first_stamp_in_six_hourly_mode() -> None:
    """Day 0 wind keeps lead 0, and day 1 wind keeps lead 18, the first multiple of 6 in its window.

    A version that passes `period_means={"ghi_w_m2"}` for wind demands a step 3 hours before each
    kept lead, so it drops lead 0 (day 0) and lead 18 (day 1, whose window starts at 18).
    """
    extract = _extract(domain="wind")
    day_0 = efh.band_steps(
        members=extract, day=0, domain="wind", six_hourly=True, ensemble_size=ENSEMBLE_SIZE
    )
    day_1 = efh.band_steps(
        members=extract, day=1, domain="wind", six_hourly=True, ensemble_size=ENSEMBLE_SIZE
    )
    assert day_0.leads.tolist() == [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]
    assert day_1.leads.tolist() == [18.0, 24.0, 30.0, 36.0, 42.0, 48.0]
    np.testing.assert_array_equal(day_0.values["speed_100m"][0], day_0.leads)


def test_a_solar_band_still_drops_lead_zero_in_six_hourly_mode() -> None:
    """Solar has no radiation at lead 0, so day 0 starts at lead 6, the mean of leads 3 and 6."""
    steps = efh.band_steps(
        members=_extract(domain="solar"),
        day=0,
        domain="solar",
        six_hourly=True,
        ensemble_size=ENSEMBLE_SIZE,
    )
    assert steps.leads.tolist() == [6.0, 12.0, 18.0, 24.0, 30.0]
    assert steps.values["ghi_w_m2"][0].tolist() == [4.5, 10.5, 16.5, 22.5, 28.5]
