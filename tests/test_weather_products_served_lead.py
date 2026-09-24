"""Tests for `studies/beam_diffuse_split/weather_products.py`'s `_served_lead`.

IFS-HRES's run cadence changes mid-record: 12 hours before Open-Meteo's ECMWF open-data cutover
(`sources.IFS_OPEN_DATA_CUTOVER`, 1 October 2025), 6 hours from it
(`weather_products.IFS_HRES_RUN_INTERVAL_HOURS`). `test_ifs_hres_lead_follows_the_cutover_date` is
the regression: a version of `_served_lead` that reads one flat interval for `ifs_hres` -- whichever
side of the cutover it picks -- gets one of the two rows below wrong, because hour 7 has a different
lead under a 12-hour and a 6-hour cadence.
"""

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Final

import polars as pl

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
"""The repo root, one level above this `tests/` directory."""

SCRIPT_PATH: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split" / "weather_products.py"
"""The study script under test, imported by path because `studies/` is not an importable package."""


def _load_script() -> ModuleType:
    """Import `weather_products.py` from its path in `studies/beam_diffuse_split/`.

    Its own directory goes on `sys.path` first, because the script does bare imports (`from
    sources import ...`, `from build_dataset import ...`) that only resolve when it is run
    directly, where Python puts the script's own directory there itself.
    """
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("weather_products", SCRIPT_PATH)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_PATH.parent))
    return module


weather_products = _load_script()


def test_ifs_hres_lead_follows_the_cutover_date() -> None:
    """Hour 7's lead differs before and after the cutover, because the cadence does.

    Before (12-hourly): `((7 - 1) % 12) + 1 == 7`. From it (6-hourly): `((7 - 1) % 6) + 1 == 1`.
    """
    frame = pl.DataFrame(
        {
            "time": [
                datetime(2025, 9, 30, 7, tzinfo=UTC),
                datetime(2025, 10, 2, 7, tzinfo=UTC),
            ]
        }
    )

    leads = frame.select(weather_products._served_lead(product="ifs_hres").alias("lead"))[
        "lead"
    ].to_list()

    assert leads == [7, 1], leads


def test_other_products_are_unaffected_by_the_cutover() -> None:
    """A flat-interval product's lead ignores the IFS cutover date entirely."""
    frame = pl.DataFrame(
        {
            "time": [
                datetime(2025, 9, 30, 7, tzinfo=UTC),
                datetime(2025, 10, 2, 7, tzinfo=UTC),
            ]
        }
    )

    leads = frame.select(weather_products._served_lead(product="icon_eu").alias("lead"))[
        "lead"
    ].to_list()

    assert leads == [1, 1], leads
