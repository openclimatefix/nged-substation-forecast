"""Tests for `studies/beam_diffuse_split/past_wind_leaderboard_charts.py`, on synthetic rows.

Each test is built to fail on the bug it exists for: a block drawn without its uncovered-month
share, the station block's contrasts drawn as if against ERA5's hub-height wind, and a figure
number typed by hand.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import polars as pl
import pytest
from studies.charts import RowSetBlock

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
SCRIPT_DIR: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split"
ALL_SHARES: Final[dict[str, float | None]] = {
    "main": 16.2,
    "icon_dream_eu": 25.1,
    "ecmwf": 3.0,
    "station": 42.2,
}


def _load() -> ModuleType:
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "past_wind_leaderboard_charts", SCRIPT_DIR / "past_wind_leaderboard_charts.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _block(*, reference_name: str) -> RowSetBlock:
    rows = pl.DataFrame(
        {
            "arm": ["ukv_wind", "icon_d2_wind"],
            "label": ["UKV", "ICON-D2"],
            "family": ["weather model", "weather model"],
            "reference": [False, False],
            "planned": [False, False],
            "difference": [-0.4, -0.6],
            "lower_95": [-0.6, -0.8],
            "upper_95": [-0.2, -0.4],
        }
    )
    planned = pl.DataFrame(
        {
            "arm": ["icon_d2_wind"],
            "reference_arm": ["ukv_wind"],
            "label": ["ICON-D2 against UKV"],
            "family": ["weather model"],
            "reference": [False],
            "planned": [True],
            "difference": [-0.1],
            "lower_95": [-0.2],
            "upper_95": [0.0],
        }
    )
    return RowSetBlock(
        "Main (100 m)",
        "August 2024",
        8,
        rows,
        planned,
        hours_unit="farm-hours",
        reference_name=reference_name,
    )


def test_a_block_with_no_uncovered_month_share_stops_the_figure() -> None:
    # Catches a block drawn without the share of its rows that no fold design can cover.
    module = _load()

    with pytest.raises(ValueError, match="ECMWF rows"):
        module.uncovered_month_note(shares=module.UNCOVERED_MONTH_SHARES)


def test_the_caption_states_each_blocks_uncovered_month_share() -> None:
    module = _load()

    lines = module.uncovered_month_note(shares=ALL_SHARES)

    assert [line.split(":")[0] for line in lines] == ["Main", "ICON-DREAM-EU", "ECMWF", "Station"]
    assert "42.2%" in lines[3]


def test_the_contrast_figure_names_the_arm_the_station_block_is_against() -> None:
    # Catches the station block's zero rule reading "same as ERA5" when it is ERA5's 10 m wind.
    module = _load()
    blocks = [_block(reference_name="ERA5"), _block(reference_name="ERA5's 10 m wind")]

    figure = module.contrasts_figure(blocks=blocks, shares=ALL_SHARES).to_dict()
    spec = json.dumps(figure, ensure_ascii=False)

    assert "same as ERA5's 10\u00a0m wind" in spec
    assert '"same as ERA5"' in spec
    assert "ICON-D2 against UKV" in spec


def test_the_figures_carry_their_numbers_from_the_wind_map() -> None:
    # Catches a title number typed by hand: leaderboard is Figure 1 and contrasts Figure 2.
    module = _load()
    blocks = [_block(reference_name="ERA5")]

    leaderboard = str(module.contrasts_figure(blocks=blocks, shares=ALL_SHARES).to_dict())

    assert "Figure 2:" in leaderboard
