"""Tests for `studies/beam_diffuse_split/past_wind_leaderboard.py`'s declarations.

The scoring, checking, and writing code is tested in `test_past_solar_leaderboard.py` and
`test_past_leaderboard_row_set_options.py`. These tests read no saved results: they check that each
row set is declared consistently, so a typo in an arm name stops here and not in a run that needs
the saved losses.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
SCRIPT_DIR: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split"


def _load() -> ModuleType:
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "past_wind_leaderboard", SCRIPT_DIR / "past_wind_leaderboard.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_each_block_is_contrasted_with_its_own_reference_arm() -> None:
    # Catches the station block contrasted with ERA5's 100 m wind, or a block left on the default
    # solar reference arm, which no wind loss file holds.
    module = _load()

    references = {row_set.key: row_set.reference_arm for row_set in module.ROW_SETS}

    assert references == {
        "main": "era5_wind",
        "icon_dream_eu": "era5_wind",
        "ecmwf": "era5_wind",
        "station": "era5_10m_wind",
    }


def test_the_reference_arm_is_a_reference_row_in_its_block_and_not_a_contrast() -> None:
    module = _load()

    for row_set in module.ROW_SETS:
        reference_rows = [arm.arm for arm in row_set.leaderboard_arms if arm.reference]
        assert reference_rows == [row_set.reference_arm], row_set.key
        assert row_set.reference_arm not in [arm.arm for arm in row_set.contrast_arms]


def test_every_planned_contrast_is_between_two_arms_of_its_block() -> None:
    # Catches a planned contrast naming an arm the block does not score, whose losses the
    # block's bootstrap would then not find.
    module = _load()

    for row_set in module.ROW_SETS:
        scored = {arm.arm for arm in row_set.leaderboard_arms}
        for contrast in row_set.planned_contrasts:
            assert {contrast.treatment.arm, contrast.reference.arm} <= scored, row_set.key


def test_the_blocks_hold_the_planned_contrasts_their_reports_name() -> None:
    module = _load()

    planned = {
        row_set.key: [
            (contrast.treatment.arm, contrast.reference.arm)
            for contrast in row_set.planned_contrasts
        ]
        for row_set in module.ROW_SETS
    }

    assert planned == {
        "main": [
            ("icon_eu_wind", "era5_wind"),
            ("ukv_wind", "era5_wind"),
            ("icon_eu_wind", "ukv_wind"),
            ("icon_d2_wind", "icon_eu_wind"),
        ],
        "icon_dream_eu": [
            ("icon_dream_eu_wind", "era5_wind"),
            ("icon_dream_eu_wind", "icon_eu_wind"),
        ],
        "ecmwf": [
            ("hres_wind", "ukv_wind"),
            ("ens_mean_day0_wind", "ukv_wind"),
            ("hres_wind", "era5_wind"),
        ],
        "station": [
            ("station_wind", "era5_10m_wind"),
            ("ukv_station_wind", "ukv_padded_wind"),
        ],
    }


def test_the_main_row_set_names_the_one_exploratory_contrast_under_its_planned_heading() -> None:
    # Catches the planned-contrast check treating ICON-D2 minus UKV as a planned contrast the
    # script forgot to list.
    module = _load()
    main = module.ROW_SETS[0]

    assert main.exploratory_in_planned == (("icon_d2_wind", "ukv_wind"),)


def test_every_block_has_a_setting_and_every_arm_a_label() -> None:
    module = _load()

    assert set(module.BLOCK_SETTINGS) == {row_set.key for row_set in module.ROW_SETS}
    for row_set in module.ROW_SETS:
        assert {arm.arm for arm in row_set.leaderboard_arms} <= set(module.ARM_LABELS)


def test_only_the_station_block_reads_contrast_tables_with_a_months_column() -> None:
    # Catches a block whose tables carry a `Months` column being read as if they did not, which
    # skips every one of its contrast tables.
    module = _load()

    wide = {row_set.key for row_set in module.ROW_SETS if row_set.wide_contrast_tables}

    assert wide == {"station"}
