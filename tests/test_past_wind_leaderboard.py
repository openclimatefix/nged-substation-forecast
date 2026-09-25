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

import polars as pl
from studies.charts import CONTRAST_COLUMNS, CONTRAST_COLUMNS_WITH_MONTHS, report_contrasts

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


def test_every_block_has_a_setting_and_every_arm_a_label() -> None:
    module = _load()

    assert set(module.BLOCK_SETTINGS) == {row_set.key for row_set in module.ROW_SETS}
    for row_set in module.ROW_SETS:
        assert {arm.arm for arm in row_set.leaderboard_arms} <= set(module.ARM_LABELS)


def _table(*, rows: list[tuple[str, str, int]], months: bool = False) -> str:
    """Return a contrast table in the layout the past-wind reports print, one row per contrast."""
    columns = CONTRAST_COLUMNS_WITH_MONTHS if months else CONTRAST_COLUMNS
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += [
        f"| {scope} | {contrast} | +0.100 | [-0.100, +0.300] | no | 3 of 5 | {n_rows:,}"
        + (" | 17 |" if months else " |")
        for scope, contrast, n_rows in rows
    ]
    return "\n".join(lines)


def _pair(*, first: str, second: str) -> str:
    return f"{first} \u2212 {second}"


REPORT_TEXTS: Final[dict[str, str]] = {
    "main": "\n\n".join(
        [
            "#### Deciding contrasts, named before the run",
            _table(
                rows=[
                    ("all", _pair(first="icon_eu_wind", second="era5_wind"), 50734),
                    ("site W1", _pair(first="icon_eu_wind", second="era5_wind"), 16886),
                    ("all", _pair(first="ukv_wind", second="era5_wind"), 50734),
                    ("all", _pair(first="icon_eu_wind", second="ukv_wind"), 50734),
                    ("all", _pair(first="icon_d2_wind", second="icon_eu_wind"), 50734),
                    ("all", _pair(first="icon_d2_wind", second="ukv_wind"), 50734),
                ]
            ),
            "#### Sensitivity: the served 100 m arm the plan specified, and the second setting",
            _table(
                rows=[
                    (
                        "served 100 m and 10 m",
                        _pair(first="icon_eu_wind", second="era5_wind"),
                        50734,
                    ),
                    ("second setting", _pair(first="icon_eu_wind", second="era5_wind"), 50734),
                    ("second setting", _pair(first="ukv_wind", second="era5_wind"), 50734),
                    ("second setting", _pair(first="icon_eu_wind", second="ukv_wind"), 50734),
                    ("second setting", _pair(first="icon_d2_wind", second="icon_eu_wind"), 50734),
                    ("second setting", _pair(first="icon_d2_wind", second="ukv_wind"), 50734),
                ]
            ),
        ]
    ),
    "icon_dream_eu": "\n\n".join(
        [
            "#### Deciding contrasts, named before the run",
            _table(
                rows=[
                    ("all", _pair(first="icon_dream_eu_wind", second="era5_wind"), 50041),
                    ("site W1", _pair(first="icon_dream_eu_wind", second="era5_wind"), 16657),
                    ("all", _pair(first="icon_dream_eu_wind", second="icon_eu_wind"), 50041),
                ]
            ),
            "#### The same two contrasts at the second hyperparameter setting",
            _table(
                rows=[
                    ("all", _pair(first="icon_dream_eu_wind", second="era5_wind"), 50041),
                    ("all", _pair(first="icon_dream_eu_wind", second="icon_eu_wind"), 50041),
                ]
            ),
        ]
    ),
    "ecmwf": "\n\n".join(
        [
            "#### Planned contrasts P1 to P3, pooled over three farms, primary setting",
            _table(
                rows=[
                    ("all", _pair(first="hres_wind", second="ukv_wind"), 43555),
                    ("all", _pair(first="ens_mean_day0_wind", second="ukv_wind"), 43555),
                    ("all", _pair(first="hres_wind", second="era5_wind"), 43555),
                ]
            ),
            "#### Planned contrasts P1 to P3, pooled over three farms, second setting",
            _table(
                rows=[
                    ("all", _pair(first="hres_wind", second="ukv_wind"), 43555),
                    ("all", _pair(first="ens_mean_day0_wind", second="ukv_wind"), 43555),
                    ("all", _pair(first="hres_wind", second="era5_wind"), 43555),
                ]
            ),
        ]
    ),
    "station": "\n\n".join(
        [
            "#### Planned contrasts S1 and S2 at the primary setting (first minus second)",
            _table(
                rows=[
                    ("all", _pair(first="station_wind", second="era5_10m_wind"), 34156),
                    ("all", _pair(first="ukv_station_wind", second="ukv_padded_wind"), 34156),
                ],
                months=True,
            ),
            "#### Planned contrasts S1 and S2 at the second setting (first minus second)",
            _table(
                rows=[
                    ("all", _pair(first="station_wind", second="era5_10m_wind"), 34156),
                    ("all", _pair(first="ukv_station_wind", second="ukv_padded_wind"), 34156),
                ],
                months=True,
            ),
        ]
    ),
}
"""The planned-contrast and second-setting tables of each real past-wind report, in its layout.

The scope cell of a second-setting row is `second setting` in the main report and `all` in the rest.
"""


def _printed(*, key: str, tmp_path: Path) -> pl.DataFrame:
    path = tmp_path / f"{key}.md"
    path.write_text(REPORT_TEXTS[key])
    return report_contrasts(report_path=path, extra_headers=(CONTRAST_COLUMNS_WITH_MONTHS,))


def test_each_blocks_planned_contrasts_are_the_rows_its_report_prints_at_scope_all(
    tmp_path: Path,
) -> None:
    # Catches a planned-contrast declaration that drifts from the pairs the report prints, or a
    # `planned_section` heading that matches none of them.
    module = _load()

    for row_set in module.ROW_SETS:
        printed = _printed(key=row_set.key, tmp_path=tmp_path)
        rows = printed.filter(
            printed["section"].str.starts_with(row_set.planned_section),
            printed["scope"] == "all",
        )
        printed_pairs = set(zip(rows["treatment"], rows["reference"], strict=True))
        declared = {
            (contrast.treatment.arm, contrast.reference.arm)
            for contrast in row_set.planned_contrasts
        } | set(row_set.exploratory_in_planned)
        assert printed_pairs == declared, row_set.key


def test_each_blocks_second_setting_scope_is_the_scope_its_report_prints(tmp_path: Path) -> None:
    # Catches a block left on the solar default `sensitivity`, which its wind report never prints:
    # the ICON-DREAM-EU, ECMWF and station reports print `all` under their second-setting headings,
    # the main report prints `second setting`.
    module = _load()

    for row_set in module.ROW_SETS:
        printed = _printed(key=row_set.key, tmp_path=tmp_path)
        rows = printed.filter(
            printed["section"].str.starts_with(row_set.second_section),
            printed["scope"] == row_set.second_scope,
        )
        pairs = set(zip(rows["treatment"], rows["reference"], strict=True))
        declared = {
            (contrast.treatment.arm, contrast.reference.arm)
            for contrast in row_set.planned_contrasts
        }
        assert declared <= pairs, row_set.key


def test_the_station_block_tells_its_two_era5_arms_apart() -> None:
    # Catches two arms both labelled "ERA5" in the station block, and a station contrast heading
    # that says "minus ERA5's" when the reference is ERA5's 10 m wind.
    module = _load()
    by_key = {row_set.key: row_set for row_set in module.ROW_SETS}

    station_labels = {arm.arm: arm.label for arm in by_key["station"].leaderboard_arms}
    assert station_labels["era5_10m_wind"] == "ERA5 10\u00a0m"
    assert station_labels["era5_wind"] == "ERA5 100\u00a0m"
    assert by_key["station"].reference_label == "ERA5's 10 m wind"
    for key in ("main", "icon_dream_eu", "ecmwf"):
        labels = {arm.arm: arm.label for arm in by_key[key].leaderboard_arms}
        assert labels["era5_wind"] == "ERA5", key
        assert by_key[key].reference_label == "ERA5's", key


def test_only_the_main_block_marks_the_three_contrasts_that_use_the_80_m_icon_arms() -> None:
    # Catches the post-hoc marks missing from the main block, or set on a block whose planned
    # contrasts were written before the run.
    module = _load()

    marked = {row_set.key: row_set.post_hoc_contrasts for row_set in module.ROW_SETS}

    assert marked == {
        "main": (
            ("icon_eu_wind", "era5_wind"),
            ("icon_eu_wind", "ukv_wind"),
            ("icon_d2_wind", "icon_eu_wind"),
        ),
        "icon_dream_eu": (),
        "ecmwf": (),
        "station": (),
    }
    main = module.ROW_SETS[0]
    planned = {(c.treatment.arm, c.reference.arm) for c in main.planned_contrasts}
    assert set(main.post_hoc_contrasts) <= planned


def test_the_icon_dream_eu_block_carries_the_post_scoring_note() -> None:
    module = _load()

    notes = {key: setting.note for key, setting in module.BLOCK_SETTINGS.items()}

    assert notes["icon_dream_eu"] == (
        "planned contrasts were written after the main block's five products were scored."
    )
    assert [key for key, note in notes.items() if note] == ["icon_dream_eu"]


def test_a_block_reads_a_months_column_exactly_where_its_report_prints_one(tmp_path: Path) -> None:
    # Catches a block whose report tables carry a `Months` column being read as if they did not,
    # which skips every one of its contrast tables, and the reverse.
    module = _load()

    for row_set in module.ROW_SETS:
        path = tmp_path / f"{row_set.key}.md"
        path.write_text(REPORT_TEXTS[row_set.key])
        read_without_months = report_contrasts(report_path=path).height > 0
        assert read_without_months != row_set.wide_contrast_tables, row_set.key
