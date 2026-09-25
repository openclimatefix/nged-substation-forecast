"""Tests for `studies/beam_diffuse_split/past_solar_leaderboard_charts.py`, on synthetic rows.

Each test is built to fail on the bug it exists for: a caveat dropped from a figure's caption, a
post hoc row drawn without its label, and a headline that states an exploratory row as planned.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import polars as pl
import pytest
from studies.charts import BlockArm, RowSetBlock

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
SCRIPT_DIR: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split"


def _load() -> ModuleType:
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "past_solar_leaderboard_charts", SCRIPT_DIR / "past_solar_leaderboard_charts.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contrast_block(*, cams: float) -> RowSetBlock:
    rows = pl.DataFrame(
        {
            "arm": ["cams_global", "ukv_global"],
            "label": ["CAMS", "UKV"],
            "family": ["satellite", "weather model"],
            "reference": [True, False],
            "planned": [False, False],
            "difference": [cams, -0.5],
            "lower_95": [cams - 0.3, -0.7],
            "upper_95": [cams + 0.3, -0.3],
        }
    )
    planned = pl.DataFrame(
        {
            "arm": ["ukv_global"],
            "reference_arm": ["cams_global"],
            "label": ["UKV against CAMS"],
            "family": ["weather model"],
            "reference": [False],
            "planned": [True],
            "difference": [0.1],
            "lower_95": [0.05],
            "upper_95": [0.15],
        }
    )
    return RowSetBlock("Main", "January 2025", 8, rows, planned)


def _caption(*, spec: dict) -> str:
    title = spec["title"]
    return " ".join([*title["text"], *title["subtitle"]])


def test_the_contrast_figures_title_and_subtitle_say_the_cams_row_is_exploratory() -> None:
    module = _load()

    spec = module.contrasts_figure(
        blocks=[_contrast_block(cams=-3.7), _contrast_block(cams=-4.1)]
    ).to_dict()

    caption = _caption(spec=spec)
    assert "CAMS beats ERA5 by 3.7 to 4.1 points in each block (exploratory)" in caption
    assert "The CAMS row is exploratory" in caption


def test_the_contrast_figure_warns_that_a_difference_from_era5_moves_between_blocks() -> None:
    module = _load()

    spec = module.contrasts_figure(
        blocks=[_contrast_block(cams=-3.7), _contrast_block(cams=-4.2)]
    ).to_dict()

    assert "not comparable across blocks either" in _caption(spec=spec)
    assert "moves by as much as 0.5 points between blocks" in _caption(spec=spec)


def test_the_spread_is_taken_from_the_data_not_a_literal() -> None:
    module = _load()

    assert "moves by as much as 1.2 points" in module.contrasts_not_comparable(
        cams_differences=[-3.0, -4.2, -3.5]
    )


@pytest.mark.parametrize(
    "caveat",
    [
        "5 to 20 hours ahead from a 00 UTC run",
        "ERA5's radiation is 1 to 12 hours ahead",
        "one pyranometer",
        "17 to 31 km away",
    ],
)
def test_both_figures_carry_the_lead_and_pyranometer_caveats(caveat: str) -> None:
    module = _load()
    blocks = [_contrast_block(cams=-3.7)]
    leaderboard = module.leaderboard_figure(
        blocks=[
            RowSetBlock(
                "Main",
                "January 2025",
                8,
                blocks[0].rows.rename({"difference": "value"}),
            )
        ]
    ).to_dict()
    contrasts = module.contrasts_figure(blocks=blocks).to_dict()

    assert caveat in _caption(spec=leaderboard)
    assert caveat in _caption(spec=contrasts)


def test_the_leaderboard_names_both_post_hoc_ukv_rebuilds_in_its_note() -> None:
    module = _load()
    rows = _contrast_block(cams=-3.7).rows.rename({"difference": "value"})

    caption = _caption(
        spec=module.leaderboard_figure(
            blocks=[RowSetBlock("Main", "January 2025", 8, rows)]
        ).to_dict()
    )

    assert "Rows marked (post hoc) were added after the first run" in caption
    assert "two ways of rebuilding UKV's hourly value from its snapshots" in caption


def test_the_contrast_figure_says_two_leads_are_unmeasured_or_unequal() -> None:
    module = _load()

    caption = _caption(spec=module.contrasts_figure(blocks=[_contrast_block(cams=-3.7)]).to_dict())

    assert "KNMI HARMONIE-AROME's lead not measured" in caption
    assert "ECMWF-IFS-HRES never shorter than ICON-EU's" in caption
    assert "ICON-DREAM-EU's 1 to 3 hours against ERA5's 1 to 12" in caption


def _intervals(*, arms: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "section": _load().ABSOLUTE_SECTION,
            "setting": "pooled",
            "treatment": arm,
            "value": value,
            "lower": value - 0.4,
            "upper": value + 0.4,
        }
        for arm, value in arms.items()
    )


def test_a_post_hoc_rebuild_is_labelled_post_hoc_on_the_leaderboard_and_others_are_not() -> None:
    module = _load()
    printed = module.PrintedRow(8.18, 7.78, 8.58, None)
    row_set = module.ROW_SETS[0]._replace(
        leaderboard_arms=(
            BlockArm("ukv_trap_global", "UKV rebuilt from its snapshots", "weather model"),
            BlockArm("icon_eu_global", "ICON-EU", "weather model"),
        )
    )

    rows = module.absolute_rows(
        frame=_intervals(arms={"ukv_trap_global": 8.18, "icon_eu_global": 8.386}),
        row_set=row_set,
        printed={
            "UKV rebuilt from its snapshots": printed,
            "ICON-EU": module.PrintedRow(8.386, 7.986, 8.786, None),
        },
    )

    labels = dict(zip(rows["arm"], rows["label"], strict=True))
    assert labels == {
        "ukv_trap_global": "UKV, snapshot mean (post hoc)",
        "icon_eu_global": "ICON-EU",
    }


def test_the_figures_carry_the_scope_caveats_added_after_review() -> None:
    module = _load()
    block = _contrast_block(cams=-3.7)
    leaderboard = module.leaderboard_figure(
        blocks=[RowSetBlock("Main", "January 2025", 8, block.rows.rename({"difference": "value"}))]
    ).to_dict()
    contrasts = module.contrasts_figure(blocks=[block]).to_dict()

    assert "not a gridded product" in _caption(spec=leaderboard)
    assert "36.9% of its hours" in _caption(spec=leaderboard)
    assert "almost entirely subsets of the main rows" in _caption(spec=contrasts)
    assert "ICON global is served up to 6 hours ahead" in _caption(spec=contrasts)
    assert "and a fitting seed" in _caption(spec=contrasts)


def _widths(*, spec: object) -> set[float]:
    """Return every numeric `width` in a Vega-Lite spec."""
    found: set[float] = set()
    if isinstance(spec, dict):
        found |= {value for key, value in spec.items() if key == "width" and isinstance(value, int)}
        for value in spec.values():
            found |= _widths(spec=value)
    elif isinstance(spec, list):
        for value in spec:
            found |= _widths(spec=value)
    return found


def _five_blocks() -> list[RowSetBlock]:
    block = _contrast_block(cams=-3.7)
    cerra = RowSetBlock("CERRA", "January 2025", 8, block.rows, block.planned_rows)
    return [block, block, block, block, cerra]


def test_the_cerra_row_set_has_a_block_label_and_both_figures_draw_five_blocks() -> None:
    module = _load()
    contrasts = module.contrasts_figure(blocks=_five_blocks()).to_dict()
    leaderboard = module.leaderboard_figure(
        blocks=[
            RowSetBlock(b.label, b.dates, b.site_hours, b.rows.rename({"difference": "value"}))
            for b in _five_blocks()
        ]
    ).to_dict()

    assert module.BLOCK_LABELS["cerra"] == "CERRA"
    assert set(module.BLOCK_LABELS) == {row_set.key for row_set in module.ROW_SETS}
    assert "CERRA: January 2025, 8 site-hours" in str(contrasts)
    assert "CERRA: January 2025, 8 site-hours" in str(leaderboard)
    assert "on each of the five row sets" in _caption(spec=leaderboard)
    assert "four row sets" not in _caption(spec=leaderboard)


def test_the_two_figures_draw_their_plots_at_the_same_width() -> None:
    module = _load()
    blocks = _five_blocks()

    contrasts = module.contrasts_figure(blocks=blocks).to_dict()
    leaderboard = module.leaderboard_figure(
        blocks=[
            RowSetBlock(b.label, b.dates, b.site_hours, b.rows.rename({"difference": "value"}))
            for b in blocks
        ]
    ).to_dict()

    assert _widths(spec=contrasts)
    assert _widths(spec=contrasts) == _widths(spec=leaderboard)


@pytest.mark.parametrize(
    "caveat",
    [
        "3-hour accumulations only",
        "lead by 0 to 3 hours",
        "00:00 UTC on 1 July 2026",
        "shorter than the main rows' window",
        "almost entirely subsets of the main rows",
        "The step width is unmatched in CERRA's contrasts against ERA5 and CAMS",
    ],
)
def test_both_figures_carry_the_cerra_row_set_caveats(caveat: str) -> None:
    module = _load()
    blocks = _five_blocks()
    leaderboard = module.leaderboard_figure(
        blocks=[
            RowSetBlock(b.label, b.dates, b.site_hours, b.rows.rename({"difference": "value"}))
            for b in blocks
        ]
    ).to_dict()
    contrasts = module.contrasts_figure(blocks=blocks).to_dict()

    assert caveat in _caption(spec=contrasts)
    if "subsets" not in caveat:
        assert caveat in _caption(spec=leaderboard)
