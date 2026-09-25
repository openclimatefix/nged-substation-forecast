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


def _all_shares() -> dict:
    module = _load()
    return {
        "main": module.MonthShares(uncovered=16.4, one_year_only=0.0),
        "icon_dream_eu": module.MonthShares(uncovered=25.1, one_year_only=0.0),
        "ecmwf": module.MonthShares(uncovered=0.0, one_year_only=9.4),
        "station": module.MonthShares(uncovered=0.0, one_year_only=42.2),
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


def _intervals(*, rows: list[dict]) -> pl.DataFrame:
    """Return an `intervals.parquet`-shaped frame holding `rows`, each a dict of its columns."""
    schema = {
        "row_set": pl.String,
        "section": pl.String,
        "setting": pl.String,
        "treatment": pl.String,
        "reference": pl.String,
        "value": pl.Float64,
        "lower": pl.Float64,
        "upper": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _row(
    *, setting: str, values: tuple[float, float, float], treatment: str = "icon_eu_wind"
) -> dict:
    value, lower, upper = values
    return {
        "row_set": "main",
        "section": "Planned contrasts",
        "setting": setting,
        "treatment": treatment,
        "reference": "ukv_wind",
        "value": value,
        "lower": lower,
        "upper": upper,
    }


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
    # Catches a block drawn without the share of its rows whose month its fold never trains on.
    module = _load()
    shares = {**module.UNCOVERED_MONTH_SHARES, "ecmwf": None}

    with pytest.raises(ValueError, match="ECMWF rows"):
        module.uncovered_month_note(shares=shares)


def test_the_caption_states_each_blocks_uncovered_month_share() -> None:
    module = _load()
    expected = {
        "main": (16.4, 0.0),
        "icon_dream_eu": (25.1, 0.0),
        "ecmwf": (0.0, 9.4),
        "station": (0.0, 42.2),
    }
    shares = module.UNCOVERED_MONTH_SHARES
    assert {
        key: (share.uncovered, share.one_year_only) for key, share in shares.items()
    } == expected

    lines = module.uncovered_month_note(shares=shares)

    assert [line.split(":")[0] for line in lines] == ["Main", "ICON-DREAM-EU", "ECMWF", "Station"]
    assert "16.4% of scored rows are in a calendar month, seen in two or more years" in lines[0]
    assert "42.2% are in a calendar month seen in one year only" in lines[3]
    assert "+0.009 and -0.028 points at the primary setting" in lines[1]
    assert (
        "absolute errors under covering folds are expected to be slightly lower, by analogy with "
        "the past-solar study; not measured for this block"
    ) in lines[1]
    assert "+0.009" not in lines[0]


def test_month_names_are_cut_to_three_letters_in_a_block_title() -> None:
    # Catches a block title so long that the panel clips it ("50,041 farm-hour").
    module = _load()

    assert module.short_months(text="August 2024 to September 2026") == "Aug 2024 to Sep 2026"


def test_the_contrast_figure_names_the_arm_the_station_block_is_against() -> None:
    # Catches the station block's zero rule reading "same as ERA5" when it is ERA5's 10 m wind.
    module = _load()
    blocks = [_block(reference_name="ERA5"), _block(reference_name="ERA5's 10 m wind")]

    figure = module.contrasts_figure(
        blocks=blocks, shares=_all_shares(), intervals=_intervals(rows=[])
    ).to_dict()
    spec = json.dumps(figure, ensure_ascii=False)

    assert "same as ERA5's 10\u00a0m wind" in spec
    assert '"same as ERA5"' in spec
    assert "ICON-D2 against UKV" in spec


def _leaderboard_block() -> RowSetBlock:
    rows = pl.DataFrame(
        {
            "arm": ["icon_d2_wind", "ukv_wind"],
            "label": ["ICON-D2", "UKV"],
            "family": ["weather model", "weather model"],
            "reference": [False, False],
            "planned": [False, False],
            "value": [8.0, 8.4],
            "lower_95": [7.0, 7.4],
            "upper_95": [9.0, 9.4],
        }
    )
    return RowSetBlock("Main", "August 2024", 8, rows, hours_unit="farm-hours")


def _title_and_cross_reference(*, figure: object) -> tuple[str, str]:
    """Return a figure's title text and the whole spec, as JSON."""
    spec = json.dumps(figure.to_dict(), ensure_ascii=False)  # ty: ignore[unresolved-attribute]
    title = spec[spec.index("Figure ") :].split(":")[0]
    return title, spec


def test_each_figure_carries_its_own_number_from_the_wind_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches a title number typed by hand, and a leaderboard figure that takes the contrasts
    # figure's number: the two figures' numbers are asserted apart, under the real map and under
    # a map with both numbers changed.
    module = _load()
    leaderboard = module.leaderboard_figure(blocks=[_leaderboard_block()], shares=_all_shares())
    contrasts = module.contrasts_figure(
        blocks=[_block(reference_name="ERA5")], shares=_all_shares(), intervals=_intervals(rows=[])
    )

    leaderboard_title, leaderboard_spec = _title_and_cross_reference(figure=leaderboard)
    contrasts_title, _ = _title_and_cross_reference(figure=contrasts)

    assert leaderboard_title == "Figure 1"
    assert contrasts_title == "Figure 2"
    assert "Figure 2's paired contrasts" in leaderboard_spec
    monkeypatch.setitem(module.WIND_FIGURE_NUMBERS, "leaderboard", 7)
    monkeypatch.setitem(module.WIND_FIGURE_NUMBERS, "contrasts", 9)
    moved_leaderboard = module.leaderboard_figure(
        blocks=[_leaderboard_block()], shares=_all_shares()
    )
    moved_contrasts = module.contrasts_figure(
        blocks=[_block(reference_name="ERA5")], shares=_all_shares(), intervals=_intervals(rows=[])
    )
    moved_title, moved_spec = _title_and_cross_reference(figure=moved_leaderboard)
    assert moved_title == "Figure 7"
    assert "Figure 9's paired contrasts" in moved_spec
    assert _title_and_cross_reference(figure=moved_contrasts)[0] == "Figure 9"


def test_no_block_title_carries_a_wind_height_or_runs_past_55_characters() -> None:
    # Catches the hub heights moved back into the block titles, which then wrap or overflow.
    module = _load()
    longest_dates = module.short_months(text="September 2026 to September 2026")

    for key, label in module.BLOCK_LABELS.items():
        setting = module.BLOCK_SETTINGS[key]
        block = RowSetBlock(
            label,
            longest_dates,
            999_999,
            pl.DataFrame(),
            hours_unit="farm-hours",
            reference_name=setting.reference_name,
        )
        assert len(block.title) < 56, block.title
        assert " m" not in block.title, block.title


def test_the_captions_state_each_blocks_wind_heights_and_the_dream_caveat() -> None:
    module = _load()

    notes = module.block_notes()

    assert notes[0] == "Wind heights of each block's arms:"
    assert "Main: 100 m; ICON 80 m." in notes
    assert "Station: 10 m station and ERA5 arm; 100 m others." in notes
    assert notes[-1].startswith("ICON-DREAM-EU: planned contrasts were written after")
    assert sum("planned contrasts were written" in note for note in notes) == 1


def test_a_contrast_that_loses_significance_at_the_second_setting_is_named_in_the_caption() -> None:
    # Catches a caption that omits a contrast whose interval crosses zero only at the second
    # setting, and one that names contrasts whose significance does not change.
    module = _load()
    intervals = _intervals(
        rows=[
            _row(setting="pooled", values=(0.126, 0.009, 0.232)),
            _row(setting="sensitivity", values=(0.082, -0.022, 0.178)),
            _row(setting="pooled", treatment="icon_d2_wind", values=(-0.3, -0.4, -0.2)),
            _row(setting="sensitivity", treatment="icon_d2_wind", values=(-0.2, -0.3, -0.1)),
        ]
    )

    notes = module.significance_change_notes(intervals=intervals)

    assert notes == [
        (
            "Main: ICON-EU minus UKV is +0.126 points [+0.009, +0.232] at the primary setting "
            "and +0.082 [-0.022, +0.178] at the second setting, so the contrast is statistically "
            "significant at the 5% level at the primary setting and not at the second setting."
        )
    ]


def test_a_contrast_that_gains_significance_at_the_second_setting_is_named_the_other_way() -> None:
    module = _load()
    intervals = _intervals(
        rows=[
            _row(setting="pooled", values=(0.08, -0.02, 0.18)),
            _row(setting="sensitivity", values=(0.13, 0.01, 0.23)),
        ]
    )

    (note,) = module.significance_change_notes(intervals=intervals)

    assert note.endswith(
        "not statistically significant at the 5% level at the primary setting and significant "
        "at the second setting."
    )


def test_the_contrast_figure_carries_the_significance_change_line() -> None:
    module = _load()
    intervals = _intervals(
        rows=[
            _row(setting="pooled", values=(0.126, 0.009, 0.232)),
            _row(setting="sensitivity", values=(0.082, -0.022, 0.178)),
        ]
    )

    figure = module.contrasts_figure(
        blocks=[_block(reference_name="ERA5")], shares=_all_shares(), intervals=intervals
    )

    assert "ICON-EU minus UKV is +0.126 points" in json.dumps(figure.to_dict())
