"""Tests for `studies/beam_diffuse_split/figure_numbers.py`."""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
MODULE_PATH: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split" / "figure_numbers.py"
ASSETS_DIR: Final[Path] = REPO_ROOT / "docs" / "studies" / "assets"
SOLAR_SVG_PREFIXES: Final[tuple[str, ...]] = ("sunshine_", "ens_past_solar_", "station_past_solar_")
WIND_SVG_PREFIXES: Final[tuple[str, ...]] = ("wind_", "ens_hres_wind_", "station_wind_")
TITLE_NUMBER: Final[re.Pattern[str]] = re.compile(r"aria-label=\"Title text 'Figure (\d+)([a-z]?):")
STALE_TITLE_SVGS: Final[frozenset[str]] = frozenset()
"""Wind SVGs still on disk with their old title number, until the redraw of the wind page. Delete a
stem from here when its SVG is redrawn; a test fails if the stem stays after the title is right."""


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("figure_numbers", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_figure_numbers_run_from_1_to_17_with_no_gap_or_duplicate() -> None:
    # Catches a renumbering that leaves a hole, or gives two figures one number.
    numbers = sorted(_load().FIGURE_NUMBERS.values())

    assert numbers == list(range(1, 18))


def test_every_solar_svg_feeds_a_figure() -> None:
    # Catches a solar-page SVG the map forgot, whose number would then be typed by hand.
    module = _load()
    stems = {
        path.stem for path in ASSETS_DIR.glob("*.svg") if path.stem.startswith(SOLAR_SVG_PREFIXES)
    }

    assert stems <= set(module.SVG_FIGURES)


def test_every_figure_key_named_by_an_svg_has_a_number() -> None:
    module = _load()
    keys = set(module.SVG_FIGURES.values())

    assert keys == set(module.FIGURE_NUMBERS)


def test_figure_numbers_follow_the_page_order_of_the_outline() -> None:
    # Catches a figure numbered against the order its section comes in on the page.
    numbers = _load().FIGURE_NUMBERS

    order = sorted(numbers, key=numbers.__getitem__)

    assert order[-8:] == [
        "weather_model_rivals",
        "neighbours",
        "own_beam",
        "ens_exploratory",
        "station_controls",
        "station_stations",
        "per_generator",
        "implied_capacity",
    ]
    assert numbers["contrasts"] == 2


def test_the_svgs_map_to_the_figures_the_outline_names() -> None:
    module = _load()
    figures = module.SVG_FIGURES

    def stems(key: str) -> set[str]:
        return {stem for stem, figure in figures.items() if figure == key}

    assert stems("leaderboard") == {"sunshine_leaderboard"}
    assert stems("contrasts") == {"sunshine_contrasts"}
    assert stems("weather_model_rivals") == {"sunshine_weather_model_rivals"}
    assert stems("per_generator") == {"station_past_solar_per_generator"}
    assert figures["sunshine_own_beam"] == "own_beam"
    assert figures["station_past_solar_controls"] == "station_controls"


def _title_label(stem: str) -> tuple[int, str] | None:
    """Return an SVG's figure number and panel letter (empty if none), or None with no title."""
    text = (ASSETS_DIR / f"{stem}.svg").read_text()
    match = TITLE_NUMBER.search(text)
    return (int(match.group(1)), match.group(2)) if match else None


def _title_number(stem: str) -> int | None:
    label = _title_label(stem)
    return None if label is None else label[0]


def test_wind_figure_numbers_run_from_1_to_15_with_no_gap_or_duplicate() -> None:
    # Catches a wind renumbering that leaves a hole, or gives two figures one number.
    numbers = sorted(_load().WIND_FIGURE_NUMBERS.values())

    assert numbers == list(range(1, 16))


def test_every_wind_svg_feeds_a_figure_or_is_listed_as_superseded() -> None:
    # Catches a wind-page SVG the map forgot, whose number would then be typed by hand.
    module = _load()
    stems = {
        path.stem for path in ASSETS_DIR.glob("*.svg") if path.stem.startswith(WIND_SVG_PREFIXES)
    }

    assert stems <= set(module.WIND_SVG_FIGURES) | module.WIND_SUPERSEDED_SVGS


def test_a_superseded_wind_svg_feeds_no_figure() -> None:
    # Catches a wind stem that is both drawn for a figure and listed for deletion.
    module = _load()

    assert not set(module.WIND_SVG_FIGURES) & module.WIND_SUPERSEDED_SVGS


def test_every_wind_figure_key_named_by_an_svg_has_a_number() -> None:
    module = _load()

    assert set(module.WIND_SVG_FIGURES.values()) == set(module.WIND_FIGURE_NUMBERS)


def test_each_wind_svg_title_carries_the_number_the_map_gives_its_figure() -> None:
    # Catches a chart whose title number was typed by hand and drifted from the map.
    module = _load()
    wrong = {}
    for stem, key in module.WIND_SVG_FIGURES.items():
        if stem in STALE_TITLE_SVGS or not (ASSETS_DIR / f"{stem}.svg").exists():
            continue
        title = _title_number(stem)
        if title is not None and title != module.WIND_FIGURE_NUMBERS[key]:
            wrong[stem] = (title, module.WIND_FIGURE_NUMBERS[key])

    assert wrong == {}


def test_a_stale_title_listing_names_only_svgs_whose_title_is_still_old() -> None:
    # Catches a redrawn SVG left in the stale list, which would hide a later drift.
    module = _load()

    for stem in STALE_TITLE_SVGS:
        key = module.WIND_SVG_FIGURES[stem]
        assert _title_number(stem) != module.WIND_FIGURE_NUMBERS[key], stem


WIND_CHART_SCRIPTS: Final[tuple[str, ...]] = (
    "wind_product_charts.py",
    "past_wind_leaderboard_charts.py",
    "wind_icon_dream_charts.py",
    "ens_hres_past_wind_charts.py",
    "station_wind_arms_charts.py",
)
WRITTEN_STEM: Final[re.Pattern[str]] = re.compile(
    r"^\s+\"((?:wind|ens_hres_wind|station_wind)_\w+)\":", re.MULTILINE
)
BLOCK_TITLE: Final[re.Pattern[str]] = re.compile(
    r"Title text '(Main|ICON-DREAM-EU|ECMWF|Station): [^']*farm-hours'"
)


def _written_stems() -> dict[str, list[str]]:
    """Return, for each wind chart script, the SVG stems its `charts` dict writes."""
    scripts_dir = MODULE_PATH.parent
    return {
        script: WRITTEN_STEM.findall((scripts_dir / script).read_text())
        for script in WIND_CHART_SCRIPTS
    }


def test_no_two_wind_chart_scripts_write_the_same_svg_stem() -> None:
    # Catches a second script overwriting a figure's SVG with an older drawing of it.
    written = _written_stems()
    owners: dict[str, list[str]] = {}
    for script, stems in written.items():
        assert stems, script
        for stem in stems:
            owners.setdefault(stem, []).append(script)

    assert {stem: scripts for stem, scripts in owners.items() if len(scripts) > 1} == {}


def test_wind_leaderboard_svg_holds_the_four_block_titles() -> None:
    # Catches the one-block leaderboard overwriting Figure 1's four-block chart.
    text = (ASSETS_DIR / "wind_leaderboard.svg").read_text()

    assert BLOCK_TITLE.findall(text) == ["Main", "ICON-DREAM-EU", "ECMWF", "Station"]


def _svgs_by_figure(module: ModuleType) -> dict[str, list[str]]:
    """Return each wind figure's SVG stems that are on disk, keyed by figure."""
    by_figure: dict[str, list[str]] = {}
    for stem, key in module.WIND_SVG_FIGURES.items():
        if (ASSETS_DIR / f"{stem}.svg").exists():
            by_figure.setdefault(key, []).append(stem)
    return by_figure


def test_a_wind_figure_drawn_by_several_svgs_letters_each_of_them() -> None:
    # Catches Figures 4, 5 and 7 being drawn by two or three SVGs under one bare number.
    module = _load()
    expected = {"models_work_timeseries": "ab", "models_work_error": "ab", "per_generator": "abc"}
    lettered = {}
    for key, stems in _svgs_by_figure(module).items():
        labels = [_title_label(stem) for stem in stems if stem not in STALE_TITLE_SVGS]
        letters = "".join(sorted(label[1] for label in labels if label is not None))
        if len(stems) > 1:
            lettered[key] = letters
        else:
            assert letters == "", (key, letters)

    assert lettered == expected


def test_a_wind_figure_drawn_by_several_svgs_is_listed_with_its_row_sets() -> None:
    # Catches a new second SVG for a figure that the letter map does not know about.
    module = _load()
    several = {key for key, stems in _svgs_by_figure(module).items() if len(stems) > 1}

    assert several == set(module.WIND_LETTERED_ROW_SETS)
    for key in several:
        stems = [stem for stem, figure in module.WIND_SVG_FIGURES.items() if figure == key]
        assert len(stems) == len(module.WIND_LETTERED_ROW_SETS[key])


def test_wind_figure_number_gives_each_row_set_its_letter_in_order() -> None:
    module = _load()

    assert module.wind_figure_number(key="per_generator", row_set="main") == "7a"
    assert module.wind_figure_number(key="per_generator", row_set="ecmwf") == "7b"
    assert module.wind_figure_number(key="per_generator", row_set="station") == "7c"
    assert module.wind_figure_number(key="models_work_timeseries", row_set="ecmwf") == "4b"
    assert module.wind_figure_number(key="models_work_error", row_set="main") == "5a"
    assert module.wind_figure_title(row_set="main", title="A finding") == "main rows - A finding"
