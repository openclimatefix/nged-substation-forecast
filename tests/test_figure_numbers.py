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
TITLE_NUMBER: Final[re.Pattern[str]] = re.compile(r"aria-label=\"Title text 'Figure (\d+):")
STALE_TITLE_SVGS: Final[frozenset[str]] = frozenset(
    {
        "ens_hres_wind_models_work",
        "ens_hres_wind_per_farm_error",
        "ens_hres_wind_robustness",
        "ens_hres_wind_reconciliation",
        "ens_hres_wind_monthly_ratio",
        "ens_hres_wind_split",
        "ens_hres_wind_by_farm",
        "station_wind_by_farm",
        "station_wind_season",
    }
)
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


def test_every_solar_svg_feeds_a_figure_or_is_listed_as_superseded() -> None:
    # Catches a solar-page SVG the map forgot, whose number would then be typed by hand.
    module = _load()
    stems = {
        path.stem for path in ASSETS_DIR.glob("*.svg") if path.stem.startswith(SOLAR_SVG_PREFIXES)
    }

    assert stems <= set(module.SVG_FIGURES) | module.SUPERSEDED_SVGS


def test_a_superseded_svg_feeds_no_figure() -> None:
    # Catches a stem that is both drawn for a figure and listed for deletion.
    module = _load()

    assert not set(module.SVG_FIGURES) & module.SUPERSEDED_SVGS


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


def _title_number(stem: str) -> int | None:
    text = (ASSETS_DIR / f"{stem}.svg").read_text()
    match = TITLE_NUMBER.search(text)
    return int(match.group(1)) if match else None


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
