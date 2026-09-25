"""Tests for `studies/beam_diffuse_split/figure_numbers.py`."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
MODULE_PATH: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split" / "figure_numbers.py"
ASSETS_DIR: Final[Path] = REPO_ROOT / "docs" / "studies" / "assets"
SOLAR_SVG_PREFIXES: Final[tuple[str, ...]] = ("sunshine_", "ens_past_solar_", "station_past_solar_")


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
