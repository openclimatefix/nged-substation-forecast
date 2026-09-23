"""Tests for the OCF Altair theme helpers."""

import re

from plotting import ocf_theme
from plotting.ocf_theme import BLUE, _ocf_theme, font_size, hex_to_rgb

_HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")

# Pairs of differently-named public constants the brand guidelines themselves print with the
# same hex value: Black 1/Data Black, and Grey 1/BACKGROUND (BACKGROUND predates the brand
# constants and keeps its own name for its existing callers).
_SANCTIONED_DUPLICATE_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"BLACK_1", "DATA_BLACK"}),
        frozenset({"GREY_1", "BACKGROUND"}),
    }
)


def test_hex_to_rgb() -> None:
    assert hex_to_rgb(BLUE) == [0x30, 0x6B, 0xFF]
    assert hex_to_rgb("#000000") == [0, 0, 0]
    assert hex_to_rgb("FFFFFF") == [255, 255, 255]  # bare hex (no "#") also accepted


def test_legend_swatches_are_fully_opaque() -> None:
    # Guards against legend swatches washing out: in layered charts Vega-Lite derives swatch
    # opacity from the layers' marks unless the config pins it.
    assert _ocf_theme()["config"]["legend"]["symbolOpacity"] == 1


def test_data_colours_light_pairs_with_data_colours() -> None:
    assert ocf_theme.DATA_COLOURS_LIGHT[0] == ocf_theme.BLUE_LIGHT  # Data Blue's light shade
    assert (
        ocf_theme.DATA_COLOURS_LIGHT[2] == ocf_theme.ORANGE_RED_LIGHT
    )  # Brand Orange's light shade


def test_no_unintended_duplicate_colour_constants() -> None:
    # Every public, uppercase, hex-valued constant should be a unique colour unless the brand
    # guidelines themselves print two names for one value.
    by_value: dict[str, list[str]] = {}
    for name, value in vars(ocf_theme).items():
        if (
            not name.startswith("_")
            and name.isupper()
            and isinstance(value, str)
            and _HEX_PATTERN.match(value)
        ):
            by_value.setdefault(value, []).append(name)
    for value, names in by_value.items():
        if len(names) > 1:
            assert frozenset(names) in _SANCTIONED_DUPLICATE_PAIRS, (value, names)


def test_font_size() -> None:
    assert font_size(style="Body", body_px=11) == 11
    assert font_size(style="Headline 1", body_px=11) == 18  # 11 * 1.65 = 18.15, rounds to 18


def test_theme_config_carries_font_stacks_where_described() -> None:
    config = _ocf_theme()["config"]
    assert config["title"]["font"] == ocf_theme.FONT_TEXT
    assert config["legend"]["titleFont"] == ocf_theme.FONT_TEXT
    assert config["axis"]["labelFont"] == ocf_theme.FONT_LABEL
    assert config["axis"]["titleFont"] == ocf_theme.FONT_LABEL
    assert config["legend"]["labelFont"] == ocf_theme.FONT_LABEL
    # FONT_NUMBER (big numbers and stats) is defined but not applied to any theme field.
    assert ocf_theme.FONT_NUMBER == "Pangram Sans Rounded, DM Sans, sans-serif"
