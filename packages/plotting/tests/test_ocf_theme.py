"""Tests for the OCF Altair theme helpers."""

import re

from plotting import ocf_theme
from plotting.ocf_theme import DATA_BLUE, _ocf_theme, font_size, hex_to_rgb

_HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")

# Chart-role constants that name a brand colour rather than define one.
_SANCTIONED_DUPLICATE_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"BLACK_1", "TEXT"}),
        frozenset({"GREY_1", "BACKGROUND"}),
    }
)


def test_hex_to_rgb() -> None:
    assert hex_to_rgb(DATA_BLUE) == [0x30, 0x6B, 0xFF]
    assert hex_to_rgb("#000000") == [0, 0, 0]
    assert hex_to_rgb("FFFFFF") == [255, 255, 255]  # bare hex (no "#") also accepted


def test_legend_swatches_are_fully_opaque() -> None:
    # Guards against legend swatches washing out: in layered charts Vega-Lite derives swatch
    # opacity from the layers' marks unless the config pins it.
    assert _ocf_theme()["config"]["legend"]["symbolOpacity"] == 1


def test_data_colours_light_pairs_with_data_colours() -> None:
    assert ocf_theme.DATA_COLOURS == (
        ocf_theme.DATA_BLUE,
        ocf_theme.DATA_SKY,
        ocf_theme.BRAND_ORANGE,
        ocf_theme.DATA_PURPLE,
        ocf_theme.DATA_GREEN,
    )
    assert ocf_theme.DATA_COLOURS_LIGHT == (
        ocf_theme.DATA_BLUE_LIGHT,
        ocf_theme.DATA_SKY_LIGHT,
        ocf_theme.BRAND_ORANGE_LIGHT,
        ocf_theme.DATA_PURPLE_LIGHT,
        ocf_theme.DATA_GREEN_LIGHT,
    )


def test_no_unintended_duplicate_colour_constants() -> None:
    # Every public, uppercase, hex-valued constant is a unique colour, apart from the chart-role
    # constants that name a brand colour.
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
    assert font_size(style="Label", body_px=10) == 9  # 10 * 0.88 = 8.8 rounds up, not down


def test_theme_config_sets_brand_fonts_and_sizes() -> None:
    config = _ocf_theme()["config"]
    assert config["font"] == ocf_theme.FONT_TEXT
    assert (config["title"]["font"], config["title"]["fontSize"]) == (ocf_theme.FONT_TEXT, 18)
    for element in ("axis", "legend"):
        assert (config[element]["titleFont"], config[element]["titleFontSize"]) == (
            ocf_theme.FONT_TEXT,
            11,
        )
        assert (config[element]["labelFont"], config[element]["labelFontSize"]) == (
            ocf_theme.FONT_LABEL,
            10,
        )
