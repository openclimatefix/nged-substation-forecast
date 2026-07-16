"""Tests for the OCF Altair theme helpers."""

from plotting.ocf_theme import BLUE, _ocf_theme, hex_to_rgb


def test_hex_to_rgb() -> None:
    assert hex_to_rgb(BLUE) == [0x30, 0x6B, 0xFF]
    assert hex_to_rgb("#000000") == [0, 0, 0]
    assert hex_to_rgb("FFFFFF") == [255, 255, 255]  # bare hex (no "#") also accepted


def test_legend_swatches_are_fully_opaque() -> None:
    # Guards against legend swatches washing out: in layered charts Vega-Lite derives swatch
    # opacity from the layers' marks unless the config pins it.
    assert _ocf_theme()["config"]["legend"]["symbolOpacity"] == 1
