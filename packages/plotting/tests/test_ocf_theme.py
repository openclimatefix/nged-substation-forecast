"""Tests for the OCF Altair theme helpers."""

from plotting.ocf_theme import BLUE, hex_to_rgb


def test_hex_to_rgb() -> None:
    assert hex_to_rgb(BLUE) == [0x30, 0x6B, 0xFF]
    assert hex_to_rgb("#000000") == [0, 0, 0]
    assert hex_to_rgb("FFFFFF") == [255, 255, 255]  # bare hex (no "#") also accepted
