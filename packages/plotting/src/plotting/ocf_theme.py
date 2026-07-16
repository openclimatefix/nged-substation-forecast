"""OCF brand Altair theme.

Import this module to register and enable the OCF theme for Altair charts.
The ``@alt.theme.register`` decorator fires at import time, so a bare
``import plotting.ocf_theme`` is sufficient to activate it.
"""

from typing import Any, Final

import altair as alt

PALETTE: Final[tuple[str, ...]] = (
    "#FF4901",  # Orange-Red
    "#306BFF",  # Blue
    "#B701FF",  # Purple
    "#17E58F",  # Spring Green
    "#10C5F7",  # Sky Blue
    "#FC9700",  # Mustard
    "#009C75",  # Dark green
    "#BF4F04",  # Brown
    "#B8F5DB",  # Mint
    "#EFC8FF",  # Lavender
)
"""OCF brand colour palette, ordered by visual priority."""

ORANGE_RED: Final[str] = PALETTE[0]
BLUE: Final[str] = PALETTE[1]
PURPLE: Final[str] = PALETTE[2]
SPRING_GREEN: Final[str] = PALETTE[3]
SKY_BLUE: Final[str] = PALETTE[4]
MUSTARD: Final[str] = PALETTE[5]
DARK_GREEN: Final[str] = PALETTE[6]
BROWN: Final[str] = PALETTE[7]
MINT: Final[str] = PALETTE[8]
LAVENDER: Final[str] = PALETTE[9]

BACKGROUND: Final[str] = "#FFFBF5"
"""Chart background colour."""

GRID: Final[str] = "#EAEAEA"
"""Axis grid line colour."""

ENSEMBLE_LINE: Final[str] = "#808080"
"""Colour for individual ensemble member lines; mid-grey stays visible at low opacity on BACKGROUND."""

_TEXT: Final[str] = "#292B2B"


def hex_to_rgb(hex_color: str) -> list[int]:
    """RGB components of a ``#RRGGBB`` colour, for libraries taking ``[r, g, b]`` lists.

    Lets non-Altair plotting libraries (e.g. lonboard, which styles map layers with RGB
    lists) use the theme palette rather than hardcoding near-miss colours.
    """
    value = hex_color.removeprefix("#")
    return [int(value[i : i + 2], 16) for i in (0, 2, 4)]


@alt.theme.register("ocf", enable=True)
def _ocf_theme() -> dict[str, Any]:
    palette = list(PALETTE)
    return {
        "config": {
            "background": BACKGROUND,
            "view": {
                "stroke": "transparent",
                "fill": BACKGROUND,
            },
            "range": {
                "category": palette,
                "ordinal": palette,
                "ramp": palette,
            },
            "axis": {
                "domainColor": _TEXT,
                "gridColor": GRID,
                "tickColor": _TEXT,
                "labelColor": _TEXT,
                "titleColor": _TEXT,
            },
            "legend": {
                "labelColor": _TEXT,
                "titleColor": _TEXT,
            },
        }
    }
