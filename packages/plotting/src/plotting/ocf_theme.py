"""OCF brand Altair theme.

Import this module to register and enable the OCF theme for Altair charts.
The ``@alt.theme.register`` decorator fires at import time, so a bare
``import plotting.ocf_theme`` is sufficient to activate it.
"""

from typing import Final

import altair as alt

# The colour constants below are annotated with a bare ``Final`` (rather than the usual
# ``Final[str]``) so they keep their literal types: Altair's ``ThemeConfig`` declares every
# colour field as ``LiteralString``, which a plain ``str`` does not satisfy.
PALETTE: Final = (
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

ORANGE_RED: Final = PALETTE[0]
BLUE: Final = PALETTE[1]
PURPLE: Final = PALETTE[2]
SPRING_GREEN: Final = PALETTE[3]
SKY_BLUE: Final = PALETTE[4]
MUSTARD: Final = PALETTE[5]
DARK_GREEN: Final = PALETTE[6]
BROWN: Final = PALETTE[7]
MINT: Final = PALETTE[8]
LAVENDER: Final = PALETTE[9]

BACKGROUND: Final = "#FFFBF5"
"""Chart background colour."""

GRID: Final = "#EAEAEA"
"""Axis grid line colour."""

ENSEMBLE_LINE: Final = "#808080"
"""Colour for individual ensemble member lines; mid-grey stays visible at low opacity on BACKGROUND."""

_TEXT: Final = "#292B2B"


def hex_to_rgb(hex_color: str) -> list[int]:
    """RGB components of a ``#RRGGBB`` colour, for libraries taking ``[r, g, b]`` lists.

    Lets non-Altair plotting libraries (e.g. lonboard, which styles map layers with RGB
    lists) use the theme palette rather than hardcoding near-miss colours.
    """
    value = hex_color.removeprefix("#")
    return [int(value[i : i + 2], 16) for i in (0, 2, 4)]


@alt.theme.register("ocf", enable=True)
def _ocf_theme() -> alt.theme.ThemeConfig:
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
                # Legend swatches must stay fully opaque whatever opacity the marks draw at.
                # In a layered chart Vega-Lite derives swatch opacity from the layers' marks
                # (washing the swatches out) and ignores a per-legend ``symbolOpacity`` — only
                # this config-level setting wins.
                "symbolOpacity": 1,
            },
        }
    }
