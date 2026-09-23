"""OCF brand Altair theme.

Import this module to register and enable the OCF theme for Altair charts. The
``@alt.theme.register`` decorator fires at import time, so a bare ``import plotting.ocf_theme``
is sufficient to activate it.

The colour and typography constants follow OCF's 2025 brand guidelines, and each colour is named as
the guidelines name it. Three of the guidelines' printed swatch labels disagree with the swatch's
own fill colour. Every other swatch's fill matches its label, so where the two disagree the
constant takes the fill, and its docstring records the printed label.
"""

import math
from typing import Final, Literal, LiteralString

import altair as alt

# The colour constants are typed ``LiteralString`` rather than ``str`` because Altair's
# ``ThemeConfig`` declares every colour field as ``LiteralString``, which a plain ``str`` does not
# satisfy — see ``_ocf_theme``.

# Brand colours.

BRAND_ORANGE: Final[LiteralString] = "#FF4901"
"""Brand guidelines: Brand Orange. Also the first of the main data colours."""

BRAND_ORANGE_LIGHT: Final[LiteralString] = "#FF8F73"
"""Brand guidelines: Brand Orange Light."""

BLACK_1: Final[LiteralString] = "#292B2B"
"""Brand guidelines: Black 1, which the additional data colours also list as Data Black."""

BLACK_2: Final[LiteralString] = "#0C0D0D"
"""Brand guidelines: Black 2."""

WHITE: Final[LiteralString] = "#FFFFFF"
"""Brand guidelines: White."""

GREY_1: Final[LiteralString] = "#FFFBF5"
"""Brand guidelines: Grey 1, a warm cream."""

GREY_2: Final[LiteralString] = "#F0ECE8"
"""Brand guidelines: Grey 2."""

GREY_3: Final[LiteralString] = "#D9D0CA"
"""The guidelines print this swatch as "Grey 2" a second time; this module names it Grey 3."""

# Main data colours: "electric contrasting colours for easy visual separation", each with a light
# shade for comparing two conditions in one hue.

DATA_BLUE: Final[LiteralString] = "#306BFF"
"""Brand guidelines: Data Blue."""

DATA_SKY: Final[LiteralString] = "#10C5F7"
"""Brand guidelines: Data Sky."""

DATA_PURPLE: Final[LiteralString] = "#B701FF"
"""Brand guidelines: Data Purple."""

DATA_GREEN: Final[LiteralString] = "#17E58F"
"""Brand guidelines: Data Green."""

DATA_BLUE_LIGHT: Final[LiteralString] = "#9CB6E1"
"""Brand guidelines: Blue Light, the light shade of both Data Blue and Visualisation Blue."""

DATA_SKY_LIGHT: Final[LiteralString] = "#A3D6E0"
"""Brand guidelines: Sky Blue Light, the light shade of both Data Sky and Visualisation Sky Blue."""

DATA_PURPLE_LIGHT: Final[LiteralString] = "#EFC8FF"
"""Brand guidelines: Data Purple Light."""

DATA_GREEN_LIGHT: Final[LiteralString] = "#B8F5DB"
"""Brand guidelines: Data Green Light. The colour overview page labels this swatch correctly,
but the data-colour pages label it "Data Purple Light #EFC8FF" a second time."""

DATA_COLOURS: Final[tuple[LiteralString, ...]] = (
    DATA_BLUE,
    DATA_SKY,
    BRAND_ORANGE,
    DATA_PURPLE,
    DATA_GREEN,
)
"""The five main data colours, in the guidelines' printed order."""

DATA_COLOURS_LIGHT: Final[tuple[LiteralString, ...]] = (
    DATA_BLUE_LIGHT,
    DATA_SKY_LIGHT,
    BRAND_ORANGE_LIGHT,
    DATA_PURPLE_LIGHT,
    DATA_GREEN_LIGHT,
)
"""The light shade of each ``DATA_COLOURS`` entry, at the same index."""

# Additional data colours, which the guidelines reserve for internal use. Data Black is
# ``BLACK_1``.

DATA_AMBER: Final[LiteralString] = "#FC9700"
"""Brand guidelines: Data Amber."""

DATA_DEEP_TEAL: Final[LiteralString] = "#009C75"
"""Brand guidelines: Data Deep Teal."""

DATA_MAGENTA: Final[LiteralString] = "#FF17EC"
"""Brand guidelines: Data Magenta. The guidelines label this swatch ``#FC9700``, Data Amber's
value, but fill it with ``#FF17EC``."""

DATA_BURNT_ORANGE: Final[LiteralString] = "#BF4F04"
"""Brand guidelines: Data Burnt Orange."""

ADDITIONAL_DATA_COLOURS: Final[tuple[LiteralString, ...]] = (
    DATA_BURNT_ORANGE,
    DATA_AMBER,
    DATA_MAGENTA,
    DATA_DEEP_TEAL,
)
"""The four coloured additional data colours, in the guidelines' printed order."""

# Visualisation colours: a gradient "from cool sky blue to warm sunlight", inspired by the
# troposphere.

VISUALISATION_BLUE: Final[LiteralString] = "#4675C1"
"""Brand guidelines: Blue (visualisation colours)."""

VISUALISATION_SKY_BLUE: Final[LiteralString] = "#65B0C9"
"""Brand guidelines: Sky Blue (visualisation colours)."""

VISUALISATION_TEAL: Final[LiteralString] = "#58B0A9"
"""Brand guidelines: Teal (visualisation colours)."""

VISUALISATION_YELLOW: Final[LiteralString] = "#FFD073"
"""Brand guidelines: Yellow (visualisation colours). The guidelines label this swatch
``#FFD480``, but fill it with ``#FFD073``."""

VISUALISATION_ORANGE: Final[LiteralString] = "#FAA056"
"""Brand guidelines: Orange (visualisation colours)."""

VISUALISATION_TEAL_LIGHT: Final[LiteralString] = "#9ED1CD"
"""Brand guidelines: Teal Light (visualisation colours)."""

VISUALISATION_YELLOW_LIGHT: Final[LiteralString] = "#FFE9BC"
"""Brand guidelines: Yellow Light (visualisation colours)."""

VISUALISATION_ORANGE_LIGHT: Final[LiteralString] = "#FFDABC"
"""Brand guidelines: Orange Light (visualisation colours)."""

VISUALISATION_COLOURS: Final[tuple[LiteralString, ...]] = (
    VISUALISATION_BLUE,
    VISUALISATION_SKY_BLUE,
    VISUALISATION_TEAL,
    VISUALISATION_YELLOW,
    VISUALISATION_ORANGE,
)
"""The five visualisation colours, cool to warm."""

VISUALISATION_COLOURS_LIGHT: Final[tuple[LiteralString, ...]] = (
    DATA_BLUE_LIGHT,
    DATA_SKY_LIGHT,
    VISUALISATION_TEAL_LIGHT,
    VISUALISATION_YELLOW_LIGHT,
    VISUALISATION_ORANGE_LIGHT,
)
"""The light shade of each ``VISUALISATION_COLOURS`` entry, at the same index."""

# Chart roles.

PALETTE: Final[tuple[LiteralString, ...]] = (
    *DATA_COLOURS[2:],
    *DATA_COLOURS[:2],
    *ADDITIONAL_DATA_COLOURS,
)
"""The theme's categorical colours: the main data colours, then the additional data colours.

Both groups keep the guidelines' printed order, with the main data colours rotated to start at
Brand Orange. Under the `dataviz` skill's validator, the closest neighbouring pair is 21.5 ΔE apart
under colour-vision deficiency within the main five, and 16.1 ΔE across all nine. Data Blue and Data
Purple are not neighbours, but are only about 2 ΔE apart under deuteranopia, so a chart that shows
both needs direct labels. The guidelines reserve the additional data colours, the sixth to ninth
entries, for internal use.
"""

BACKGROUND: Final[LiteralString] = GREY_1
"""Chart background colour."""

TEXT: Final[LiteralString] = BLACK_1
"""Colour of chart text, and of axis lines and ticks."""

GRID: Final[LiteralString] = "#EAEAEA"
"""Axis grid line colour."""

ENSEMBLE_LINE: Final[LiteralString] = "#808080"
"""Colour for individual ensemble-member lines.

Mid-grey stays visible at low opacity against BACKGROUND.
"""

# Typography. Matter XH, Matter Semi Mono and Pangram Sans Rounded are commercial fonts that most
# readers will not have installed, so each stack falls back to the free font OCF's own slide
# template substitutes for it (confirmed with `pdffonts` against that template): DM Sans for
# Matter XH, Roboto Mono for Matter Semi Mono. Pangram Sans Rounded has no in-house substitute on
# record, so it falls back to DM Sans too. A browser or an SVG renderer silently skips to the next
# name in the stack when a font is not installed.
FONT_TEXT: Final[LiteralString] = "Matter XH, DM Sans, sans-serif"
"""Brand guidelines' main typeface, as a CSS font-family stack Vega-Lite accepts as ``font``."""

FONT_LABEL: Final[LiteralString] = "Matter Semi Mono, Roboto Mono, monospace"
"""Brand guidelines' typeface for "labels, shorter texts, data and navigation"."""

FONT_NUMBER: Final[LiteralString] = "Pangram Sans Rounded, DM Sans, sans-serif"
"""Brand guidelines' typeface for big numbers and stats."""

TypeScaleStyleType = Literal[
    "Headline 4",
    "Headline 3",
    "Headline 2",
    "Headline 1",
    "Body Large",
    "Body",
    "Body Small",
    "Label",
    "Micro",
]
"""Brand guidelines' named type-scale styles."""

TYPE_SCALE: Final[dict[TypeScaleStyleType, float]] = {
    "Headline 4": 4.5,
    "Headline 3": 3.0,
    "Headline 2": 2.25,
    "Headline 1": 1.65,
    "Body Large": 1.25,
    "Body": 1.0,
    "Body Small": 0.88,
    "Label": 0.88,
    "Micro": 0.77,
}
"""Brand guidelines' type-size scale: each style's size as a multiple of the body size. "Body
Small" and "Label" share a multiplier in the guidelines. The guidelines set Label and Micro in
``FONT_LABEL`` (the Semi Mono), and every other style in ``FONT_TEXT``.
"""


def font_size(*, style: TypeScaleStyleType, body_px: float) -> int:
    """Pixel size for a type-scale style, given the body size.

    Args:
        style: One of the brand guidelines' named type-scale styles.
        body_px: The body text size, in pixels, that every other style is a multiple of.

    Returns:
        The style's size in pixels, rounded to the nearest whole number, halves up, as the
        guidelines themselves prescribe ("round up/down to the nearest number for ease of use").
    """
    return math.floor(TYPE_SCALE[style] * body_px + 0.5)


def hex_to_rgb(hex_color: str) -> list[int]:
    """RGB components of a ``#RRGGBB`` colour, for libraries taking ``[r, g, b]`` lists.

    The leading ``#`` is optional: ``hex_to_rgb("FFFFFF")`` and ``hex_to_rgb("#FFFFFF")`` both
    return ``[255, 255, 255]``.

    Lets non-Altair plotting libraries (e.g. lonboard, which styles map layers with RGB lists)
    use the theme palette rather than hardcoding near-miss colours.
    """
    value = hex_color.removeprefix("#")
    return [int(value[i : i + 2], 16) for i in (0, 2, 4)]


_SEQUENTIAL: Final[tuple[LiteralString, ...]] = (
    VISUALISATION_BLUE,
    VISUALISATION_TEAL,
    VISUALISATION_YELLOW,
)

_BODY_PX: Final[float] = 11
"""Vega-Lite's default axis-title font size, taken as the type scale's body size."""


@alt.theme.register("ocf", enable=True)
def _ocf_theme() -> alt.theme.ThemeConfig:
    title_px = font_size(style="Headline 1", body_px=_BODY_PX)
    body_px = font_size(style="Body", body_px=_BODY_PX)
    label_px = font_size(style="Label", body_px=_BODY_PX)
    return {
        "config": {
            "background": BACKGROUND,
            # The default for every text element the entries below do not set: subtitles, facet
            # headers, and text marks.
            "font": FONT_TEXT,
            "view": {
                "stroke": "transparent",
                "fill": BACKGROUND,
            },
            "range": {
                "category": list(PALETTE),
                # Continuous colour scales interpolate between these three visualisation colours,
                # which run cool to warm with lightness rising at every step. The five
                # visualisation colours together do not: Yellow is lighter than Orange.
                "ramp": list(_SEQUENTIAL),
                "heatmap": list(_SEQUENTIAL),
            },
            "title": {
                "color": TEXT,
                "font": FONT_TEXT,
                "fontSize": title_px,
                # Vega-Lite defaults every title to bold, which the guidelines' type specimens
                # never use.
                "fontWeight": "normal",
                "subtitleColor": TEXT,
                "subtitleFont": FONT_TEXT,
                "subtitleFontSize": body_px,
            },
            "header": {
                "labelColor": TEXT,
                "titleColor": TEXT,
            },
            "axis": {
                "domainColor": TEXT,
                "gridColor": GRID,
                "tickColor": TEXT,
                "labelColor": TEXT,
                "labelFont": FONT_LABEL,
                "labelFontSize": label_px,
                "titleColor": TEXT,
                "titleFont": FONT_TEXT,
                "titleFontSize": body_px,
                "titleFontWeight": "normal",
            },
            "legend": {
                "labelColor": TEXT,
                "labelFont": FONT_LABEL,
                "labelFontSize": label_px,
                "titleColor": TEXT,
                "titleFont": FONT_TEXT,
                "titleFontSize": body_px,
                "titleFontWeight": "normal",
                # Legend swatches must stay fully opaque whatever opacity the marks draw at.
                # In a layered chart Vega-Lite derives swatch opacity from the layers' marks
                # (washing the swatches out) and ignores a per-legend ``symbolOpacity`` — only
                # this config-level setting wins.
                "symbolOpacity": 1,
            },
        }
    }
