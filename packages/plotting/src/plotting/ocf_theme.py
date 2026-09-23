"""OCF brand Altair theme.

Import this module to register and enable the OCF theme for Altair charts. The
``@alt.theme.register`` decorator fires at import time, so a bare ``import plotting.ocf_theme``
is sufficient to activate it.

The colour and typography constants below follow OCF's 2025 brand guidelines. Some of the
guidelines' printed swatch labels are typos. Where a label is evidently wrong, the constant takes
the swatch's own fill colour, and each constant's docstring notes the discrepancy.
"""

from typing import Final, Literal, LiteralString

import altair as alt

# The colour constants below are typed ``LiteralString`` rather than ``str`` because Altair's
# ``ThemeConfig`` declares every colour field as ``LiteralString``, which a plain ``str`` does
# not satisfy — see ``_ocf_theme``.
PALETTE: Final[tuple[LiteralString, ...]] = (
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

ORANGE_RED: Final[LiteralString] = PALETTE[0]
"""Brand guidelines: Brand Orange (main data colours group)."""

BLUE: Final[LiteralString] = PALETTE[1]
"""Brand guidelines: Data Blue (main data colours group)."""

PURPLE: Final[LiteralString] = PALETTE[2]
"""Brand guidelines: Data Purple (main data colours group)."""

SPRING_GREEN: Final[LiteralString] = PALETTE[3]
"""Brand guidelines: Data Green (main data colours group)."""

SKY_BLUE: Final[LiteralString] = PALETTE[4]
"""Brand guidelines: Data Sky (main data colours group)."""

MUSTARD: Final[LiteralString] = PALETTE[5]
"""Brand guidelines: Data Amber (additional data colours group), which the guidelines reserve
for internal use. ``PALETTE`` still includes it, so a chart with six or more categories uses it."""

DARK_GREEN: Final[LiteralString] = PALETTE[6]
"""Brand guidelines: Data Deep Teal (additional data colours group), which the guidelines reserve
for internal use. ``PALETTE`` still includes it, so a chart with seven or more categories uses
it."""

BROWN: Final[LiteralString] = PALETTE[7]
"""Brand guidelines: Data Burnt Orange (additional data colours group), which the guidelines
reserve for internal use. ``PALETTE`` still includes it, so a chart with eight or more categories
uses it."""

MINT: Final[LiteralString] = PALETTE[8]
"""Brand guidelines: Data Green Light (main data colours group, light shade)."""

LAVENDER: Final[LiteralString] = PALETTE[9]
"""Brand guidelines: Data Purple Light (main data colours group, light shade)."""

# Brand colours outside PALETTE. Adding a colour to PALETTE would change the category, ordinal
# and ramp range of every chart.
ORANGE_RED_LIGHT: Final[LiteralString] = "#FF8F73"
"""Brand guidelines: Brand Orange Light (brand colours group)."""

BLACK_1: Final[LiteralString] = "#292B2B"
"""Brand guidelines: Black 1 (brand colours group). Same value as ``DATA_BLACK`` below, which the
guidelines also print at ``#292B2B``."""

BLACK_2: Final[LiteralString] = "#0C0D0D"
"""Brand guidelines: Black 2 (brand colours group)."""

WHITE: Final[LiteralString] = "#FFFFFF"
"""Brand guidelines: White (brand colours group)."""

GREY_1: Final[LiteralString] = "#FFFBF5"
"""Brand guidelines: Grey 1 (brand colours group). Same value as ``BACKGROUND`` below, which
keeps its own name for its existing callers."""

GREY_2: Final[LiteralString] = "#F0ECE8"
"""Brand guidelines: Grey 2 (brand colours group)."""

GREY_3: Final[LiteralString] = "#D9D0CA"
"""Brand guidelines: printed as "Grey 2" a second time, which is a typo in the guidelines PDF —
called Grey 3 here to keep every constant name unique (brand colours group)."""

BLUE_LIGHT: Final[LiteralString] = "#9CB6E1"
"""Brand guidelines: Blue Light. Listed under both the visualisation and the main data colours
groups with the same value, so it is defined once here."""

SKY_BLUE_LIGHT: Final[LiteralString] = "#A3D6E0"
"""Brand guidelines: Sky Blue Light. Listed under both the visualisation and the main data
colours groups with the same value, so it is defined once here."""

VISUALISATION_BLUE: Final[LiteralString] = "#4675C1"
"""Brand guidelines: Blue (visualisation colours group)."""

VISUALISATION_SKY_BLUE: Final[LiteralString] = "#65B0C9"
"""Brand guidelines: Sky Blue (visualisation colours group)."""

VISUALISATION_TEAL: Final[LiteralString] = "#58B0A9"
"""Brand guidelines: Teal (visualisation colours group)."""

VISUALISATION_YELLOW: Final[LiteralString] = "#FFD073"
"""Brand guidelines: Yellow (visualisation colours group). The guidelines label the swatch
``#FFD480`` but fill it with ``#FFD073``; this constant follows the fill."""

VISUALISATION_ORANGE: Final[LiteralString] = "#FAA056"
"""Brand guidelines: Orange (visualisation colours group)."""

VISUALISATION_TEAL_LIGHT: Final[LiteralString] = "#9ED1CD"
"""Brand guidelines: Teal Light (visualisation colours group)."""

VISUALISATION_YELLOW_LIGHT: Final[LiteralString] = "#FFE9BC"
"""Brand guidelines: Yellow Light (visualisation colours group)."""

VISUALISATION_ORANGE_LIGHT: Final[LiteralString] = "#FFDABC"
"""Brand guidelines: Orange Light (visualisation colours group)."""

DATA_BLACK: Final[LiteralString] = BLACK_1
"""Brand guidelines: Data Black (additional data colours group). Same value as ``BLACK_1``."""

DATA_MAGENTA: Final[LiteralString] = "#FF17EC"
"""Brand guidelines: Data Magenta (additional data colours group), which the guidelines reserve
for internal use. The guidelines label the swatch ``#FC9700``, Data Amber's value, but fill it
with ``#FF17EC``; this constant follows the fill."""

DATA_COLOURS: Final[tuple[LiteralString, ...]] = (
    BLUE,
    SKY_BLUE,
    ORANGE_RED,
    PURPLE,
    SPRING_GREEN,
)
"""Brand guidelines' five main data colours, in their printed order: Data Blue, Data Sky, Brand
Orange, Data Purple, Data Green. "Electric contrasting colours for easy visual separation."
"""

DATA_COLOURS_LIGHT: Final[tuple[LiteralString, ...]] = (
    BLUE_LIGHT,
    SKY_BLUE_LIGHT,
    ORANGE_RED_LIGHT,
    LAVENDER,
    MINT,
)
"""Light shade of each ``DATA_COLOURS`` entry at the same index, e.g. ``DATA_COLOURS_LIGHT[0]``
is the light shade of ``DATA_COLOURS[0]``, for comparing two conditions in one hue. The
guidelines label Data Green Light's mint swatch "Data Purple Light #EFC8FF" a second time, but
fill it with ``#B8F5DB`` (``MINT``), which this tuple follows.
"""

VISUALISATION_COLOURS: Final[tuple[LiteralString, ...]] = (
    VISUALISATION_BLUE,
    VISUALISATION_SKY_BLUE,
    VISUALISATION_TEAL,
    VISUALISATION_YELLOW,
    VISUALISATION_ORANGE,
)
"""Brand guidelines' five visualisation colours, cool to warm: a gradient from cool sky blue to
warm sunlight."""

BACKGROUND: Final[LiteralString] = GREY_1
"""Chart background colour. Same value as the brand guidelines' Grey 1 (``GREY_1``)."""

GRID: Final[LiteralString] = "#EAEAEA"
"""Axis grid line colour."""

ENSEMBLE_LINE: Final[LiteralString] = "#808080"
"""Colour for individual ensemble-member lines.

Mid-grey stays visible at low opacity against BACKGROUND.
"""

_TEXT: Final[LiteralString] = BLACK_1

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
        The style's size in pixels, rounded to the nearest whole number as the guidelines
        themselves prescribe ("round up/down to the nearest number for ease of use").
    """
    return round(TYPE_SCALE[style] * body_px)


def hex_to_rgb(hex_color: str) -> list[int]:
    """RGB components of a ``#RRGGBB`` colour, for libraries taking ``[r, g, b]`` lists.

    The leading ``#`` is optional: ``hex_to_rgb("FFFFFF")`` and ``hex_to_rgb("#FFFFFF")`` both
    return ``[255, 255, 255]``.

    Lets non-Altair plotting libraries (e.g. lonboard, which styles map layers with RGB lists)
    use the theme palette rather than hardcoding near-miss colours.
    """
    value = hex_color.removeprefix("#")
    return [int(value[i : i + 2], 16) for i in (0, 2, 4)]


_BODY_PX: Final[float] = 11
"""Vega-Lite's default axis-title font size, taken as the type scale's body size."""


@alt.theme.register("ocf", enable=True)
def _ocf_theme() -> alt.theme.ThemeConfig:
    palette = list(PALETTE)
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
                "category": palette,
                "ordinal": palette,
                "ramp": palette,
            },
            "title": {
                "color": _TEXT,
                "font": FONT_TEXT,
                "fontSize": title_px,
                # The guidelines set headlines in the regular weight, where Vega-Lite defaults
                # to bold.
                "fontWeight": "normal",
            },
            "axis": {
                "domainColor": _TEXT,
                "gridColor": GRID,
                "tickColor": _TEXT,
                "labelColor": _TEXT,
                "labelFont": FONT_LABEL,
                "labelFontSize": label_px,
                "titleColor": _TEXT,
                "titleFont": FONT_TEXT,
                "titleFontSize": body_px,
            },
            "legend": {
                "labelColor": _TEXT,
                "labelFont": FONT_LABEL,
                "labelFontSize": label_px,
                "titleColor": _TEXT,
                "titleFont": FONT_TEXT,
                "titleFontSize": body_px,
                # Legend swatches must stay fully opaque whatever opacity the marks draw at.
                # In a layered chart Vega-Lite derives swatch opacity from the layers' marks
                # (washing the swatches out) and ignores a per-legend ``symbolOpacity`` — only
                # this config-level setting wins.
                "symbolOpacity": 1,
            },
        }
    }
