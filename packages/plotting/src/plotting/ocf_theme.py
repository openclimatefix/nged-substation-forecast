"""OCF brand Altair theme.

Import this module to register and enable the OCF theme for Altair charts. The
``@alt.theme.register`` decorator fires at import time, so a bare ``import plotting.ocf_theme``
is sufficient to activate it.

The colour and typography constants below follow OCF's 2025 brand guidelines. Two values in the
guidelines are typos, handled as noted beside each: "Grey 2" is printed twice for two different
hex values, and Data Magenta is printed with the same hex as Data Amber.
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
"""Brand guidelines: Data Amber (additional data colours group), internal use only in OCF's
brand guidelines, so keep it out of anything published."""

DARK_GREEN: Final[LiteralString] = PALETTE[6]
"""Brand guidelines: Data Deep Teal (additional data colours group), internal use only in OCF's
brand guidelines, so keep it out of anything published."""

BROWN: Final[LiteralString] = PALETTE[7]
"""Brand guidelines: Data Burnt Orange (additional data colours group), internal use only in
OCF's brand guidelines, so keep it out of anything published."""

MINT: Final[LiteralString] = PALETTE[8]
"""Brand guidelines: Data Green Light (main data colours group, light shade)."""

LAVENDER: Final[LiteralString] = PALETTE[9]
"""Brand guidelines: Data Purple Light (main data colours group, light shade)."""

# Additional brand colours not currently in PALETTE. Kept out of PALETTE deliberately: adding a
# colour there changes every chart's category/ordinal/ramp range, which is out of scope here.
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

VISUALISATION_YELLOW: Final[LiteralString] = "#FFD480"
"""Brand guidelines: Yellow (visualisation colours group)."""

VISUALISATION_ORANGE: Final[LiteralString] = "#FAA056"
"""Brand guidelines: Orange (visualisation colours group)."""

VISUALISATION_TEAL_LIGHT: Final[LiteralString] = "#9ED1CD"
"""Brand guidelines: Teal Light (visualisation colours group)."""

VISUALISATION_YELLOW_LIGHT: Final[LiteralString] = "#FFE9BC"
"""Brand guidelines: Yellow Light (visualisation colours group)."""

VISUALISATION_ORANGE_LIGHT: Final[LiteralString] = "#FFDABC"
"""Brand guidelines: Orange Light (visualisation colours group)."""

DATA_BLACK: Final[LiteralString] = BLACK_1
"""Brand guidelines: Data Black (additional data colours group), internal use only in OCF's
brand guidelines, so keep it out of anything published. Same value as ``BLACK_1``.

The additional data colours group also lists Data Deep Teal (``DARK_GREEN`` above) and Data
Burnt Orange (``BROWN`` above) — both already named constants, so no separate constant is added
for either. It lists a fifth colour, Data Magenta, printed with the same hex value as Data Amber
(``#FC9700``, ``MUSTARD`` above) — evidently a typo, since two differently-named swatches on the
same page cannot share a value. No ``DATA_MAGENTA`` constant is defined here rather than guess at
the intended value.
"""

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
is the light shade of ``DATA_COLOURS[0]``. "Each has a lighter shade to create mono-coloured
comparative graphs."
"""

VISUALISATION_COLOURS: Final[tuple[LiteralString, ...]] = (
    VISUALISATION_BLUE,
    VISUALISATION_SKY_BLUE,
    VISUALISATION_TEAL,
    VISUALISATION_YELLOW,
    VISUALISATION_ORANGE,
)
"""Brand guidelines' five visualisation colours, cool to warm: a gradient from cool sky blue to
warm sunlight, for visualisations such as weather maps."""

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

TypeStyle = Literal[
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

TYPE_SCALE: Final[dict[TypeStyle, float]] = {
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
Small" and "Label" share a multiplier in the guidelines. Label and Micro are set in
``FONT_LABEL`` (the Semi Mono), every other style in ``FONT_TEXT``.
"""


def font_size(*, style: TypeStyle, body_px: float) -> int:
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
                "font": FONT_TEXT,
                "fontSize": title_px,
            },
            "axis": {
                "domainColor": _TEXT,
                "gridColor": GRID,
                "tickColor": _TEXT,
                "labelColor": _TEXT,
                "labelFont": FONT_LABEL,
                "labelFontSize": label_px,
                "titleColor": _TEXT,
                "titleFont": FONT_LABEL,
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
