"""Draw the map of which parts of the UK the ICON-D2 and AROME France domains leave out.

One-off script for the map on
<https://openclimatefix.github.io/nged-substation-forecast/roadmap/data-sources/#weather-data>.
Every other weather product that page names covers the whole UK, so the map draws only these two
limited-area models' domains over the UK, with Ireland drawn lighter for context.

The five outlines sit in `domains/` beside this script, rounded to 0.01°, with every exterior ring
wound clockwise: Vega's map projection reads a counter-clockwise ring as the whole globe minus the
polygon. Vega also joins consecutive vertices by great-circle arcs rather than along parallels, so a
rectangle's edges are densified to 0.5° steps before they are stored.

Run it with `uv run python studies/beam_diffuse_split/weather_product_domains.py`, then optimise
the SVG with `npx svgo@4 --multipass --precision=1 --final-newline` before committing it.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Final, Literal

import altair as alt

# Importing the theme module registers and enables the OCF Altair theme as a side effect.
import plotting.ocf_theme as ocf

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DomainProductType = Literal["ICON-D2", "AROME France"]
"""The limited-area weather models whose domains the map draws."""

DOMAINS_DIR: Final[Path] = Path(__file__).resolve().parent / "domains"
"""Where the five GeoJSON outlines live."""

GREAT_BRITAIN_PATH: Final[Path] = DOMAINS_DIR / "great_britain.geojson"
"""England, Scotland, and Wales, from `england_scotland_wales.geojson` in the `geo` package.

The `geo` package's file is the boundary the `h3_grid_weights` asset clips the NWP grid to. The
outline here is that boundary dissolved into one shape, simplified to a 0.02° tolerance, and
stripped of islands smaller than 0.003 square degrees.
"""

# The Northern Ireland and Ireland outlines come from Natural Earth 5.1.1's public-domain 1:10m
# Admin 0 map units (`ne_10m_admin_0_map_units`, <https://www.naturalearthdata.com>), which split
# the UK into its four countries. Each outline is clipped to 11.5°W to 4.5°E and 48.5°N to 61.5°N,
# simplified to a 0.01° tolerance, and stripped of islands smaller than 0.001 square degrees.

NORTHERN_IRELAND_PATH: Final[Path] = DOMAINS_DIR / "northern_ireland.geojson"
"""Northern Ireland, from Natural Earth's `Northern Ireland` map unit."""

IRELAND_PATH: Final[Path] = DOMAINS_DIR / "ireland.geojson"
"""Ireland, from Natural Earth's `Ireland` map unit, drawn lighter than the UK for context."""

DOMAIN_PATHS: Final[dict[DomainProductType, Path]] = {
    "ICON-D2": DOMAINS_DIR / "icon_d2.geojson",
    "AROME France": DOMAINS_DIR / "arome_france.geojson",
}
"""Each product's domain outline, and where each outline came from.

ICON-D2: the valid-data mask of one file the German weather service (DWD) publishes on
<https://opendata.dwd.de>, the regular latitude-longitude regrid of the 2026-09-23 00 UTC run's
`t_2m` at step 0. For every tenth row of the 0.02° grid, the outline takes the westernmost and
easternmost grid point carrying data. The native grid is rotated, so the domain is a tilted
quadrilateral whose western edge runs from 1.8°W at 49.9°N to 3.9°W at 57.3°N and passes 2.6°W at
53.2°N. Open-Meteo, which serves the same regrid, returns no data west of about 2.5°W at that
latitude.

AROME France: the rectangle 37.5°N to 55.4°N, 12°W to 16°E, from the grid description Open-Meteo
publishes at <https://api.open-meteo.com/data/meteofrance_arome_france0025/static/meta.json>.
Open-Meteo's forecast API returns data at 55.3°N and none at 55.5°N, which confirms the 55.4°N
northern edge.
"""

DOMAIN_COLOURS: Final[dict[DomainProductType, str]] = {
    "ICON-D2": ocf.DATA_BLUE,
    "AROME France": ocf.BRAND_ORANGE,
}
"""Two of the brand guidelines' main data colours, 32.5 ΔE apart under protanopia."""

DOMAIN_DASHES: Final[dict[DomainProductType, list[int]]] = {
    "ICON-D2": [1, 0],
    "AROME France": [6, 3],
}
"""A different edge pattern for each product, so the map still reads without colour."""

DOMAIN_LABELS: Final[tuple[tuple[DomainProductType, float, float, str], ...]] = (
    ("ICON-D2", 0.3, 56.9, "ICON-D2 covers\nthe area east of\nthe blue line"),
    ("AROME France", -8.4, 51.0, "AROME France\ncovers the area\nsouth of 55.4°N"),
)
"""Each product's direct label: the product, the label's longitude and latitude, and its text."""

WEST: Final[float] = -11.0
EAST: Final[float] = 4.0
SOUTH: Final[float] = 49.0
NORTH: Final[float] = 61.0
"""The map's extent in degrees: the whole of Great Britain, and the sea around it."""

GRATICULE_STEP_DEGREES: Final[int] = 2
"""Spacing of the latitude and longitude lines, in degrees."""

WIDTH_PX: Final[int] = 560
HEIGHT_PX: Final[int] = 720
"""The map's plot size. The extent is taller than it is wide, so the map is drawn in portrait."""

OUTPUT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "roadmap"
    / "assets"
    / "weather_product_domains.svg"
)
"""Where the data-sources page reads the map from."""


def _feature(*, path: Path, properties: dict[str, str]) -> dict[str, object]:
    """Reads one GeoJSON feature and replaces its properties.

    Args:
        path: The GeoJSON file, holding a single `Feature`.
        properties: The properties the chart encodes, such as the product's name.

    Returns:
        The feature, as the mapping Vega-Lite reads inline.
    """
    feature = json.loads(path.read_text())
    feature["properties"] = properties
    return feature


def _extent() -> dict[str, object]:
    """The map's extent as a clockwise polygon, for the projection to fit to.

    Returns:
        A GeoJSON feature whose edges are densified to one vertex per degree, so the fitted extent
        follows the parallels rather than great circles.
    """
    west_to_east = [float(lon) for lon in range(int(WEST), int(EAST) + 1)]
    north_edge = [[lon, NORTH] for lon in west_to_east]
    south_edge = [[lon, SOUTH] for lon in reversed(west_to_east)]
    ring = [*north_edge, *south_edge, north_edge[0]]
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


_EAST_OR_WEST: Final[dict[int, str]] = {-1: "W", 0: "", 1: "E"}
"""The hemisphere letter for a longitude's sign. The prime meridian takes none."""


def _graticule_labels() -> alt.LayerChart:
    """Degree labels along the map's northern and western edges.

    The longitude labels sit along the northern edge, each to the west of its meridian, because
    the ICON-D2 domain's western edge crosses the southern edge at 2°W and Shetland sits just east
    of 2°W at the northern edge.

    Returns:
        Two text layers: the longitude labels just inside the northern edge, and the latitude labels
        just inside the western edge.
    """
    longitude_rows = [
        {
            "lon": lon,
            "lat": NORTH - 0.3,
            "text": f"{abs(lon)}°{_EAST_OR_WEST[(lon > 0) - (lon < 0)]}",
        }
        for lon in range(-10, int(EAST), GRATICULE_STEP_DEGREES)
    ]
    latitude_rows = [
        {"lon": WEST + 0.35, "lat": lat, "text": f"{lat}°N"}
        for lat in range(50, int(NORTH) + 1, GRATICULE_STEP_DEGREES)
    ]
    return alt.LayerChart(
        layer=[
            _degree_labels(rows=longitude_rows, align="right", dx=-3),
            _degree_labels(rows=latitude_rows, align="left", dx=3),
        ]
    )


def _degree_labels(
    *, rows: list[dict[str, float | str]], align: Literal["left", "right"], dx: int
) -> alt.Chart:
    """One text layer of degree labels.

    Args:
        rows: One mapping per label, holding its `lon`, its `lat` and its `text`.
        align: Which side of the label its anchor point sits on.
        dx: How far to shift each label from its anchor point, in pixels.

    Returns:
        The text layer.
    """
    return (
        alt.Chart(alt.Data(values=rows))
        .mark_text(font=ocf.FONT_LABEL, fontSize=10, color=ocf.TEXT, align=align, dx=dx)
        .encode(longitude="lon:Q", latitude="lat:Q", text="text:N")  # ty: ignore[unresolved-attribute]
    )


def domain_map() -> alt.LayerChart:
    """Draws Great Britain, the ICON-D2 domain, and the AROME France domain on one map.

    Returns:
        The layered chart, projected onto a conic conformal projection fitted to the map's extent.
    """
    products = list(DOMAIN_PATHS)
    domains = [
        _feature(path=path, properties={"product": product})
        for product, path in DOMAIN_PATHS.items()
    ]
    united_kingdom = [
        _feature(path=GREAT_BRITAIN_PATH, properties={"name": "Great Britain"}),
        _feature(path=NORTHERN_IRELAND_PATH, properties={"name": "Northern Ireland"}),
    ]
    ireland = _feature(path=IRELAND_PATH, properties={"name": "Ireland"})
    colour_scale = alt.Scale(domain=products, range=[DOMAIN_COLOURS[p] for p in products])
    dash_scale = alt.Scale(domain=products, range=[DOMAIN_DASHES[p] for p in products])
    legend = alt.Legend(
        title="Domain edge", orient="top", direction="horizontal", symbolType="stroke"
    )

    graticule = alt.Chart(
        alt.graticule(
            extent=[[WEST - 1, SOUTH - 1], [EAST + 1, NORTH + 1]], step=[GRATICULE_STEP_DEGREES] * 2
        )
    ).mark_geoshape(filled=False, stroke=ocf.GREY_3, strokeWidth=0.75, clip=True)
    context_land = alt.Chart(alt.Data(values=[ireland])).mark_geoshape(
        fill=ocf.GREY_2, stroke=ocf.GREY_3, strokeWidth=0.5, clip=True
    )
    land = alt.Chart(alt.Data(values=united_kingdom)).mark_geoshape(
        fill=ocf.GREY_3, stroke=ocf.TEXT, strokeWidth=0.5, clip=True
    )
    domains_layer = (
        alt.Chart(alt.Data(values=domains))
        .mark_geoshape(fillOpacity=0.12, strokeWidth=2.5, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            fill=alt.Fill("properties.product:N", scale=colour_scale, legend=legend),
            stroke=alt.Stroke("properties.product:N", scale=colour_scale, legend=legend),
            strokeDash=alt.StrokeDash("properties.product:N", scale=dash_scale, legend=legend),
        )
    )
    labels = (
        alt.Chart(
            alt.Data(
                values=[
                    {"lon": lon, "lat": lat, "text": text.split("\n")}
                    for _, lon, lat, text in DOMAIN_LABELS
                ]
            )
        )
        .mark_text(font=ocf.FONT_TEXT, fontSize=12, color=ocf.TEXT, align="left", lineHeight=15)
        .encode(longitude="lon:Q", latitude="lat:Q", text="text:N")  # ty: ignore[unresolved-attribute]
    )
    return (
        # `graticule` and the two land layers carry no `.encode()` to hang the usual suppression
        # on, so ty's loss of the chart type after `mark_geoshape()` (astral-sh/ty#2520) surfaces
        # here.
        alt.LayerChart(
            layer=[graticule, context_land, land, domains_layer, labels, _graticule_labels()]  # ty: ignore[invalid-argument-type]
        )
        .project(
            type="conicConformal",
            parallels=[50, 60],
            rotate=[3.5, 0, 0],
            fit=_extent(),
        )
        .properties(
            width=WIDTH_PX,
            height=HEIGHT_PX,
            title=alt.Title(
                text="ICON-D2 and AROME France each leave out part of the UK",
                subtitle=[
                    "Shaded: the area each weather model's data covers.",
                    "Dark grey: the UK. Light grey: Ireland, for context.",
                ],
                anchor="start",
            ),
        )
    )


def main() -> int:
    """Writes the map to `OUTPUT_PATH`.

    Returns:
        The process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    domain_map().save(OUTPUT_PATH)
    _LOG.info("wrote %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
