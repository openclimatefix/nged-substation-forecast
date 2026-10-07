"""Put every published capacity figure for each solar BMU side by side, one row per BMU.

A BMU is a Balancing Mechanism Unit. Each figure answers a different question: the generation
capacity the lead party declared, the installed capacity in the Installed Generation Capacity per
Unit (IGCPU) report, the project's Transmission Entry Capacity (TEC), the largest Maximum Export
Limit, and the installed capacity in the Renewable Energy Planning Database (REPD). A sixth column,
`p99_output_mw`, is the 99th percentile of the BMU's own half-hourly output. The columns are never
merged or added to each other. Run after `fetch_sources.py` and `classify.py`:
`uv run python studies/solar_bmu_census/collate.py`. The table is written as `solar_bmus.csv` and
`solar_bmus.parquet` in the study's data folder, beside `classes.parquet`.
"""

import difflib
import math
import re
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, Literal, TypedDict

import polars as pl
from classify import empty_output, p99_output_mw
from fetch_sources import (
    OUTPUT_DIR,
    STUDY_DIR,
    fetch_bmu_reference,
    fetch_dno_areas,
    fetch_igcpu,
    fetch_mels,
    fetch_repd,
    fetch_tec,
    recorded_run,
)
from pyproj import Transformer

MATCH_THRESHOLD: Final[float] = 0.85
"""The lowest name-similarity ratio at which a site name counts as a match.

At 0.8 a REPD row for a different farm with a one-letter-different name matched; at 0.85 that REPD
row did not match.
"""
MIN_SUBSTRING_NOISE_LENGTH: Final[int] = 4
NOISE_WORDS: Final[frozenset[str]] = frozenset(
    {"solar", "farm", "pv", "park", "project", "power", "ltd", "limited", "array", "energy"}
    | {"photovoltaic", "photovoltaics", "plant", "station", "extension", "bess", "battery"}
    | {"storage"}
)
REPD_BUILT_STATUSES: Final[tuple[str, ...]] = ("Operational", "Under Construction")
"""REPD statuses that count as a site that exists.

A site can be generating while REPD still lists it as under construction, so a solar site that is
operational or under construction is a candidate match. A battery at the site is found through
REPD's `Storage Co-location REPD Ref ID` column, not by name, and only a battery with one of these
statuses counts.
"""
STORAGE_OUTPUT_MWH: Final[float] = 0.1
"""A storage BMU has output in the window if any half-hour reaches this many megawatt-hours.

The output can be in either direction, charging or discharging.
"""
REVIEWED_MATCHES_PATH: Final[Path] = Path(__file__).parent / "site_matches_reviewed.csv"
"""The hand-reviewed table, kept beside this script, of BMU to TEC project and REPD reference.

Columns `elexon_bmu_id`, `tec_project_id`, `repd_ref_id`, `storage_bmu_ids` (any may be blank), and
`note`. The table has a row for each BMU whose Elexon name resembles the site's name in neither the
TEC register nor REPD. A non-blank `tec_project_id` replaces that BMU's TEC name match, and a
non-blank `repd_ref_id` replaces its REPD name match. `storage_bmu_ids` names the separately
registered storage BMUs at the site.
"""
TEC_STATUS_ORDER: Final[tuple[str, ...]] = (
    "Built",
    "Under Construction/Commissioning",
    "Consents Approved",
    "Awaiting Consents",
    "Scoping",
)
"""TEC project statuses from the most to the least advanced.

`best_tec_rows` keeps each project's row at the status earliest in this tuple, and a status not
listed ranks last.
"""
PV_PLANT_TYPE: Final[str] = "PV Array"
NON_GENERATING_PLANT_TYPES: Final[tuple[str, ...]] = ("Demand", "Reactive Compensation")
"""TEC plant types that do not make a PV site a hybrid: its own demand and reactive plant."""
CAPACITY_COLUMNS: Final[tuple[str, ...]] = (
    "generation_capacity_mw",
    "igcpu_installed_capacity_mw",
    "tec_mw",
    "largest_mel_mw",
    "repd_installed_capacity_mw",
)
P99_COLUMN: Final[str] = "p99_output_mw"
"""The 99th percentile of the BMU's own half-hourly output, in MW (`classify.p99_output_mw`).

This column measures observed output. It is not a registered capacity, so it is not in
`CAPACITY_COLUMNS` and is never added to another column.
"""
DISPARITY_FIGURES: Final[tuple[str, ...]] = (*CAPACITY_COLUMNS, P99_COLUMN)
"""The six figures whose spread `capacity_disparity` ranks."""
GSP_GROUP_AREAS: Final[dict[str, tuple[str, str]]] = {
    "_A": ("East England", "UKPN"),
    "_B": ("East Midlands", "NGED"),
    "_C": ("London", "UKPN"),
    "_D": ("North Wales, Merseyside and Cheshire", "SP Energy Networks"),
    "_E": ("West Midlands", "NGED"),
    "_F": ("North East England", "Northern Powergrid"),
    "_G": ("North West England", "Electricity North West"),
    "_H": ("Southern England", "SSEN"),
    "_J": ("South East England", "UKPN"),
    "_K": ("South Wales", "NGED"),
    "_L": ("South West England", "NGED"),
    "_M": ("Yorkshire", "Northern Powergrid"),
    "_N": ("South and Central Scotland", "SP Energy Networks"),
    "_P": ("North Scotland", "SSEN"),
}
"""Each Elexon grid supply point (GSP) group identifier mapped to its area name and the
distribution network operator (DNO) that holds the licence for that area.

The BMU register names its GSP group by identifier, and spells some groups' names more than one way,
so the lookup is by identifier. Elexon defines a GSP group as the part of one distributor's network
that a set of grid supply points feeds. The mapping is the attribute table of NESO's map of the 14
DNO licence areas (`fetch_sources.NESO_DNO_AREAS_GEOJSON`), which labels each area with its GSP
group identifier, and the area names are NESO's. UKPN is UK Power Networks, NGED is National Grid
Electricity Distribution, and SSEN is Scottish and Southern Electricity Networks.
"""
TechnologyType = Literal["pure PV", "hybrid", "unknown"]
StorageEvidenceType = Literal[
    "storage BMU with output",
    "operational battery in REPD",
    "storage planned or under construction",
    "TEC plant type lists PV only",
    "REPD solar row, no battery row",
    "none",
]


def normalise(name: str) -> str:
    """Lower-case a site name and drop punctuation, spaces, and generic words.

    Whole generic words are dropped first. Generic words of at least `MIN_SUBSTRING_NOISE_LENGTH`
    letters are then also dropped inside a longer token, so "Energyfarm" and "Energy Farm"
    normalise alike.
    """
    words = [word for word in re.findall(r"[a-z0-9]+", name.lower()) if word not in NOISE_WORDS]
    joined = "".join(words)
    for noise in sorted(NOISE_WORDS, key=len, reverse=True):
        if len(noise) >= MIN_SUBSTRING_NOISE_LENGTH:
            joined = joined.replace(noise, "")
    return joined


def best_match(*, site_name: str, candidates: dict[str, str]) -> tuple[str, float] | None:
    """Return the key and similarity of the candidate name closest to `site_name`, or None.

    Args:
        site_name: The BMU's site name.
        candidates: Maps a key (a project identifier) to that candidate's name.

    Returns:
        The key and the similarity ratio (rounded to 2 decimal places) of the best candidate at or
        above `MATCH_THRESHOLD`. None when no candidate reaches the threshold, or when `site_name`
        is made only of generic words and normalises to nothing. A tie goes to the key that sorts
        first, so the result does not depend on dict order.
    """
    target = normalise(site_name)
    if not target:
        return None
    best: tuple[str, float] | None = None
    for key in sorted(candidates):
        score = difflib.SequenceMatcher(None, target, normalise(candidates[key])).ratio()
        if score >= MATCH_THRESHOLD and (best is None or score > best[1]):
            best = (key, score)
    return None if best is None else (best[0], round(best[1], 2))


def technology_from_tec_plant_type(*, plant_type: str) -> TechnologyType:
    """Say whether a TEC `Plant Type` string describes a pure photovoltaic (PV) site or a hybrid.

    Args:
        plant_type: The register's semicolon-separated technologies, such as
            `Energy Storage System;PV Array (Photo Voltaic/solar)`.

    Returns:
        `hybrid` if the site lists any technology other than PV, demand, or reactive compensation;
        `pure PV` if PV is the only technology listed besides demand and reactive compensation; and
        `unknown` if PV is not listed at all.
    """
    parts = [part.strip() for part in plant_type.split(";") if part.strip()]
    if not any(part.startswith(PV_PLANT_TYPE) for part in parts):
        return "unknown"
    others = [
        part
        for part in parts
        if not part.startswith(PV_PLANT_TYPE) and part not in NON_GENERATING_PLANT_TYPES
    ]
    return "hybrid" if others else "pure PV"


def site_technology(
    *,
    tec_plant_type: str | None,
    storage_bmu_with_output: bool,
    repd_battery_status: str | None,
    repd_solar_found: bool,
) -> tuple[TechnologyType, StorageEvidenceType]:
    """Say whether a site is hybrid or pure PV, on the strongest evidence of storage first.

    The evidence runs from a storage BMU with output at the site, to an operational battery in
    REPD, to storage that is planned or under construction (listed in the TEC plant type or in a
    REPD battery row not yet operational). A site with no evidence of storage is pure PV when the
    TEC plant type lists PV only, or REPD has a solar row and no battery row; otherwise the
    technology is unknown.

    Args:
        tec_plant_type: The matched TEC project's plant type, or None if no project matched.
        storage_bmu_with_output: Whether a storage BMU at the site has output in the window.
        repd_battery_status: The development status of the REPD battery at the site, or None.
        repd_solar_found: Whether a REPD solar row matched.

    Returns:
        The technology and the evidence for it.
    """
    if storage_bmu_with_output:
        return "hybrid", "storage BMU with output"
    if repd_battery_status == "Operational":
        return "hybrid", "operational battery in REPD"
    tec_technology = (
        technology_from_tec_plant_type(plant_type=tec_plant_type) if tec_plant_type else "unknown"
    )
    if tec_technology == "hybrid" or repd_battery_status is not None:
        return "hybrid", "storage planned or under construction"
    if tec_technology == "pure PV":
        return "pure PV", "TEC plant type lists PV only"
    if repd_solar_found:
        return "pure PV", "REPD solar row, no battery row"
    return "unknown", "none"


def _to_float(value: Any) -> float | None:
    """Parse a number from a cell, returning None for a blank or unparseable cell."""
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


@cache
def _osgb_to_wgs84() -> Transformer:
    """Return the transformer from the Ordnance Survey National Grid to longitude and latitude."""
    return Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


def osgb_to_lon_lat(*, easting_m: float, northing_m: float) -> tuple[float, float]:
    """Convert an Ordnance Survey National Grid position to degrees east and north.

    Args:
        easting_m: Metres east of the grid's origin, as REPD's `X-coordinate` column states it.
        northing_m: Metres north of the grid's origin, as REPD's `Y-coordinate` column states it.

    Returns:
        `(longitude, latitude)` in degrees.
    """
    longitude, latitude = _osgb_to_wgs84().transform(easting_m, northing_m)
    return float(longitude), float(latitude)


def _ring_contains(*, ring: list[list[float]], x: float, y: float) -> bool:
    """Say whether a point is inside a closed ring, counting the edges a ray east crosses."""
    inside = False
    for (x1, y1), (x2, y2) in pairwise(ring):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def licence_area_at(
    *, easting_m: float, northing_m: float, areas: dict[str, Any]
) -> dict[str, Any] | None:
    """Return the properties of the licence area that contains a position, or None.

    Args:
        easting_m: Metres east, on the same grid as the GeoJSON's coordinates.
        northing_m: Metres north.
        areas: NESO's licence-area GeoJSON: a feature collection of `MultiPolygon` or `Polygon`
            features. In a polygon, the first ring is the outer boundary and any later ring is a
            hole.

    Returns:
        The feature's `properties` (for NESO's map: `Name`, `DNO`, `Area`), or None when no area
        contains the position. A position on a boundary can fall in either neighbour.
    """
    for feature in areas["features"]:
        geometry = feature["geometry"]
        polygons = (
            [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        )
        for outer, *holes in polygons:
            if _ring_contains(ring=outer, x=easting_m, y=northing_m) and not any(
                _ring_contains(ring=hole, x=easting_m, y=northing_m) for hole in holes
            ):
                return feature["properties"]
    return None


def _distance_to_segment_m(*, x: float, y: float, start: list[float], end: list[float]) -> float:
    """Return the distance from a point to the line segment from `start` to `end`."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    along = (
        0.0 if length_squared == 0 else ((x - start[0]) * dx + (y - start[1]) * dy) / length_squared
    )
    along = min(1.0, max(0.0, along))
    return float(math.hypot(x - (start[0] + along * dx), y - (start[1] + along * dy)))


def distance_to_other_areas_m(
    *, easting_m: float, northing_m: float, areas: dict[str, Any], own_name: str
) -> float:
    """Return the distance from a position to the nearest edge of any other licence area.

    Args:
        easting_m: Metres east, on the same grid as the GeoJSON's coordinates.
        northing_m: Metres north.
        areas: NESO's licence-area GeoJSON, as for `licence_area_at`.
        own_name: The `Name` of the area that contains the position, whose edges are skipped.

    Returns:
        The distance in metres, which says how far inside its own area a position sits.
    """
    nearest = math.inf
    for feature in areas["features"]:
        if feature["properties"]["Name"] == own_name:
            continue
        geometry = feature["geometry"]
        polygons = (
            [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        )
        for polygon in polygons:
            for ring in polygon:
                for start, end in pairwise(ring):
                    nearest = min(
                        nearest,
                        _distance_to_segment_m(x=easting_m, y=northing_m, start=start, end=end),
                    )
    return nearest


def connection_type(*, elexon_bmu_id: str) -> str:
    """Classify a BMU's connection from its identifier prefix."""
    if elexon_bmu_id.startswith("T_"):
        return "transmission-connected"
    if elexon_bmu_id.startswith("E_"):
        return "embedded"
    return "other"


def dno_area(*, gsp_group_id: str | None) -> str | None:
    """Return the distribution network operator whose licence area a GSP group names.

    Args:
        gsp_group_id: The BMU register's GSP group identifier, such as `_B`, or None when the
            register leaves the group empty.

    Returns:
        The operator's short name, or None when the identifier is empty or not one of the 14 groups.
    """
    area = GSP_GROUP_AREAS.get(gsp_group_id or "")
    return area[1] if area else None


def _display_name(
    *, site_name: str, bmu_id: str, tec_name: str | None, repd_name: str | None
) -> str:
    """Return the name to show for a BMU: its register name, or its project's name if it has none.

    A few BMUs carry their own identifier as their register name, which says nothing to a reader,
    so the matched TEC project's name, or else the matched REPD site's name, stands in.
    """
    if site_name not in {"", bmu_id}:
        return site_name
    return tec_name or repd_name or bmu_id


class ReviewedMatch(TypedDict):
    """One row of the hand-reviewed matches."""

    tec_project_id: str | None
    repd_ref_id: str | None
    storage_bmu_ids: list[str]


def _reviewed_matches() -> dict[str, ReviewedMatch]:
    """Read the hand-reviewed BMU matches."""
    table = pl.read_csv(REVIEWED_MATCHES_PATH, infer_schema_length=0)
    return {
        row["elexon_bmu_id"]: ReviewedMatch(
            tec_project_id=row["tec_project_id"] or None,
            repd_ref_id=row["repd_ref_id"] or None,
            storage_bmu_ids=[bmu for bmu in (row["storage_bmu_ids"] or "").split(";") if bmu],
        )
        for row in table.iter_rows(named=True)
    }


def _has_output(*, bmu_id: str, window_label: str) -> bool:
    """Say whether a BMU's output reaches `STORAGE_OUTPUT_MWH` in either direction."""
    output = pl.read_parquet(OUTPUT_DIR / f"{bmu_id}_{window_label}.parquet")
    return bool((output["output_mwh"].abs() >= STORAGE_OUTPUT_MWH).any())


def _repd_grid_position(*, repd_row: dict[str, Any] | None) -> tuple[float, float] | None:
    """Return a REPD row's (easting, northing) in metres, or None when it has no position."""
    if repd_row is None:
        return None
    easting = _to_float(repd_row["X-coordinate"])
    northing = _to_float(repd_row["Y-coordinate"])
    if easting is None or northing is None:
        return None
    return easting, northing


def _repd_position(*, repd_row: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """Return a REPD row's longitude and latitude, or None for a missing or blank position."""
    position = _repd_grid_position(repd_row=repd_row)
    if position is None:
        return None, None
    return osgb_to_lon_lat(easting_m=position[0], northing_m=position[1])


def best_tec_rows(*, tec: pl.DataFrame) -> pl.DataFrame:
    """Return one row for each TEC project that lists PV: the row at its most advanced status.

    The register holds one row for each stage of a project. A later stage carries a cumulative
    capacity that includes capacity not yet built. A project built at 99.4 MW with a later 20.6 MW
    increase has a second row of 120 MW, so the study takes the row at the most advanced status.
    The `tec_mw` column is the capacity connected for a project whose most advanced status is
    Built. For a project at any other status (under construction, consented, awaiting consents, or
    scoping), `tec_mw` is the agreed cumulative capacity, because a built row's cumulative figure
    can include a later increase.

    Args:
        tec: The TEC register with all-string columns.

    Returns:
        The PV rows, one for each `Project ID`, with the added column `tec_mw`.
    """
    rank = {status: index for index, status in enumerate(TEC_STATUS_ORDER)}
    return (
        tec.filter(pl.col("Plant Type").str.contains(PV_PLANT_TYPE))
        .with_columns(
            status_rank=pl.col("Project Status").replace_strict(
                rank, default=len(rank), return_dtype=pl.Int64
            )
        )
        .sort("status_rank", maintain_order=True)
        .unique(subset="Project ID", keep="first", maintain_order=True)
        .with_columns(
            tec_mw=pl.when(pl.col("Project Status") == "Built")
            .then(pl.col("MW Connected").cast(pl.Float64, strict=False))
            .otherwise(pl.col("Cumulative Total Capacity (MW)").cast(pl.Float64, strict=False))
        )
        .drop("status_rank")
    )


def _tec_candidates(*, tec: pl.DataFrame) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Return the names and the rows of the TEC register's PV projects, keyed by Project ID."""
    rows = {str(row["Project ID"]): row for row in best_tec_rows(tec=tec).iter_rows(named=True)}
    return {key: str(row["Project Name"]) for key, row in rows.items()}, rows


def _repd_candidates(
    *, repd: pl.DataFrame
) -> tuple[dict[str, str], dict[str, tuple[str, float | None]]]:
    """Return the solar projects' names by Ref ID, and the linked battery at each.

    The solar projects are those whose status is in `REPD_BUILT_STATUSES`.

    REPD links a solar row and a battery row at one site through the column `Storage Co-location
    REPD Ref ID`, which holds the other row's Ref ID. A solar row's battery status is the best
    status among the battery rows linked to it in either direction, with Operational the best.

    Args:
        repd: The REPD register with all-string columns.

    Returns:
        The names of the solar projects, and for each the status and capacity in megawatts
        of the battery linked to it.
    """
    built = repd.filter(pl.col("Development Status (short)").is_in(REPD_BUILT_STATUSES))
    solar = built.filter(pl.col("Technology Type") == "Solar Photovoltaics")
    batteries = {
        str(row["Ref ID"]): row
        for row in built.filter(pl.col("Technology Type") == "Battery").iter_rows(named=True)
    }
    battery_status: dict[str, tuple[str, float | None]] = {}

    def record(solar_ref: str, battery: dict[str, Any]) -> None:
        status = str(battery["Development Status (short)"])
        current = battery_status.get(solar_ref)
        if current is None or current[0] != "Operational":
            capacity = _to_float(battery["Installed Capacity (MWelec)"])
            battery_status[solar_ref] = (status, capacity)

    for battery in batteries.values():
        linked = battery["Storage Co-location REPD Ref ID"]
        if linked:
            record(str(linked), battery)
    for row in solar.iter_rows(named=True):
        linked = row["Storage Co-location REPD Ref ID"]
        if linked and str(linked) in batteries:
            record(str(row["Ref ID"]), batteries[str(linked)])
    return {
        str(r["Ref ID"]): str(r["Site Name"]) for r in solar.iter_rows(named=True)
    }, battery_status


def capacity_disparity(*, table: pl.DataFrame) -> pl.DataFrame:
    """Rank the BMUs with output by how far apart their six capacity figures are.

    The rule: for each single-site BMU whose output follows the sun, take the figures in
    `DISPARITY_FIGURES` that exist and are above zero (a missing figure, and a Maximum Export Limit
    of zero, are left out), and divide the largest by the smallest. A BMU with fewer than two such
    figures is left out.

    Args:
        table: The census table, with `scope`, `basis`, and the `DISPARITY_FIGURES` columns.

    Returns:
        One row for each ranked BMU, `elexon_bmu_id`, `display_name`, the six figures, `lowest_mw`,
        `highest_mw`, and `ratio` (highest over lowest), sorted by `ratio`, largest first, then by
        `elexon_bmu_id`.
    """
    followers = table.filter(
        (pl.col("scope") == "single-site") & pl.col("basis").str.contains("behaviour")
    )
    figures = pl.concat_list(
        [pl.when(pl.col(name) > 0).then(pl.col(name)) for name in DISPARITY_FIGURES]
    ).list.drop_nulls()
    return (
        followers.with_columns(figures=figures)
        .filter(pl.col("figures").list.len() >= 2)
        .with_columns(
            lowest_mw=pl.col("figures").list.min(), highest_mw=pl.col("figures").list.max()
        )
        .with_columns(ratio=pl.col("highest_mw") / pl.col("lowest_mw"))
        .select(
            "elexon_bmu_id", "display_name", *DISPARITY_FIGURES, "lowest_mw", "highest_mw", "ratio"
        )
        .sort(["ratio", "elexon_bmu_id"], descending=[True, False])
    )


def build_table() -> pl.DataFrame:
    """Build the one-row-per-solar-BMU table of capacity figures.

    The run date and window come from `fetch_sources.py`'s `lineage.json`, and the classes from
    `classify.py`.

    Returns:
        One row per BMU in the census (IGCPU type Solar, or output that follows the sun), with the
        BMU's five capacity figures and the P99 of its output, the matched TEC project and REPD
        reference, the BMU's technology (pure PV, hybrid, or unknown) and the evidence for
        it, the licence area (from NESO's map) that contains the matched REPD row's position and how
        far that position is from the nearest other area, the join methods, and the match scores.
    """
    reference = {str(r["elexonBmUnit"]): r for r in fetch_bmu_reference() if r["elexonBmUnit"]}
    today, window = recorded_run()
    igcpu = fetch_igcpu(today=today)
    latest_igcpu: dict[str, dict[str, Any]] = {}
    for row in sorted(igcpu, key=lambda r: str(r["publishTime"])):
        if row["bmUnit"]:
            latest_igcpu[str(row["bmUnit"])] = row
    classes = pl.read_parquet(STUDY_DIR / "classes.parquet").filter(pl.col("is_solar"))
    bmu_ids = classes["elexon_bmu_id"].to_list()
    mel = fetch_mels(bmu_ids=bmu_ids, today=today)
    tec_names, tec_rows = _tec_candidates(tec=fetch_tec())
    repd = fetch_repd()
    repd_names, repd_battery_status = _repd_candidates(repd=repd)
    repd_rows = {str(r["Ref ID"]): r for r in repd.iter_rows(named=True)}
    reviewed = _reviewed_matches()
    licence_areas = fetch_dno_areas()

    out = []
    for class_row in classes.iter_rows(named=True):
        bmu_id = class_row["elexon_bmu_id"]
        output_path = OUTPUT_DIR / f"{bmu_id}_{window.label}.parquet"
        output = pl.read_parquet(output_path) if output_path.exists() else empty_output()
        ref = reference.get(bmu_id, {})
        igcpu_row = latest_igcpu.get(bmu_id)
        site_name = ref.get("bmUnitName") or (igcpu_row or {}).get("registeredResourceName") or ""
        tec_match = best_match(site_name=site_name, candidates=tec_names)
        repd_match = best_match(site_name=site_name, candidates=repd_names)
        hand = reviewed.get(bmu_id)
        storage_ids: list[str] = []
        if hand is not None:
            tec_match = (hand["tec_project_id"], 1.0) if hand["tec_project_id"] else tec_match
            repd_match = (hand["repd_ref_id"], 1.0) if hand["repd_ref_id"] else repd_match
            storage_ids = [
                bmu
                for bmu in hand["storage_bmu_ids"]
                if _has_output(bmu_id=bmu, window_label=window.label)
            ]
        battery = repd_battery_status.get(repd_match[0]) if repd_match else None
        technology, evidence = site_technology(
            tec_plant_type=str(tec_rows[tec_match[0]]["Plant Type"]) if tec_match else None,
            storage_bmu_with_output=bool(storage_ids),
            repd_battery_status=battery[0] if battery else None,
            repd_solar_found=repd_match is not None,
        )
        longitude, latitude = _repd_position(
            repd_row=repd_rows[repd_match[0]] if repd_match else None
        )
        grid_position = _repd_grid_position(
            repd_row=repd_rows[repd_match[0]] if repd_match else None
        )
        area_at_position = (
            licence_area_at(
                easting_m=grid_position[0], northing_m=grid_position[1], areas=licence_areas
            )
            if grid_position
            else None
        )
        out.append(
            {
                "elexon_bmu_id": bmu_id,
                "national_grid_bmu_id": ref.get("nationalGridBmUnit") or bmu_id.split("_", 1)[-1],
                "site_name": site_name,
                "display_name": _display_name(
                    site_name=site_name,
                    bmu_id=bmu_id,
                    tec_name=tec_names.get(tec_match[0]) if tec_match else None,
                    repd_name=repd_names.get(repd_match[0]) if repd_match else None,
                ),
                "igcpu_name": igcpu_row["registeredResourceName"] if igcpu_row else None,
                "tec_name": tec_names.get(tec_match[0]) if tec_match else None,
                "repd_name": repd_names.get(repd_match[0]) if repd_match else None,
                "lead_party": ref.get("leadPartyName"),
                "connection_type": connection_type(elexon_bmu_id=bmu_id),
                "gsp_group": ref.get("gspGroupId"),
                "dno_area": dno_area(gsp_group_id=ref.get("gspGroupId")),
                "position_gsp_group": area_at_position["Name"] if area_at_position else None,
                "licence_area_by_position": dno_area(gsp_group_id=area_at_position["Name"])
                if area_at_position
                else None,
                "km_to_nearest_other_area": round(
                    distance_to_other_areas_m(
                        easting_m=grid_position[0],
                        northing_m=grid_position[1],
                        areas=licence_areas,
                        own_name=area_at_position["Name"],
                    )
                    / 1000,
                    1,
                )
                if grid_position and area_at_position
                else None,
                "scope": class_row["scope"],
                "basis": class_row["basis"],
                "correlation": class_row["correlation"],
                "technology": technology,
                "technology_evidence": evidence,
                "repd_battery_status": battery[0] if battery else None,
                "repd_battery_mw": battery[1] if battery else None,
                "storage_bmu_ids": ";".join(storage_ids),
                "generation_capacity_mw": _to_float(ref.get("generationCapacity")),
                "igcpu_installed_capacity_mw": igcpu_row["installedCapacity"]
                if igcpu_row
                else None,
                "tec_project_id": tec_match[0] if tec_match else None,
                "tec_mw": tec_rows[tec_match[0]]["tec_mw"] if tec_match else None,
                "tec_status": tec_rows[tec_match[0]]["Project Status"] if tec_match else None,
                "tec_connected_mw": _to_float(tec_rows[tec_match[0]]["MW Connected"])
                if tec_match
                else None,
                "largest_mel_mw": mel.get(bmu_id),
                "repd_ref_id": repd_match[0] if repd_match else None,
                "repd_installed_capacity_mw": _to_float(
                    repd_rows[repd_match[0]]["Installed Capacity (MWelec)"]
                )
                if repd_match
                else None,
                P99_COLUMN: p99_output_mw(output=output, window_start=window.start),
                "repd_county": repd_rows[repd_match[0]]["County"] if repd_match else None,
                "repd_region": repd_rows[repd_match[0]]["Region"] if repd_match else None,
                "longitude": longitude,
                "latitude": latitude,
                "join_method": "igcpu and MEL: BMU id; tec and repd: "
                + ("hand-reviewed where a match is stated, else " if hand else "")
                + "fuzzy site name",
                "match_score": f"tec={tec_match[1] if tec_match else ''}; "
                f"repd={repd_match[1] if repd_match else ''}",
            }
        )
    return pl.DataFrame(
        out,
        schema_overrides={
            "correlation": pl.Float64,
            "igcpu_installed_capacity_mw": pl.Float64,
            "largest_mel_mw": pl.Float64,
            "tec_mw": pl.Float64,
            "tec_connected_mw": pl.Float64,
            "repd_battery_mw": pl.Float64,
            "repd_installed_capacity_mw": pl.Float64,
            "generation_capacity_mw": pl.Float64,
            P99_COLUMN: pl.Float64,
            "longitude": pl.Float64,
            "latitude": pl.Float64,
        },
    )


def main() -> None:
    """Build the table and write it as a CSV and a parquet file."""
    table = build_table()
    table.write_csv(STUDY_DIR / "solar_bmus.csv")
    table.write_parquet(STUDY_DIR / "solar_bmus.parquet")
    print(f"{table.height} solar BMUs written to {STUDY_DIR / 'solar_bmus.csv'}")


if __name__ == "__main__":
    main()
