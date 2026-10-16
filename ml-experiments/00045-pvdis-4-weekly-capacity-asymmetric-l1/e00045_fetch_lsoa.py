"""Download the outside geography and DESNZ small-scale solar files for experiment 00045.

Run with `uv run python ml-experiments/00045-.../e00045_fetch_lsoa.py`. Resumable: every chunk is
one file under `outside/_cache/`, written to a `.partial` name and renamed, and a re-run skips every
chunk already on disk.

It fetches, for the 16 Primary service-area polygons in `TimeSeriesMetadata.area_wkt`:

1. The ONS lower-layer super output area (LSOA) December 2021 boundaries (generalised clipped,
   BGC V5) that intersect any polygon, and their 2021 population-weighted centroids.
2. The ONS Westminster parliamentary constituency July 2024 boundaries (BGC) that intersect any
   polygon. DESNZ publishes its small-scale solar capacity no finer than constituency and local
   authority, never by LSOA, so these are the boundaries the DESNZ table can be joined to.
3. The DESNZ "Solar photovoltaics deployment" workbook (Energy Trends table 6.3) and the DESNZ
   "Renewable electricity by local authority" workbook, with the constituency table and the
   local-authority solar column extracted to parquet.

It then writes the area shares, the validation report, `lineage.json`, and `README.md` into
`outside/`. Licence: Open Government Licence v3.0.
"""

import hashlib
import json
import re
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from xml.etree import ElementTree

import e00045_common as common
import numpy as np
import polars as pl
import requests
import shapely
from pyproj import Transformer
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
METADATA_PATH: Final[Path] = REPOSITORY_ROOT / "data" / "NGED" / "metadata.parquet"
OUTSIDE_DIR: Final[Path] = common.DATA_DIR / "outside"
CACHE_DIR: Final[Path] = OUTSIDE_DIR / "_cache"
SCRIPT_PATH: Final[str] = (
    "ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_fetch_lsoa.py"
)

ARCGIS_ROOT: Final[str] = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services"
LSOA_SERVICE: Final[str] = "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5"
LSOA_CENTROID_SERVICE: Final[str] = "LSOA_PopCentroids_EW_2021_V4"
CONSTITUENCY_SERVICE: Final[str] = (
    "Westminster_Parliamentary_Constituencies_July_2024_Boundaries_UK_BGC"
)
PORTAL_PAGES: Final[dict[str, str]] = {
    "LSOA boundaries (BGC V5)": "https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bgc-v5",
    "LSOA population-weighted centroids": "https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-ew-population-weighted-centroids",
    "Constituency boundaries (BGC)": "https://geoportal.statistics.gov.uk/datasets/ons::westminster-parliamentary-constituencies-july-2024-boundaries-uk-bgc",
}
"""Portal pages describing each layer; the query URLs themselves are in `lineage.json`."""

DESNZ_SOLAR_URL: Final[str] = (
    "https://assets.publishing.service.gov.uk/media/6aba729a734e2435f202f19e/"
    "ET_6.3_SOLAR_PV_SEP_26.xlsx"
)
DESNZ_LOCAL_AUTHORITY_URL: Final[str] = (
    "https://assets.publishing.service.gov.uk/media/6aba6ebafe72ed1e2b02f151/"
    "Renewable_electricity_by_local_authority_2014_2025.xlsx"
)
DESNZ_SOLAR_PAGE: Final[str] = (
    "https://www.gov.uk/government/statistics/energy-trends-section-6-renewables"
)
DESNZ_LOCAL_AUTHORITY_PAGE: Final[str] = (
    "https://www.gov.uk/government/statistics/regional-renewable-statistics"
)
DESNZ_CONSTITUENCY_SHEET: Final[str] = "Table_3_dom_by_PC"
DESNZ_LOCAL_AUTHORITY_SHEET: Final[str] = "LA - Capacity, 2025"

TO_BRITISH_NATIONAL_GRID: Final = Transformer.from_crs(
    "EPSG:4326", "EPSG:27700", always_xy=True
).transform
CENTROID_BATCH_SIZE: Final[int] = 200
PAGE_SIZE: Final[int] = 1000
REQUEST_ATTEMPTS: Final[int] = 4
REQUEST_TIMEOUT_S: Final[int] = 120
MIN_INTERSECTION_M2: Final[float] = 1.0
"""An intersection smaller than this is a shared boundary, not an overlap."""
COORDINATE_DECIMALS: Final[int] = 1
"""Decimals of a metre kept in the written well-known text."""
MAX_PLAUSIBLE_CONSTITUENCY_DOMESTIC_MW: Final[float] = 100.0
MAX_PLAUSIBLE_LOCAL_AUTHORITY_SOLAR_MW: Final[float] = 1000.0
CONSTITUENCY_CODE_PATTERN: Final[str] = r"(E14|W07|S14|N05)\d{6}"
"""July 2024 constituency codes; the sheet's region and country rows have other prefixes."""
GB_CODE_PREFIXES: Final[tuple[str, ...]] = ("E", "W", "S")

_SPREADSHEET_NS: Final[str] = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RELATIONSHIP_NS: Final[str] = (
    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
)
_PACKAGE_RELATIONSHIP_NS: Final[str] = (
    "{http://schemas.openxmlformats.org/package/2006/relationships}"
)


def _atomic_write_bytes(*, path: Path, content: bytes) -> None:
    """Write `content` to `path` through a `.partial` file, so a kill leaves no truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(content)
    partial.rename(path)


def _request(*, url: str, params: dict[str, Any] | None, post: bool = False) -> requests.Response:
    """Send a request, retrying a connection error, a timeout or a 5xx; any other status raises.

    ArcGIS queries are sent with POST because a `LSOA21CD IN (...)` clause can exceed the length a
    server accepts in a URL.
    """
    last_error: Exception | None = None
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            response = (
                requests.post(url, data=params, timeout=REQUEST_TIMEOUT_S)
                if post
                else requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
            )
        except (requests.ConnectionError, requests.Timeout) as error:
            last_error = error
        else:
            if response.status_code < 500:
                response.raise_for_status()
                return response
            last_error = requests.HTTPError(f"HTTP {response.status_code} from {url}")
        time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _query_features(*, service: str, params: dict[str, str]) -> dict[str, Any]:
    """Return the GeoJSON feature collection of one ArcGIS query, following result paging.

    The server answers `error` inside a 200 response for a bad parameter, which is raised here.
    """
    url = f"{ARCGIS_ROOT}/{service}/FeatureServer/0/query"
    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        page_params = params | {
            "f": "geojson",
            "outSR": "27700",
            "resultOffset": str(offset),
            "resultRecordCount": str(PAGE_SIZE),
        }
        payload = _request(url=url, params=page_params, post=True).json()
        if "error" in payload:
            msg = f"{service}: {payload['error']}"
            raise RuntimeError(msg)
        features.extend(payload["features"])
        if not payload.get("exceededTransferLimit"):
            return {"type": "FeatureCollection", "features": features}
        offset += len(payload["features"])


def _envelope_json(*, geometry: BaseGeometry) -> str:
    """Return the bounding box of a British National Grid geometry as an Esri envelope."""
    xmin, ymin, xmax, ymax = geometry.bounds
    return json.dumps(
        {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "spatialReference": {"wkid": 27700},
        }
    )


def primary_polygons() -> dict[int, BaseGeometry]:
    """Return each Primary series' service-area polygon in British National Grid metres.

    Returns:
        `time_series_id` to polygon, for the 16 Disaggregated Demand series.
    """
    metadata = (
        pl.read_parquet(METADATA_PATH)
        .filter(pl.col("time_series_type") == "Disaggregated Demand")
        .select("time_series_id", "area_wkt")
        .sort("time_series_id")
    )
    return {
        row["time_series_id"]: shapely.transform(
            shapely.from_wkt(row["area_wkt"]),
            lambda xy: np.column_stack(TO_BRITISH_NATIONAL_GRID(xy[:, 0], xy[:, 1])),
        )
        for row in metadata.iter_rows(named=True)
    }


def fetch_boundary_chunks(
    *, kind: str, service: str, code_field: str, polygons: dict[int, BaseGeometry]
) -> dict[str, dict[str, Any]]:
    """Fetch, one cached file per distinct polygon, the features intersecting each polygon.

    Args:
        kind: The cache file prefix.
        service: The ArcGIS feature service name.
        code_field: The attribute holding the feature's code.
        polygons: `time_series_id` to polygon in British National Grid metres.

    Returns:
        Feature code to its GeoJSON feature, over the union of all chunks.
    """
    done: dict[str, int] = {}
    features: dict[str, dict[str, Any]] = {}
    for series_id, polygon in polygons.items():
        key = hashlib.sha256(polygon.wkb).hexdigest()[:8]
        if key in done:
            print(f"{kind} {series_id}: shares the polygon of series {done[key]}")
            continue
        done[key] = series_id
        path = CACHE_DIR / f"{kind}_{key}.geojson"
        if path.exists():
            print(f"{kind} {series_id}: already cached, skipping")
        else:
            collection = _query_features(
                service=service,
                params={
                    "where": "1=1",
                    "geometry": _envelope_json(geometry=polygon),
                    "geometryType": "esriGeometryEnvelope",
                    "inSR": "27700",
                    "spatialRel": "esriSpatialRelIntersects",
                    "outFields": "*",
                },
            )
            _atomic_write_bytes(path=path, content=json.dumps(collection).encode())
            print(f"{kind} {series_id}: fetched {len(collection['features'])} features")
        for feature in json.loads(path.read_text())["features"]:
            features[feature["properties"][code_field]] = feature
    return features


def fetch_centroid_chunks(*, codes: list[str]) -> dict[str, tuple[float, float]]:
    """Fetch the population-weighted centroids of the given LSOAs, in cached batches.

    Args:
        codes: LSOA 2021 codes.

    Returns:
        LSOA code to its centroid's British National Grid (easting, northing).
    """
    centroids: dict[str, tuple[float, float]] = {}
    ordered = sorted(codes)
    for start in range(0, len(ordered), CENTROID_BATCH_SIZE):
        batch = ordered[start : start + CENTROID_BATCH_SIZE]
        key = hashlib.sha256(",".join(batch).encode()).hexdigest()[:8]
        path = CACHE_DIR / f"lsoa_centroids_{start // CENTROID_BATCH_SIZE:03d}_{key}.geojson"
        if path.exists():
            print(f"lsoa centroids batch {start}: already cached, skipping")
        else:
            quoted = ",".join(f"'{code}'" for code in batch)
            collection = _query_features(
                service=LSOA_CENTROID_SERVICE,
                params={"where": f"LSOA21CD IN ({quoted})", "outFields": "LSOA21CD"},
            )
            _atomic_write_bytes(path=path, content=json.dumps(collection).encode())
            print(f"lsoa centroids batch {start}: fetched {len(collection['features'])}")
        for feature in json.loads(path.read_text())["features"]:
            easting, northing = feature["geometry"]["coordinates"][:2]
            centroids[feature["properties"]["LSOA21CD"]] = (easting, northing)
    return centroids


def download_file(*, url: str, path: Path) -> None:
    """Download `url` to `path` unless it is already there."""
    if path.exists():
        print(f"{path.name}: already cached, skipping")
        return
    _atomic_write_bytes(path=path, content=_request(url=url, params=None).content)
    print(f"{path.name}: downloaded {path.stat().st_size:,} bytes")


def read_workbook_rows(*, path: Path, sheet_name: str) -> list[list[str | None]]:
    """Read every row of one sheet of an `.xlsx` workbook as text.

    Args:
        path: The workbook.
        sheet_name: The sheet's name.

    Returns:
        One list per row, one text cell per column from the first column, `None` where empty.
    """
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in item.iter(f"{_SPREADSHEET_NS}t"))
                for item in root.iter(f"{_SPREADSHEET_NS}si")
            ]
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationship_id = next(
            sheet.get(f"{_RELATIONSHIP_NS}id")
            for sheet in workbook.iter(f"{_SPREADSHEET_NS}sheet")
            if sheet.get("name") == sheet_name
        )
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(
            str(rel.get("Target"))
            for rel in relationships.iter(f"{_PACKAGE_RELATIONSHIP_NS}Relationship")
            if rel.get("Id") == relationship_id
        )
        part = "xl/" + target.removeprefix("/xl/")
        rows: list[list[str | None]] = []
        with archive.open(part) as stream:
            for _, element in ElementTree.iterparse(stream):
                if element.tag != f"{_SPREADSHEET_NS}row":
                    continue
                cells: dict[int, str | None] = {}
                for cell in element.findall(f"{_SPREADSHEET_NS}c"):
                    letters = re.match(r"[A-Z]+", str(cell.get("r")))
                    assert letters is not None
                    column = 0
                    for letter in letters.group():
                        column = column * 26 + ord(letter) - ord("A") + 1
                    value = cell.find(f"{_SPREADSHEET_NS}v")
                    text = None if value is None else value.text
                    if cell.get("t") == "s" and text is not None:
                        text = shared[int(text)]
                    elif cell.get("t") == "inlineStr":
                        text = "".join(t.text or "" for t in cell.iter(f"{_SPREADSHEET_NS}t"))
                    cells[column - 1] = text
                width = max(cells, default=-1) + 1
                rows.append([cells.get(i) for i in range(width)])
                element.clear()
    return rows


def _float_or_none(*, text: str | None) -> float | None:
    """Return the number a cell holds, or None for an empty or non-numeric cell."""
    try:
        return None if text is None else float(text)
    except ValueError:
        return None


def parse_constituency_table(*, path: Path) -> pl.DataFrame:
    """Return DESNZ's domestic solar capacity and installation count per constituency.

    Args:
        path: The Energy Trends table 6.3 workbook.

    Returns:
        One row per row of the sheet that has a code starting with a letter and digits.
    """
    rows = read_workbook_rows(path=path, sheet_name=DESNZ_CONSTITUENCY_SHEET)
    header_index = next(i for i, row in enumerate(rows) if row and row[0] == "Constituency Code")
    records = [
        {
            "constituency_code": row[0],
            "country_or_region": row[1],
            "constituency_name": row[2],
            "domestic_installed_capacity_mw": _float_or_none(text=row[3]),
            "domestic_installations": _float_or_none(text=row[4]),
        }
        for row in rows[header_index + 1 :]
        if len(row) >= 5 and row[0] is not None and re.fullmatch(CONSTITUENCY_CODE_PATTERN, row[0])
    ]
    return pl.DataFrame(
        records,
        schema={
            "constituency_code": pl.String,
            "country_or_region": pl.String,
            "constituency_name": pl.String,
            "domestic_installed_capacity_mw": pl.Float64,
            "domestic_installations": pl.Float64,
        },
    )


def parse_local_authority_solar(*, path: Path) -> pl.DataFrame:
    """Return DESNZ's end-2025 solar PV installed capacity (MW) per local authority.

    Args:
        path: The local-authority renewable electricity workbook.

    Returns:
        One row per local authority with a code.
    """
    rows = read_workbook_rows(path=path, sheet_name=DESNZ_LOCAL_AUTHORITY_SHEET)
    header_index = next(
        i for i, row in enumerate(rows) if row and (row[0] or "").startswith("Local Authority Code")
    )
    header = rows[header_index]
    solar_column = header.index("Solar PV")
    records = [
        {
            "local_authority_code": row[0],
            "local_authority_name": row[1],
            "solar_pv_installed_capacity_mw": _float_or_none(text=row[solar_column]),
        }
        for row in rows[header_index + 1 :]
        if len(row) > solar_column and row[0] is not None and re.fullmatch(r"[EWSN]\d{8}", row[0])
    ]
    return pl.DataFrame(
        records,
        schema={
            "local_authority_code": pl.String,
            "local_authority_name": pl.String,
            "solar_pv_installed_capacity_mw": pl.Float64,
        },
    )


def desnz_publication_facts(*, path: Path) -> dict[str, str]:
    """Return the publication date, data period and capacity-basis lines of the DESNZ workbook."""
    cover = [
        " ".join(c for c in row if c)
        for row in read_workbook_rows(path=path, sheet_name="Cover_sheet")
    ]
    commentary = [
        " ".join(c for c in row if c)
        for row in read_workbook_rows(path=path, sheet_name="Commentary")
    ]
    constituency_title = next(
        (
            row[0] or ""
            for row in read_workbook_rows(path=path, sheet_name=DESNZ_CONSTITUENCY_SHEET)
        ),
        "",
    )
    published = " ".join(next((line for line in cover if "was published on" in line), "").split())
    period = " ".join(
        next((line for line in cover if "contains monthly data including" in line), "").split()
    )
    basis_text = " ".join(next((line for line in cover if "(TIC)" in line), "").split())
    basis = next(
        (sentence for sentence in re.split(r"(?<=\.) ", basis_text) if "(TIC)" in sentence), ""
    )
    return {
        "constituency_sheet_title": constituency_title,
        "published": published,
        "data_period": period,
        "capacity_basis_sentence": basis,
        "commentary_first_line": next((c for c in commentary if c.strip()), ""),
    }


def build_geography_frames(
    *, features: dict[str, dict[str, Any]], code_field: str, name_field: str
) -> tuple[pl.DataFrame, dict[str, BaseGeometry]]:
    """Return a boundary table (code, name, WKT in EPSG:27700) and the geometries by code.

    The table's `is_valid` records whether the downloaded geometry was valid. The returned
    geometries and the written well-known text are repaired with `shapely.make_valid`.
    """
    raw_geometries = {code: shape(feature["geometry"]) for code, feature in features.items()}
    geometries = {code: shapely.make_valid(geometry) for code, geometry in raw_geometries.items()}
    frame = pl.DataFrame(
        {
            "code": list(features),
            "name": [f["properties"][name_field] for f in features.values()],
            "latitude": [f["properties"].get("LAT") for f in features.values()],
            "longitude": [f["properties"].get("LONG") for f in features.values()],
            "area_km2": [geometries[c].area / 1e6 for c in features],
            "is_valid": [raw_geometries[c].is_valid for c in features],
            "geometry_wkt_epsg27700": [
                shapely.to_wkt(geometries[c], rounding_precision=COORDINATE_DECIMALS)
                for c in features
            ],
        }
    ).rename({"code": code_field})
    return frame.sort(code_field), geometries


def membership_table(
    *,
    polygons: dict[int, BaseGeometry],
    geometries: dict[str, BaseGeometry],
    centroids: dict[str, tuple[float, float]] | None,
    code_column: str,
) -> pl.DataFrame:
    """Return, for each polygon and each boundary feature overlapping it, the area shares.

    Args:
        polygons: `time_series_id` to Primary polygon in metres.
        geometries: Feature code to boundary geometry in metres.
        centroids: Feature code to population-weighted centroid, or None where there are none.
        code_column: The name of the feature code column in the result.

    Returns:
        One row per overlapping pair, with the overlap area, the share of the feature inside the
        polygon, the share of the polygon the feature covers, and whether the centroid is inside.
    """
    records = []
    for series_id, polygon in polygons.items():
        for code, geometry in geometries.items():
            overlap = polygon.intersection(geometry).area
            if overlap <= MIN_INTERSECTION_M2:
                continue
            centroid_inside = (
                None
                if centroids is None or code not in centroids
                else bool(polygon.contains(shapely.Point(*centroids[code])))
            )
            records.append(
                {
                    "time_series_id": series_id,
                    code_column: code,
                    "overlap_km2": overlap / 1e6,
                    "feature_area_km2": geometry.area / 1e6,
                    "share_of_feature_inside_polygon": overlap / geometry.area,
                    "share_of_polygon_covered_by_feature": overlap / polygon.area,
                    "centroid_inside_polygon": centroid_inside,
                }
            )
    return pl.DataFrame(
        records,
        schema={
            "time_series_id": pl.Int64,
            code_column: pl.String,
            "overlap_km2": pl.Float64,
            "feature_area_km2": pl.Float64,
            "share_of_feature_inside_polygon": pl.Float64,
            "share_of_polygon_covered_by_feature": pl.Float64,
            "centroid_inside_polygon": pl.Boolean,
        },
    ).sort("time_series_id", code_column)


def coverage_by_polygon(
    *, polygons: dict[int, BaseGeometry], geometries: dict[str, BaseGeometry]
) -> dict[int, float]:
    """Return the share of each polygon's area that the union of the boundary features covers."""
    union = shapely.union_all(list(geometries.values()))
    return {
        series_id: polygon.intersection(union).area / polygon.area
        for series_id, polygon in polygons.items()
    }


def validate(
    *,
    polygons: dict[int, BaseGeometry],
    lsoa_frame: pl.DataFrame,
    lsoa_geometries: dict[str, BaseGeometry],
    lsoa_membership: pl.DataFrame,
    centroids: dict[str, tuple[float, float]],
    constituency_frame: pl.DataFrame,
    constituency_geometries: dict[str, BaseGeometry],
    constituency_membership: pl.DataFrame,
    constituency_table: pl.DataFrame,
    local_authority_table: pl.DataFrame,
) -> dict[str, Any]:
    """Run the data-validation checks that apply and return them as a JSON-ready dict."""
    lsoa_coverage = coverage_by_polygon(polygons=polygons, geometries=lsoa_geometries)
    constituency_coverage = coverage_by_polygon(
        polygons=polygons, geometries=constituency_geometries
    )
    lsoas_per_polygon = lsoa_membership.group_by("time_series_id").agg(
        pl.len().alias("lsoas_intersecting"),
        pl.col("centroid_inside_polygon").sum().alias("lsoas_with_centroid_inside"),
        pl.col("share_of_polygon_covered_by_feature").sum().alias("sum_of_polygon_area_shares"),
    )
    constituencies_per_polygon = constituency_membership.group_by("time_series_id").agg(
        pl.len().alias("constituencies_intersecting"),
        pl.col("share_of_polygon_covered_by_feature").sum().alias("sum_of_polygon_area_shares"),
    )
    joined = constituency_frame.join(
        constituency_table, left_on="PCON24CD", right_on="constituency_code", how="left"
    )
    lsoa_centroid_cross_check = [
        bool(lsoa_geometries[code].buffer(1.0).contains(shapely.Point(*xy)))
        for code, xy in centroids.items()
        if code in lsoa_geometries
    ]
    gb_table = constituency_table.filter(
        pl.col("constituency_code").str.slice(0, 1).is_in(GB_CODE_PREFIXES)
    )
    kw_per_installation = constituency_table.filter(pl.col("domestic_installations") > 0).select(
        (pl.col("domestic_installed_capacity_mw") * 1000 / pl.col("domestic_installations")).alias(
            "kw"
        )
    )["kw"]
    return {
        "lsoa": {
            "rows": lsoa_frame.height,
            "duplicate_codes": lsoa_frame.height - lsoa_frame["LSOA21CD"].n_unique(),
            "invalid_geometries": int((~lsoa_frame["is_valid"]).sum()),
            "null_cells": int(sum(lsoa_frame.null_count().row(0))),
            "area_km2_min_max": [lsoa_frame["area_km2"].min(), lsoa_frame["area_km2"].max()],
            "centroids_fetched": len(centroids),
            "lsoas_without_centroid": sorted(set(lsoa_geometries) - set(centroids))[:10],
            "centroid_inside_or_on_own_lsoa_share": (
                sum(lsoa_centroid_cross_check) / len(lsoa_centroid_cross_check)
            ),
            "coverage_of_polygon_by_union": {str(k): round(v, 4) for k, v in lsoa_coverage.items()},
            "membership_per_polygon": lsoas_per_polygon.sort("time_series_id").to_dicts(),
            "sum_of_polygon_shares_above_1_001": int(
                (lsoas_per_polygon["sum_of_polygon_area_shares"] > 1.001).sum()
            ),
        },
        "constituency": {
            "rows": constituency_frame.height,
            "duplicate_codes": constituency_frame.height
            - constituency_frame["PCON24CD"].n_unique(),
            "invalid_geometries": int((~constituency_frame["is_valid"]).sum()),
            "coverage_of_polygon_by_union": {
                str(k): round(v, 4) for k, v in constituency_coverage.items()
            },
            "membership_per_polygon": constituencies_per_polygon.sort("time_series_id").to_dicts(),
            "codes_missing_from_desnz_table": joined.filter(
                pl.col("domestic_installed_capacity_mw").is_null()
            )["PCON24CD"].to_list(),
            "domestic_capacity_mw_of_matched_constituencies_min_median_max": [
                joined["domestic_installed_capacity_mw"].min(),
                joined["domestic_installed_capacity_mw"].median(),
                joined["domestic_installed_capacity_mw"].max(),
            ],
            "matched_above_plausible_maximum": joined.filter(
                pl.col("domestic_installed_capacity_mw") > MAX_PLAUSIBLE_CONSTITUENCY_DOMESTIC_MW
            )["PCON24CD"].to_list(),
        },
        "desnz_constituency_table": {
            "rows": constituency_table.height,
            "duplicate_codes": constituency_table.height
            - constituency_table["constituency_code"].n_unique(),
            "null_capacity": int(constituency_table["domestic_installed_capacity_mw"].null_count()),
            "null_installations": int(constituency_table["domestic_installations"].null_count()),
            "negative_values": int(
                (constituency_table["domestic_installed_capacity_mw"] < 0).sum()
            ),
            "gb_rows": gb_table.height,
            "gb_capacity_mw_total": gb_table["domestic_installed_capacity_mw"].sum(),
            "mean_kw_per_installation_min_median_max": [
                kw_per_installation.min(),
                kw_per_installation.median(),
                kw_per_installation.max(),
            ],
        },
        "desnz_local_authority_table": {
            "rows": local_authority_table.height,
            "duplicate_codes": local_authority_table.height
            - local_authority_table["local_authority_code"].n_unique(),
            "null_solar_capacity": int(
                local_authority_table["solar_pv_installed_capacity_mw"].null_count()
            ),
            "above_plausible_maximum": local_authority_table.filter(
                pl.col("solar_pv_installed_capacity_mw") > MAX_PLAUSIBLE_LOCAL_AUTHORITY_SOLAR_MW
            )["local_authority_code"].to_list(),
        },
    }


def write_readme(
    *, facts: dict[str, Any], sizes: dict[str, tuple[int, int | None]], retrieved: str
) -> None:
    """Write `outside/README.md`, with every number read from the written files."""
    file_lines = "\n".join(
        f"- `{name}`: {size:,} bytes" + ("" if rows is None else f", {rows:,} rows")
        for name, (size, rows) in sizes.items()
    )
    validation = facts["validation"]
    lsoa = validation["lsoa"]
    constituency = validation["constituency"]
    desnz_table = validation["desnz_constituency_table"]
    la_table = validation["desnz_local_authority_table"]
    pages = PORTAL_PAGES
    lsoa_counts = "\n".join(
        f"  - `time_series_id` {row['time_series_id']}: {row['lsoas_intersecting']} LSOAs "
        f"intersecting, {row['lsoas_with_centroid_inside']} with the centroid inside"
        for row in lsoa["membership_per_polygon"]
    )
    text = f"""# Outside checks for experiment 00045: ONS boundaries and DESNZ small-scale solar

Written by `{SCRIPT_PATH}` on {retrieved}. Re-run that script to regenerate this folder; the
`_cache/` folder holds the raw chunks, and a re-run skips any chunk already there.

## What DESNZ publishes

**DESNZ publishes no small-scale solar table by LSOA.** The finest geography we found is the
parliamentary constituency, for domestic installations only (`domestic_installed_capacity_mw`,
`domestic_installations`), and the local authority, for solar PV of every size. The ONS LSOA
boundaries are still here, because they give the area shares of the Primary polygons, but the DESNZ
numbers must be joined at constituency or local-authority level.

- Constituency table: workbook `{DESNZ_SOLAR_URL}`, sheet `{DESNZ_CONSTITUENCY_SHEET}`.
  {facts["desnz"]["published"]}
  {facts["desnz"]["data_period"]}
  Sheet title: {facts["desnz"]["constituency_sheet_title"]}. Publication page: {DESNZ_SOLAR_PAGE}
- Local-authority table: workbook `{DESNZ_LOCAL_AUTHORITY_URL}`, sheet
  `{DESNZ_LOCAL_AUTHORITY_SHEET}`, column `Solar PV`, in MW, at the end of 2025. Publication page:
  {DESNZ_LOCAL_AUTHORITY_PAGE}
- Capacity basis: the publication does not say whether its capacity is DC or AC. Its cover sheet
  states: "{facts["desnz"]["capacity_basis_sentence"]}" Total installed capacity (TIC) for
  Microgeneration Certification Scheme (MCS) installations is the module (DC) rating in the MCS
  database, but the workbook does not say so, so treat DC as an assumption.
- Size bands: the constituency table has none; it is all domestic installations. The monthly
  national Tables 1 and 2 have size bands but no geography.

## Sources and licence

- ONS LSOA December 2021 boundaries, generalised clipped (BGC V5):
  {pages["LSOA boundaries (BGC V5)"]}
- ONS LSOA 2021 population-weighted centroids:
  {pages["LSOA population-weighted centroids"]}
- ONS Westminster parliamentary constituencies July 2024 boundaries (BGC):
  {pages["Constituency boundaries (BGC)"]}
- Queried through the ONS Open Geography Portal ArcGIS feature services under `{ARCGIS_ROOT}`; the
  exact query parameters are in `lineage.json`.
- Licence: Open Government Licence v3.0
  (<https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/>). Required
  attributions: "Source: Office for National Statistics licensed under the Open Government Licence
  v3.0" and "Contains OS data (c) Crown copyright and database right 2026". DESNZ tables are also
  Open Government Licence v3.0, Crown copyright.

## Files

{file_lines}

## Columns

- `lsoa_2021_bgc.parquet`: `LSOA21CD` code, `LSOA21NM` name, `latitude` and `longitude` (degrees,
  the ONS label point), `area_km2` (computed from the geometry), `is_valid`, and
  `geometry_wkt_epsg27700` (well-known text in British National Grid metres).
- `lsoa_2021_population_weighted_centroids.parquet`: `LSOA21CD`, `easting` and `northing` in
  British National Grid metres.
- `constituency_2024_bgc.parquet`: the same columns as the LSOA file, with `PCON24CD`, `PCON24NM`.
- `*_primary_membership.parquet`: one row per Primary polygon (`time_series_id`) and overlapping
  feature, with `overlap_km2`, `feature_area_km2`, `share_of_feature_inside_polygon`,
  `share_of_polygon_covered_by_feature`, and `centroid_inside_polygon` (null for constituencies).
- `desnz_constituency_domestic_solar.parquet`: `constituency_code`, `country_or_region`,
  `constituency_name`, `domestic_installed_capacity_mw` (MW), `domestic_installations` (count).
- `desnz_local_authority_solar_2025.parquet`: `local_authority_code`, `local_authority_name`,
  `solar_pv_installed_capacity_mw` (MW).

## Missing values

Polars null. The DESNZ constituency table has {desnz_table["null_capacity"]} null capacities
and {desnz_table["null_installations"]} null installation counts; the local-authority table has
{la_table["null_solar_capacity"]} null solar capacities.
Rows for region or country headings are dropped because they carry no eight-digit code.

## Validation

The full results are in `validation.json`. Summary:

- LSOAs fetched: {lsoa["rows"]}; duplicate codes {lsoa["duplicate_codes"]}; invalid geometries
  {lsoa["invalid_geometries"]}.
- Constituencies fetched: {constituency["rows"]}; duplicate codes
  {constituency["duplicate_codes"]}; codes missing from the DESNZ table
  {constituency["codes_missing_from_desnz_table"]}.
- Share of each Primary polygon's area covered by the fetched LSOAs (BGC boundaries are clipped to
  the coastline, so a polygon reaching into the sea or the Wash is covered less than 1):
  {lsoa["coverage_of_polygon_by_union"]}
- LSOAs per Primary polygon:
{lsoa_counts}

## Gotchas

- The BGC boundaries are generalised and clipped to the mean high water mark, so an LSOA's area is
  its land area and the polygon shares above are shares of the land inside the polygon.
- Two Primary series (the two Wrangle transformers) share one polygon, so their rows repeat.
- The constituency codes are the July 2024 codes (`E14...`); the DESNZ table uses the same codes.
- The DESNZ constituency table is domestic installations only, with MCS and Feed-in Tariff
  registrations assigned by postcode. It has no size bands and no non-domestic small-scale solar.
- DESNZ revises recent months upward as late notifications arrive.
"""
    (OUTSIDE_DIR / "README.md").write_text(text)


def main() -> None:
    """Fetch every chunk, write the combined files, validate, and write the notes."""
    retrieved = datetime.now(UTC).strftime("%Y-%m-%d")
    polygons = primary_polygons()
    print(f"{len(polygons)} Primary polygons")

    lsoa_features = fetch_boundary_chunks(
        kind="lsoa", service=LSOA_SERVICE, code_field="LSOA21CD", polygons=polygons
    )
    constituency_features = fetch_boundary_chunks(
        kind="constituency",
        service=CONSTITUENCY_SERVICE,
        code_field="PCON24CD",
        polygons=polygons,
    )
    centroids = fetch_centroid_chunks(codes=list(lsoa_features))
    download_file(url=DESNZ_SOLAR_URL, path=OUTSIDE_DIR / "desnz" / "ET_6.3_SOLAR_PV_SEP_26.xlsx")
    download_file(
        url=DESNZ_LOCAL_AUTHORITY_URL,
        path=OUTSIDE_DIR / "desnz" / "Renewable_electricity_by_local_authority_2014_2025.xlsx",
    )

    lsoa_frame, lsoa_geometries = build_geography_frames(
        features=lsoa_features, code_field="LSOA21CD", name_field="LSOA21NM"
    )
    constituency_frame, constituency_geometries = build_geography_frames(
        features=constituency_features, code_field="PCON24CD", name_field="PCON24NM"
    )
    centroid_frame = pl.DataFrame(
        {
            "LSOA21CD": list(centroids),
            "easting": [xy[0] for xy in centroids.values()],
            "northing": [xy[1] for xy in centroids.values()],
        }
    ).sort("LSOA21CD")
    lsoa_membership = membership_table(
        polygons=polygons,
        geometries=lsoa_geometries,
        centroids=centroids,
        code_column="LSOA21CD",
    )
    constituency_membership = membership_table(
        polygons=polygons,
        geometries=constituency_geometries,
        centroids=None,
        code_column="PCON24CD",
    )
    constituency_table = parse_constituency_table(
        path=OUTSIDE_DIR / "desnz" / "ET_6.3_SOLAR_PV_SEP_26.xlsx"
    )
    local_authority_table = parse_local_authority_solar(
        path=OUTSIDE_DIR / "desnz" / "Renewable_electricity_by_local_authority_2014_2025.xlsx"
    )

    outputs: dict[str, pl.DataFrame] = {
        "lsoa_2021_bgc.parquet": lsoa_frame,
        "lsoa_2021_population_weighted_centroids.parquet": centroid_frame,
        "lsoa_primary_membership.parquet": lsoa_membership,
        "constituency_2024_bgc.parquet": constituency_frame,
        "constituency_primary_membership.parquet": constituency_membership,
        "desnz_constituency_domestic_solar.parquet": constituency_table,
        "desnz_local_authority_solar_2025.parquet": local_authority_table,
    }
    for name, frame in outputs.items():
        partial = OUTSIDE_DIR / f"{name}.partial"
        frame.write_parquet(partial)
        partial.rename(OUTSIDE_DIR / name)

    validation = validate(
        polygons=polygons,
        lsoa_frame=lsoa_frame,
        lsoa_geometries=lsoa_geometries,
        lsoa_membership=lsoa_membership,
        centroids=centroids,
        constituency_frame=constituency_frame,
        constituency_geometries=constituency_geometries,
        constituency_membership=constituency_membership,
        constituency_table=constituency_table,
        local_authority_table=local_authority_table,
    )
    desnz = desnz_publication_facts(path=OUTSIDE_DIR / "desnz" / "ET_6.3_SOLAR_PV_SEP_26.xlsx")
    facts = {"validation": validation, "desnz": desnz}
    (OUTSIDE_DIR / "validation.json").write_text(json.dumps(validation, indent=2, default=str))
    (OUTSIDE_DIR / "lineage.json").write_text(
        json.dumps(
            {
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
                "script": SCRIPT_PATH,
                "arcgis_services": {
                    "lsoa": f"{ARCGIS_ROOT}/{LSOA_SERVICE}/FeatureServer/0/query",
                    "lsoa_centroids": (
                        f"{ARCGIS_ROOT}/{LSOA_CENTROID_SERVICE}/FeatureServer/0/query"
                    ),
                    "constituency": f"{ARCGIS_ROOT}/{CONSTITUENCY_SERVICE}/FeatureServer/0/query",
                },
                "arcgis_query": (
                    "where=1=1; geometry=bounding box of each distinct Primary polygon in "
                    "EPSG:27700; spatialRel=esriSpatialRelIntersects; outSR=27700; f=geojson; "
                    f"resultRecordCount={PAGE_SIZE}; centroids by LSOA21CD IN (...) in batches "
                    f"of {CENTROID_BATCH_SIZE}"
                ),
                "desnz_urls": [DESNZ_SOLAR_URL, DESNZ_LOCAL_AUTHORITY_URL],
                "desnz_publication": desnz,
                "licence": "Open Government Licence v3.0",
            },
            indent=2,
        )
    )
    sizes: dict[str, tuple[int, int | None]] = {
        name: ((OUTSIDE_DIR / name).stat().st_size, frame.height) for name, frame in outputs.items()
    }
    for workbook in sorted((OUTSIDE_DIR / "desnz").glob("*.xlsx")):
        sizes[f"desnz/{workbook.name}"] = (workbook.stat().st_size, None)
    write_readme(facts=facts, sizes=sizes, retrieved=retrieved)
    print(json.dumps(validation, indent=2, default=str))


if __name__ == "__main__":
    main()
