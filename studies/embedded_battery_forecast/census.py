"""Count the batteries connected in NGED's four licence areas, and how many of them are BMUs.

A BMU is a Balancing Mechanism Unit. NGED's Embedded Capacity Register (ECR) lists every generator
and store connected to its distribution network, but it carries no BMU identifier, and no public
register maps a meter point number to a BMU. This module therefore proposes matches between the
ECR's connected storage rows and the reviewed storage BMUs, and reports how many matches each of
three keys supports: the licence area (which fixes the BMU register's grid supply point group), the
site name, and the capacity. A second route goes through the Renewable Energy Planning Database
(REPD), which gives a BMU's site a position that can be compared with the ECR's own eastings and
northings. Only counts and bands go into a report that is committed; the site-level table goes
beside the other results under the study's data folder.
"""

import io
import json
import re
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.name_matching import NOISE_WORDS, best_match
from studies.sources import (
    BATTERY_PV_SEPARATION_DIR,
    MARKET_DOWNLOADS_DIR,
    SOLAR_BMU_CENSUS_INPUTS_DIR,
)

LICENCE_AREAS: Final[tuple[str, ...]] = (
    "East Midlands",
    "West Midlands",
    "South Wales",
    "South West",
)
"""NGED's four licence areas, as the ECR names them in brackets."""

GSP_GROUP_OF_LICENCE_AREA: Final[dict[str, str]] = {
    "East Midlands": "East Midlands",
    "West Midlands": "Midlands",
    "South Wales": "South Wales",
    "South West": "South Western",
}
"""The Elexon grid supply point group of each licence area. The BMU register names two of them
differently from the ECR."""

SizeClassType = Literal["under 1 MW", "1 to 10 MW", "10 to 50 MW", "50 to 100 MW", "100 MW or more"]

SIZE_CLASS_EDGES_MW: Final[tuple[float, ...]] = (1.0, 10.0, 50.0, 100.0)
"""The upper edges of the size classes. A battery on an edge belongs to the class above it."""

SIZE_CLASSES: Final[tuple[SizeClassType, ...]] = (
    "under 1 MW",
    "1 to 10 MW",
    "10 to 50 MW",
    "50 to 100 MW",
    "100 MW or more",
)

CAPACITY_TOLERANCE: Final[float] = 0.10
"""How far, as a share of the BMU's generation capacity, an ECR export capacity may differ."""

MIN_PLACE_TOKEN_LENGTH: Final[int] = 4
"""The fewest letters a word of a BMU's name needs to count as a place name."""

ADDRESS_COLUMNS: Final[tuple[str, ...]] = (
    "customer_name",
    "customer_site",
    "address_line_1",
    "address_line_2",
    "town",
    "county",
)
"""The ECR columns searched for a BMU's place name."""

REPD_DISTANCE_LIMIT_M: Final[float] = 5000.0
"""The farthest an ECR site may lie from a REPD battery and still be taken as the same site's
neighbour, which places the REPD battery in a licence area."""

REPD_SAME_SITE_M: Final[float] = 1000.0
"""A REPD battery this close to a connected ECR storage row is taken to be that row."""

REPD_BUILT_STATUSES: Final[tuple[str, ...]] = ("Operational",)
"""REPD statuses that count as a battery that exists."""

BmuClassType = Literal[
    "own BMU, FPN submitted",
    "own BMU, no FPN",
    "response provider, no BMU found",
    "no BMU found",
]

BMU_CLASSES: Final[tuple[BmuClassType, ...]] = (
    "own BMU, FPN submitted",
    "own BMU, no FPN",
    "response provider, no BMU found",
    "no BMU found",
)

STORAGE_PATTERN: Final[str] = "(?i)stor"
"""Matches an ECR energy source or technology that names storage."""

CENSUS_DIR: Final[Path] = BATTERY_PV_SEPARATION_DIR.parent / "embedded_battery_forecast"
"""Where the census writes its tables and `census_report.md`."""

REPD_PATH: Final[Path] = SOLAR_BMU_CENSUS_INPUTS_DIR / "raw" / "repd_register.json"
TEC_PATH: Final[Path] = SOLAR_BMU_CENSUS_INPUTS_DIR / "raw" / "tec_register.json"
BMU_LIST_PATH: Final[Path] = BATTERY_PV_SEPARATION_DIR / "bmu_list.csv"
BMU_REFERENCE_PATH: Final[Path] = (
    MARKET_DOWNLOADS_DIR / "elexon_bmu_reference" / ("bmu_reference.parquet")
)
PN_PATH: Final[Path] = MARKET_DOWNLOADS_DIR / "elexon_pn" / "elexon_pn_half_hourly.parquet"
EAC_UNIT_PATH: Final[Path] = MARKET_DOWNLOADS_DIR / "neso_eac_unit_to_bmu" / "eac_unit_to_bmu.csv"

_STORAGE_SLOTS: Final[tuple[tuple[str, str], ...]] = (
    ("energy_source_1", "technology_1"),
    ("energy_source_2", "technology_2"),
    ("energy_source_3", "technology_3"),
)
EMPTY_LABELS: Final[tuple[str, ...]] = ("data not applicable", "data not available")
"""The ECR's text for a cell with no value."""


def licence_area_of(*, text: str) -> str:
    """Return the short licence area name inside the brackets of the ECR's `Licence Area` text.

    Args:
        text: For example `National Grid Electricity Distribution (South West) Plc`.

    Returns:
        The text inside the brackets, such as `South West`.

    Raises:
        ValueError: If the text has no brackets.
    """
    found = re.search(r"\(([^)]+)\)", text)
    if found is None:
        msg = "The licence area text has no bracketed name."
        raise ValueError(msg)
    return found.group(1)


def size_class_of(*, megawatts: float | None) -> SizeClassType | None:
    """Return the size class of an export capacity, or None when the capacity is unknown.

    Args:
        megawatts: The maximum export capacity.

    Returns:
        The class. The edges follow the thresholds that decide whether a battery must be a BMU.
    """
    if megawatts is None or np.isnan(megawatts):
        return None
    return SIZE_CLASSES[int(np.searchsorted(SIZE_CLASS_EDGES_MW, megawatts, side="right"))]


def with_storage_flags(*, ecr: pl.DataFrame) -> pl.DataFrame:
    """Add the columns that say whether a register row lists storage, and what else.

    Args:
        ecr: The frame `ecr.read_ecr` returns.

    Returns:
        The frame with `ecr_row` (its position), `licence_area_short`, `lists_storage` (any of
        the three technology slots is storage), `storage_first` (the first slot is storage),
        `hybrid` (storage and another technology are both listed), `electrochemical` (a slot is a
        battery), `export_mw` (the maximum export capacity, or the connected registered capacity
        where that is missing), and `size_class`.
    """
    storage_flags = [
        pl.col(source).str.contains(STORAGE_PATTERN)
        | pl.col(technology).str.contains(STORAGE_PATTERN)
        for source, technology in _STORAGE_SLOTS
    ]
    filled_flags = [
        ~pl.col(technology).is_in(list(EMPTY_LABELS)) for _, technology in _STORAGE_SLOTS
    ]
    other_flags = [
        filled & ~storage for filled, storage in zip(filled_flags, storage_flags, strict=True)
    ]
    return (
        ecr.with_row_index("ecr_row")
        .with_columns(
            licence_area_short=pl.col("licence_area").map_elements(
                lambda text: licence_area_of(text=text), return_dtype=pl.String
            ),
            lists_storage=pl.any_horizontal(*storage_flags),
            storage_first=storage_flags[0],
            hybrid=pl.any_horizontal(*storage_flags) & pl.any_horizontal(*other_flags),
            electrochemical=pl.any_horizontal(
                *[pl.col(technology).str.contains("(?i)batter") for _, technology in _STORAGE_SLOTS]
            ),
            export_mw=pl.coalesce("maximum_export_capacity_mw", "connected_registered_capacity_mw"),
        )
        .with_columns(
            size_class=pl.col("export_mw").map_elements(
                lambda value: size_class_of(megawatts=value), return_dtype=pl.String
            )
        )
    )


def connected_storage(*, ecr: pl.DataFrame) -> pl.DataFrame:
    """Return the connected rows that list storage, with the storage flags added."""
    return with_storage_flags(ecr=ecr).filter(
        pl.col("lists_storage") & (pl.col("connection_status") == "Connected")
    )


def storage_bmus() -> pl.DataFrame:
    """Return the reviewed storage BMUs with their licence-area group and FPN flag.

    Returns:
        One row per reviewed BMU, with `elexon_bmu_id`, `bmu_name`, `lead_party_name`, `bmu_type`
        (E embedded, T transmission, S and V other), `generation_capacity_mw`, `gsp_group_name`,
        `fpn_flag` (the register's flag that the BMU submits Physical Notifications), `pn_rows` (the
        half-hours of Physical Notification on disk), and `licence_area` (one of `LICENCE_AREAS`, or
        null when the BMU's group is not one of NGED's four).
    """
    bmus = pl.read_csv(BMU_LIST_PATH).select(
        "elexon_bmu_id",
        "bmu_name",
        "lead_party_name",
        "bmu_type",
        "generation_capacity_mw",
        "gsp_group_name",
    )
    flags = pl.read_parquet(BMU_REFERENCE_PATH).select("elexon_bmu_id", "fpn_flag")
    pn_rows = (pl.scan_parquet(PN_PATH).group_by("bmu_id").agg(pn_rows=pl.len()).collect()).rename(
        {"bmu_id": "elexon_bmu_id"}
    )
    area_of_group = {group: area for area, group in GSP_GROUP_OF_LICENCE_AREA.items()}
    return (
        bmus.join(flags, on="elexon_bmu_id", how="left")
        .join(pn_rows, on="elexon_bmu_id", how="left")
        .with_columns(
            pn_rows=pl.col("pn_rows").fill_null(0),
            licence_area=pl.col("gsp_group_name").replace_strict(area_of_group, default=None),
        )
    )


def place_tokens(*, bmu_name: str) -> list[str]:
    """Return the words of a BMU's name that could be a place name.

    Args:
        bmu_name: The registered BMU name, such as `Torquay` or `Berkeley BESS`.

    Returns:
        The lower-case alphabetic words of at least `MIN_PLACE_TOKEN_LENGTH` letters that are not
        generic words (`studies.name_matching.NOISE_WORDS`).
    """
    return [
        word
        for word in re.findall(r"[a-z]+", bmu_name.lower())
        if len(word) >= MIN_PLACE_TOKEN_LENGTH and word not in NOISE_WORDS
    ]


def propose_matches(*, ecr_storage: pl.DataFrame, bmus: pl.DataFrame) -> pl.DataFrame:
    """Propose, for each BMU in NGED's area, the ECR storage rows it could be.

    A candidate is an ECR connected storage row in the BMU's licence area. Three keys are tested:
    the row's customer name or site name is similar to the BMU's name (at the solar census's
    similarity threshold), every place word of the BMU's name appears in the row's address text,
    and the row's export capacity lies within `CAPACITY_TOLERANCE` of the BMU's generation capacity.

    Args:
        ecr_storage: The frame `connected_storage` returns.
        bmus: The frame `storage_bmus` returns.

    Returns:
        One row per BMU and candidate row that satisfies at least one key, with `elexon_bmu_id`,
        `ecr_row`, `name_score` (null when below the threshold), `capacity_ratio` (the row's export
        capacity over the BMU's generation capacity), `name_hit`, `place_hit`, `capacity_hit`, and
        `grade`: `text and capacity` when the name or the place and the capacity agree,
        `text only`, or `capacity only`.
    """
    rows = []
    for bmu in bmus.filter(pl.col("licence_area").is_not_null()).iter_rows(named=True):
        area_rows = ecr_storage.filter(pl.col("licence_area_short") == bmu["licence_area"])
        ids = area_rows["ecr_row"].to_list()
        scores: dict[int, float] = {}
        for column in ("customer_site", "customer_name"):
            found = best_match(
                site_name=bmu["bmu_name"],
                candidates=dict(zip(ids, area_rows[column].to_list(), strict=True)),
            )
            if found is not None:
                scores[int(found[0])] = max(scores.get(int(found[0]), 0.0), found[1])
        tokens = place_tokens(bmu_name=bmu["bmu_name"])
        for row in area_rows.iter_rows(named=True):
            haystack = " ".join(str(row[c] or "") for c in ADDRESS_COLUMNS).lower()
            place_hit = bool(tokens) and all(token in haystack for token in tokens)
            ratio = (
                None
                if row["export_mw"] is None
                else row["export_mw"] / bmu["generation_capacity_mw"]
            )
            capacity_hit = ratio is not None and abs(ratio - 1.0) <= CAPACITY_TOLERANCE
            name_hit = row["ecr_row"] in scores
            if name_hit or place_hit or capacity_hit:
                rows.append(
                    {
                        "elexon_bmu_id": bmu["elexon_bmu_id"],
                        "ecr_row": row["ecr_row"],
                        "name_score": scores.get(row["ecr_row"]),
                        "capacity_ratio": ratio,
                        "name_hit": name_hit,
                        "place_hit": place_hit,
                        "capacity_hit": capacity_hit,
                    }
                )
    schema = {
        "elexon_bmu_id": pl.String,
        "ecr_row": pl.UInt32,
        "name_score": pl.Float64,
        "capacity_ratio": pl.Float64,
        "name_hit": pl.Boolean,
        "place_hit": pl.Boolean,
        "capacity_hit": pl.Boolean,
    }
    text_hit = pl.col("name_hit") | pl.col("place_hit")
    return pl.DataFrame(rows, schema=schema).with_columns(
        grade=pl.when(text_hit & pl.col("capacity_hit"))
        .then(pl.lit("text and capacity"))
        .when(text_hit)
        .then(pl.lit("text only"))
        .otherwise(pl.lit("capacity only"))
    )


def repd_batteries() -> pl.DataFrame:
    """Return the operational batteries in the Renewable Energy Planning Database.

    Returns:
        Columns `repd_site_name`, `repd_capacity_mw`, `easting`, and `northing` (OSGB metres), for
        England and Wales only (NGED's licence areas lie in both).
    """
    body = json.loads(REPD_PATH.read_text())["body"]
    repd = pl.read_csv(io.StringIO(body), infer_schema_length=0)
    return (
        repd.filter(pl.col("Technology Type") == "Battery")
        .filter(pl.col("Development Status (short)").is_in(list(REPD_BUILT_STATUSES)))
        .filter(pl.col("Country").is_in(["England", "Wales"]))
        .select(
            repd_site_name=pl.col("Site Name"),
            repd_capacity_mw=pl.col("Installed Capacity (MWelec)").cast(pl.Float64, strict=False),
            easting=pl.col("X-coordinate").str.replace_all(",", "").cast(pl.Float64, strict=False),
            northing=pl.col("Y-coordinate").str.replace_all(",", "").cast(pl.Float64, strict=False),
        )
        .drop_nulls(["easting", "northing"])
        .with_row_index("repd_row")
    )


def nearest_ecr_row(
    *, repd: pl.DataFrame, ecr_rows: pl.DataFrame, limit_m: float = REPD_DISTANCE_LIMIT_M
) -> pl.DataFrame:
    """Find, for each REPD battery, the nearest ECR row and its distance.

    Args:
        repd: The frame `repd_batteries` returns.
        ecr_rows: ECR rows with `ecr_row`, `easting`, and `northing`.
        limit_m: The farthest distance, in metres, at which a row counts.

    Returns:
        `repd` with `nearest_ecr_row` and `distance_m`, both null where no ECR row lies within
        `limit_m`.
    """
    ecr_xy = np.column_stack([ecr_rows["easting"].to_numpy(), ecr_rows["northing"].to_numpy()])
    ids = ecr_rows["ecr_row"].to_list()
    nearest: list[int | None] = []
    distances: list[float | None] = []
    for easting, northing in zip(
        repd["easting"].to_list(), repd["northing"].to_list(), strict=True
    ):
        distance = np.hypot(ecr_xy[:, 0] - easting, ecr_xy[:, 1] - northing)
        distance = np.where(np.isnan(distance), np.inf, distance)
        best = int(np.argmin(distance)) if distance.size else -1
        if best >= 0 and distance[best] <= limit_m:
            nearest.append(ids[best])
            distances.append(float(distance[best]))
        else:
            nearest.append(None)
            distances.append(None)
    return repd.with_columns(
        nearest_ecr_row=pl.Series(nearest, dtype=pl.UInt32),
        distance_m=pl.Series(distances, dtype=pl.Float64),
    )


def propose_repd_route(*, bmus: pl.DataFrame, repd_located: pl.DataFrame) -> pl.DataFrame:
    """Match each BMU to a REPD battery by name, and so to a place and an ECR row.

    Args:
        bmus: The frame `storage_bmus` returns.
        repd_located: The frame `nearest_ecr_row` returns.

    Returns:
        One row per BMU matched to a REPD battery by name at the similarity threshold, with
        `elexon_bmu_id`, `repd_row`, `name_score`, `capacity_ratio` (REPD over BMU capacity),
        `nearest_ecr_row`, and `distance_m`.
    """
    names = dict(
        zip(
            repd_located["repd_row"].to_list(),
            repd_located["repd_site_name"].to_list(),
            strict=True,
        )
    )
    by_row = {row["repd_row"]: row for row in repd_located.iter_rows(named=True)}
    rows = []
    for bmu in bmus.iter_rows(named=True):
        found = best_match(site_name=bmu["bmu_name"], candidates=names)
        if found is None:
            continue
        repd_row = by_row[int(found[0])]
        capacity = repd_row["repd_capacity_mw"]
        rows.append(
            {
                "elexon_bmu_id": bmu["elexon_bmu_id"],
                "repd_row": repd_row["repd_row"],
                "name_score": found[1],
                "capacity_ratio": None
                if capacity is None
                else capacity / bmu["generation_capacity_mw"],
                "nearest_ecr_row": repd_row["nearest_ecr_row"],
                "distance_m": repd_row["distance_m"],
            }
        )
    schema = {
        "elexon_bmu_id": pl.String,
        "repd_row": pl.UInt32,
        "name_score": pl.Float64,
        "capacity_ratio": pl.Float64,
        "nearest_ecr_row": pl.UInt32,
        "distance_m": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def response_participants() -> frozenset[str]:
    """Return the NESO auction participants that hold a battery unit with no BMU.

    Returns:
        The participants' names, upper-cased. An ECR customer with the same name may be selling
        frequency response as a non-BM unit; the register cannot say which of the participant's
        sites.
    """
    units = pl.read_csv(EAC_UNIT_PATH)
    held = units.filter(
        (pl.col("technology_type") == "Batteries")
        & pl.col("services").str.contains("Response")
        & pl.col("candidate_bmu_id").is_null()
    )
    return frozenset(held["auction_participant"].drop_nulls().str.to_uppercase().to_list())


def classify_connected_storage(
    *,
    ecr_storage: pl.DataFrame,
    accepted_matches: pl.DataFrame,
    bmus: pl.DataFrame,
    participants: frozenset[str],
) -> pl.DataFrame:
    """Give each connected ECR storage row one of the four BMU classes.

    Args:
        ecr_storage: The frame `connected_storage` returns.
        accepted_matches: Rows of `elexon_bmu_id` and `ecr_row`: the matches the census accepts.
        bmus: The frame `storage_bmus` returns.
        participants: The output of `response_participants`.

    Returns:
        `ecr_storage` with `bmu_class` and `elexon_bmu_id` (null unless a BMU was matched). A row
        with more than one accepted BMU takes the first by identifier.
    """
    matched = (
        accepted_matches.join(
            bmus.select("elexon_bmu_id", "fpn_flag", "pn_rows"), on="elexon_bmu_id"
        )
        .sort("elexon_bmu_id")
        .unique("ecr_row", keep="first", maintain_order=True)
    )
    joined = ecr_storage.join(matched, on="ecr_row", how="left")
    return joined.with_columns(
        bmu_class=pl.when(pl.col("elexon_bmu_id").is_not_null() & (pl.col("pn_rows") > 0))
        .then(pl.lit("own BMU, FPN submitted"))
        .when(pl.col("elexon_bmu_id").is_not_null())
        .then(pl.lit("own BMU, no FPN"))
        .when(pl.col("customer_name").str.to_uppercase().is_in(list(participants)))
        .then(pl.lit("response provider, no BMU found"))
        .otherwise(pl.lit("no BMU found"))
    ).drop("fpn_flag", "pn_rows")


def bmu_share_band(*, classified: pl.DataFrame, recall: float) -> dict[str, float]:
    """Return the BMU share by count and by megawatts, as a lower and an upper bound.

    The lower bound counts only the matched batteries. The upper bound adds the batteries without
    a BMU (with or without a response contract), multiplied by the miss rate (one minus `recall`).

    Args:
        classified: The frame `classify_connected_storage` returns.
        recall: The share of known BMUs that the matching finds, from 0 to 1.

    Returns:
        `count_low`, `count_high`, `megawatts_low`, and `megawatts_high`, each a share from 0 to 1.
    """
    is_bmu = classified["bmu_class"].is_in(["own BMU, FPN submitted", "own BMU, no FPN"])
    unknown = classified["bmu_class"].is_in(["no BMU found", "response provider, no BMU found"])
    weight = classified["export_mw"].fill_null(0.0)
    miss = 1.0 - recall
    n = classified.height
    bmu_count = float(is_bmu.sum())
    unknown_count = float(unknown.sum())
    bmu_mw = float(weight.filter(is_bmu).sum())
    unknown_mw = float(weight.filter(unknown).sum())
    total_mw = float(weight.sum())
    return {
        "count_low": bmu_count / n,
        "count_high": (bmu_count + miss * unknown_count) / n,
        "megawatts_low": bmu_mw / total_mw,
        "megawatts_high": (bmu_mw + miss * unknown_mw) / total_mw,
    }
