"""Put every published capacity figure for each solar BMU side by side, one row per BMU.

The figures mean different things and are never merged or added across columns. Run after
`fetch_sources.py` and `classify.py`: `uv run python studies/solar_bmu_census/collate.py`. The
table is written beside the other study outputs.
"""

import difflib
import re
from functools import cache
from pathlib import Path
from typing import Any, Final, Literal

import polars as pl
from fetch_sources import (
    STUDY_DIR,
    fetch_bmu_reference,
    fetch_igcpu,
    fetch_mels,
    fetch_repd,
    fetch_tec,
    recorded_run,
)
from pyproj import Transformer

MATCH_THRESHOLD: Final[float] = 0.85
"""The lowest name-similarity ratio at which a site name counts as a match.

At 0.8 a REPD row for a different farm with a one-letter-different name matched; at 0.85 it did not.
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
operational or under construction is a candidate match, and so is a battery that shares its name.
"""
REVIEWED_MATCHES_PATH: Final[Path] = Path(__file__).parent / "site_matches_reviewed.csv"
"""The hand-reviewed table, kept beside this script, of BMU to TEC project and REPD reference.

Columns `elexon_bmu_id`, `tec_project_id`, `repd_ref_id` (either may be blank), and `note`. A row
replaces the name match for its BMU, for the BMUs whose Elexon name does not resemble the site's
name in either register.
"""
TEC_STATUS_ORDER: Final[tuple[str, ...]] = (
    "Built",
    "Under Construction/Commissioning",
    "Consents Approved",
    "Awaiting Consents",
    "Scoping",
)
"""TEC project statuses from the most to the least advanced, which picks one row per project."""
PV_PLANT_TYPE: Final[str] = "PV Array"
NON_GENERATING_PLANT_TYPES: Final[tuple[str, ...]] = ("Demand", "Reactive Compensation")
"""TEC plant types that do not make a PV site a hybrid: its own demand and reactive plant."""
SINGLE_SITE_PREFIXES: Final[tuple[str, ...]] = ("T_", "E_", "M_")
"""A BMU with one of these prefixes is one generating site. `2_` (supplier), `V_`, and `C_` BMUs
can aggregate many sites."""
CAPACITY_COLUMNS: Final[tuple[str, ...]] = (
    "generation_capacity_mw",
    "igcpu_installed_capacity_mw",
    "tec_mw",
    "largest_mel_mw",
    "repd_installed_capacity_mw",
)
TechnologyType = Literal["pure PV", "hybrid", "unknown"]


def normalise(name: str) -> str:
    """Lower-case a site name and drop punctuation, spaces, and generic words.

    Whole generic words are dropped first. Long generic words are then also dropped inside a longer
    token, so "Energyfarm" and "Energy Farm" normalise alike.
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
        The best candidate at or above `MATCH_THRESHOLD`, or None when no candidate reaches it. A
        tie goes to the key that sorts first, so the result does not depend on dict order.
    """
    target = normalise(site_name)
    if not target:
        return None
    best: tuple[str, float] | None = None
    for key in sorted(candidates):
        score = difflib.SequenceMatcher(None, target, normalise(candidates[key])).ratio()
        if score >= MATCH_THRESHOLD and (best is None or score > best[1]):
            best = (key, round(score, 2))
    return best


def technology_from_tec_plant_type(*, plant_type: str) -> TechnologyType:
    """Say whether a TEC `Plant Type` string describes a pure PV site or a hybrid.

    Args:
        plant_type: The register's semicolon-separated technologies, such as
            `Energy Storage System;PV Array (Photo Voltaic/solar)`.

    Returns:
        `hybrid` if the site lists any generating or storage technology besides PV, `pure PV` if
        PV is the only one, and `unknown` if PV is not listed at all.
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


def connection_type(*, elexon_bmu_id: str) -> str:
    """Classify a BMU's connection from its identifier prefix."""
    if elexon_bmu_id.startswith("T_"):
        return "transmission-connected"
    if elexon_bmu_id.startswith("E_"):
        return "embedded"
    return "other"


def _display_name(
    *, site_name: str, bmu_id: str, tec_name: str | None, repd_name: str | None
) -> str:
    """Return the name to show for a BMU: its register name, or its project's name if it has none.

    A few BMUs carry their own identifier as their register name, which says nothing to a reader,
    so the matched TEC project's name, or else the matched REPD site's name, stands in.
    """
    if site_name not in {"", bmu_id, "None"}:
        return site_name
    return tec_name or repd_name or bmu_id


def _reviewed_matches() -> dict[str, tuple[str | None, str | None]]:
    """Read the hand-reviewed BMU matches."""
    table = pl.read_csv(REVIEWED_MATCHES_PATH, infer_schema_length=0)
    return {
        row["elexon_bmu_id"]: (row["tec_project_id"] or None, row["repd_ref_id"] or None)
        for row in table.iter_rows(named=True)
    }


def _repd_position(*, repd_row: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """Return a REPD row's longitude and latitude, or None for a missing or blank position."""
    if repd_row is None:
        return None, None
    easting = _to_float(repd_row["X-coordinate"])
    northing = _to_float(repd_row["Y-coordinate"])
    if easting is None or northing is None:
        return None, None
    return osgb_to_lon_lat(easting_m=easting, northing_m=northing)


def best_tec_rows(*, tec: pl.DataFrame) -> pl.DataFrame:
    """Return one row for each TEC project that lists PV: the row at its most advanced status.

    The register holds one row for each stage of a project, and a later stage carries a cumulative
    capacity that includes capacity not yet built. A project built at 99.4 MW with a later 20.6 MW
    increase has a second row of 120 MW, so the study takes the row at the most advanced status.

    Args:
        tec: The TEC register with all-string columns.

    Returns:
        The PV rows, one for each `Project ID`.
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
        .drop("status_rank")
    )


def _tec_candidates(*, tec: pl.DataFrame) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Return the names and the rows of the TEC register's PV projects, keyed by Project ID."""
    rows = {str(row["Project ID"]): row for row in best_tec_rows(tec=tec).iter_rows(named=True)}
    return {key: str(row["Project Name"]) for key, row in rows.items()}, rows


def _repd_candidates(*, repd: pl.DataFrame) -> tuple[dict[str, str], pl.DataFrame]:
    """Return the built solar projects' names, keyed by Ref ID, and the built batteries.

    The battery rows are the evidence that a solar project shares its site with storage.
    """
    built = repd.filter(pl.col("Development Status (short)").is_in(REPD_BUILT_STATUSES))
    solar = built.filter(pl.col("Technology Type") == "Solar Photovoltaics")
    batteries = built.filter(pl.col("Technology Type") == "Battery")
    return {str(r["Ref ID"]): str(r["Site Name"]) for r in solar.iter_rows(named=True)}, batteries


def build_table() -> pl.DataFrame:
    """Build the one-row-per-solar-BMU table of capacity figures.

    The run date and window come from `fetch_sources.py`'s `lineage.json`, and the classes from
    `classify.py`.

    Returns:
        One row per BMU in the census (IGCPU type Solar, or output that follows the sun), with the
        BMU's five capacity figures, the matched TEC project and REPD reference, the BMU's
        technology (pure PV, hybrid, or unknown) and the evidence for it, and the join methods and
        match scores.
    """
    reference = {str(r["elexonBmUnit"]): r for r in fetch_bmu_reference() if r["elexonBmUnit"]}
    today, _ = recorded_run()
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
    repd_names, repd_batteries = _repd_candidates(repd=repd)
    repd_rows = {str(r["Ref ID"]): r for r in repd.iter_rows(named=True)}
    battery_names = {normalise(str(n)) for n in repd_batteries["Site Name"].to_list()}
    reviewed = _reviewed_matches()

    out = []
    for class_row in classes.iter_rows(named=True):
        bmu_id = class_row["elexon_bmu_id"]
        ref = reference.get(bmu_id, {})
        igcpu_row = latest_igcpu.get(bmu_id)
        site_name = str(ref.get("bmUnitName") or (igcpu_row or {}).get("registeredResourceName"))
        tec_match = best_match(site_name=site_name, candidates=tec_names)
        repd_match = best_match(site_name=site_name, candidates=repd_names)
        hand = reviewed.get(bmu_id)
        if hand is not None:
            tec_match = (hand[0], 1.0) if hand[0] else tec_match
            repd_match = (hand[1], 1.0) if hand[1] else repd_match
        technology: TechnologyType = "unknown"
        evidence = "none"
        if tec_match:
            tec_row = tec_rows[tec_match[0]]
            technology = technology_from_tec_plant_type(plant_type=str(tec_row["Plant Type"]))
            evidence = "TEC plant type"
        if technology == "unknown" and repd_match:
            has_battery = normalise(repd_rows[repd_match[0]]["Site Name"]) in battery_names
            technology = "hybrid" if has_battery else "pure PV"
            evidence = "REPD battery row" if has_battery else "REPD solar row, no battery row"
        longitude, latitude = _repd_position(
            repd_row=repd_rows[repd_match[0]] if repd_match else None
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
                "lead_party": ref.get("leadPartyName"),
                "connection_type": connection_type(elexon_bmu_id=bmu_id),
                "scope": "single-site" if bmu_id.startswith(SINGLE_SITE_PREFIXES) else "aggregate",
                "basis": class_row["basis"],
                "correlation": class_row["correlation"],
                "technology": technology,
                "technology_evidence": evidence,
                "generation_capacity_mw": _to_float(ref.get("generationCapacity")),
                "igcpu_installed_capacity_mw": igcpu_row["installedCapacity"]
                if igcpu_row
                else None,
                "tec_project_id": tec_match[0] if tec_match else None,
                "tec_mw": _to_float(tec_rows[tec_match[0]]["Cumulative Total Capacity (MW)"])
                if tec_match
                else None,
                "largest_mel_mw": mel.get(bmu_id),
                "repd_ref_id": repd_match[0] if repd_match else None,
                "repd_installed_capacity_mw": _to_float(
                    repd_rows[repd_match[0]]["Installed Capacity (MWelec)"]
                )
                if repd_match
                else None,
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
            "repd_installed_capacity_mw": pl.Float64,
            "generation_capacity_mw": pl.Float64,
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
