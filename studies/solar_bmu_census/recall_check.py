"""Check the census against the TEC register's PV sites: which has a BMU in the census?

A transmission-connected solar site must hold Transmission Entry Capacity (TEC), so the register's
PV rows are a short, complete list to measure recall against. A TEC row and a BMU match
many-to-many, and a BMU's lead party is a trading party rather than the TEC customer, so the match
cannot be automated. The script writes a draft mapping from TEC Project ID to BMU identifiers to the
study folder, and reads the reviewed mapping beside it when that file exists. Run:
`uv run python studies/solar_bmu_census/recall_check.py`.
"""

from pathlib import Path
from typing import Any, Final, Literal

import polars as pl
from collate import PV_PLANT_TYPE, best_match, technology_from_tec_plant_type
from fetch_sources import STUDY_DIR, fetch_bmu_reference, fetch_tec

DRAFT_PATH: Final[Path] = STUDY_DIR / "tec_mapping_draft.csv"
REVIEWED_PATH: Final[Path] = Path(__file__).parent / "tec_mapping_reviewed.csv"
"""The hand-reviewed mapping from TEC Project ID to BMU identifiers, kept beside this script."""
RECALL_STATUSES: Final[tuple[str, ...]] = ("Built", "Under Construction/Commissioning")
SOLAR_NAME_HINTS: Final[tuple[str, ...]] = ("solar", "pv", "photovoltaic")
OutcomeType = Literal["no BMU identified", "BMU identified, not in census", "census BMU identified"]


def tec_solar_rows(*, tec: pl.DataFrame) -> pl.DataFrame:
    """Return the TEC rows to check: PV in the plant type, or a solar name on a storage-only row.

    Only rows that are Built or Under Construction/Commissioning are kept, because earlier stages
    have no BMU yet.
    """
    in_scope = tec.filter(pl.col("Project Status").is_in(RECALL_STATUSES))
    has_pv = pl.col("Plant Type").str.contains(PV_PLANT_TYPE)
    solar_name = pl.col("Project Name").str.to_lowercase().str.contains("|".join(SOLAR_NAME_HINTS))
    return in_scope.filter(has_pv | solar_name)


def draft_mapping(*, tec_rows: pl.DataFrame, reference: list[dict[str, Any]]) -> pl.DataFrame:
    """Draft a mapping from each TEC row to the BMUs whose names match its project name.

    A draft is a starting point for a hand check, never the final mapping.
    """
    names = {str(r["elexonBmUnit"]): str(r["bmUnitName"]) for r in reference if r["bmUnitName"]}
    rows = []
    for row in tec_rows.iter_rows(named=True):
        project = str(row["Project Name"])
        matched = sorted(
            bmu_id
            for bmu_id, name in names.items()
            if best_match(site_name=project, candidates={bmu_id: name}) is not None
        )
        rows.append(
            {
                "project_id": row["Project ID"],
                "project_name": project,
                "project_status": row["Project Status"],
                "plant_type": row["Plant Type"],
                "bmu_ids": ";".join(matched),
            }
        )
    return pl.DataFrame(rows)


def load_mapping(
    *, tec_rows: pl.DataFrame, reference: list[dict[str, Any]]
) -> tuple[pl.DataFrame, str]:
    """Return the mapping to use and its provenance: the reviewed file, else a fresh draft."""
    if REVIEWED_PATH.exists():
        return pl.read_csv(REVIEWED_PATH, infer_schema_length=0), "reviewed by hand"
    draft = draft_mapping(tec_rows=tec_rows, reference=reference)
    draft.write_csv(DRAFT_PATH)
    return draft, "unreviewed draft"


def outcome_for(*, bmu_ids: str, solar_ids: set[str]) -> OutcomeType:
    """Say what the census found for one TEC row, from its mapped BMUs.

    Args:
        bmu_ids: The row's mapped BMU identifiers, separated by semicolons, or empty.
        solar_ids: The BMUs in the census.

    Returns:
        `no BMU identified` when the row maps to no BMU, `BMU identified, not in census` when it
        maps only to BMUs outside the census, and `census BMU identified` when any mapped BMU is in
        it.
    """
    mapped = [part for part in (bmu_ids or "").split(";") if part]
    if not mapped:
        return "no BMU identified"
    in_census = any(m in solar_ids for m in mapped)
    return "census BMU identified" if in_census else "BMU identified, not in census"


def recall_table(*, today_solar_ids: set[str]) -> tuple[pl.DataFrame, str]:
    """Return one row per checked TEC row with its technology and outcome, and the provenance."""
    tec_rows = tec_solar_rows(tec=fetch_tec())
    mapping, provenance = load_mapping(tec_rows=tec_rows, reference=fetch_bmu_reference())
    joined = tec_rows.select("Project ID", "Project Status", "Plant Type", "Agreement Type").join(
        mapping.select(pl.col("project_id").alias("Project ID"), "bmu_ids"),
        on="Project ID",
        how="left",
    )
    table = joined.with_columns(
        technology=pl.col("Plant Type").map_elements(
            lambda plant_type: technology_from_tec_plant_type(plant_type=plant_type),
            return_dtype=pl.String,
        ),
        census_bmus=pl.col("bmu_ids").map_elements(
            lambda ids: sum(m in today_solar_ids for m in (ids or "").split(";") if m),
            return_dtype=pl.Int64,
        ),
        other_bmus=pl.col("bmu_ids").map_elements(
            lambda ids: sum(m not in today_solar_ids for m in (ids or "").split(";") if m),
            return_dtype=pl.Int64,
        ),
        outcome=pl.struct("bmu_ids").map_elements(
            lambda row: outcome_for(bmu_ids=row["bmu_ids"], solar_ids=today_solar_ids),
            return_dtype=pl.String,
        ),
    )
    return table, provenance


def main() -> None:
    """Print the recall table's outcome counts."""
    solar_ids = set(
        pl.read_parquet(STUDY_DIR / "classes.parquet").filter("is_solar")["elexon_bmu_id"]
    )
    table, provenance = recall_table(today_solar_ids=solar_ids)
    print(f"Mapping: {provenance}")
    print(table.group_by("Project Status", "technology", "outcome").len().sort("Project Status"))


if __name__ == "__main__":
    main()
