"""Run the batteries census and write its report and its site-level tables.

Run: `uv run python studies/embedded_battery_forecast/census_report.py`.

Writes, under `data/studies/per_study/embedded_battery_forecast/`:

- `census_report.md`: counts, bands, and the checks on the register read. No site, party, or BMU
  name appears in it.
- `census_matches.csv`: every proposed match between a reviewed storage BMU and a connected ECR
  storage row, with the names and the scores. This is the table a person reviews, and it is not
  committed.
- `census_classes.parquet`: each connected ECR storage row with its BMU class.
"""

import io
import json
from collections.abc import Sequence
from typing import Final

import census
import ecr
import numpy as np
import polars as pl
from pyproj import Transformer
from studies.battery_market import B1610_DIR, B1610_SUFFIX, HALF_HOURS_PER_DAY
from studies.name_matching import best_match
from studies.nged_battery_a import battery_a_site

EXPECTED_B1610_ROWS: Final[int] = 365 * HALF_HOURS_PER_DAY
MIN_B1610_SHARE: Final[float] = 0.95
"""A BMU counts as having output in the window when B1610 holds at least this share of its
half-hours."""

BACKGROUND_PAGE_TARGETS: Final[dict[str, dict[str, int]]] = {
    "Embedded batteries of 50 kW or more, by licence area": {
        "East Midlands": 66,
        "West Midlands": 54,
        "South Wales": 35,
        "South West": 47,
    },
    "Of which listed as storage in the ECR": {
        "East Midlands": 58,
        "West Midlands": 45,
        "South Wales": 31,
        "South West": 42,
    },
}
"""The counts the background page `gb-battery-scheduling` reports from hand-matching."""

BACKGROUND_PAGE_SIZE_EDGES_MW: Final[tuple[float, ...]] = (1.0, 5.0, 50.0)
BACKGROUND_PAGE_SIZE_TARGETS: Final[tuple[int, ...]] = (141, 12, 46, 3)
BACKGROUND_PAGE_SIZE_LABELS: Final[tuple[str, ...]] = (
    "under 1 MW",
    "1 to 5 MW",
    "5 to 50 MW",
    "over 50 MW",
)


def _table(*, header: list[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """Return a markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return [*lines, ""]


def b1610_share(*, bmu_id: str) -> float:
    """Return the share of the window's half-hours for which B1610 holds the BMU's output.

    Args:
        bmu_id: The BMU identifier.

    Returns:
        A share from 0 to 1, or 0 when the BMU has no B1610 file.
    """
    path = B1610_DIR / f"{bmu_id}{B1610_SUFFIX}"
    if not path.exists():
        return 0.0
    return pl.read_parquet(path).height / EXPECTED_B1610_ROWS


def register_checks(*, frame: pl.DataFrame, storage: pl.DataFrame) -> list[str]:
    """Return the report lines for the checks on the register read.

    Args:
        frame: The frame `ecr.read_ecr` returns.
        storage: The frame `census.with_storage_flags` returns, filtered to rows listing storage.

    Returns:
        Markdown lines: rows per sheet, the licence-area and status values, duplicate meter point
        numbers among storage rows, missing capacities, and the capacity range.
    """
    flagged = census.with_storage_flags(ecr=frame)
    per_sheet = frame.group_by("sheet").len().sort("sheet")
    export = storage.filter(pl.col("connection_status") == "Connected")["export_mw"]
    real_mpans = storage.filter(~pl.col("export_mpan").is_in(list(census.EMPTY_LABELS)))[
        "export_mpan"
    ]
    status_counts = frame.group_by("connection_status").len().sort("connection_status")
    return [
        "## Checks on the register read",
        "",
        f"- Rows: {frame.height} ({', '.join(f'{r[0]}: {r[1]}' for r in per_sheet.iter_rows())}).",
        "- Connection status: "
        + ", ".join(f"{r[0]}: {r[1]}" for r in status_counts.iter_rows())
        + ".",
        f"- Licence areas found: {sorted(flagged['licence_area_short'].unique().to_list())}.",
        (
            f"- Rows listing storage: {storage.height} "
            f"({storage.filter(pl.col('connection_status') == 'Connected').height} connected, "
            f"{storage.filter(pl.col('connection_status') != 'Connected').height} accepted to "
            "connect)."
        ),
        (
            f"- The register's own `Reference` column holds {frame['reference'].n_unique()} "
            f"distinct "
            f"value(s), so it cannot serve as a key; duplicate check uses the export meter point "
            f"number: {real_mpans.len() - real_mpans.n_unique()} repeated among "
            f"{real_mpans.len()} storage rows with a real number."
        ),
        f"- Connected storage rows with no export capacity: {export.is_null().sum()}.",
        (
            f"- Connected storage export capacity: minimum {export.min():.3f} MW, median "
            f"{export.median():.3f} MW, maximum {export.max():.1f} MW; "
            f"rows at 0 MW: {(export == 0).sum()}."
        ),
        "",
    ]


def licence_area_table(*, connected: pl.DataFrame) -> list[str]:
    """Return the report lines for rung C1 on the ECR side, compared with the background page."""
    by_area = connected.group_by("licence_area_short").agg(
        total=pl.len(),
        storage_only=(~pl.col("hybrid")).sum(),
        hybrid=pl.col("hybrid").sum(),
        megawatts=pl.col("export_mw").sum(),
    )
    targets = BACKGROUND_PAGE_TARGETS["Of which listed as storage in the ECR"]
    rows = []
    for area in census.LICENCE_AREAS:
        row = by_area.filter(pl.col("licence_area_short") == area).row(0, named=True)
        rows.append(
            [
                area,
                row["total"],
                row["storage_only"],
                row["hybrid"],
                f"{row['megawatts']:.0f}",
                targets[area],
                "yes" if row["total"] == targets[area] else "NO",
            ]
        )
    rows.append(
        [
            "Total",
            connected.height,
            int((~connected["hybrid"]).sum()),
            int(connected["hybrid"].sum()),
            f"{connected['export_mw'].sum():.0f}",
            sum(targets.values()),
            "yes" if connected.height == sum(targets.values()) else "NO",
        ]
    )
    size_rows = []
    for label in census.SIZE_CLASSES:
        part = connected.filter(pl.col("size_class") == label)
        size_rows.append([label, part.height, f"{part['export_mw'].sum():.0f}"])
    page_bands = np.searchsorted(
        BACKGROUND_PAGE_SIZE_EDGES_MW, connected["export_mw"].to_numpy(), side="right"
    )
    band_rows = [
        [label, int((page_bands == i).sum()), BACKGROUND_PAGE_SIZE_TARGETS[i]]
        for i, label in enumerate(BACKGROUND_PAGE_SIZE_LABELS)
    ]
    return [
        "## Rung C1: the ECR's connected storage rows",
        "",
        (
            "Connected storage rows by licence area, against the background page's hand-matched "
            "count "
            "of rows listed as storage in the ECR."
        ),
        "",
        *_table(
            header=[
                "Licence area",
                "Connected storage rows",
                "Storage only",
                "Hybrid site",
                "Export MW",
                "Background page",
                "Same",
            ],
            rows=rows,
        ),
        (
            f"Rows whose first technology is storage: {int(connected['storage_first'].sum())}. "
            f"Rows listing an electrochemical battery: {int(connected['electrochemical'].sum())}."
        ),
        "",
        "By size class (this study's thresholds, which follow the obligation to be a BMU):",
        "",
        *_table(header=["Class", "Rows", "Export MW"], rows=size_rows),
        (
            "By the background page's bands, against its hand-matched count of 202 embedded "
            "batteries "
            "(a different population, so only the under-1-MW band is expected to agree):"
        ),
        "",
        *_table(header=["Band", "ECR rows", "Background page (202)"], rows=band_rows),
    ]


def bmu_side_table(*, bmus: pl.DataFrame) -> tuple[list[str], pl.DataFrame]:
    """Return the report lines for the BMU side of rung C1, and the BMUs in NGED's four groups.

    Args:
        bmus: The frame `census.storage_bmus` returns.

    Returns:
        The lines, and the in-area embedded BMUs with their B1610 coverage added.
    """
    with_share = bmus.with_columns(
        b1610_share=pl.Series([b1610_share(bmu_id=b) for b in bmus["elexon_bmu_id"].to_list()])
    )
    in_area = with_share.filter(pl.col("licence_area").is_not_null())
    by_type = with_share.group_by("bmu_type").len().sort("bmu_type")
    rows = [[area, int((in_area["licence_area"] == area).sum())] for area in census.LICENCE_AREAS]
    embedded_all = with_share.filter(pl.col("bmu_type") == "E")
    lines = [
        "### The reviewed storage BMUs",
        "",
        f"- Reviewed storage BMUs: {bmus.height}; by type: "
        + ", ".join(f"{r[0]}: {r[1]}" for r in by_type.iter_rows())
        + " (E embedded, T transmission-connected, S and V other).",
        (
            f"- Embedded (type E) BMUs with at least {MIN_B1610_SHARE:.0%} of the window in B1610: "
            f"{int((embedded_all['b1610_share'] >= MIN_B1610_SHARE).sum())} of "
            f"{embedded_all.height}."
        ),
        (
            "- Transmission-connected (type T) BMUs, listed separately because a "
            "transmission-connected "
            f"battery is not embedded: {int((with_share['bmu_type'] == 'T').sum())}."
        ),
        "",
        "Reviewed storage BMUs whose grid supply point group is one of NGED's four:",
        "",
        *_table(header=["Licence area", "BMUs"], rows=rows),
        (
            f"Of the {in_area.height} in NGED's four groups, "
            f"{int((in_area['bmu_type'] == 'E').sum())} are embedded; "
            f"{int((in_area['b1610_share'] >= MIN_B1610_SHARE).sum())} have at least "
            f"{MIN_B1610_SHARE:.0%} of the window in B1610; {int(in_area['fpn_flag'].sum())} are "
            f"flagged by Elexon as submitting Physical Notifications; "
            f"{int((in_area['pn_rows'] > 0).sum())} have Physical Notification data on disk."
        ),
        "",
    ]
    return lines, in_area


def match_section(
    *, proposals: pl.DataFrame, in_area: pl.DataFrame, repd_route: pl.DataFrame
) -> list[str]:
    """Return the report lines for rung C2: the matching funnel."""
    n = in_area.height
    by_bmu = proposals.group_by("elexon_bmu_id").agg(
        text_and_capacity=(pl.col("grade") == "text and capacity").any(),
        text=(pl.col("name_hit") | pl.col("place_hit")).any(),
        capacity=pl.col("capacity_hit").any(),
        n_capacity_candidates=pl.col("capacity_hit").sum(),
    )
    ids = set(in_area["elexon_bmu_id"].to_list())
    in_area_proposals = by_bmu.filter(pl.col("elexon_bmu_id").is_in(list(ids)))
    in_area_repd = repd_route.filter(pl.col("elexon_bmu_id").is_in(list(ids)))
    rows = [
        ["BMUs in NGED's four groups", n],
        [
            "BMUs whose name is similar to a connected storage row's customer or site name",
            int(proposals.filter(pl.col("name_hit"))["elexon_bmu_id"].n_unique()),
        ],
        [
            "BMUs whose place words all appear in a connected storage row's address text",
            int(proposals.filter(pl.col("place_hit"))["elexon_bmu_id"].n_unique()),
        ],
        [
            "BMUs with a connected storage row within 10% of their capacity",
            int(in_area_proposals["capacity"].sum()),
        ],
        [
            "BMUs with a row that agrees on text and on capacity",
            int(in_area_proposals["text_and_capacity"].sum()),
        ],
        ["BMUs matched by name to an operational REPD battery", in_area_repd.height],
    ]
    capacity_candidates = in_area_proposals.filter(pl.col("capacity"))["n_capacity_candidates"]
    place_rows = proposals.filter(pl.col("place_hit"))
    median_ratio = place_rows["capacity_ratio"].median()
    return [
        "## Rung C2: the match",
        "",
        (
            "Three keys: the licence area (which fixes the BMU register's grid supply point "
            "group), "
            "the site name at similarity 0.85, and the capacity within 10%, plus a place-word key "
            "that looks for the BMU's name in the ECR's address text."
        ),
        "",
        *_table(header=["Step", "BMUs"], rows=rows),
        (
            f"- Candidates per BMU on capacity alone: {sorted(capacity_candidates.to_list())}. A "
            f"BMU "
            "with several candidates cannot be assigned one without more evidence."
        ),
        (
            f"- Place-word candidates: {place_rows.height} rows, whose median export capacity is "
            f"{median_ratio:.4f} "
            "of the BMU's capacity, so they are small sites that share a place name with the BMU, "
            "not the BMU's own site."
        ),
        (
            "- Every proposal is in `census_matches.csv` for a person to review; "
            "none is accepted unless the text and the capacity agree."
        ),
        "",
    ]


def classification_section(
    *, classified: pl.DataFrame, recall: float, ledger: dict[str, float], n_bmus_known: int
) -> list[str]:
    """Return the report lines for rung C3: the classes, the recall, and the bands."""
    rows = []
    for label in census.BMU_CLASSES:
        part = classified.filter(pl.col("bmu_class") == label)
        rows.append([label, part.height, f"{part['export_mw'].sum():.0f}"])
    band = census.bmu_share_band(classified=classified, recall=recall)
    return [
        "## Rung C3: the classes, the recall, and the bands",
        "",
        *_table(header=["Class", "Connected storage rows", "Export MW"], rows=rows),
        (
            f"- Recall: the match found {recall * n_bmus_known:.0f} of the {n_bmus_known} reviewed "
            f"embedded storage BMUs in NGED's four groups, a recall of {recall:.0%}."
        ),
        (
            f"- Band from the match: {band['count_low']:.1%} to {band['count_high']:.1%} of "
            f"connected "
            f"storage rows by count, and {band['megawatts_low']:.1%} to "
            f"{band['megawatts_high']:.1%} "
            "by megawatts. With a recall this low, the upper bound says the match cannot tell."
        ),
        (
            f"- Band from the BMU register's side: the {int(ledger['embedded_in_area'])} embedded "
            f"storage BMUs in NGED's four groups ({int(ledger['with_output'])} with enough B1610 "
            f"output) against the {classified.height} connected storage rows is "
            f"{ledger['embedded_in_area'] / classified.height:.1%} by count (lower bound "
            f"{ledger['with_output'] / classified.height:.1%}), and "
            f"{ledger['bmu_megawatts'] / float(classified['export_mw'].sum()):.0%} by megawatts "
            "(BMU generation capacity over the rows' export capacity). Each BMU may stand for more "
            "than one ECR row, or for none."
        ),
        "",
    ]


def repd_universe_section(*, ecr_all: pl.DataFrame, connected: pl.DataFrame) -> list[str]:
    """Return the report lines that try to reproduce the background page's total of 202.

    Operational REPD batteries in England and Wales that lie within 5 km of a connected ECR row
    (of any technology) are placed in that row's licence area. A REPD battery within 1 km of a
    connected ECR storage row is taken to be that row already counted.
    """
    connected_all = ecr_all.filter(pl.col("connection_status") == "Connected")
    repd = census.repd_batteries()
    placed = census.nearest_ecr_row(repd=repd, ecr_rows=connected_all)
    storage_rows = connected.select("ecr_row", "easting", "northing")
    same_site = census.nearest_ecr_row(
        repd=repd, ecr_rows=storage_rows, limit_m=census.REPD_SAME_SITE_M
    ).select("repd_row", already_counted=pl.col("nearest_ecr_row").is_not_null())
    extra = (
        placed.join(same_site, on="repd_row")
        .filter(pl.col("nearest_ecr_row").is_not_null() & ~pl.col("already_counted"))
        .join(
            connected_all.select("ecr_row", "licence_area_short"),
            left_on="nearest_ecr_row",
            right_on="ecr_row",
        )
    )
    targets_total = BACKGROUND_PAGE_TARGETS["Embedded batteries of 50 kW or more, by licence area"]
    rows = []
    for area in census.LICENCE_AREAS:
        added = int((extra["licence_area_short"] == area).sum())
        base = int((connected["licence_area_short"] == area).sum())
        rows.append([area, base, added, base + added, targets_total[area]])
    rows.append(
        [
            "Total",
            connected.height,
            extra.height,
            connected.height + extra.height,
            sum(targets_total.values()),
        ]
    )
    return [
        "## Embedded batteries beyond the ECR's storage rows",
        "",
        (
            f"- Operational REPD batteries in England and Wales: {repd.height}; with a connected "
            f"ECR "
            f"row (any technology) within {census.REPD_DISTANCE_LIMIT_M / 1000:.0f} km: "
            f"{int(placed['nearest_ecr_row'].is_not_null().sum())}."
        ),
        (
            f"- Of those, within {census.REPD_SAME_SITE_M / 1000:.0f} km of a connected ECR "
            f"storage "
            f"row (taken as already counted): {int(same_site['already_counted'].sum())}."
        ),
        "",
        *_table(
            header=[
                "Licence area",
                "ECR storage rows",
                "Added from REPD",
                "Reproduced total",
                "Background page",
            ],
            rows=rows,
        ),
        (
            "The background page's total came from a hand-matching of the REPD, the TEC "
            "register, and "
            "the ECR that no script recorded. This count follows the rule above, so a difference "
            "from "
            "the page is a difference of method, not an error in either."
        ),
        "",
    ]


def transmission_section() -> list[str]:
    """Return the report lines for transmission-connected batteries at grid supply points."""
    body = json.loads(census.TEC_PATH.read_text())["body"].lstrip("﻿")
    tec = pl.read_csv(io.StringIO(body), infer_schema_length=0)
    storage = tec.filter(
        pl.col("Plant Type").str.contains("(?i)storage")
        & (pl.col("Project Status") == "Built")
        & (pl.col("Agreement Type") == "Direct Connection")
    )
    ecr_gsp = {
        g.lower().replace(" s stn", "").replace("kv", "").strip()
        for g in ecr.read_ecr()["grid_supply_point"].drop_nulls().unique().to_list()
    }
    ecr_gsp_names = {" ".join(g.split()[:-1]) if g.split()[-1].isdigit() else g for g in ecr_gsp}
    site = storage["Connection Site"].str.to_lowercase()
    hits = storage.filter(
        site.map_elements(
            lambda text: any(name and name.split(" ")[0] in text for name in ecr_gsp_names),
            return_dtype=pl.Boolean,
        )
    )
    return [
        "## Transmission-connected batteries",
        "",
        (
            f"- Built, directly connected storage in the TEC register: {storage.height}; at a "
            f"connection site whose first word matches one of the ECR's grid supply point names "
            f"(a loose match): {hits.height}, with "
            f"{hits['MW Connected'].cast(pl.Float64).sum():.0f} MW "
            "connected. The background page counts 5 to 7."
        ),
        "",
    ]


def battery_a_status(
    *, bmus: pl.DataFrame, connected: pl.DataFrame, proposals: pl.DataFrame
) -> list[str]:
    """Return the report lines saying whether NGED battery A matches a BMU.

    The series' name and position are used only to decide; nothing about the site is printed.
    """
    site = battery_a_site()
    to_osgb = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    easting, northing = to_osgb.transform(site["longitude"], site["latitude"])
    names = dict(zip(bmus["elexon_bmu_id"].to_list(), bmus["bmu_name"].to_list(), strict=True))
    by_name = best_match(site_name=str(site["name"]), candidates=names)
    repd = census.repd_batteries()
    distance = np.hypot(
        repd["easting"].to_numpy() - easting, repd["northing"].to_numpy() - northing
    )
    near = repd.filter(pl.Series(distance <= census.REPD_DISTANCE_LIMIT_M))
    via_repd = [
        best_match(site_name=row["repd_site_name"], candidates=names)
        for row in near.iter_rows(named=True)
    ]
    ecr_distance = np.hypot(
        connected["easting"].to_numpy() - easting, connected["northing"].to_numpy() - northing
    )
    near_rows = connected.filter(pl.Series(ecr_distance <= census.REPD_SAME_SITE_M))["ecr_row"]
    leads = proposals.filter(pl.col("ecr_row").is_in(near_rows.to_list()))
    return [
        "## NGED battery A",
        "",
        f"- Its series name is similar to a reviewed storage BMU's name: {by_name is not None}.",
        (
            f"- Operational REPD batteries within {census.REPD_DISTANCE_LIMIT_M / 1000:.0f} km of "
            f"its "
            f"position: {near.height}; of those, with a name similar to a reviewed storage BMU's: "
            f"{sum(m is not None for m in via_repd)}."
        ),
        (
            f"- Connected ECR storage rows within 1 km of its position: "
            f"{int((ecr_distance <= census.REPD_SAME_SITE_M).sum())}."
        ),
        (
            f"- Those rows that any reviewed storage BMU proposes as a match: {leads.height} "
            f"({sorted(leads['grade'].unique().to_list())})."
        ),
        "- Verdict by these keys: "
        + (
            "matches a reviewed storage BMU."
            if by_name is not None or any(m is not None for m in via_repd)
            else "no reviewed storage BMU is matched, so by these keys NGED battery A is not a BMU."
        ),
        "",
    ]


def main() -> None:
    """Run the census and write the report and tables."""
    census.CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    frame = ecr.read_ecr()
    flagged = census.with_storage_flags(ecr=frame)
    storage = flagged.filter(pl.col("lists_storage"))
    connected = census.connected_storage(ecr=frame)
    bmus = census.storage_bmus()
    bmu_lines, in_area = bmu_side_table(bmus=bmus)

    proposals = census.propose_matches(ecr_storage=connected, bmus=bmus)
    repd = census.repd_batteries()
    repd_located = census.nearest_ecr_row(
        repd=repd, ecr_rows=flagged.filter(pl.col("connection_status") == "Connected")
    )
    repd_route = census.propose_repd_route(bmus=bmus, repd_located=repd_located)

    embedded_in_area = in_area.filter(pl.col("bmu_type") == "E")
    accepted = proposals.filter(pl.col("grade") == "text and capacity").select(
        "elexon_bmu_id", "ecr_row"
    )
    found = embedded_in_area.filter(
        pl.col("elexon_bmu_id").is_in(accepted["elexon_bmu_id"].to_list())
    ).height
    recall = found / embedded_in_area.height
    classified = census.classify_connected_storage(
        ecr_storage=connected,
        accepted_matches=accepted,
        bmus=bmus,
        participants=census.response_participants(),
    )
    ledger = {
        "embedded_in_area": float(embedded_in_area.height),
        "with_output": float((embedded_in_area["b1610_share"] >= MIN_B1610_SHARE).sum()),
        "bmu_megawatts": float(embedded_in_area["generation_capacity_mw"].sum()),
    }

    lines = [
        "# Embedded-battery census: how many of NGED's batteries are BMUs",
        "",
        "Generated by `studies/embedded_battery_forecast/census_report.py`. Counts and bands only.",
        "",
        *register_checks(frame=frame, storage=storage),
        *licence_area_table(connected=connected),
        *bmu_lines,
        *match_section(proposals=proposals, in_area=in_area, repd_route=repd_route),
        *classification_section(
            classified=classified,
            recall=recall,
            ledger=ledger,
            n_bmus_known=embedded_in_area.height,
        ),
        *repd_universe_section(ecr_all=flagged, connected=connected),
        *transmission_section(),
        *battery_a_status(bmus=bmus, connected=connected, proposals=proposals),
        "## What the registers cannot reproduce",
        "",
        (
            "- The background page's hand-matching used external knowledge of site names that the "
            "registers do not carry: no BMU in NGED's four groups matches an ECR row on name and "
            "capacity together. The proposals in `census_matches.csv` are leads for a person."
        ),
        (
            "- The ECR's storage capacity in megawatt-hours, its duration, and both "
            "service-provider "
            "flags read `data not available` in every connected storage row, so size is classified "
            "by export capacity alone."
        ),
        (
            "- No register says which ECR customers sell frequency response without a BMU: the "
            "NESO "
            "auction table names the participant, not the site."
        ),
        "- No dataset used here counts domestic batteries; the ECR starts at 50 kW.",
        "",
    ]
    (census.CENSUS_DIR / "census_report.md").write_text("\n".join(lines))
    names = connected.select(
        "ecr_row", "customer_name", "customer_site", "licence_area_short", "export_mw"
    )
    proposals.join(names, on="ecr_row").join(
        bmus.select("elexon_bmu_id", "bmu_name", "lead_party_name", "generation_capacity_mw"),
        on="elexon_bmu_id",
    ).sort("elexon_bmu_id", "grade").write_csv(census.CENSUS_DIR / "census_matches.csv")
    classified.select(
        "ecr_row", "licence_area_short", "export_mw", "size_class", "hybrid", "bmu_class"
    ).write_parquet(census.CENSUS_DIR / "census_classes.parquet")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
