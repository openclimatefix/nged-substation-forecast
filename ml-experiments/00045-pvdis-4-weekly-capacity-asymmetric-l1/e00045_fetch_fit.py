"""Download the Ofgem Feed-in Tariff installation report and Code-Point Open for experiment 00045.

Run with `uv run python ml-experiments/00045-.../e00045_fetch_fit.py`. Resumable: every download and
every parsed report part is one file under `outside/_cache/`, written to a `.partial` name and
renamed, and a re-run skips every file already on disk.

It fetches:

1. The Ofgem "Feed-in Tariff Installation Report" (installations accredited from 1 April 2010 to
   31 March 2025, three workbook parts), one row per installation, with postcode district, installed
   and declared net capacity, and commissioning date.
2. Ordnance Survey Code-Point Open, the position of every postcode unit in Great Britain. No open
   polygon boundary for postcode districts exists, so each district is allocated to the Primary
   polygons by the share of its postcode units whose point lies inside each polygon.

It writes `fit_solar_pv_installations.parquet`, `fit_district_polygon_shares.parquet`,
`fit_validation.json`, `fit_lineage.json`, and a "Ofgem Feed-in Tariff" section of `README.md` in
`outside/`. Licence: Open Government Licence v3.0 for both sources.
"""

import io
import json
import re
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import e00045_fetch_lsoa as lsoa
import polars as pl
import shapely

OUTSIDE_DIR: Final[Path] = lsoa.OUTSIDE_DIR
FIT_CACHE_DIR: Final[Path] = lsoa.CACHE_DIR / "fit"
CODEPOINT_CACHE_DIR: Final[Path] = lsoa.CACHE_DIR / "codepoint"
SCRIPT_PATH: Final[str] = (
    "ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_fetch_fit.py"
)
FIT_PAGE: Final[str] = (
    "https://www.ofgem.gov.uk/publications/feed-tariff-installation-report-31-march-2025"
)
FIT_PUBLISHED: Final[str] = "6 June 2025"
FIT_PART_URLS: Final[dict[int, str]] = {
    part: "https://www.ofgem.gov.uk/sites/default/files/2025-06/"
    f"Feed-in%20Tariff%20Installation%20Report%20Part%20{part}.xlsx"
    for part in (1, 2, 3)
}
FIT_COPYRIGHT_PAGE: Final[str] = "https://www.ofgem.gov.uk/copyright"
CODEPOINT_PAGE: Final[str] = "https://api.os.uk/downloads/v1/products/CodePointOpen"
CODEPOINT_URL: Final[str] = (
    "https://api.os.uk/downloads/v1/products/CodePointOpen/downloads?area=GB&format=CSV&redirect"
)
HEADER_FIRST_CELL: Final[str] = "Extension (Y/N)"
PV_TECHNOLOGY: Final[str] = "Photovoltaic"
FIT_COLUMNS: Final[dict[str, str]] = {
    "Extension (Y/N)": "extension",
    "PostCode": "postcode",
    "Technology": "technology",
    "Installed capacity": "installed_capacity_kw",
    "Declared net capacity": "declared_net_capacity_kw",
    "Application date": "application_date",
    "Commissioning date": "commissioning_date",
    "Installation Type": "installation_type",
    "Local Authority": "local_authority",
    "Constituency": "constituency",
    "LLSOA Code": "llsoa_code",
    "FIT ID": "fit_id",
}
EXCEL_EPOCH: Final[date] = date(1899, 12, 30)
SCHEME_START: Final[date] = date(2010, 4, 1)
SCHEME_CLOSED_TO_NEW: Final[date] = date(2019, 4, 1)
REPORT_END: Final[date] = date(2025, 3, 31)
MAX_PLAUSIBLE_PV_KW: Final[float] = 5000.0
SMALL_SCALE_KW: Final[float] = 50.0
CODEPOINT_COLUMNS: Final[list[str]] = [
    "postcode",
    "quality",
    "easting",
    "northing",
    "country",
    "nhs_region",
    "nhs_authority",
    "county",
    "district_code",
    "ward",
]


def parse_fit_part(*, part: int) -> pl.DataFrame:
    """Read one report workbook into a frame of the columns in `FIT_COLUMNS`.

    Args:
        part: The part number, 1 to 3.

    Returns:
        One row per installation, capacities in kW, dates as `Date`, other cells as text.
    """
    rows = lsoa.read_workbook_rows(
        path=FIT_CACHE_DIR / f"part{part}.xlsx", sheet_name=f"Part {part}"
    )
    header_index = next(i for i, row in enumerate(rows) if row and row[0] == HEADER_FIRST_CELL)
    header = [(cell or "").strip() for cell in rows[header_index]]
    wanted = {header.index(source): target for source, target in FIT_COLUMNS.items()}
    width = len(header)
    data = [row for row in rows[header_index + 1 :] if any(cell is not None for cell in row)]
    frame = pl.DataFrame(
        {
            target: [row[i] if i < len(row) else None for row in data]
            for i, target in wanted.items()
        },
        schema=dict.fromkeys(wanted.values(), pl.String),
    )
    assert frame.width == len(FIT_COLUMNS)
    assert width >= max(wanted)
    return frame.with_columns(
        pl.col("postcode").str.strip_chars(),
        pl.col("technology").str.strip_chars(),
        pl.col("installed_capacity_kw", "declared_net_capacity_kw").cast(pl.Float64, strict=False),
        *[
            (pl.lit(EXCEL_EPOCH) + pl.duration(days=pl.col(name).cast(pl.Float64, strict=False)))
            .cast(pl.Date)
            .alias(name)
            for name in ("application_date", "commissioning_date")
        ],
    ).with_columns(pl.lit(part).alias("report_part"))


def cached_fit_parts() -> pl.DataFrame:
    """Parse each report part once, caching it as parquet, and return all parts together."""
    frames = []
    for part in FIT_PART_URLS:
        path = FIT_CACHE_DIR / f"part{part}.parquet"
        if not path.exists():
            frame = parse_fit_part(part=part)
            partial = path.with_name(path.name + ".partial")
            frame.write_parquet(partial)
            partial.rename(path)
            print(f"part {part}: parsed {frame.height:,} rows")
        frames.append(pl.read_parquet(path))
    return pl.concat(frames)


def district_polygon_shares(*, polygons: dict[int, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Allocate postcode districts to the Primary polygons by the postcode units inside each.

    Args:
        polygons: `time_series_id` to polygon in British National Grid metres.

    Returns:
        The share table (`district`, `time_series_id`, `units_inside`, `units_in_district`,
        `share`), and one row per district touching any polygon with its unit centroid.
    """
    with zipfile.ZipFile(CODEPOINT_CACHE_DIR / "codepo_gb.zip") as archive:
        names = [n for n in archive.namelist() if re.fullmatch(r"Data/CSV/[a-z]+\.csv", n)]
        units = pl.concat(
            [
                pl.read_csv(
                    io.BytesIO(archive.read(name)),
                    has_header=False,
                    new_columns=CODEPOINT_COLUMNS,
                    schema_overrides={"postcode": pl.String},
                ).select("postcode", "easting", "northing")
                for name in names
            ]
        )
    units = units.with_columns(
        district=pl.col("postcode").str.replace(r"\s*\d[A-Z]{2}$", "")
    ).filter(pl.col("easting") > 0)
    easting = units["easting"].to_numpy().astype(float)
    northing = units["northing"].to_numpy().astype(float)
    totals = units.group_by("district").agg(units_in_district=pl.len())
    shares = []
    for series_id, polygon in polygons.items():
        inside = shapely.contains_xy(polygon, easting, northing)
        shares.append(
            units.filter(pl.Series(inside))
            .group_by("district")
            .agg(units_inside=pl.len())
            .with_columns(time_series_id=pl.lit(series_id, dtype=pl.Int64))
        )
    share_table = (
        pl.concat(shares)
        .join(totals, on="district")
        .with_columns(share=pl.col("units_inside") / pl.col("units_in_district"))
        .sort("district", "time_series_id")
    )
    centroids = (
        units.filter(pl.col("district").is_in(share_table["district"].unique().implode()))
        .group_by("district")
        .agg(
            centroid_easting=pl.col("easting").mean(),
            centroid_northing=pl.col("northing").mean(),
        )
        .sort("district")
    )
    return share_table, centroids


def validate(*, everything: pl.DataFrame, solar: pl.DataFrame, shares: pl.DataFrame) -> dict:
    """Run the data-validation checks on the parsed report and return a JSON-ready report."""
    capacity = pl.col("declared_net_capacity_kw")
    by_year = (
        solar.group_by(pl.col("commissioning_date").dt.year().alias("year"))
        .agg(n=pl.len(), declared_net_mw=capacity.sum() / 1000)
        .sort("year")
    )
    duplicates = solar.group_by("fit_id", "extension").len().filter(pl.col("len") > 1)
    full_row_duplicates = solar.height - solar.unique().height
    in_scheme = solar.filter(pl.col("commissioning_date") < SCHEME_CLOSED_TO_NEW)
    small = solar.filter(
        pl.col("commissioning_date") < SCHEME_CLOSED_TO_NEW,
        pl.col("installed_capacity_kw") < SMALL_SCALE_KW,
    )
    district_pattern = r"^[A-Z]{1,2}\d[A-Z\d]?\s"
    return {
        "rows_all_technologies": everything.height,
        "rows_by_technology": everything.group_by("technology").len().sort("len").to_dicts(),
        "rows_by_part": everything.group_by("report_part").len().sort("report_part").to_dicts(),
        "solar_pv_rows": solar.height,
        "duplicate_fit_id_and_extension": duplicates.height,
        "fully_duplicated_rows": full_row_duplicates,
        "nulls_in_solar_pv": solar.null_count().to_dicts()[0],
        "commissioning_date_min": str(solar["commissioning_date"].min()),
        "commissioning_date_max": str(solar["commissioning_date"].max()),
        "commissioned_before_scheme_start": solar.filter(
            pl.col("commissioning_date") < SCHEME_START
        ).height,
        "commissioned_after_report_end": solar.filter(
            pl.col("commissioning_date") > REPORT_END
        ).height,
        "commissioned_from_2019_04_01": solar.filter(
            pl.col("commissioning_date") >= SCHEME_CLOSED_TO_NEW
        ).height,
        "declared_net_kw_quantiles": {
            str(q): solar["declared_net_capacity_kw"].quantile(q)
            for q in (0.0, 0.01, 0.5, 0.99, 0.999, 1.0)
        },
        "installed_kw_max": solar["installed_capacity_kw"].max(),
        "rows_declared_net_above_installed": solar.filter(
            capacity > pl.col("installed_capacity_kw") * 1.001
        ).height,
        "rows_capacity_not_positive": solar.filter(capacity <= 0).height,
        "rows_above_plausible_pv_kw": solar.filter(
            pl.col("installed_capacity_kw") > MAX_PLAUSIBLE_PV_KW
        ).height,
        "postcodes_not_matching_district_pattern": solar.filter(
            ~(pl.col("postcode") + " ").str.contains(district_pattern)
        ).height,
        "solar_pv_commissioned_by_year": by_year.to_dicts(),
        "declared_net_mw_before_2019_04": in_scheme["declared_net_capacity_kw"].sum() / 1000,
        "share_of_pre_2019_declared_net_under_50kw_installed": (
            float(small["declared_net_capacity_kw"].sum())
            / float(in_scheme["declared_net_capacity_kw"].sum())
        ),
        "districts_in_report": solar["postcode"].n_unique(),
        "report_districts_touching_a_polygon": solar.filter(
            pl.col("postcode").is_in(shares["district"].unique().implode())
        )["postcode"].n_unique(),
        "share_sums_over_one": shares.group_by("district")
        .agg(pl.col("share").sum())
        .filter(pl.col("share") > 1.0 + 1e-9)
        .height,
        "districts_touching_a_polygon_without_fit_rows": shares.filter(
            ~pl.col("district").is_in(solar["postcode"].unique().implode())
        )["district"].n_unique(),
    }


def write_readme_section(*, validation: dict, sizes: dict[str, int], retrieved: str) -> None:
    """Replace the "Ofgem Feed-in Tariff" section of `outside/README.md`, numbers from the files."""
    marker = "## Ofgem Feed-in Tariff and Code-Point Open"
    by_year = "\n".join(
        f"    - {row['year']}: {row['n']:,} installations, {row['declared_net_mw']:.1f} MW"
        for row in validation["solar_pv_commissioned_by_year"]
    )
    files = "\n".join(f"- `{name}`: {size:,} bytes" for name, size in sizes.items())
    section = f"""{marker}

Written by `{SCRIPT_PATH}` on {retrieved}.

**None of the outside photovoltaic datasets is complete, so every outside check is descriptive.**
The Feed-in Tariff (FiT) register, the DESNZ tables, and the Embedded Capacity Register each
count a subset of the installed photovoltaic capacity. A mismatch with the weekly capacity series
therefore does not show that the series is wrong, and no outside check can decide the verdict.

- Source: Ofgem, "Feed-in Tariff: installation report 31 March 2025", published {FIT_PUBLISHED},
  three workbook parts. Page: {FIT_PAGE}
- Part URLs: {", ".join(FIT_PART_URLS.values())}
- Licence: Crown copyright, reusable under the Open Government Licence v3.0 (Ofgem copyright page:
  {FIT_COPYRIGHT_PAGE}). No login or terms click-through is needed to download the files.
- Coverage gap: the report covers installations accredited from 1 April 2010 to 31 March 2025.
  The FiT closed to new applicants on 31 March 2019, so the
  {validation["commissioned_from_2019_04_01"]:,} photovoltaic rows commissioned from 1 April 2019
  are late accreditations or extensions. The register is a floor after March 2019, and it excludes
  every installation that never took the tariff (for example installations that exported under the
  Smart Export Guarantee). The latest photovoltaic commissioning date in the report is
  {validation["commissioning_date_max"]}, and {validation["commissioned_before_scheme_start"]:,}
  rows were commissioned before the scheme started on 1 April 2010 (installations that moved onto
  the tariff).
- Position data: Ordnance Survey Code-Point Open ({CODEPOINT_PAGE}), the point of each postcode
  unit, Open Government Licence v3.0. No openly licensed polygon boundary for postcode districts
  was found among the Office for National Statistics and Ordnance Survey open products. Each
  district is allocated to a Primary polygon by the share of its postcode units whose point lies
  inside the polygon, so the allocation counts units, not area, and a district straddling a polygon
  edge adds error.
  Wrangle T1 and Wrangle T2 share one polygon and so carry identical shares, which makes the
  shares of one district sum to more than 1 ({validation["share_sums_over_one"]} districts do).
  In total
  {validation["report_districts_touching_a_polygon"]} report districts touch a polygon.
- Columns of `fit_solar_pv_installations.parquet`: `extension` (Y if the row extends an earlier
  installation), `postcode` (the district, for example `LD3`), `technology`,
  `installed_capacity_kw`, `declared_net_capacity_kw` (the Ofgem name for AC capacity, in kW),
  `application_date`, `commissioning_date`, `installation_type`, `local_authority`,
  `constituency`, `llsoa_code` (2011 lower-layer super output area, GSS code; the ONS 2021 LSOA
  boundaries in this folder use different codes for the areas that changed), `fit_id`
  (not unique alone: an extension reuses the identifier), `report_part`.
- Columns of `fit_district_polygon_shares.parquet`: `district`, `time_series_id`, `units_inside`,
  `units_in_district`, `share`. `fit_district_centroids.parquet` holds the unit-mean position
  (`centroid_easting`, `centroid_northing`, British National Grid metres) of each district that
  touches a polygon.
- Missing values: Polars null. {validation["nulls_in_solar_pv"]["postcode"]}
  photovoltaic rows have no postcode and no `llsoa_code`; no other column has a null.
- Rows: {validation["rows_all_technologies"]:,} across all technologies, of which
  {validation["solar_pv_rows"]:,} are `{PV_TECHNOLOGY}`; only those are written.
- Declared net capacity commissioned before 1 April 2019:
  {validation["declared_net_mw_before_2019_04"]:,.0f} MW.
  Share of that capacity in installations under {SMALL_SCALE_KW:.0f} kW installed capacity:
  {validation["share_of_pre_2019_declared_net_under_50kw_installed"]:.1%}.
- Photovoltaic installations and declared net capacity by commissioning year:

{by_year}

- Checks run (full numbers in `fit_validation.json`): duplicate keys, nulls, capacity per
  installation, commissioning dates against the scheme dates, the technology filter, postcode
  format, and the share table.

{files}
"""
    path = OUTSIDE_DIR / "README.md"
    text = path.read_text()
    head = text.split(marker)[0].rstrip("\n")
    partial = path.with_name("README.md.partial")
    partial.write_text(head + "\n\n" + section)
    partial.rename(path)


def main() -> None:
    """Fetch, parse, validate, and write the Feed-in Tariff files."""
    retrieved = datetime.now(UTC).strftime("%Y-%m-%d")
    for part, url in FIT_PART_URLS.items():
        lsoa.download_file(url=url, path=FIT_CACHE_DIR / f"part{part}.xlsx")
    lsoa.download_file(url=CODEPOINT_URL, path=CODEPOINT_CACHE_DIR / "codepo_gb.zip")

    everything = cached_fit_parts()
    solar = everything.filter(pl.col("technology") == PV_TECHNOLOGY)
    shares, centroids = district_polygon_shares(polygons=lsoa.primary_polygons())
    validation = validate(everything=everything, solar=solar, shares=shares)
    outputs = {
        "fit_solar_pv_installations.parquet": solar.sort("commissioning_date", "fit_id"),
        "fit_district_polygon_shares.parquet": shares,
        "fit_district_centroids.parquet": centroids,
    }
    for name, frame in outputs.items():
        partial = OUTSIDE_DIR / f"{name}.partial"
        frame.write_parquet(partial)
        partial.rename(OUTSIDE_DIR / name)
    (OUTSIDE_DIR / "fit_validation.json").write_text(json.dumps(validation, indent=2, default=str))
    (OUTSIDE_DIR / "fit_lineage.json").write_text(
        json.dumps(
            {
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
                "script": SCRIPT_PATH,
                "source_address": list(FIT_PART_URLS.values()),
                "publication_page": FIT_PAGE,
                "published": FIT_PUBLISHED,
                "position_source": CODEPOINT_URL,
                "licence": "Open Government Licence v3.0",
                "report_covers": "2010-04-01 to 2025-03-31",
            },
            indent=2,
        )
    )
    sizes = {name: (OUTSIDE_DIR / name).stat().st_size for name in outputs}
    sizes |= {
        f"_cache/fit/part{part}.xlsx": (FIT_CACHE_DIR / f"part{part}.xlsx").stat().st_size
        for part in FIT_PART_URLS
    }
    sizes["_cache/codepoint/codepo_gb.zip"] = (CODEPOINT_CACHE_DIR / "codepo_gb.zip").stat().st_size
    write_readme_section(validation=validation, sizes=sizes, retrieved=retrieved)
    print(json.dumps(validation, indent=2, default=str))


if __name__ == "__main__":
    main()
