r"""Download Elexon's Aggregated GSP Group Take Volumes (AGV) for 2025-09-01 to 2026-09-30.

Written for the study of Elexon's indicated generation and demand. AGV (Elexon data flow
CDCA-I029) is the settled energy that each of the 14 Grid Supply Point (GSP) groups takes from the
transmission system in each half-hour. Elexon publishes it in its Open Settlement Data, as one zip
for each year in which a settlement run was published:

    uv run python studies/market_downloads/fetch_agv.py

The script keeps only the rows of one settlement run (`SF`, the interim settlement run), so the
whole window has one vintage. The latest run of each settlement period would mix runs from the
initial run to the final reconciliation across the window. Pass `--start`, `--end` (both
inclusive), and `--output-root` for a small test run.
"""

import argparse
import hashlib
import io
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl
from fetch_system_series import add_period_time
from market_common import (
    WINDOW_END,
    WINDOW_START,
    expected_settlement_starts,
    get_response,
    summarise_gaps,
    write_lineage,
    write_parquet_atomic,
    write_readme,
)
from studies.sources import MARKET_DOWNLOADS_DIR

SCRIPT_PATH: Final[str] = "studies/market_downloads/fetch_agv.py"
SOURCE_PAGE: Final[str] = "https://elexon.co.uk/data/open-settlement-data"
ZIP_URL: Final[str] = "https://www.elexon.co.uk/open-data/AGV_{year}.zip"
"""The URL of one year's zip. It redirects to Elexon's S3 bucket, so the client follows it."""
SETTLEMENT_RUN: Final[str] = "SF"
"""The interim settlement run, which has one row for every settlement period of the window."""
GSP_GROUPS: Final[tuple[str, ...]] = tuple(f"_{letter}" for letter in "ABCDEFGHJKLMNP")
"""The 14 GSP groups, which Elexon labels `_A` to `_P` and skips `_I` and `_O`."""
PURPOSE: Final[str] = (
    "Public data for the study of Elexon's indicated generation and demand "
    "(<https://github.com/openclimatefix/nged-substation-forecast/issues/1108>)."
)
LICENCE: Final[str] = (
    "Elexon announced the Open Settlement Data as open data "
    "(<https://www.elexon.co.uk/bsc/article/making-electricity-market-data-more-openly-available-2/>)."
    " The licence text and its attribution wording have not been independently verified. The "
    "maintainer accepted the licence for this download."
)
CSV_COLUMNS: Final[dict[str, str]] = {
    "GSP Group Id": "gsp_group",
    "Settlement Date": "settlement_date",
    "Settlement Run Type": "settlement_run",
    "CDCA Run Number": "cdca_run_number",
    "Settlement Period": "settlement_period",
    "Estimate Indicator": "estimate_indicator",
    "Import/Export Indicator": "import_export",
    "GSP Group Take Volume": "take_mwh",
    "Flow Run Date": "flow_run_date",
}
COLUMN_DOCS: Final[dict[str, str]] = {
    "time": "UTC start of the settlement period.",
    "settlement_date": "Settlement date (UK clock-time day).",
    "settlement_period": "Settlement period within the settlement date, 1 to 48 (46 or 50 on "
    "clock-change days).",
    "gsp_group": "GSP group, `_A` to `_P` (`_I` and `_O` do not exist).",
    "import_export": "`I` if the group imports from the transmission system in the period, "
    "`E` if it exports. A group, date, period, and run has one row, so never both.",
    "take_mwh": "Energy in the period, MWh. Always positive; the sign is in `import_export`. "
    "The unit is not stated in the source and was checked against the national demand "
    "outturn (INDO): twice the sum over the 14 groups is about 0.93 of INDO.",
    "estimate_indicator": "Elexon's estimate flag (`T` on most rows). Its meaning is not "
    "documented in the files.",
    "settlement_run": "Always `SF`.",
    "cdca_run_number": "The run number of the settlement run.",
    "flow_run_date": "The date Elexon ran the data flow.",
}


def read_year_zip(*, content: bytes) -> pl.DataFrame:
    """Read every monthly CSV of one AGV zip into one table with the columns in `CSV_COLUMNS`."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return pl.concat(
            [
                pl.read_csv(archive.read(name), infer_schema=False)
                .select(list(CSV_COLUMNS))
                .rename(CSV_COLUMNS)
                for name in archive.namelist()
                if name.endswith(".csv")
            ]
        )


def tidy(*, raw: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    """Keep the `SF` rows inside the window and give every column its type and the UTC `time`."""
    frame = (
        raw.filter(pl.col("settlement_run") == SETTLEMENT_RUN)
        .with_columns(
            settlement_date=pl.col("settlement_date").str.to_date(format="%Y%m%d"),
            flow_run_date=pl.col("flow_run_date").str.to_date(format="%Y%m%d"),
            settlement_period=pl.col("settlement_period").cast(pl.Int16),
            cdca_run_number=pl.col("cdca_run_number").cast(pl.Int16),
            take_mwh=pl.col("take_mwh").cast(pl.Float64),
        )
        .filter(pl.col("settlement_date").is_between(start, end))
    )
    keys = ["gsp_group", "settlement_date", "settlement_period"]
    duplicated = frame.filter(pl.struct(keys).is_duplicated())
    if duplicated.height:
        raise ValueError(
            f"{duplicated.height} {SETTLEMENT_RUN} rows share a group, date, and period"
        )
    return add_period_time(frame=frame).sort("time", "gsp_group")


def completeness(*, frame: pl.DataFrame, start: date, end: date) -> dict[str, Any]:
    """Compare the table with 14 groups in every settlement period of the window."""
    expected = expected_settlement_starts(start=start, end=end)
    per_group = frame.group_by("gsp_group").agg(periods=pl.col("time").n_unique())
    last_date = frame["settlement_date"].max()
    return {
        "gsp_groups": sorted(frame["gsp_group"].unique().to_list()),
        "gsp_groups_expected": list(GSP_GROUPS),
        "last_settlement_date_present": None if last_date is None else str(last_date),
        "periods_per_group": dict(sorted(per_group.iter_rows())),
        "period_coverage": summarise_gaps(expected=expected, actual=frame["time"]),
        "import_export_counts": dict(
            frame.group_by("import_export").agg(rows=pl.len()).sort("import_export").iter_rows()
        ),
        "estimate_indicator_counts": dict(
            frame.group_by("estimate_indicator")
            .agg(rows=pl.len())
            .sort("estimate_indicator")
            .iter_rows()
        ),
    }


def run(*, root: Path, start: date, end: date) -> pl.DataFrame:
    """Download the zips, write the parquet, the lineage note, and the README."""
    output_dir = root / "elexon_agv"
    zip_dir = output_dir / "_zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    raw_frames = []
    zips = {}
    for year in range(start.year, end.year + 1):
        url = ZIP_URL.format(year=year)
        content = get_response(url=url).content
        (zip_dir / f"AGV_{year}.zip").write_bytes(content)
        zips[f"AGV_{year}.zip"] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        raw_frames.append(read_year_zip(content=content))
    frame = tidy(raw=pl.concat(raw_frames), start=start, end=end)
    write_parquet_atomic(frame=frame, path=output_dir / "elexon_agv.parquet")
    checks = completeness(frame=frame, start=start, end=end)
    write_lineage(
        product_dir=output_dir,
        note={
            "source_address": ZIP_URL,
            "window": [start.isoformat(), end.isoformat()],
            "script": SCRIPT_PATH,
            "settlement_run_kept": SETTLEMENT_RUN,
            "zips": zips,
            "rows": frame.height,
            "checks": checks,
        },
    )
    coverage = checks["period_coverage"]
    write_readme(
        product_dir=output_dir,
        title="Elexon Aggregated GSP Group Take Volumes (AGV, CDCA-I029), SF run",
        source_page=SOURCE_PAGE,
        script_path=SCRIPT_PATH,
        attribution=None,
        cache_hint="_zips/",
        licence=LICENCE,
        timestamp_convention=(
            "The source gives the settlement date and period. `time` is computed by this script "
            "as the UTC start of that settlement period (period 1 starts at 00:00 UK local time)."
        ),
        columns=COLUMN_DOCS,
        row_summary=(
            f"- {frame.height} rows for {len(checks['periods_per_group'])} GSP groups.\n"
            f"- The last settlement date present is {checks['last_settlement_date_present']}.\n"
            f"- Settlement periods with at least one row: {coverage['distinct_rows']} of "
            f"{coverage['expected_rows']} expected; {coverage['missing_count']} are missing "
            f"(first missing: {coverage['missing_first'][:5] or 'none'})."
        ),
        gotchas=[
            (
                f"Only the `{SETTLEMENT_RUN}` run is kept. Elexon publishes seven runs (II, SF, "
                "R1, R2, R3, RF, DF), and each settlement period has one row for every run "
                "published so far. The SF run appears about 28 days after the settlement date, "
                "so the last weeks of any download have no SF rows yet."
            ),
            (
                "AGV is the energy taken at the GSP group's boundary with the transmission "
                "system, so it is net of generation embedded in the distribution network."
            ),
            (
                "Each zip is named for the year the settlement run was published, so a zip holds "
                "settlement dates from earlier years. The script reads every year from the "
                "window start to the window end."
            ),
            (
                "`AGV_<current year>.zip` is rewritten daily, so `lineage.json` records a "
                "sha256 for each zip."
            ),
        ],
        purpose=PURPOSE,
    )
    print(f"elexon_agv: wrote {frame.height} rows")
    return frame


def main() -> None:
    """Parse the command line and run the download."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, default=MARKET_DOWNLOADS_DIR)
    parser.add_argument("--start", type=date.fromisoformat, default=WINDOW_START)
    parser.add_argument("--end", type=date.fromisoformat, default=WINDOW_END)
    args = parser.parse_args()
    run(root=args.output_root, start=args.start, end=args.end)
    print(f"Finished at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
