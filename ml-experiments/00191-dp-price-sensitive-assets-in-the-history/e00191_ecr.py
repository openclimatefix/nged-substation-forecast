# ruff: noqa: E501
"""Experiment 191, step 1: price-following assets under each metered Primary, from NGED's ECR.

Reads the August 2026 Embedded Capacity Register (both Part 1 sheets, and Part 2 for demand-side
response) and prints, per Primary's D-label, the count and registered MW of gas or oil engines (CHP
and not), other gas or oil plant, and batteries, with their connection dates. It prints counts and
MW only, never a customer's name, so no metered generator is named beside a series. It writes
`work/ecr_assets.parquet` (one row per asset, no name) for the scoring scripts.

    uv run --no-sync python ml-experiments/00191-*/e00191_ecr.py
"""

from datetime import date, timedelta
from typing import Final

import e00191_common as common
import polars as pl
from pyproj import Transformer
from studies.ecr_reader import ECR_PATH, REGISTER_SHEETS, read_sheet

EXCEL_EPOCH: Final[date] = date(1899, 12, 30)
EAST: Final[str] = "Location (X-coordinate):Eastings (where data is held)"
NORTH: Final[str] = "Location (y-coordinate):Northings (where data is held)"
PRIMARY_NAMES: Final[dict[str, str]] = {
    "D5": "Kirton", "D9": "Trusthorpe", "D10": "Spilsby", "D11": "Horncastle",
    "D14": "Mount Bridge Boston", "D15": "Wrangle T1", "D16": "Donington", "D18": "Wrangle T2",
    "D2": "Warth Lane Skegness", "D3": "Sleaford Road Boston", "D4": "Ingoldmells",
    "D8": "Marsh Lane", "D12": "Stickney", "D19": "Chapel St Leonards", "D21": "Alford",
    "D22": "Langrick",
}  # fmt: skip
SITE_LATLON: Final[dict[str, tuple[float, float]]] = {
    "D8": (52.96159, -0.01362), "D14": (52.969059, -0.01441), "D5": (52.921421, -0.05731),
    "D10": (53.16853, 0.10461), "D11": (53.218079, -0.11218), "D16": (52.915459, -0.18896),
    "D9": (53.325821, 0.26435),
}  # fmt: skip


def asset_class(*, source: str, technology: str, chp: str) -> str | None:
    """Return the price-following class of one technology slot, or None for other plant."""
    text = f"{source} {technology}".lower()
    if "storage" in text and "electrochemical" in text:
        return "battery"
    if "gas" in text and "biofuel" in text:
        return None
    if ("gas" in text or "oil" in text or "fossil" in text) and "engine" in text:
        return "gas or oil engine, CHP" if chp == "Yes" else "gas or oil engine, not CHP"
    if "fossil" in text and ("turbine" in text or "gas" in text or "oil" in text):
        return "other gas or oil plant"
    return None


def main() -> None:
    """Print the asset table and store it."""
    to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    pieces = []
    for sheet in REGISTER_SHEETS:
        frame = read_sheet(path=ECR_PATH, sheet_name=sheet)
        for slot in (1, 2, 3):
            suffix = "" if slot == 1 else f" {slot}"
            pieces.append(
                frame.select(
                    primary=pl.col("Primary"),
                    source=pl.col(f"Energy Source {slot}"),
                    technology=pl.col(f"Energy Conversion Technology {slot}"),
                    chp=pl.col(f"CHP Cogeneration{suffix} (Yes/No)"),
                    storage_mwh=pl.col(f"Storage Capacity {slot} (MWh)"),
                    mw=pl.col(
                        f"Energy Source & Energy Conversion Technology {slot} - Registered Capacity (MW)"
                    ),
                    status=pl.col("Connection Status"),
                    connected=pl.col("Date Connected"),
                    east=pl.col(EAST),
                    north=pl.col(NORTH),
                )
            )
    assets = pl.concat(pieces).filter(pl.col("source").is_not_null())
    assets = assets.with_columns(
        mw=pl.col("mw").cast(pl.Float64, strict=False),
        storage_mwh=pl.col("storage_mwh").cast(pl.Float64, strict=False),
        east=pl.col("east").cast(pl.Float64, strict=False),
        north=pl.col("north").cast(pl.Float64, strict=False),
        connected_date=pl.col("connected")
        .cast(pl.Float64, strict=False)
        .map_elements(
            lambda v: EXCEL_EPOCH + timedelta(days=int(v)), return_dtype=pl.Date, skip_nulls=True
        ),
    )
    labelled = []
    for label, name in PRIMARY_NAMES.items():
        part = assets.filter(pl.col("primary").str.to_lowercase().str.contains(name.lower()))
        labelled.append(part.with_columns(site=pl.lit(label)))
    table = pl.concat(labelled).with_columns(
        asset_class=pl.struct("source", "technology", "chp").map_elements(
            lambda r: asset_class(source=r["source"], technology=r["technology"], chp=r["chp"]),
            return_dtype=pl.String,
        )
    )
    distances = []
    for row in table.iter_rows(named=True):
        site = SITE_LATLON.get(row["site"])
        if site is None or row["east"] is None:
            distances.append(None)
            continue
        e, n = to_bng.transform(site[1], site[0])
        distances.append(((row["east"] - e) ** 2 + (row["north"] - n) ** 2) ** 0.5 / 1000.0)
    table = table.with_columns(distance_km=pl.Series(distances, dtype=pl.Float64)).drop(
        "primary", "connected", "east", "north"
    )
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(common.WORK_DIR / "ecr_assets.parquet")
    print("Part 2 (demand-side response) register rows: 0 (the sheet holds headings only)")
    print(f"ECR asset slots under the 16 named Primaries: {table.height}")
    flagged = table.filter(pl.col("asset_class").is_not_null())
    summary = (
        flagged.group_by("site", "asset_class", "status")
        .agg(n=pl.len(), mw=pl.col("mw").sum(), storage_mwh=pl.col("storage_mwh").sum())
        .sort("site", "asset_class", "status")
    )
    with pl.Config(tbl_rows=100, tbl_cols=10, fmt_str_lengths=40):
        print(summary)
        print("Connected price-following assets: date connected and distance from the substation")
        print(
            flagged.filter(pl.col("status") == "Connected")
            .select("site", "asset_class", "mw", "storage_mwh", "connected_date", "distance_km")
            .sort("site", "connected_date")
        )
        print("Sites with no price-following asset at all:")
        have = set(flagged["site"].to_list())
        print(sorted(set(PRIMARY_NAMES) - have))


if __name__ == "__main__":
    main()
