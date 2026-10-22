# ruff: noqa: E501
"""Embedded Capacity Register assets near the decision Primaries, read from the workbook.

Defines the engine-evidence exception list from the register alone, before any flagged period is
inspected. Prints classes, MW, connection date and distance only, never a customer name.

    uv run --no-sync python ml-experiments/00199-*/e00199_ecr.py
"""

from datetime import date, timedelta
from typing import Final

import polars as pl
from pyproj import Transformer
from studies.ecr_reader import ECR_PATH, REGISTER_SHEETS, read_sheet

EXCEL_EPOCH: Final[date] = date(1899, 12, 30)
EAST: Final[str] = "Location (X-coordinate):Eastings (where data is held)"
NORTH: Final[str] = "Location (y-coordinate):Northings (where data is held)"
SITE_LATLON: Final[dict[str, tuple[float, float]]] = {
    "D8": (52.96159, -0.01362), "D5": (52.921421, -0.05731), "D10": (53.16853, 0.10461),
    "D11": (53.218079, -0.11218),
}  # fmt: skip
RADIUS_KM: Final[float] = 5.0
PRIMARIES: Final[dict[str, str]] = {"D5": "Kirton", "D10": "Spilsby", "D11": "Horncastle"}


def read_assets() -> pl.DataFrame:
    """One row per technology slot with class, MW, status, connection date, easting, northing."""
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
                    mw=pl.col(
                        f"Energy Source & Energy Conversion Technology {slot} - Registered Capacity (MW)"
                    ).cast(pl.Float64, strict=False),
                    status=pl.col("Connection Status"),
                    connected=pl.col("Date Connected").cast(pl.Float64, strict=False),
                    east=pl.col(EAST).cast(pl.Float64, strict=False),
                    north=pl.col(NORTH).cast(pl.Float64, strict=False),
                )
            )
    return (
        pl.concat(pieces)
        .filter(pl.col("source").is_not_null())
        .with_columns(
            connected_date=pl.col("connected").map_elements(
                lambda v: EXCEL_EPOCH + timedelta(days=int(v)),
                return_dtype=pl.Date,
                skip_nulls=True,
            ),
            text=(pl.col("source") + " " + pl.col("technology")).str.to_lowercase(),
        )
    )


def main() -> None:
    """Print gas, oil, CHP, and storage assets within the radius of each Primary."""
    to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    assets = read_assets()
    interesting = assets.filter(
        pl.col("text").str.contains("gas|oil|fossil|biogas|biomass|storage|engine|chp")
        | (pl.col("chp") == "Yes")
    )
    with pl.Config(tbl_rows=60, tbl_cols=12, fmt_str_lengths=45):
        for label in PRIMARIES:
            lat, lon = SITE_LATLON[label]
            e, n = to_bng.transform(lon, lat)
            near = (
                interesting.with_columns(
                    distance_km=(((pl.col("east") - e) ** 2 + (pl.col("north") - n) ** 2).sqrt())
                    / 1000.0
                )
                .filter(pl.col("distance_km") <= RADIUS_KM)
                .select(
                    "source", "technology", "chp", "mw", "status", "connected_date", "distance_km"
                )
                .sort("distance_km")
            )
            print(
                f"== {label}: {near.height} gas, oil, CHP, biogas, or storage slots within {RADIUS_KM} km"
            )
            print(near)


if __name__ == "__main__":
    main()
