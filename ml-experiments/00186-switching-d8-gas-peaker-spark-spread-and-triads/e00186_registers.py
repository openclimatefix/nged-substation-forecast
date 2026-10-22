# ruff: noqa: E501
"""Experiment 186, step 5: do the public registers list a gas plant near D8's coordinates?

Reads the Renewable Energy Planning Database (REPD, DESNZ quarterly extract, Q2 2026) and NESO's
Capacity Market register (components file of 2026-10-21), both stored under `PUBLIC_DIR/registers`.
Prints counts and capacities only, never plant names, so no metered generator is named beside a
series. Run from the repository root: `uv run --no-sync python <this file>` with this folder on
`PYTHONPATH`.
"""

from __future__ import annotations

import re
from typing import Final

import e00186_common as common
import numpy as np
import polars as pl
from pyproj import Transformer

D8_LAT: Final[float] = 52.96158981323242
D8_LON: Final[float] = -0.013620000332593918
RADIUS_KM: Final[tuple[float, ...]] = (5.0, 10.0, 20.0)
D8_POSTCODE_AREAS: Final[tuple[str, ...]] = ("PE20", "PE21", "PE22", "PE24", "PE23", "PE25")
"""Postcode districts around Boston and the Skegness coast, the area D8's 33/11 kV substation serves."""
GAS_WORDS: Final[tuple[str, ...]] = ("gas", "chp", "reciprocating", "ocgt", "ccgt", "engine")
REPORT: list[str] = []


def say(text: str = "") -> None:
    """Print a line and keep it for the report."""
    print(text, flush=True)
    REPORT.append(text)


def main() -> None:
    """Count gas plants near D8 in the two registers."""
    reg = common.PUBLIC_DIR / "registers"
    to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    d8_e, d8_n = to_bng.transform(D8_LON, D8_LAT)
    say(f"D8 on the British National Grid: easting {d8_e:.0f}, northing {d8_n:.0f}")

    repd = pl.read_csv(
        reg / "repd.csv", encoding="latin1", infer_schema_length=0, ignore_errors=True
    )
    repd = repd.with_columns(
        east=pl.col("X-coordinate").str.replace_all(",", "").cast(pl.Float64, strict=False),
        north=pl.col("Y-coordinate").str.replace_all(",", "").cast(pl.Float64, strict=False),
        mw=pl.col("Installed Capacity (MWelec)")
        .str.replace_all(",", "")
        .cast(pl.Float64, strict=False),
    ).with_columns(
        km=(((pl.col("east") - d8_e) ** 2 + (pl.col("north") - d8_n) ** 2).sqrt() / 1000.0)
    )
    operating = repd.filter(pl.col("Development Status (short)") == "Operational")
    say(f"REPD rows: {len(repd)}; operational {len(operating)}")
    for radius in RADIUS_KM:
        near = operating.filter(pl.col("km") <= radius)
        by = (
            near.group_by("Technology Type")
            .agg(n=pl.len(), mw=pl.col("mw").sum())
            .sort("mw", descending=True)
        )
        say(f"REPD operational plants within {radius:.0f} km of D8, by technology (MW electric):")
        for row in by.iter_rows(named=True):
            say(f"  {row['Technology Type']}: {row['n']} plants, {row['mw']:.1f} MW")
        gasy = near.filter(
            pl.col("Technology Type").str.contains("(?i)landfill gas|biogas|anaerobic|sewage|gas")
            | (pl.col("CHP Enabled") == "Yes")
        )
        say(
            f"  gas-fuelled or combined-heat-and-power plants (landfill gas, biogas, anaerobic "
            f"digestion, sewage gas, or CHP flagged): {len(gasy)} plants, {gasy['mw'].sum():.1f} MW"
        )

    comp = pl.read_csv(reg / "component.csv", encoding="utf8-lossy", infer_schema_length=0)
    comp = comp.with_columns(
        text=pl.concat_str(
            [
                "Primary Fuel of Component",
                "Generating Technology Class",
                "Description of CMU Components",
            ],
            separator=" ",
            ignore_nulls=True,
        ).str.to_lowercase(),
        place=pl.col("Location and Post Code").str.to_uppercase().fill_null(""),
        grid=pl.col("OS Grid Reference").fill_null(""),
    )
    gas = comp.filter(pl.col("text").str.contains("|".join(GAS_WORDS)))
    say(f"\nCapacity Market components rows {len(comp)}; gas-like rows {len(gas)}")
    pattern = "|".join(rf"\b{p}\b" for p in D8_POSTCODE_AREAS)
    hits = gas.filter(pl.col("place").str.contains(pattern))
    say(f"Gas-like Capacity Market components with a postcode in {D8_POSTCODE_AREAS}: {len(hits)}")
    if len(hits):
        by = hits.group_by(["Auction Name", "Primary Fuel of Component", "Type"]).agg(
            n=pl.len(),
            capacity=pl.col("Connection or DSR Capacity")
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False)
            .sum(),
        )
        for row in by.sort("Auction Name").iter_rows(named=True):
            say(
                f"  {row['Auction Name']}: {row['n']} components, fuel {row['Primary Fuel of Component']}, type {row['Type']}, connection capacity {row['capacity']}"
            )
    # Capacity Market gas components with an Ordnance Survey grid reference, within 30 km of D8.
    cmu = pl.read_csv(reg / "cmu.csv", encoding="utf8-lossy", infer_schema_length=0)
    cmu_gas = (
        cmu.filter(pl.col("Primary Fuel Type") == "Gas")
        .with_columns(
            mw=pl.col("Connection or DSR Capacity")
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False)
        )
        .group_by("CMU ID")
        .agg(
            mw=pl.col("mw").max(),
            transmission_or_distribution=pl.col("Transmission or Distribution").first(),
            first_delivery_year=pl.col("Delivery Year").min(),
            last_delivery_year=pl.col("Delivery Year").max(),
        )
    )
    squares = {
        "TF": (500_000, 300_000),
        "TA": (500_000, 400_000),
        "SK": (400_000, 300_000),
        "TG": (600_000, 300_000),
        "TL": (500_000, 200_000),
        "TQ": (500_000, 100_000),
    }

    def metres(ref: str) -> tuple[float, float] | None:
        match = re.match(r"^\s*([A-Z]{2})\s*(\d{3})\s*(\d{3})\s*$", ref.upper())
        if match is None or match.group(1) not in squares:
            return None
        east0, north0 = squares[match.group(1)]
        return east0 + int(match.group(2)) * 100.0, north0 + int(match.group(3)) * 100.0

    seen = set()
    near_rows = []
    for row in comp.filter(pl.col("CMU ID").is_in(cmu_gas["CMU ID"].to_list())).iter_rows(
        named=True
    ):
        xy = metres(row["grid"])
        if xy is None or (row["CMU ID"], row["grid"]) in seen:
            continue
        seen.add((row["CMU ID"], row["grid"]))
        km = float(np.hypot(xy[0] - d8_e, xy[1] - d8_n) / 1000.0)
        if km <= 30:
            info = cmu_gas.filter(pl.col("CMU ID") == row["CMU ID"]).row(0, named=True)
            if km < 1.0:
                text = (row["Description of CMU Components"] or "").replace("\n", " ")[:90]
                say(f"  description of a unit at {km:.1f} km: {text}")
            near_rows.append(
                (
                    km,
                    row["Generating Technology Class"],
                    info["mw"],
                    info["transmission_or_distribution"],
                    info["first_delivery_year"],
                    info["last_delivery_year"],
                )
            )
    say(
        "\nCapacity Market gas-fuelled CM units with a grid reference within 30 km of D8 (unit names withheld):"
    )
    for km, tech, mw, td, first, last in sorted(near_rows):
        say(
            f"  {km:.1f} km from D8: {tech}, connection capacity {mw} MW, {td}, delivery years {first} to {last}"
        )
    say(f"Gas-fuelled CM units with any grid reference in the register: {len(seen)}")
    (common.WORK_DIR / "printed_registers.txt").write_text("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
