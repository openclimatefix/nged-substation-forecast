"""List the public onshore wind farms that can join experiment 00288's pool.

A candidate is a Balancing Mechanism Unit (BMU) whose register fuel type is WIND and whose name
matches an operational onshore row of the Renewable Energy Planning Database (REPD), and whose REPD
position falls inside a cell of the ENS table. The candidates are written to
`data/ml-experiments/00288-*/work/farm_candidates.parquet` for the B1610 fetch to read. Positions
stay in that private file and are never printed.

    uv run --no-sync python ml-experiments/00288-*/e00288_farms.py
"""

import json
from typing import Final

import h3
import polars as pl
from pyproj import Transformer
from studies.name_matching import best_match
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

RAW_DIR: Final = REPO_DATA_DIR / "studies/per_study/solar_bmu_census/inputs/raw"
WORK_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00288-pooled-head-on-public-wind-farms" / "work"
ENS_RESOLUTION: Final[int] = 5


def ens_cells() -> set[int]:
    """Return the H3 cells of the ENS table, read from one run and one valid time."""
    cells = (
        pl.scan_delta(str(REPO_DATA_DIR / "NWP"))
        .filter(
            pl.col("init_time") == pl.datetime(2025, 3, 3, time_zone="UTC"),
            pl.col("ensemble_member") == 0,
            pl.col("valid_time") == pl.datetime(2025, 3, 3, 12, time_zone="UTC"),
        )
        .select("h3_index")
        .unique()
        .collect()
    )
    return set(cells["h3_index"].to_list())


def repd_wind() -> pl.DataFrame:
    """Return operational onshore wind rows of REPD with a position, capacity, and turbines."""
    body = json.loads((RAW_DIR / "repd_register.json").read_text())["body"]
    frame = pl.read_csv(body.encode(), infer_schema_length=0)
    return (
        frame.filter(
            pl.col("Technology Type") == "Wind Onshore",
            pl.col("Development Status (short)") == "Operational",
        )
        .select(
            repd_ref=pl.col("Ref ID"),
            repd_name=pl.col("Site Name"),
            capacity_mw=pl.col("Installed Capacity (MWelec)")
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False),
            turbine_mw=pl.col("Turbine Capacity (MW)")
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False),
            n_turbines=pl.col("No. of Turbines")
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False),
            hub_height_m=pl.col("Height of Turbines (m)").cast(pl.Float64, strict=False),
            operational=pl.col("Operational").str.to_date("%d/%m/%Y", strict=False),
            easting=pl.col("X-coordinate").str.replace_all(",", "").cast(pl.Float64, strict=False),
            northing=pl.col("Y-coordinate").str.replace_all(",", "").cast(pl.Float64, strict=False),
        )
        .drop_nulls(["easting", "northing", "capacity_mw"])
    )


def main() -> None:
    """Match BMUs to REPD rows, keep those inside the ENS table, and write the candidates."""
    register = json.loads((RAW_DIR / "bmunits_all.json").read_text())["body"]
    bmus = pl.DataFrame(
        json.loads(register) if isinstance(register, str) else register, infer_schema_length=None
    )
    bmus = bmus.filter(pl.col("fuelType") == "WIND", pl.col("interconnectorId").is_null())
    repd = repd_wind()
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(repd["easting"].to_numpy(), repd["northing"].to_numpy())
    cells = ens_cells()
    in_footprint = [
        h3.str_to_int(h3.latlng_to_cell(la, lo, ENS_RESOLUTION)) in cells
        for la, lo in zip(lat, lon, strict=True)
    ]
    repd = repd.with_columns(lat=pl.Series(lat), lon=pl.Series(lon), in_ens=pl.Series(in_footprint))
    names = dict(zip(repd["repd_ref"], repd["repd_name"], strict=True))
    rows = []
    for bmu in bmus.iter_rows(named=True):
        match = best_match(site_name=str(bmu["bmUnitName"]), candidates=names)
        rows.append(
            {
                "elexon_bmu_id": bmu["elexonBmUnit"],
                "bmu_name": bmu["bmUnitName"],
                "bmu_type": bmu["bmUnitType"],
                "generation_capacity_mw": bmu["generationCapacity"],
                "repd_ref": None if match is None else match[0],
                "match_ratio": None if match is None else match[1],
            }
        )
    matched = pl.DataFrame(rows, infer_schema_length=None).join(repd, on="repd_ref", how="left")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    matched.write_parquet(WORK_DIR / "farm_candidates.parquet")
    matched_count = matched["repd_name"].is_not_null().sum()
    print(f"WIND BMUs {bmus.height}; matched to operational onshore REPD {matched_count}")
    print(f"matched and inside the ENS table: {matched.filter(pl.col('in_ens')).height}")
    print(
        matched.filter(pl.col("in_ens"))
        .select(
            "elexon_bmu_id",
            "bmu_name",
            "match_ratio",
            "capacity_mw",
            "turbine_mw",
            "hub_height_m",
            "operational",
        )
        .sort("capacity_mw", descending=True)
    )


if __name__ == "__main__":
    main()
