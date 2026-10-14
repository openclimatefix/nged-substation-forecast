"""Check what stage 0 did to the ENS shortwave of the validation runs (descriptive).

For each validation month, the mean daylight shortwave of the raw ENS members, of the members after
the ENS-to-CAMS mapping, and of CAMS, over the 6 solar farms; and the mean spread (standard
deviation across the 51 members) of the raw and mapped shortwave in daylight.

Output: `work/results/stage0_check.md`.

    uv run python ml-experiments/00020-*/e00020_stage0_check.py
"""

import numpy as np
import polars as pl
from e00020_common import (
    ANALYSIS_PATH,
    ENS_VALIDATION_DIR,
    MAX_ZENITH_DEGREES,
    WORK_DIR,
    site_table,
)
from e00020_fit import apply_mapping

STAGE0_PATH = WORK_DIR / "stage0" / "cams.npz"
OUTPUT_PATH = WORK_DIR / "results" / "stage0_check.md"


def main() -> None:
    """Write the monthly ratio table and the spread comparison."""
    sites = site_table().filter(pl.col("technology") == "solar")
    solar_ids = sites["time_series_id"].to_list()
    loaded = np.load(STAGE0_PATH)
    mapping = (loaded["xs"], loaded["ys"])
    cams = pl.read_parquet(ANALYSIS_PATH).select("time_series_id", "valid_time", "shortwave_cams")
    pieces = []
    for path in sorted(ENS_VALIDATION_DIR.glob("init=*.parquet")):
        frame = (
            pl.read_parquet(path)
            .filter(pl.col("time_series_id").is_in(solar_ids))
            .filter(pl.col("solar_zenith") < MAX_ZENITH_DEGREES)
            .with_columns(
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0
            )
        )
        mapped = apply_mapping(
            mapping=mapping,
            ens=frame["shortwave"].to_numpy(),
            lead_hours=frame["lead_hours"].to_numpy(),
            cos_zenith=np.cos(np.radians(frame["solar_zenith"].to_numpy())),
        )
        frame = frame.with_columns(mapped=pl.Series(mapped))
        spread = frame.group_by("time_series_id", "power_fcst_init_time", "valid_time").agg(
            raw_mean=pl.col("shortwave").mean(),
            mapped_mean=pl.col("mapped").mean(),
            raw_sd=pl.col("shortwave").std(),
            mapped_sd=pl.col("mapped").std(),
        )
        pieces.append(spread)
    table = (
        pl.concat(pieces)
        .join(cams, on=["time_series_id", "valid_time"], how="left")
        .drop_nulls("shortwave_cams")
        .with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
    )
    monthly = (
        table.group_by("month")
        .agg(
            raw=pl.col("raw_mean").mean(),
            mapped=pl.col("mapped_mean").mean(),
            cams=pl.col("shortwave_cams").mean(),
            raw_sd=pl.col("raw_sd").mean(),
            mapped_sd=pl.col("mapped_sd").mean(),
        )
        .sort("month")
        .with_columns(
            raw_ratio=pl.col("raw") / pl.col("cams"), mapped_ratio=pl.col("mapped") / pl.col("cams")
        )
    )
    year = table.select(
        raw=pl.col("raw_mean").mean() / pl.col("shortwave_cams").mean(),
        mapped=pl.col("mapped_mean").mean() / pl.col("shortwave_cams").mean(),
        sd_ratio=pl.col("mapped_sd").mean() / pl.col("raw_sd").mean(),
    ).row(0, named=True)
    lines = [
        "# Stage 0 check (solar farms, daylight, validation year)",
        "",
        (
            f"- Year: raw ENS / CAMS = {year['raw']:.3f}, mapped / CAMS = {year['mapped']:.3f}, "
            f"member spread after / before = {year['sd_ratio']:.3f}"
        ),
        "",
        "| month | raw / CAMS | mapped / CAMS | raw spread | mapped spread |",
        "| --- | --- | --- | --- | --- |",
        *[
            f"| {r['month']} | {r['raw_ratio']:.3f} | {r['mapped_ratio']:.3f} | "
            f"{r['raw_sd']:.1f} | {r['mapped_sd']:.1f} |"
            for r in monthly.iter_rows(named=True)
        ],
    ]
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
