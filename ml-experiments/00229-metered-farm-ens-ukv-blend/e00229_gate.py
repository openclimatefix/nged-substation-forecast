"""Data gate for the UKV hours of experiment 229, run before any arm is scored.

Checks, each printed and written to `gate.csv` under the experiment's data folder:

1. Run completeness: the share of the needed 00 UTC runs whose status is complete, by month.
2. Valid-time convention and sun position: the mean UKV shortwave by hour of day against clear-sky
   irradiance at the farm (the UKV curve must peak with the clear-sky curve, not an hour away).
3. Weather against the cloudless baseline at the freshest-run leads 0 to 5 hours (stop rule: UKV
   shortwave through the plant must not score worse than the cloudless sky).
4. Step at the era boundary: the monthly mean ratio of UKV shortwave to clear-sky around 2025-12 and
   2026-02, which the experiment never fits across.

    uv run --no-sync python ml-experiments/00229-*/e00229_gate.py
"""

from datetime import UTC, datetime

import e00229_arms as arms
import e00229_common as common
import e00229_ukv as ukv
import numpy as np
import polars as pl
from studies.ukv_ceda_stores import read_at_instants


def main() -> None:
    """Run the four checks and write the table."""
    stores, group = ukv.open_stores()
    cells = ukv.farm_cells(group=group)
    farm_rows = pl.read_parquet(common.FARM_ROWS)
    start, end = datetime(2025, 3, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC)
    instants = pl.datetime_range(
        start, end, interval="1h", time_zone="UTC", eager=True, closed="left"
    )
    reads = read_at_instants(
        stores=stores,
        instants=instants,
        variables=["shortwave_down", "temperature_1p5m"],
        cells=cells,
    )
    usable = reads.usable
    month = instants.dt.strftime("%Y-%m")
    out = []
    completeness = (
        pl.DataFrame({"month": month, "usable": usable})
        .group_by("month")
        .agg(share=pl.col("usable").mean())
        .sort("month")
    )
    out.extend(
        {"check": "usable_instants_share", "key": r["month"], "value": r["share"]}
        for r in completeness.iter_rows(named=True)
    )
    for k, site in enumerate(common.FARMS):
        rows = farm_rows.filter(pl.col("site") == site).with_columns(
            instant=pl.col("time").dt.truncate("1h")
        )
        frame = pl.DataFrame(
            {
                "instant": instants,
                "lead_hours": reads.lead_hours,
                "usable": usable,
                "ghi": reads.values["shortwave_down"][:, k],
                "temperature_k": reads.values["temperature_1p5m"][:, k],
            }
        )
        joined = (
            rows.filter((pl.col("time") >= start) & (pl.col("time") < end))
            .join(frame, on="instant", how="inner")
            .filter(pl.col("usable") & pl.col("ghi").is_finite())
        )
        # Check 2: shape against the sun.
        by_hour = (
            joined.group_by(pl.col("time").dt.hour().alias("hour"))
            .agg(ukv=pl.col("ghi").mean(), clear=pl.col("clear_sky").mean())
            .sort("hour")
        )
        peak_ukv = by_hour.sort("ukv", descending=True)["hour"][0]
        peak_clear = by_hour.sort("clear", descending=True)["hour"][0]
        out.append(
            {
                "check": f"peak_hour_offset_{site}",
                "key": "ukv_minus_clear",
                "value": float(peak_ukv - peak_clear),
            }
        )
        # Check 3: plant fed UKV vs plant fed the cloudless sky, by lead.
        tuning = joined.filter(pl.col("time") < datetime(2025, 9, 1, tzinfo=UTC))

        def plant(frame_: pl.DataFrame, ghi: np.ndarray) -> np.ndarray:
            return arms.unit_response(
                ghi=ghi,
                temperature_c=frame_["temperature_k"].to_numpy() - 273.15,
                zenith=frame_["zenith_deg"].to_numpy(),
                azimuth=frame_["sun_azimuth_deg"].to_numpy(),
                doy=frame_["time"].dt.ordinal_day().to_numpy(),
            )

        for label, column in (("ukv", "ghi"), ("clear", "clear_sky")):
            unit_all = plant(joined, joined[column].to_numpy())
            unit_tune = plant(tuning, tuning[column].to_numpy())
            y_tune = tuning["output"].to_numpy().astype(np.float64)
            scale = float(np.sum(y_tune * unit_tune) / np.sum(unit_tune**2))
            held = (joined["time"] >= datetime(2025, 9, 1, tzinfo=UTC)).to_numpy()
            error = np.abs(joined["output"].to_numpy().astype(np.float64) - scale * unit_all)
            for lead in range(6):
                rows_lead = held & (joined["lead_hours"].to_numpy() == lead)
                if rows_lead.sum() > 0:
                    out.append(
                        {
                            "check": f"mae_{label}_{site}",
                            "key": f"lead_{lead}",
                            "value": 100.0 * float(error[rows_lead].mean()),
                        }
                    )
        # Check 4: ratio of UKV shortwave to clear-sky by month.
        ratio = (
            joined.filter(pl.col("clear_sky") > common.SUNNY_CLEAR_SKY_W_M2)
            .group_by(pl.col("time").dt.strftime("%Y-%m").alias("month"))
            .agg(ratio=(pl.col("ghi") / pl.col("clear_sky")).mean())
            .sort("month")
        )
        out.extend(
            {"check": f"ukv_over_clear_{site}", "key": r["month"], "value": r["ratio"]}
            for r in ratio.iter_rows(named=True)
        )
    table = pl.DataFrame(out)
    table.write_csv(common.DATA_DIR / "gate.csv")
    with pl.Config(tbl_rows=400, fmt_str_lengths=40):
        print(table.filter(pl.col("check") == "usable_instants_share"))
        print(table.filter(pl.col("check").str.starts_with("peak_hour")))
        mae = (
            table.filter(pl.col("check").str.starts_with("mae_"))
            .with_columns(source=pl.col("check").str.extract(r"mae_(\w+?)_", 1))
            .group_by("source", "key")
            .agg(mae=pl.col("value").mean())
            .sort("source", "key")
        )
        print(mae)
        print(
            table.filter(pl.col("check").str.starts_with("ukv_over_clear_"))
            .group_by("key")
            .agg(r=pl.col("value").mean())
            .sort("key")
        )


if __name__ == "__main__":
    main()
