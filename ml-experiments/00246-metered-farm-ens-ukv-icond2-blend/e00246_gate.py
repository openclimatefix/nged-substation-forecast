"""Data gate for the ICON-D2 hours of experiment 246, run before any arm is scored.

Checks, each printed and written to `gate.csv` under the experiment's data folder:

1. Coverage: the share of half-hours in the scored period that have a usable ICON-D2 row, by month.
2. Valid-time convention and sun position: the mean ICON-D2 shortwave by hour of day, and the offset
   between its peak hour and the clear-sky peak hour (the curves must peak together).
3. Weather against the cloudless baseline: the plant fed ICON-D2 shortwave must score better than
   the plant fed the cloudless sky (stop rule: otherwise report a data problem first).
4. Step in the product: the monthly mean ratio of ICON-D2 shortwave to clear-sky irradiance.

The UKV hours passed the equivalent gate in experiment 229, so they are not re-checked.

    uv run --no-sync python ml-experiments/00246-*/e00246_gate.py
"""

from datetime import UTC, datetime

import e00246_arms as arms
import e00246_common as common
import numpy as np
import polars as pl

TUNING_END = datetime(2025, 9, 1, tzinfo=UTC)


def plant(frame: pl.DataFrame, ghi: np.ndarray) -> np.ndarray:
    """Return the unit plant response to `ghi` at the frame's sun positions."""
    return arms.unit_response(
        ghi=ghi,
        temperature_c=frame["icon_temperature"].to_numpy(),
        zenith=frame["zenith_deg"].to_numpy(),
        azimuth=frame["sun_azimuth_deg"].to_numpy(),
        doy=frame["time"].dt.ordinal_day().to_numpy(),
    )


def main() -> None:
    """Run the four checks and write the table."""
    farm_rows = pl.read_parquet(common.FARM_ROWS)
    out: list[dict[str, object]] = []
    for site in common.FARMS:
        rows = farm_rows.filter(pl.col("site") == site)
        icon = pl.read_parquet(common.UKV_DIR / f"{site}_icon.parquet").unique(
            subset="time", keep="last"
        )
        joined = rows.join(icon, on="time", how="left").with_columns(
            usable=pl.col("icon_usable").fill_null(False) & pl.col("icon_ghi").is_finite()
        )
        out.extend(
            {"check": f"usable_share_{site}", "key": r["month"], "value": r["share"]}
            for r in joined.filter(pl.col("clear_sky") > 0)
            .group_by(pl.col("time").dt.strftime("%Y-%m").alias("month"))
            .agg(share=pl.col("usable").mean())
            .sort("month")
            .iter_rows(named=True)
        )
        used = joined.filter(pl.col("usable"))
        by_hour = (
            used.group_by(pl.col("time").dt.hour().alias("hour"))
            .agg(icon=pl.col("icon_ghi").mean(), clear=pl.col("clear_sky").mean())
            .sort("hour")
        )
        peak_icon = by_hour.sort("icon", descending=True)["hour"][0]
        peak_clear = by_hour.sort("clear", descending=True)["hour"][0]
        out.append(
            {
                "check": f"peak_hour_offset_{site}",
                "key": "icon_minus_clear",
                "value": float(peak_icon - peak_clear),
            }
        )
        # Check 3 needs the sun position columns, which the farm rows carry.
        used = used.filter(pl.col("zenith_deg").is_not_null())
        tuning = used.filter(pl.col("time") < TUNING_END)
        held = (used["time"] >= TUNING_END).to_numpy()
        y_tune = tuning["output"].to_numpy().astype(np.float64)
        for label, column in (("icon", "icon_ghi"), ("clear", "clear_sky")):
            unit_tune = plant(tuning, tuning[column].to_numpy())
            scale = float(np.sum(y_tune * unit_tune) / np.sum(unit_tune**2))
            unit_all = plant(used, used[column].to_numpy())
            error = np.abs(used["output"].to_numpy().astype(np.float64) - scale * unit_all)
            out.append(
                {
                    "check": f"mae_{label}_{site}",
                    "key": "held_out",
                    "value": 100.0 * float(error[held].mean()),
                }
            )
        ratio = (
            used.filter(pl.col("clear_sky") > common.SUNNY_CLEAR_SKY_W_M2)
            .group_by(pl.col("time").dt.strftime("%Y-%m").alias("month"))
            .agg(ratio=(pl.col("icon_ghi") / pl.col("clear_sky")).mean())
            .sort("month")
        )
        out.extend(
            {"check": f"icon_over_clear_{site}", "key": r["month"], "value": r["ratio"]}
            for r in ratio.iter_rows(named=True)
        )
    table = pl.DataFrame(out)
    table.write_csv(common.DATA_DIR / "gate.csv")
    with pl.Config(tbl_rows=400, fmt_str_lengths=40):
        print(
            table.filter(pl.col("check").str.starts_with("usable_share_"))
            .group_by("key")
            .agg(share=pl.col("value").mean())
            .sort("key")
        )
        print(table.filter(pl.col("check").str.starts_with("peak_hour")))
        print(table.filter(pl.col("check").str.starts_with("mae_")))
        print(
            table.filter(pl.col("check").str.starts_with("icon_over_clear_"))
            .group_by("key")
            .agg(r=pl.col("value").mean())
            .sort("key")
        )


if __name__ == "__main__":
    main()
