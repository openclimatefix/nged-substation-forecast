"""Part 1 of experiment 299: when does arm D fall behind BG2, by week, lead, hour, and sunshine?

OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 systemd-run --user --scope -p MemoryMax=8G \
        -p OOMPolicy=continue uv run --no-sync python ml-experiments/00299-*/e00299_timing.py
"""

import e00299_common as common  # isort: skip
import numpy as np
import polars as pl


def main() -> None:
    """Write `work/timing.json` and print the headline numbers."""
    m = common.read_master()
    gap = m["gap"].to_numpy()
    blocks = m["init_week"].to_numpy()
    daylight = m["day"].to_numpy()
    month = m["init_month"].to_numpy()
    init_day = m["init_day"].to_numpy()
    lead = m["lead_hours"].to_numpy()
    out: dict = {"rows": m.height}
    # Reproduction gate: arm D minus BG2 in April to June (experiment 266: +0.319).
    amj = np.isin(month, common.AMJ)
    out["gate_amj_all"] = common.block_mean_interval(values=gap, mask=amj, blocks=blocks)
    # Weekly gap per site, daylight and night, by valid week.
    weekly = (
        m.group_by("site", "valid_week", "day")
        .agg(pl.col("gap").mean(), pl.len())
        .sort("site", "valid_week", "day")
    )
    out["weekly"] = weekly.to_dicts()
    out["by_site"] = {}
    for site in common.SITES:
        s = m["site"].to_numpy() == site
        r: dict = {}
        r["amj_daylight"] = common.block_mean_interval(
            values=gap, mask=s & amj & daylight, blocks=blocks
        )
        r["jas_daylight"] = common.block_mean_interval(
            values=gap, mask=s & np.isin(month, common.JAS) & daylight, blocks=blocks
        )
        r["amj_night"] = common.block_mean_interval(
            values=gap, mask=s & amj & ~daylight, blocks=blocks
        )
        # Share of the AMJ daylight excess (sum of gap) valid within 4 weeks after the change,
        # and its share of rows, so a flat excess scores its row share.
        sel = s & amj & daylight & np.isfinite(gap)
        total = gap[sel].sum()
        early = sel & (init_day >= "2026-04-01") & (init_day < "2026-04-26")
        short = sel & (lead < 24)
        r["share_excess_first_4_weeks"] = float(gap[early].sum() / total) if total else None
        r["share_rows_first_4_weeks"] = float(early.sum() / sel.sum())
        r["share_excess_leads_under_24h"] = float(gap[short].sum() / total) if total else None
        r["share_rows_leads_under_24h"] = float(short.sum() / sel.sum())
        # Step test: 4 weeks before versus 4 weeks after each clock change, daylight and night.
        for name, day in (
            ("spring_2026", common.CLOCK_CHANGE_SPRING_2026),
            ("autumn_2025", common.CLOCK_CHANGE_AUTUMN_2025),
        ):
            lo = str(np.datetime64(day) - np.timedelta64(28, "D"))
            hi = str(np.datetime64(day) + np.timedelta64(28, "D"))
            for label, mask in (("daylight", daylight), ("night", ~daylight)):
                before = s & mask & (init_day >= lo) & (init_day < day)
                after = s & mask & (init_day >= day) & (init_day < hi)
                r[f"{name}_{label}_before"] = common.block_mean_interval(
                    values=gap, mask=before, blocks=blocks
                )
                r[f"{name}_{label}_after"] = common.block_mean_interval(
                    values=gap, mask=after, blocks=blocks
                )
        # AMJ daylight by lead band, by local hour, by realised clear-sky band.
        for band in ("0-6", "6-24", "24-48", "48-168"):
            r[f"amj_daylight_lead_{band}"] = common.block_mean_interval(
                values=gap,
                mask=s & amj & daylight & (m["lead_band"].to_numpy() == band),
                blocks=blocks,
            )
        csi = m["realised_csi"].to_numpy()
        for name, lo, hi in (("overcast", 0.0, 0.3), ("mixed", 0.3, 0.7), ("clear", 0.7, 9.0)):
            r[f"amj_daylight_csi_{name}"] = common.block_mean_interval(
                values=gap, mask=s & amj & daylight & (csi >= lo) & (csi < hi), blocks=blocks
            )
        hours = (
            m.filter((pl.col("site") == site) & pl.col("init_month").is_in(list(common.AMJ)))
            .group_by("local_hour")
            .agg(pl.col("gap").mean(), pl.len())
            .sort("local_hour")
        )
        r["amj_by_local_hour"] = hours.to_dicts()
        out["by_site"][site] = r
    # Gap by lead band, whole year versus AMJ, all sites.
    out["lead_share_all_rows_daylight_amj"] = (
        m.filter(pl.col("day") & pl.col("init_month").is_in(list(common.AMJ)))
        .group_by("lead_band")
        .agg(pl.col("gap").sum(), pl.len())
        .sort("lead_band")
        .to_dicts()
    )
    out["checks"] = (
        m.group_by("site")
        .agg(
            (pl.col("med_d") - pl.col("mean_d")).mean().alias("median_minus_mean"),
            (pl.col("mean_d") - pl.col("actual")).abs().mean().alias("mean_forecast_mae_mw"),
            (pl.col("med_d") - pl.col("actual")).abs().mean().alias("median_forecast_mae_mw"),
        )
        .sort("site")
        .to_dicts()
    )
    print(out["checks"])
    common.write_json_atomic(path=common.WORK_DIR / "timing.json", content=out)
    print(out["gate_amj_all"])
    for site, r in out["by_site"].items():
        print(
            site,
            "AMJ day",
            round(r["amj_daylight"]["est"], 3),
            "JAS day",
            round(r["jas_daylight"]["est"], 3),
            "night",
            round(r["amj_night"]["est"], 3),
            "first-4wk share",
            r["share_excess_first_4_weeks"] and round(r["share_excess_first_4_weeks"], 2),
            "(rows",
            round(r["share_rows_first_4_weeks"], 2),
            ") <24h share",
            r["share_excess_leads_under_24h"] and round(r["share_excess_leads_under_24h"], 2),
            "(rows",
            round(r["share_rows_leads_under_24h"], 2),
            ")",
        )


if __name__ == "__main__":
    main()
