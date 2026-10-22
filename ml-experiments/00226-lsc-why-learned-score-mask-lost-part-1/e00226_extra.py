# ruff: noqa: PLR0915
"""The gap decomposition and the harm split by Primary substation, for experiment 226.

Prints (and stores in `extra.json`) (1) the 7.31 against 6.32 gap on experiment 163's 7 shared
load shifts split into each load shift's row-weighted contribution, and (2) each combined and split
arm's ordinary-day contrast against no mask by Primary substation and by season.

    OMP_NUM_THREADS=1 uv run --no-sync python e00226_extra.py
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from e00226_common import WORK_DIR
from e00226_report import SITES, WINTER, bg2, boot, fmt, load

ARMS = (
    "PAIR_NEW",
    "PAIR_GATED",
    "LEARNED_R",
    "U1",
    "U2",
    "U3",
    "U4",
    "L_BOTH",
    "L_ONLY",
    "L_ONLY_NOBAL",
)
KEYS = ["site", "power_fcst_init_time", "valid_time"]


def main() -> None:
    """Print the decomposition and the harm split."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    meta, after, scores, _tails, _spans = load()
    meta = meta.with_columns(
        mo=pl.col("valid_time").dt.month(), month=pl.col("valid_time").dt.strftime("%Y-%m")
    )
    report: dict = {}

    # ---- ordinary harm by site and by season ----
    sc = scores.join(meta.select("site", "row", "ordinary", "month", "mo"), on=["site", "row"])
    sc = sc.filter(pl.col("ordinary"))
    wide = (
        sc.group_by("site", "row", "arm")
        .agg(qs=pl.col("qs").mean())
        .pivot(on="arm", index=["site", "row"], values="qs")
        .join(meta.select("site", "row", "month", "mo"), on=["site", "row"])
    )
    print("== ordinary-day contrast by Primary (arm minus no mask, points) ==")
    by_site = {}
    for arm in ARMS:
        by_site[arm] = {
            s: boot(num=(g[arm] - g["K0"]).to_numpy().astype(float), groups=g["month"].to_numpy())
            for s in SITES
            for g in [wide.filter(pl.col("site") == s)]
        }
        print(arm, {s: round(v[0], 3) for s, v in by_site[arm].items()})
    report["by_site"] = by_site
    print("\n== leave one Primary out, ordinary days (arm minus no mask, pooled) ==")
    loso = {}
    for arm in ("PAIR_NEW", "LEARNED_R", "U1", "U2", "U4", "L_ONLY"):
        loso[arm] = {
            s: boot(num=(g[arm] - g["K0"]).to_numpy().astype(float), groups=g["month"].to_numpy())
            for s in SITES
            for g in [wide.filter(pl.col("site") != s)]
        }
        print(arm, {s: round(v[0], 3) for s, v in loso[arm].items()})
    report["loso_harm"] = loso
    print("\n== ordinary-day contrast by season (arm minus no mask) ==")
    seasons = {
        "winter DJF": [12, 1, 2],
        "spring MAM": [3, 4, 5],
        "summer JJA": [6, 7, 8],
        "autumn SON": [9, 10, 11],
    }
    by_season = {}
    for arm in ARMS:
        by_season[arm] = {}
        for name, months in seasons.items():
            g = wide.filter(pl.col("mo").is_in(months))
            by_season[arm][name] = boot(
                num=(g[arm] - g["K0"]).to_numpy().astype(float), groups=g["month"].to_numpy()
            )
        print(arm, {k: fmt(v) for k, v in by_season[arm].items()})
    report["by_season"] = by_season
    print("\nNovember to March:", WINTER)

    # ---- gap decomposition on the 7 shared load shifts ----
    ev = after.join(meta, on=["site", "row"], how="inner")
    cols = ["PAIR_NEW", "LEARNED_R", "U1", "U4"]
    parts = []
    for arm in ["K0", *cols]:
        s = scores.filter(pl.col("arm") == arm)
        j = (
            ev.join(s.drop("fold"), on=["site", "row"], how="left")
            if arm in ("K0", "PAIR_NEW")
            else ev.join(s, on=["site", "row", "fold"], how="left")
        )
        parts.append(
            j.select(
                "site",
                "card",
                "row",
                "power_fcst_init_time",
                "valid_time",
                "run_period",
                pl.lit(arm).alias("arm"),
                "qs",
            )
        )
    long = pl.concat(parts).pivot(
        on="arm",
        index=["site", "card", "row", "power_fcst_init_time", "valid_time", "run_period"],
        values="qs",
    )
    ext = bg2().select(*KEYS, bg2_qs=pl.col("qs"))
    long = long.with_columns(
        pl.col("power_fcst_init_time").dt.replace_time_zone("UTC"),
        pl.col("valid_time").dt.replace_time_zone("UTC"),
    ).join(ext, on=KEYS, how="left")
    shared = long.filter(
        pl.col("run_period").is_in(["scored", "replicate"]) & pl.col("bg2_qs").is_not_null()
    )
    n = shared.height
    dec = (
        shared.group_by("site", "card")
        .agg(
            rows=pl.len(),
            K0=pl.col("K0").mean(),
            PAIR_NEW=pl.col("PAIR_NEW").mean(),
            LEARNED_R=pl.col("LEARNED_R").mean(),
            U1=pl.col("U1").mean(),
            gap_R_minus_pair=(pl.col("LEARNED_R") - pl.col("PAIR_NEW")).sum() / n,
            gap_U1_minus_pair=(pl.col("U1") - pl.col("PAIR_NEW")).sum() / n,
        )
        .sort("gap_R_minus_pair", descending=True)
    )
    print(
        "\n== gap decomposition on the 7 shared load shifts (contribution = row-weighted points) =="
    )
    print(dec)
    print(
        "sum of contributions R minus pair:",
        float(dec["gap_R_minus_pair"].sum()),
        "U1 minus pair:",
        float(dec["gap_U1_minus_pair"].sum()),
        "rows",
        n,
    )
    report["decomposition"] = dec.to_dicts()
    print("\n== leave one Primary out, 7 shared load shifts, arm minus pair-alarm mask ==")
    loso_gain = {}
    shared = shared.with_columns(
        pl.col("U1"), pl.col("LEARNED_R"), pl.col("PAIR_NEW"), pl.col("U4")
    )
    for arm in ("LEARNED_R", "U1", "U4"):
        loso_gain[arm] = {}
        for s in SITES:
            g = shared.filter(pl.col("site") != s)
            if g.height:
                loso_gain[arm][s] = boot(
                    num=(g[arm] - g["PAIR_NEW"]).to_numpy().astype(float),
                    groups=g["card"].to_numpy(),
                )
        print(arm, {s: round(v[0], 3) for s, v in loso_gain[arm].items()})
    report["loso_gain"] = loso_gain
    (WORK_DIR / "extra.json").write_text(json.dumps(report, default=str, indent=1))
    _ = np


if __name__ == "__main__":
    main()
