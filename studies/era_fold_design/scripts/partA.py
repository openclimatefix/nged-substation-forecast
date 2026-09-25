"""Part A: coverage arithmetic for D0, D1, D2 (no fits)."""
import json
from common import *  # noqa: F403
from studies.cross_validation import (
    UKV_UPGRADE_MONTH, calendar_month_coverage, uncovered_months, search_fold_offsets, cut_eras,
)

result = {}
for study, builder in (("wind", wind_pre_fold), ("solar", solar_pre_fold)):
    pre = builder()
    print(study, pre.height, pre["time"].min(), pre["time"].max(), flush=True)
    zero = cut_eras(frame=pre, first_months=[UKV_UPGRADE_MONTH], fold_offsets={0: 0, 1: 0})
    d0f = d0(pre)
    key = ["site", "time"]
    same = zero.sort(key).select(key + ["fold"]).equals(d0f.sort(key).select(key + ["fold"]))
    found = search_fold_offsets(frame=pre, first_months=[UKV_UPGRADE_MONTH])
    designs = {"D0": d0f, "D2": d2(pre)}
    all_offsets = {}
    for rot in range(5):
        f = d1(pre, {0: 0, 1: rot})
        cov = calendar_month_coverage(frame=f)
        all_offsets[rot] = int(uncovered_months(coverage=cov)["n_scored"].sum())
        designs[f"D1_rot{rot}"] = f
    best = min(all_offsets, key=all_offsets.get)
    designs["D1"] = designs[f"D1_rot{best}"]
    out = {"cut_eras_zero_equals_with_eras": same, "search_found": [dict(o) for o in found],
           "uncovered_hours_by_rotation": all_offsets, "best_rotation": best, "designs": {}}
    total = pre.height
    for name, f in designs.items():
        cov = calendar_month_coverage(frame=f)
        unc = uncovered_months(coverage=cov)
        per_site = {}
        for site in sorted(f["site"].unique()):
            n = int(f.filter(pl.col("site") == site).height)
            u = unc.filter(pl.col("site") == site)
            per_site[site] = {"hours": n, "uncovered": int(u["n_scored"].sum()),
                              "months": sorted(set(u["calendar_month"].to_list())),
                              "cells": u.height}
        entry = {"hours": total, "uncovered": int(unc["n_scored"].sum()),
                 "months": sorted(set(unc["calendar_month"].to_list())), "per_site": per_site}
        # post-upgrade hours scored by models trained on no post rows
        post = f.filter(pl.col("month") >= UKV_UPGRADE_MONTH)
        trained_post = f.group_by("site", "fold").agg(
            n_post=(pl.col("month") >= UKV_UPGRADE_MONTH).sum(), n=pl.len())
        tot_by_site = f.group_by("site").agg(pl.col("month").ge(UKV_UPGRADE_MONTH).sum().alias("tp"))
        j = post.join(tot_by_site, on="site").join(
            post.group_by("site", "fold").agg(sp=pl.len()), on=["site", "fold"])
        # training post rows for a held-out fold = site total post - post rows in that fold
        noP = j.filter(pl.col("tp") - pl.col("sp") == 0)
        entry["post_hours"] = post.height
        entry["post_scored_with_no_post_training"] = noP.height
        out["designs"][name] = entry
    result[study] = out
    print(json.dumps({k: (v if k != "designs" else {n: (d["uncovered"], d["months"], d["post_scored_with_no_post_training"], d["post_hours"]) for n, d in v.items()}) for k, v in out.items()}, default=str, indent=1), flush=True)
(OUT / "partA.json").write_text(json.dumps(result, indent=1, default=str))
