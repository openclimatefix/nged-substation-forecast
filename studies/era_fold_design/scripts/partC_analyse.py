"""Part C analysis: intervals under each design and setting, and the paired change of each design.

Usage: partC_analyse.py {pooled|both} [domain...]
Every call filters on the setting explicitly and asserts each arm's row count (sites x hours x seeds).
The second setting is analysed for the planned contrasts, the references and the contrasts listed in
out/sens_extra_contrasts.json (those near the 5% line under the primary setting).
"""
import json
import sys

import numpy as np
from common import *  # noqa: F403
import ens_forecast_horizons as eh
from studies.bootstrap import (
    _resample_bounds,
    bootstrap_absolute,
    bootstrap_difference,
    paired_differences,
)

M = eh.METRIC
DAYS = list(eh.BAND_DAYS)
COMPARE = {"D0trim": "D0", "I": "D0trim", "I2": "D0trim", "II": "D0", "IIr": "D0"}
mode = sys.argv[1]
domains = sys.argv[2:] or ["wind", "solar"]
BASELINE_PREFIXES = ("climatology", "persistence", "diurnal_persistence", "smart_persistence")
extra_path = OUT / "sens_extra_contrasts.json"
extra = {tuple(x) for x in json.loads(extra_path.read_text())} if extra_path.exists() else set()


def is_baseline(arm: str) -> bool:
    return arm.startswith(BASELINE_PREFIXES)


rows, abs_rows = [], []
for domain in domains:
    designs = [d for d in ("D0", "D0trim", "I", "I2", "II", "IIr") if (OUT / f"losses_ens_{domain}_{d}.parquet").exists()]
    prim = {d: pl.read_parquet(OUT / f"losses_ens_{domain}_{d}.parquet") for d in designs}
    sens = {}
    for d in designs:
        p = OUT / f"losses_ens_{domain}_{d}_sens.parquet"
        p2 = OUT / f"losses_ens_{domain}_{d}_sens_extra.parquet"
        if mode == "both" and p.exists():
            sens[d] = pl.concat(
                [
                    pl.read_parquet(p),
                    *([pl.read_parquet(p2)] if p2.exists() else []),
                    # baselines have no hyperparameters: the primary file's rows stand under both settings
                    prim[d].filter(pl.col("arm").str.contains("^(climatology|persistence|diurnal_persistence|smart_persistence)")).with_columns(setting=pl.lit("sensitivity")),
                ],
                how="diagonal_relaxed",
            ).unique(subset=["arm", "setting", "site", "time", "seed"], keep="first", maintain_order=True)
    n_exp = {d: expected_rows(kind="ens_", study=domain, design=d) for d in designs}
    best = eh.best_baselines(losses=prim["D0"].filter(pl.col("setting") == "pooled"))
    mean = lambda d: eh.ens_arm(way="mean", day=d)
    contrasts = [(t, r, True, "planned") for t, r in eh.PLANNED]
    contrasts += [
        (mean(0), "era5", False, "reference"), (mean(1), "era5", False, "reference"),
        (mean(0), "live_gb_rich_xgb", False, "reference"), (mean(1), "live_gb_rich_xgb", False, "reference"),
        (eh.ens_arm(way="control", day=0), "era5", False, "reference"), ("live_gb_rich_xgb", "era5", False, "reference"),
        (eh.ens_arm(way="members", day=1), eh.ens_arm(way="control", day=1), False, "members"),
    ]
    for d in DAYS:
        contrasts += [(mean(d), eh.ens_arm(way="control", day=d), False, "mean vs control"),
                      (mean(d), "climatology", False, "mean vs climatology"),
                      (mean(d), best[d], False, f"mean vs best baseline (D0's: {best[d]})"),
                      (mean(d), eh.CALENDAR_ONLY, False, "mean vs calendar-only (post hoc)"),
                      (mean(d), mean(0), False, "horizon growth"),
                      (mean(d), eh.upsampling_arm(method="linear", day=d), False, "chosen upsampling vs linear")]
    contrasts += [(eh.CALENDAR_ONLY, "climatology", False, "calendar-only vs climatology")]
    seen = set()
    contrasts = [c for c in contrasts if c[0] != c[1] and not ((c[0], c[1]) in seen or seen.add((c[0], c[1])))]

    def run(setting, store, design, t, r, planned, kind):
        """One contrast at one setting, plus its change from the comparison design's."""
        lo = store[design]
        exp = n_exp[design]
        arms = set(lo.filter(pl.col("setting") == setting)["arm"].unique().to_list())
        if t not in arms or r not in arms:
            return
        if design in COMPARE:
            base_arms = set(store[COMPARE[design]].filter(pl.col("setting") == setting)["arm"].unique().to_list())
            if t not in base_arms or r not in base_arms:
                return
        sub = take(losses=lo, setting=setting, arms=[t, r], expected=exp)
        b = bootstrap_difference(losses=sub, treatment=t, reference=r, metric=M)
        rows.append({"domain": domain, "setting": setting, "design": design, "treatment": t, "reference": r,
                     "kind": kind, "planned": planned, "diff_pp": b["difference"] * 100, "lo": b["lower_95"] * 100,
                     "hi": b["upper_95"] * 100, "n_rows": b["n_rows"], "n_months": b["n_months"]})
        if design in COMPARE:
            other = COMPARE[design]
            base = take(
                losses=store[other].join(lo.select("site", "time").unique(), on=["site", "time"], how="semi"),
                setting=setting, arms=[t, r], expected=exp,
            )
            d1, m1 = paired_differences(losses=sub, treatment=t, reference=r, metric=M)
            d0, m0 = paired_differences(losses=base, treatment=t, reference=r, metric=M)
            assert d1.shape == d0.shape and (m1 == m0).all()
            ch = d1 - d0
            l, h = _resample_bounds(values=ch, months=m1)
            rows.append({"domain": domain, "setting": setting, "design": f"{design}-{other}", "treatment": t, "reference": r,
                         "kind": kind, "planned": planned, "diff_pp": float(ch.mean()) * 100, "lo": l * 100, "hi": h * 100,
                         "n_rows": ch.shape[1], "n_months": len(np.unique(m1))})

    for design in designs:
        for t, r, planned, kind in contrasts:
            run("pooled", prim, design, t, r, planned, kind)
            if design in sens and (planned or kind == "reference" or (domain, t, r) in {(x[0], x[1], x[2]) for x in extra}):
                run("sensitivity", sens, design, t, r, planned, kind)
        show = [mean(d) for d in DAYS] + [eh.ens_arm(way="control", day=d) for d in DAYS] + [
            "era5", "live_gb_rich_xgb", "climatology", eh.CALENDAR_ONLY, eh.ens_arm(way="members", day=1)
        ] + sorted(set(best.values()))
        for setting, store in (("pooled", prim), ("sensitivity", sens)):
            if design not in store:
                continue
            present = set(store[design].filter(pl.col("setting") == setting)["arm"].unique().to_list())
            for a in dict.fromkeys(show):
                if a in present:
                    sub = take(losses=store[design], setting=setting, arms=[a], expected=n_exp[design])
                    b = bootstrap_absolute(losses=sub, arm=a, metric=M)
                    abs_rows.append({"domain": domain, "setting": setting, "design": design, "arm": a,
                                     "value": b["value"] * 100, "lo": b["lower_95"] * 100, "hi": b["upper_95"] * 100,
                                     "n_rows": b["n_rows"]})
    print(domain, "best baselines (D0)", best, flush=True)
tag = "" if mode == "pooled" else "_both"
pl.DataFrame(rows).write_parquet(OUT / f"partC_contrasts{tag}.parquet")
pl.DataFrame(abs_rows).write_parquet(OUT / f"partC_absolute{tag}.parquet")
