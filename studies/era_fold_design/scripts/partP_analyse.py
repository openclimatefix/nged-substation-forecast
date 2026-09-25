"""Analyse the panel D0/D1 fits. Usage: partP_analyse.py PANEL DEVICE {pooled|sensitivity}
Writes out/partP_<panel>_<device>_<setting>.md (and json). With DEVICE cpu and pooled it also
compares D0 with the published per-row losses."""
import json
import sys

import numpy as np
from common import *  # noqa: F403
import weather_products as wp
import wind_icon_dream as wdr
from studies.bootstrap import _resample_bounds, bootstrap_absolute, bootstrap_difference, paired_differences

panel, dev, setting = sys.argv[1:4]
tag = dev + ("_sens" if setting == "sensitivity" else "")
METRIC = wp.METRIC
planned = list(wp.NEW_PLANNED_CONTRASTS["all"]) if panel == "solar_all" else list(wdr.DECIDING_CONTRASTS)
pub_path = DATA / "past_weather_v2" / ("solar_all" if panel == "solar_all" else "wind_icon_dream") / "losses.parquet"
los = {d: pl.read_parquet(OUT / f"panel_{panel}_{d}_{tag}.parquet") for d in ("D0", "D1")}
n_exp = pl.read_parquet(OUT / f"panel_{panel}_D0_folds.parquet").height * 3
assert n_exp == pl.read_parquet(OUT / f"panel_{panel}_D1_folds.parquet").height * 3
arms = sorted(los["D0"]["arm"].unique().to_list())
for d in los:
    take(losses=los[d], setting=setting, arms=arms, expected=n_exp)
lines = [f"### {panel}, {setting} setting, device {dev}: planned contrasts (percentage points of capacity; `*` = 95% interval excludes zero)\n",
         "| Contrast | D0 | D1 (post-upgrade era rotated by 2) | D1 − D0 |\n|---|---|---|---|"]
res = {"contrasts": [], "abs": []}
f = lambda v, l, h: f"{v:+.3f} [{l:+.3f}, {h:+.3f}]" + ("" if l <= 0 <= h else "*")
for t, r in planned:
    arr, cells = {}, []
    for d in ("D0", "D1"):
        sub = take(losses=los[d], setting=setting, arms=[t, r], expected=n_exp)
        b = bootstrap_difference(losses=sub, treatment=t, reference=r, metric=METRIC)
        arr[d] = paired_differences(losses=sub, treatment=t, reference=r, metric=METRIC)
        cells.append(f(b["difference"] * 100, b["lower_95"] * 100, b["upper_95"] * 100))
        res["contrasts"].append({"t": t, "r": r, "design": d, "diff": b["difference"] * 100, "lo": b["lower_95"] * 100, "hi": b["upper_95"] * 100})
    assert (arr["D0"][1] == arr["D1"][1]).all()
    ch = arr["D1"][0] - arr["D0"][0]
    lo, hi = _resample_bounds(values=ch, months=arr["D0"][1])
    cells.append(f(float(ch.mean()) * 100, lo * 100, hi * 100))
    res["contrasts"].append({"t": t, "r": r, "design": "D1-D0", "diff": float(ch.mean()) * 100, "lo": lo * 100, "hi": hi * 100})
    lines.append(f"| {t} − {r} | " + " | ".join(cells) + " |")
lines += ["", "Absolute mean absolute error (percent of capacity) with 95% interval:\n", "| Arm | D0 | D1 |\n|---|---|---|"]
for a in arms:
    cells = []
    for d in ("D0", "D1"):
        b = bootstrap_absolute(losses=take(losses=los[d], setting=setting, arms=[a], expected=n_exp), arm=a, metric=METRIC)
        cells.append(f"{b['value'] * 100:.3f} [{b['lower_95'] * 100:.2f}, {b['upper_95'] * 100:.2f}]")
        res["abs"].append({"arm": a, "design": d, "value": b["value"] * 100})
    lines.append(f"| {a} | " + " | ".join(cells) + " |")
info = {d: json.loads((OUT / f"panel_{panel}_{d}_{tag}_info.json").read_text()) for d in ("D0", "D1")}
lines += ["", f"Coverage: D0 leaves {info['D0']['uncovered_hours']:,} of {info['D0']['rows']:,} hours ({100 * info['D0']['share']:.1f}%) in uncovered cells, calendar months {info['D0']['months']}; D1 leaves {info['D1']['uncovered_hours']:,}. Rows per arm: {n_exp:,} (hours x 3 seeds), asserted for every arm and design."]
if dev == "cpu" and setting == "pooled":
    pub = take(losses=pl.read_parquet(pub_path), setting="pooled", arms=arms, expected=n_exp)
    k = ["arm", "site", "time", "seed"]
    j = los["D0"].select(*k, new=METRIC).join(pub.select(*k, old=METRIC), on=k, how="full", coalesce=True)
    diff = (j["new"] - j["old"]).abs()
    lines.append(f"\nReproduction of the published per-row losses under D0 (CPU): {j.height:,} rows, unmatched {int(j['new'].is_null().sum() + j['old'].is_null().sum())}, bit-identical share {float((diff == 0).mean()):.6f}, max absolute difference {float(diff.max()):.3g}.")
md = "\n".join(lines)
(OUT / f"partP_{panel}_{tag}.md").write_text(md)
(OUT / f"partP_{panel}_{tag}.json").write_text(json.dumps(res, indent=1))
print(md)
