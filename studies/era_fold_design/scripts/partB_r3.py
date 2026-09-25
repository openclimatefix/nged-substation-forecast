"""D1 with rotation 3 (a second covering rotation): planned contrasts at the primary setting, with the change from D0 and from D1 (rotation 2)."""
import numpy as np
from common import *  # noqa: F403
import weather_products as wp
import wind_products as wd
from studies.bootstrap import _resample_bounds, bootstrap_absolute, bootstrap_difference, paired_differences

M = wp.METRIC
CONTRASTS = {"wind": list(wd.REPORTED_CONTRASTS), "solar": list(wp.PANELS["long"].planned)}
NP = {"wind": 4, "solar": 6}
L = ["### D1 with rotation 3 versus rotation 2 (Part B, primary setting, CPU)\n",
     "| Study | Contrast | D0 | D1 (rot 2) | D1 (rot 3) | rot 3 − D0 | rot 3 − rot 2 |\n|---|---|---|---|---|---|---|"]
f = lambda v, l, h: f"{v:+.3f} [{l:+.3f}, {h:+.3f}]" + ("" if l <= 0 <= h else "*")
absl = ["\n| Study | Arm | D0 | D1 (rot 2) | D1 (rot 3) |\n|---|---|---|---|---|"]
for study in ("wind", "solar"):
    los = {d: pl.read_parquet(OUT / f"losses_{study}_{d}.parquet") for d in ("D0", "D1", "D1r3")}
    n = expected_rows(kind="", study=study, design="D0")
    assert n == expected_rows(kind="", study=study, design="D1r3")
    arms = sorted(los["D0"]["arm"].unique())
    for d in los:
        take(losses=los[d], setting="pooled", arms=arms, expected=n)
    for a in arms:
        cells = []
        for d in ("D0", "D1", "D1r3"):
            b = bootstrap_absolute(losses=take(losses=los[d], setting="pooled", arms=[a], expected=n), arm=a, metric=M)
            cells.append(f"{b['value'] * 100:.3f} [{b['lower_95'] * 100:.2f}, {b['upper_95'] * 100:.2f}]")
        absl.append(f"| {study} | {a} | " + " | ".join(cells) + " |")
    for i, (t, r) in enumerate(CONTRASTS[study]):
        arr, cells = {}, []
        for d in ("D0", "D1", "D1r3"):
            sub = take(losses=los[d], setting="pooled", arms=[t, r], expected=n)
            b = bootstrap_difference(losses=sub, treatment=t, reference=r, metric=M)
            arr[d] = paired_differences(losses=sub, treatment=t, reference=r, metric=M)
            cells.append(f(b["difference"] * 100, b["lower_95"] * 100, b["upper_95"] * 100))
        for other in ("D0", "D1"):
            assert (arr["D1r3"][1] == arr[other][1]).all()
            ch = arr["D1r3"][0] - arr[other][0]
            lo, hi = _resample_bounds(values=ch, months=arr["D0"][1])
            cells.append(f(float(ch.mean()) * 100, lo * 100, hi * 100))
        L.append(f"| {study} | {t} − {r}{'' if i < NP[study] else ' (exploratory)'} | " + " | ".join(cells) + " |")
(OUT / "partB_r3.md").write_text("\n".join(L + absl))
print("\n".join(L + absl))
