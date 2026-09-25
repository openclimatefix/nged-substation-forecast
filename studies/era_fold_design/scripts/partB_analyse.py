"""Part B analysis: reproduction check, intervals at both settings, and paired changes from D0.

Every call filters on the setting explicitly and asserts each arm's row count.
"""

import json

import numpy as np
from common import *  # noqa: F403
import weather_products as wp
import wind_products as wd
from studies.bootstrap import (
    _resample_bounds,
    bootstrap_absolute,
    bootstrap_difference,
    paired_differences,
)

METRIC = wp.METRIC
PUB = {
    "wind": DATA / "beam_diffuse_wind_products" / "losses.parquet",
    "solar": DATA / "past_weather_v2" / "solar_long" / "losses.parquet",
}
CONTRASTS = {"wind": list(wd.REPORTED_CONTRASTS), "solar": list(wp.PANELS["long"].planned)}
PLANNED_N = {"wind": 4, "solar": 6}
SETTINGS = {"pooled": "", "sensitivity": "_sens"}
result = {"repro": {}, "abs": [], "contrasts": []}
for study in ("wind", "solar"):
    for setting, suffix in SETTINGS.items():
        losses = {
            d: pl.read_parquet(OUT / f"losses_{study}_{d}{suffix}.parquet")
            for d in ("D0", "D1", "D2")
        }
        exp = {d: expected_rows(kind="", study=study, design=d) for d in losses}
        assert len(set(exp.values())) == 1, exp
        n_exp = exp["D0"]
        arms = sorted(losses["D0"]["arm"].unique().to_list())
        for d in losses:
            take(losses=losses[d], setting=setting, arms=arms, expected=n_exp)
        pub = take(
            losses=pl.read_parquet(PUB[study]), setting=setting, arms=arms, expected=n_exp
        )
        k = ["arm", "site", "time", "seed"]
        j = (
            losses["D0"]
            .select(*k, new=pl.col(METRIC))
            .join(pub.select(*k, old=pl.col(METRIC)), on=k, how="full", coalesce=True)
        )
        diff = (j["new"] - j["old"]).abs()
        rep = {
            "rows_new": j["new"].len() - j["new"].null_count(),
            "rows_pub": j["old"].len() - j["old"].null_count(),
            "unmatched": int(j["new"].is_null().sum() + j["old"].is_null().sum()),
            "exact_equal_share": float((diff == 0).mean()),
            "max_abs_diff": float(diff.max()),
            "arms": arms,
        }
        result["repro"][f"{study}/{setting}"] = rep
        print(study, setting, {a: b for a, b in rep.items() if a != "arms"}, flush=True)
        for a in arms:
            for d in ("D0", "D1", "D2"):
                sub = take(losses=losses[d], setting=setting, arms=[a], expected=n_exp)
                r = bootstrap_absolute(losses=sub, arm=a, metric=METRIC)
                result["abs"].append(
                    {
                        "study": study,
                        "setting": setting,
                        "arm": a,
                        "design": d,
                        "value": r["value"] * 100,
                        "lo": r["lower_95"] * 100,
                        "hi": r["upper_95"] * 100,
                        "n_rows": r["n_rows"],
                    }
                )
        for n, (t, ref) in enumerate(CONTRASTS[study]):
            planned = n < PLANNED_N[study]
            arrs = {}
            for d in ("D0", "D1", "D2"):
                sub = take(losses=losses[d], setting=setting, arms=[t, ref], expected=n_exp)
                r = bootstrap_difference(losses=sub, treatment=t, reference=ref, metric=METRIC)
                arrs[d] = paired_differences(losses=sub, treatment=t, reference=ref, metric=METRIC)
                result["contrasts"].append(
                    {
                        "study": study,
                        "setting": setting,
                        "treatment": t,
                        "reference": ref,
                        "design": d,
                        "planned": planned,
                        "diff_pp": r["difference"] * 100,
                        "lo": r["lower_95"] * 100,
                        "hi": r["upper_95"] * 100,
                        "n_rows": r["n_rows"],
                        "n_months": r["n_months"],
                    }
                )
            for d in ("D1", "D2"):
                assert arrs[d][0].shape == arrs["D0"][0].shape
                assert (arrs[d][1] == arrs["D0"][1]).all()
                ch = arrs[d][0] - arrs["D0"][0]
                lo, hi = _resample_bounds(values=ch, months=arrs["D0"][1])
                result["contrasts"].append(
                    {
                        "study": study,
                        "setting": setting,
                        "treatment": t,
                        "reference": ref,
                        "design": f"{d}-D0",
                        "planned": planned,
                        "diff_pp": float(ch.mean()) * 100,
                        "lo": lo * 100,
                        "hi": hi * 100,
                        "n_rows": ch.shape[1],
                        "n_months": len(np.unique(arrs["D0"][1])),
                    }
                )
(OUT / "partB.json").write_text(json.dumps(result["repro"], indent=1))
pl.DataFrame(result["contrasts"]).write_parquet(OUT / "partB_contrasts.parquet")
pl.DataFrame(result["abs"]).write_parquet(OUT / "partB_absolute.parquet")
