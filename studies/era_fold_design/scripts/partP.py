"""D1 (post-upgrade era rotated) on the past-solar `all` panel and on wind_icon_dream, planned contrasts, primary setting.

Usage: partP.py {solar_all|wind_dream} {D0|D1}   fits (CPU, so D0 can reproduce the published losses bit for bit)
       partP.py analyse
"""
import json
import sys

import numpy as np
from common import *  # noqa: F403
import weather_products as wp
from common import _add_time_features, _wind_sites
import wind_icon_dream as wdr
from run_experiment import run_all
from studies.bootstrap import (
    _resample_bounds, bootstrap_absolute, bootstrap_difference, paired_differences,
)
from studies.cross_validation import (
    calendar_month_coverage, rotate_folds, search_fold_offsets, uncovered_months,
)

import os
DEV = os.environ.get("DEVICE", "cpu")
if DEV == "cuda":
    import studies.cross_validation as _cv
    _orig = _cv.booster_parameters
    _cv.booster_parameters = lambda **kw: {**_orig(**kw), "device": "cuda"}
METRIC = wp.METRIC
PANEL = wp.PANELS["all"]
CONFIG = {
    "solar_all": {"planned": list(wp.NEW_PLANNED_CONTRASTS["all"]), "pub": DATA / "past_weather_v2" / "solar_all" / "losses.parquet"},
    "wind_dream": {"planned": list(wdr.DECIDING_CONTRASTS), "pub": DATA / "past_weather_v2" / "wind_icon_dream" / "losses.parquet"},
}


def pre_fold(panel: str) -> pl.DataFrame:
    if panel == "solar_all":
        rows = solar_common_rows(frame=solar_joined(products=PANEL.products))
        rows = rows.filter(pl.col("time") >= PANEL.first_time)
        return _add(rows)
    sites = _wind_sites()
    base = wind_common_rows(frame=wind_joined(sites=sites))
    return _add(base.join(wdr.icon_dream_site_frame(sites=sites), on=["site", "time"], how="inner"))


def _add(rows):
    return _add_time_features(dataset=rows)


def design_frame(panel: str, design: str) -> pl.DataFrame:
    frame = with_eras(frame=pre_fold(panel))
    if design == "D1":
        frame = rotate_folds(frame=frame, fold_offsets={0: 0, 1: 2})
    if panel == "solar_all":
        frame = finish_solar(frame)
    else:
        frame = frame.with_columns(constrained=pl.lit(False), cap_mw=pl.lit(None, dtype=pl.Float64)) if "cap_mw" not in frame.columns else frame
    return frame


def arms_features(panel: str) -> dict[str, tuple[str, ...]]:
    if panel == "solar_all":
        cols = wp._arm_columns(products=PANEL.products, with_snapshot_arms=False)
        shared = (*wp.SHARED_FEATURES, "era_code")
        return {a: (*shared, *cols[a]) for a in {x for c in CONFIG[panel]["planned"] for x in c}}
    return {a: (*wdr.SHARED_FEATURES, *wdr._wind_columns(product=a.removesuffix("_wind"))) for a in {x for c in CONFIG[panel]["planned"] for x in c}}


SENS = "--sens" in sys.argv
SETTING = "sensitivity" if SENS else "pooled"
TAG = f"{DEV}{'_sens' if SENS else ''}"
if sys.argv[1] != "analyse":
    panel, design = sys.argv[1], sys.argv[2]
    frame = design_frame(panel, design)
    cov = calendar_month_coverage(frame=frame)
    unc = uncovered_months(coverage=cov)
    info = {"panel": panel, "design": design, "rows": frame.height, "uncovered_hours": int(unc["n_scored"].sum()),
            "share": float(unc["n_scored"].sum()) / frame.height, "months": sorted(set(unc["calendar_month"].to_list())),
            "n_eras": frame["era_code"].n_unique(), "months_by_era": frame.group_by("era").agg(pl.col("month").n_unique()).to_dicts(),
            "search_offsets": [dict(o) for o in search_fold_offsets(frame=frame.drop("era_code", "era", "fold").pipe(lambda f: f), first_months=[wp.UKV_UPGRADE_MONTH])] if design == "D0" else None}
    print(json.dumps(info, default=str), flush=True)
    (OUT / f"panel_{panel}_{design}_{TAG}_info.json").write_text(json.dumps(info, default=str))
    feats = arms_features(panel)
    print({a: f for a, f in feats.items()}, flush=True)
    import os
    if os.environ.get("DRY"):
        raise SystemExit
    hp = wp.SENSITIVITY_HYPER_PARAMETERS if SENS else wp.PRIMARY_HYPER_PARAMETERS
    jobs_ = [(a, SETTING, "power_mw", f, hp, False) for a, f in feats.items()]
    losses = run_all(dataset=frame, jobs=jobs_, max_workers=4)
    losses.write_parquet(OUT / f"panel_{panel}_{design}_{TAG}.parquet")
    frame.select("site", "time", "month", "fold").write_parquet(OUT / f"panel_{panel}_{design}_folds.parquet")
    print("done", losses.height, flush=True)
    raise SystemExit

