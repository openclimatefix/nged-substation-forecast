"""Part B: fit the planned arms at the primary setting under one fold design.

Usage: partB_fit.py {wind,solar} {D0,D1,D2}
"""
import logging
import sys
import time
from common import *  # noqa: F403
import weather_products as wp
import wind_products as wd
from run_experiment import run_all

study, design = sys.argv[1], sys.argv[2]
setting = sys.argv[3] if len(sys.argv) > 3 else "pooled"
suffix = "" if setting == "pooled" else "_sens"
logging.basicConfig(level=logging.INFO)
if study == "wind":
    pre = wind_pre_fold()
    planned = {a for c in wd.REPORTED_CONTRASTS for a in c}
    all_jobs = wd.jobs()
    finish = lambda f: f.with_columns(constrained=pl.lit(False), cap_mw=pl.lit(None, dtype=pl.Float64)) if "cap_mw" not in f.columns else f
else:
    pre = solar_pre_fold()
    panel = wp.PANELS["long"]
    planned = {a for c in panel.planned for a in c}
    all_jobs = wp.jobs(products=panel.products, with_snapshot_arms=True) + wp.sensitivity_jobs(panel=panel)
    finish = finish_solar
jobs_ = [j for j in all_jobs if j[0] in planned and j[1] == setting]
print(study, design, sorted(j[0] for j in jobs_), flush=True)
for j in jobs_:
    print(j[0], j[3], flush=True)
frame = {"D0": d0, "D1": lambda f: d1(f, {0: 0, 1: 2}), "D2": d2, "D1r3": lambda f: d1(f, {0: 0, 1: 3})}[design](pre)
frame = finish(frame)
print("rows", frame.height, frame["fold"].value_counts().sort("fold").to_dicts(), flush=True)
t = time.time()
losses = run_all(dataset=frame, jobs=jobs_, max_workers=4)
losses = losses.with_columns(design=pl.lit(design), study=pl.lit(study))
losses.write_parquet(OUT / f"losses_{study}_{design}{suffix}.parquet")
frame.select("site", "time", "month", "fold", "era_code").write_parquet(OUT / f"folds_{study}_{design}.parquet")
print("done", losses.height, time.time() - t, flush=True)
